use pyo3::prelude::*;
use ndarray::prelude::*;
use nalgebra::{Vector2, Matrix2, OMatrix, U1, U2};
use nalgebra_mvn::MultivariateNormal;
use ndarray::{OwnedRepr};

#[pymodule]
pub fn moebius(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(olr_wrapper, m)?)?;

    Ok(())
}

#[pyfunction()]
#[pyo3(name = "olr")]
pub fn olr_wrapper(w: Vec<f64>, means: Vec<Vec<f64>>, covs: Vec<Vec<Vec<f64>>>) -> PyResult<Vec<f64>> {
    Ok(
        olr(
            w,
            vec_to_array2(means),
            vec_to_array3(covs)
        )
    )
}

fn vec_to_array2<T: Clone>(v: Vec<Vec<T>>) -> Array2<T> {
    if v.is_empty() {
        return Array2::from_shape_vec((0, 0), Vec::new()).unwrap();
    }
    let nrows = v.len();
    let ncols = v[0].len();
    let mut data = Vec::with_capacity(nrows * ncols);
    for row in &v {
        assert_eq!(row.len(), ncols);
        data.extend_from_slice(&row);
    }
    Array2::from_shape_vec((nrows, ncols), data).unwrap()
}

fn vec_to_array3<T: Clone>(v: Vec<Vec<Vec<T>>>) -> Array3<T> {
    if v.is_empty() {
        return Array3::from_shape_vec((0, 0, 0), Vec::new()).unwrap();
    }
    let nrows = v.len();
    let ncols = v[0].len();
    let nitems = v[0][0].len();
    let mut data = Vec::with_capacity(nrows * ncols * nitems);
    for row in &v {
        assert_eq!(row.len(), ncols);

        for items in row {
            assert_eq!(items.len(), nrows);
            data.extend_from_slice(&items);
        }
    }

    Array3::from_shape_vec((nrows, ncols, nitems), data).unwrap()
}

pub fn olr(w: Vec<f64>, means: Array2<f64>, covs: Array3<f64>) -> Vec<f64> {
    let n_comp = w.len();
    let mut olr_values = Vec::new();

    for i in 0..n_comp {
        for j in (i + 1)..n_comp {
            let means_slice_i = &means.slice(s![i, ..]).to_owned();
            let means_slice_j = &means.slice(s![j, ..]).to_owned();

            let delta = (means_slice_j - means_slice_i) * 1.0 / 1000.0;
            let mut points = vec![means_slice_i - 10.0 * &delta];
            let mut curr_point: ArrayBase<OwnedRepr<f64>, Ix1> = means_slice_i - 10.0 * &delta;

            for _ in 0..1030 {
                let new_point: ArrayBase<OwnedRepr<f64>, Ix1> = &curr_point + &delta;
                curr_point = new_point.clone();
                points.push(new_point);
            }

            let w1 = w[i];
            let w2 = w[j];
            let w1_new = w1 / (w1 + w2);
            let w2_new = 1.0 - w1_new;

            let w_new = vec![w1_new, w2_new];
            let m_new = vec![means_slice_i, means_slice_j];

            let covs_slice_i = &covs.slice(s![i, .., ..]).to_owned();
            let covs_slice_j = &covs.slice(s![j, .., ..]).to_owned();

            let cov_new = vec![covs_slice_i, covs_slice_j];
            let mut peaks = Vec::<f64>::new();
            let mut saddles = Vec::<f64>::new();

            for k in 1..1030 {
                let pdf_k = pdf_gmm(&points[k], &w_new, &m_new, &cov_new);
                let pdf_prev_k = pdf_gmm(&points[k - 1], &w_new, &m_new, &cov_new);
                let pdf_next_k = pdf_gmm(&points[k + 1], &w_new, &m_new, &cov_new);

                if ((pdf_k - pdf_prev_k) > 0.0) & ((pdf_k - pdf_next_k) > 0.0) {
                    peaks.push(pdf_k);
                }
                if ((pdf_k - pdf_prev_k) < 0.0) & ((pdf_k - pdf_next_k) < 0.0) {
                    saddles.push(pdf_k);
                }
            }

            let olr_current;
            if peaks.len() == 1 {
                olr_current = 1.0;
            } else {
                if saddles.len() == 0 {
                    olr_current = 1.0;
                } else {
                    olr_current = saddles[0] / peaks.into_iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
                }
            }

            olr_values.push(olr_current);
        }
    }

    return olr_values
}

fn pdf_gmm(x: &Array1<f64>, w: &Vec<f64>, means: &Vec<&Array1<f64>>, covs: &Vec<&Array2<f64>>) -> f64 {
    let mut p = 0.0;

    for i in 0..w.len() {
        p += w[i] * pdf_mvn(x, means[i], covs[i]);
    }

    p
}

fn pdf_mvn(x: &Array1<f64>, mean: &Array1<f64>, cov: &Array2<f64>) -> f64 {
    let mu = Vector2::from_column_slice(mean.as_slice().unwrap());
    let sigma = Matrix2::from_column_slice(cov.as_slice().unwrap());

    let mvn = MultivariateNormal::from_mean_and_covariance(&mu, &sigma).unwrap();

    let xs = OMatrix::<_,U1,U2>::new(
        x[0], x[1]
    );

    mvn.pdf(&xs).as_slice()[0]
}
