use pyo3::prelude::*;
use ndarray::prelude::*;
use nalgebra::DVector;
use ndarray::{OwnedRepr};
use pyo3::exceptions::PyException;
use statrs::distribution::{Continuous, MultivariateNormal};
use statrs::StatsError;


/// Entry point for the Python module.
///
/// # Arguments
///
/// * `_py` - The Python interpreter.
/// * `m` - The module object to add functions to.
///
/// # Returns
///
/// PyResult indicating success or failure.
#[pymodule]
pub fn moebius(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // Add the Python function to the module
    m.add_function(wrap_pyfunction!(olr_wrapper, m)?)?;

    Ok(())
}

/// Calculates the Overlap Rate (OLR) values for a Gaussian mixture model.
///
/// # Arguments
///
/// * `w` - Vector of weights for each component.
/// * `means` - Array of means for each component.
/// * `covs` - Array of covariances for each component.
///
/// # Returns
///
/// Vector of OLR values.
///
/// # Errors
///
/// Returns a `StatsError` if there's an issue with the computation.
#[pyfunction()]
#[pyo3(name = "olr")]
pub fn olr_wrapper(w: Vec<f64>, means: Vec<Vec<f64>>, covs: Vec<Vec<Vec<f64>>>) -> PyResult<Vec<f64>> {
    olr(
        w,
        vec_to_array2(means),
        vec_to_array3(covs)
    ).map_err(|e| PyException::new_err(e.to_string()))
}

/// Converts a vector of vectors into a 2D array.
///
/// # Arguments
///
/// * `v` - A vector of vectors.
///
/// # Returns
///
/// A 2D array.
fn vec_to_array2<T: Clone>(v: Vec<Vec<T>>) -> Array2<T> {
    if v.is_empty() {
        return Array2::from_shape_vec((0, 0), Vec::new()).unwrap();
    }
    let nrows = v.len();
    let ncols = v[0].len();
    let mut data = Vec::with_capacity(nrows * ncols);
    for row in &v {
        data.extend_from_slice(&row);
    }
    Array2::from_shape_vec((nrows, ncols), data).unwrap()
}

/// Converts a vector of vectors of vectors into a 3D array.
///
/// # Arguments
///
/// * `v` - A vector of vectors of vectors.
///
/// # Returns
///
/// A 3D array.
fn vec_to_array3<T: Clone>(v: Vec<Vec<Vec<T>>>) -> Array3<T> {
    if v.is_empty() {
        return Array3::from_shape_vec((0, 0, 0), Vec::new()).unwrap();
    }
    let nrows = v.len();
    let ncols = v[0].len();
    let nitems = v[0][0].len();
    let mut data = Vec::with_capacity(nrows * ncols * nitems);
    for row in &v {
        for col in row {
            data.extend_from_slice(&col);
        }
    }

    Array3::from_shape_vec((nrows, ncols, nitems), data).unwrap()
}

/// Calculates the Overlap Rate (OLR) values for a Gaussian mixture model.
///
/// # Arguments
///
/// * `w` - Vector of weights for each component.
/// * `means` - Array of means for each component.
/// * `covs` - Array of covariances for each component.
///
/// # Returns
///
/// Vector of OLR values.
///
/// # Errors
///
/// Returns a `StatsError` if there's an issue with the computation.
pub fn olr(w: Vec<f64>, means: Array2<f64>, covs: Array3<f64>) -> Result<Vec<f64>, StatsError> {
    let n_comp = w.len();
    let mut olr_values = Vec::new();

    for i in 0..n_comp {
        for j in (i + 1)..n_comp {
            // Calculate means current components
            let means_slice_i = &means.slice(s![i, ..]).to_owned();
            let means_slice_j = &means.slice(s![j, ..]).to_owned();

            // Create points along the line between means
            let delta = (means_slice_j - means_slice_i) * 1.0 / 1000.0;
            let mut points = vec![means_slice_i - 10.0 * &delta];
            let mut curr_point: ArrayBase<OwnedRepr<f64>, Ix1> = means_slice_i - 10.0 * &delta;

            for _ in 0..1030 {
                let new_point: ArrayBase<OwnedRepr<f64>, Ix1> = &curr_point + &delta;
                curr_point = new_point.clone();
                points.push(new_point);
            }

            // Calculate weights, means, and covariances for the new components
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

            // Find peaks and saddles along the line
            for k in 1..1030 {
                let pdf_k = pdf_gmm(&points[k], &w_new, &m_new, &cov_new)?;
                let pdf_prev_k = pdf_gmm(&points[k - 1], &w_new, &m_new, &cov_new)?;
                let pdf_next_k = pdf_gmm(&points[k + 1], &w_new, &m_new, &cov_new)?;

                if ((pdf_k - pdf_prev_k) > 0.0) & ((pdf_k - pdf_next_k) > 0.0) {
                    peaks.push(pdf_k);
                }
                if ((pdf_k - pdf_prev_k) < 0.0) & ((pdf_k - pdf_next_k) < 0.0) {
                    saddles.push(pdf_k);
                }
            }

            // Calculate OLR for the current components
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

    Ok(olr_values)
}

/// Calculates the probability density function for a Gaussian mixture model at a given point.
///
/// # Arguments
///
/// * `x` - The value at which to evaluate the PDF.
/// * `w` - Vector of weights for each component.
/// * `means` - Vector of means for each component.
/// * `covs` - Vector of covariances for each component.
///
/// # Returns
///
/// The probability density function value at `x`.
///
/// # Errors
///
/// Returns a `StatsError` if there's an issue with the computation.
fn pdf_gmm(x: &Array1<f64>, w: &Vec<f64>, means: &Vec<&Array1<f64>>, covs: &Vec<&Array2<f64>>) -> Result<f64, StatsError> {
    let mut p = 0.0;

    for i in 0..w.len() {
        p += w[i] * pdf_mvn(x, means[i], covs[i])?;
    }

    Ok(p)
}

/// Calculates the probability density function for a multivariate normal distribution at a given point.
///
/// # Arguments
///
/// * `x` - The value at which to evaluate the PDF.
/// * `mean` - The mean of the multivariate normal distribution.
/// * `cov` - The covariance matrix of the multivariate normal distribution.
///
/// # Returns
///
/// The probability density function value at `x`.
///
/// # Errors
///
/// Returns a `StatsError` if there's an issue with the computation.
fn pdf_mvn(x: &Array1<f64>, mean: &Array1<f64>, cov: &Array2<f64>) -> Result<f64, StatsError> {
    let cov: Vec<f64> = cov.iter().map(|x| *x).collect();
    let mvn = MultivariateNormal::new(mean.to_vec(), cov.clone())?;

    Ok(mvn.pdf(&DVector::from_vec(x.to_vec())))
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::{arr2, arr3};
    use crate::olr;

    #[test]
    fn two_comps_two_dims() {
        let w = vec![5.2194e-01,  4.7806e-01];
        let means = arr2(&[
            [1.1987e+00, 1.1542e+00],
            [4.1592e+00, 4.1487e+00]
        ]);
        let covs = arr3(&[
            [
                [1.9455e+00, -9.1612e-04],
                [-9.1612e-04, 1.9703e+00]
            ],
            [
                [1.5160e+00, 1.1011e+00],
                [1.1011e+00, 1.5178e+00]
            ]
        ]);

        assert_abs_diff_eq!(0.9205257521646449, olr(w, means, covs).unwrap()[0], epsilon = 1e-4);
    }

    #[test]
    fn two_comps_one_dim() {
        let w = vec![0.5, 0.5];
        let means = arr2(&[
            [5.0],
            [2.0]
        ]);
        let covs = arr3(&[
            [
                [0.5]
            ],
            [
                [0.5]
            ]
        ]);

        assert_abs_diff_eq!(0.21077243773848037, olr(w, means, covs).unwrap()[0], epsilon = 1e-4)
    }

    #[test]
    fn three_comps_two_dims() {
        let w = vec![5.2194e-01,  4.7806e-01, 5.2194e-01];
        let means = arr2(&[
            [1.1987e+00, 1.1542e+00],
            [4.1592e+00, 4.1487e+00],
            [4.1592e+00, 4.1487e+00]
        ]);
        let covs = arr3(&[
            [
                [1.9455e+00, -9.1612e-04],
                [-9.1612e-04, 1.9703e+00]
            ],
            [
                [1.5160e+00, 1.1011e+00],
                [1.1011e+00, 1.5178e+00]
            ],
            [
                [1.5160e+00, 1.1009e+00],
                [1.1009e+00, 1.5178e+00]
            ]
        ]);

        let olrs = olr(w, means, covs).unwrap();

        assert_abs_diff_eq!(0.9205257521646449, olrs[0], epsilon = 1e-4);
        assert_abs_diff_eq!(0.9464977842655895, olrs[1], epsilon = 1e-4);
        assert_abs_diff_eq!(1.0, olrs[2], epsilon = 1e-4);
    }

    #[test]
    #[should_panic]
    fn singular_0() {
        let w = vec![0.2, 0.2];
        let means = arr2(&[
            [6f64],
            [11f64]
        ]);
        let covs = arr3(&[
            [
                [-0.006577556145946767]
            ],
            [
                [0.5448831829968969]
            ]
        ]);

        olr(w, means, covs).unwrap();
    }

    #[test]
    #[should_panic]
    fn singular_1() {
        let w = vec![0.22222222, 0.77777778];
        let means = arr2(&[
            [18.83333334, 18.16666668, 24.16666662, 41.83333333, 84.16666664, 44.16666665,
                         41.33333325, 69.33333339, 40.83333336],
            [42.57142856, 45.19047617, 47.95238095, 53.47619047, 49.28571431, 40.23809524,
                         52.00000002, 55.04761904, 43.28571428]
        ]);
        let covs = arr3(&[
            [[ 219.8055559 ,   -6.63888894,  -89.4722222 , -232.02777816,
                -132.47222233,  124.52777804,  150.05555611,  140.3888889 ,
                82.63888893],
                [  -6.63888894,  134.47222238,   41.3055559 ,  147.6944447 ,
                    52.80555583,  108.63888915,  225.27777864,  158.27777768,
                    234.86111134],
                [ -89.4722222 ,   41.3055559 ,  241.80555476,  168.86111134,
                    134.30555498,  -22.3611115 ,  287.61110946, -105.72222082,
                    49.52777857],
                [-232.02777816,  147.6944447 ,  168.86111134,  850.1388903 ,
                    359.36111168, -451.97222299, -149.94444479, -267.44444482,
                    301.97222276],
                [-132.47222233,   52.80555583,  134.30555498,  359.36111168,
                    266.80555547,  -34.19444473,  208.94444338, -283.22222164,
                    37.19444498],
                [ 124.52777804,  108.63888915,  -22.3611115 , -451.97222299,
                    -34.19444473,  843.13889019,  905.27777866,  264.94444535,
                    24.86111136],
                [ 150.05555611,  225.27777864,  287.61110946, -149.94444479,
                    208.94444338,  905.27777866, 1593.88888776,   98.2222252 ,
                    275.38889061],
                [ 140.3888889 ,  158.27777768, -105.72222082, -267.44444482,
                    -283.22222164,  264.94444535,   98.2222252 ,  682.55555461,
                    323.3888885 ],
                [  82.63888893,  234.86111134,   49.52777857,  301.97222276,
                    37.19444498,   24.86111136,  275.38889061,  323.3888885 ,
                    523.47222268]],

            [[ 963.38775501,  395.51020433, -135.73469385, -317.60544189,
                167.83673392,   69.81632634, -171.57142898, -372.40816281,
                -43.73469362],
                [ 395.51020433,  784.63038564,   17.53287976, -268.51927408,
                    392.65986292,  -38.56916117,  -87.8571434 ,  -69.29478421,
                    316.13605456],
                [-135.73469385,   17.53287976,  637.66439879,  111.54648519,
                    -20.510204  ,   57.48752833, -190.33333319,  -44.6643991 ,
                    127.63265297],
                [-317.60544189, -268.51927408,  111.54648519,  979.01133745,
                    52.14965958,   91.12471645,   20.52380933,  402.83446703,
                    90.91156467],
                [ 167.83673392,  392.65986292,  -20.510204  ,   52.14965958,
                    552.39455888, -128.87755066,   35.71428663,   29.74829878,
                    302.96598579],
                [  69.81632634,  -38.56916117,   57.48752833,   91.12471645,
                    -128.87755066,  938.84807218, -360.76190438,   49.13151913,
                    -229.02040816],
                [-171.57142898,  -87.8571434 , -190.33333319,   20.52380933,
                    35.71428663, -360.76190438,  774.66666695,  121.428571  ,
                    398.47618996],
                [-372.40816281,  -69.29478421,  -44.6643991 ,  402.83446703,
                    29.74829878,   49.13151913,  121.428571  ,  668.14058946,
                    -5.63265288],
                [ -43.73469362,  316.13605456,  127.63265297,   90.91156467,
                    302.96598579, -229.02040816,  398.47618996,   -5.63265288,
                    835.06122425]]
        ]);

        olr(w, means, covs).unwrap();
    }
}
