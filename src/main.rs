use std::time::Instant;
use ndarray::prelude::*;
use ndarray::{arr3};

fn main() {
    // let w = vec![5.2194e-01,  4.7806e-01];
    // let means = arr2(&[
    //     [1.1987e+00, 1.1542e+00],
    //     [4.1592e+00, 4.1487e+00]
    // ]);
    // let covs = arr3(&[
    //     [
    //         [1.9455e+00, 1.5219e-04],
    //         [-9.1612e-04, 1.9703e+00]
    //     ],
    //     [
    //         [1.5160e+00, 1.1009e+00],
    //         [1.1011e+00, 1.5178e+00]
    //     ]
    // ]);

    let w = vec![5.2194e-01,  4.7806e-01, 5.2194e-01];
    let means = vec![
        vec![1.1987e+00, 1.1542e+00],
        vec![4.1592e+00, 4.1487e+00],
        vec![4.1592e+00, 4.1487e+00]
    ];
    let covs = vec![
        vec![
            vec![1.9455e+00, 1.5219e-04],
            vec![-9.1612e-04, 1.9703e+00]
        ],
        vec![
            vec![1.5160e+00, 1.1009e+00],
            vec![1.1011e+00, 1.5178e+00]
        ],
        vec![
            vec![1.5160e+00, 1.1009e+00],
            vec![1.1011e+00, 1.5178e+00]
        ]
    ];

    let before = Instant::now();
    let olr = moebius::olr_wrapper(w, means, covs);
    println!("Elapsed time: {:.2?}", before.elapsed());

    println!("{:?}", olr);
}

// fn olr_from_plain(x: &[f64]) -> f64 {
//     olr(vec![x[0],  x[1]],  arr2(&[[x[2],  x[3]],  [x[4],
//         x[5]]]),  arr3(&[[[x[6], x[7]],  [x[8],  x[9]]],
//             [[x[10],  x[11]],  [x[12], x[13]]]]))[0]
// }
