use crate::constants::*;
use crate::demo_data::DemoData;
use nalgebra::{Dynamic, Matrix, VecStorage};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

pub fn learning_thread(data: Arc<Mutex<DemoData>>) {
    loop {
        if USE_NORMAL_EQUATION {
            let mut data = data.lock().unwrap();
            data.theta = normal_equation_solve(&data.xs, &data.ys);
        } else {
            for _ in 0..GRADIENT_STEPS_PER_UPDATE {
                let mut data = data.lock().unwrap();
                data.theta = helpers::math::gradient_descent_step(
                    data.theta, LEARNING_RATE, &data.xs, &data.ys, |x: [f64; 2]| {
                        let mut result = 0.0;

                        for (theta_i, theta_val) in data.theta.iter().enumerate() {
                            result += theta_val * if theta_i > 0 { x[theta_i - 1] } else { 1.0 };
                        }

                        result
                    });
            }
        }

        thread::sleep(Duration::from_millis(GRADIENT_UPDATE_PERIOD as u64));
    }
}

type MatrixFDynamic = Matrix<f64, Dynamic, Dynamic, VecStorage<f64, Dynamic, Dynamic>>;

// normal_equation_solve finds theta using normal equation
// the formula is `(x_transpose * x)_inverse * x_transpose * y`
fn normal_equation_solve<const N_FEATURES: usize>(
    xs: &[[f64; N_FEATURES]], ys: &[f64],
) -> [f64; N_FEATURES + 1] {
    let n_rows = xs.len();
    let n_columns = xs[0].len();

    let x = MatrixFDynamic::from_iterator(
        n_columns + 1,
        n_rows,
        xs.iter()
            .map(|x| {
                // TODO: refactor this piece of shit to be in functional style
                let mut res = vec![1.0];
                for element in x {
                    res.push(*element);
                }
                res
            })
            .flatten(),
    )
        .transpose();
    let y = MatrixFDynamic::from_iterator(ys.len(), 1, ys.iter().map(|x| *x));
    let x_transpose = x.transpose();

    let result = (x_transpose.clone() * x).pseudo_inverse(0.0).unwrap() * x_transpose * y;
    let result = result.data.as_vec();
    let mut result_array = [Default::default(); N_FEATURES + 1];

    for (i, val) in result.iter().enumerate() {
        result_array[i] = *val;
    }

    result_array
}
