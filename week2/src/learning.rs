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
                data.theta = gradient_descent_step(&data.theta, LEARNING_RATE, &data.xs, &data.ys);
            }
        }

        thread::sleep(Duration::from_millis(GRADIENT_UPDATE_PERIOD as u64));
    }
}

// gradient_descent_step makes one step of gradient descent with provided parameters
fn gradient_descent_step(
    theta: &[f64],
    learning_rate: f64,
    xs: &[Vec<f64>],
    ys: &[f64],
) -> Vec<f64> {
    assert!(theta.len() > 0);
    assert!(xs.len() > 0);
    assert!(ys.len() > 0);
    assert!(xs[0].len() > 0);
    assert_eq!(xs.len(), ys.len());

    assert_eq!(theta.len(), xs[0].len() + 1);

    let f = |x: &[f64]| {
        let mut result = 0.0;

        for (theta_i, theta_val) in theta.iter().enumerate() {
            result += theta_val * if theta_i > 0 { x[theta_i - 1] } else { 1.0 };
        }

        result
    };

    let mut diff = vec![0.0; theta.len()];

    for theta_i in 0..theta.len() {
        for (point_i, x) in xs.iter().enumerate() {
            let y = ys[point_i];

            diff[theta_i] += (f(x) - y) * if theta_i > 0 { x[theta_i - 1] } else { 1.0 };
        }

        // divide by number of points
        diff[theta_i] /= ys.len() as f64;
        diff[theta_i] *= learning_rate;
    }

    let mut result = Vec::with_capacity(theta.len());

    for (theta_i, theta_val) in theta.iter().enumerate() {
        result.push(theta_val - diff[theta_i]);
    }

    result
}

type MatrixFDynamic = Matrix<f64, Dynamic, Dynamic, VecStorage<f64, Dynamic, Dynamic>>;

// normal_equation_solve finds theta using normal equation
// the formula is `(x_transpose * x)_inverse * x_transpose * y`
fn normal_equation_solve(xs: &[Vec<f64>], ys: &[f64]) -> Vec<f64> {
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

    result.clone()
}
