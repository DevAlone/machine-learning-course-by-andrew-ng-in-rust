use std::sync::{Arc, Mutex};
use crate::demo_data::DemoData;
use std::thread;
use std::time::Duration;
use crate::constants::*;

pub fn learning_thread(data: Arc<Mutex<DemoData>>) {
    loop {
        for _ in 0..GRADIENT_STEPS_PER_UPDATE {
            let mut data = data.lock().unwrap();
            data.theta = gradient_descent_step(
                &data.theta, LEARNING_RATE, &data.xs, &data.ys,
            );
        }

        thread::sleep(Duration::from_millis(GRADIENT_UPDATE_PERIOD as u64));
    }
}

// gradient_descent_step makes one step of gradient descent with provided parameters
fn gradient_descent_step(theta: &[f64], learning_rate: f64, xs: &[Vec<f64>], ys: &[f64]) -> Vec<f64> {
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
