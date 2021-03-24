use std::sync::{Arc, Mutex};
use crate::demo_data::DemoData;
use std::thread;
use std::time::Duration;
use crate::constants::*;

pub fn learning_thread(data: Arc<Mutex<DemoData>>) {
    loop {
        for _ in 0..GRADIENT_STEPS_PER_UPDATE {
            let mut data = data.lock().unwrap();
            let (new_theta0, new_theta1) = gradient_descent_step(
                data.theta0, data.theta1, LEARNING_RATE, &data.points,
            );
            data.theta0 = new_theta0;
            data.theta1 = new_theta1;
        }

        thread::sleep(Duration::from_millis(GRADIENT_UPDATE_PERIOD as u64));
    }
}

fn gradient_descent_step(theta0: f64, theta1: f64, learning_rate: f64, points: &[(f64, f64)]) -> (f64, f64) {
    let mut diff0 = 0.0;
    let mut diff1 = 0.0;
    let f = |x: f64| theta0 + theta1 * x;

    for point in points {
        let point = *point;
        diff0 += f(point.0) - point.1;
        diff1 += (f(point.0) - point.1) * point.0;
    }

    diff0 /= points.len() as f64;
    diff1 /= points.len() as f64;

    diff0 *= learning_rate;
    diff1 *= learning_rate;

    (theta0 - diff0, theta1 - diff1)
}
