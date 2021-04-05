use crate::constants::*;
use crate::demo_data::DemoDataNFeatures;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

pub fn learning_thread<const N_FEATURES: usize>(data: Arc<Mutex<DemoDataNFeatures<N_FEATURES>>>)
    where [(); N_FEATURES + 1]:
{
    loop {
        for _ in 0..GRADIENT_STEPS_PER_UPDATE {
            let mut data = data.lock().unwrap();
            data.theta = helpers::math::gradient_descent_step(
                data.theta,
                LEARNING_RATE,
                &data.xs,
                &data.ys,
                |x: [f64; N_FEATURES]| {
                    helpers::math::logistic_regression_predict(data.theta, x)
                },
            );
        }

        thread::sleep(Duration::from_millis(GRADIENT_UPDATE_PERIOD as u64));
    }
}
