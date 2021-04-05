pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + std::f64::consts::E.powf(-x))
}

pub fn logistic_regression_predict<const N_FEATURES: usize>(
    theta: [f64; N_FEATURES + 1],
    x: [f64; N_FEATURES],
) -> f64 {
    let mut result = 0.0;

    for (theta_i, theta_val) in theta.iter().enumerate() {
        result += theta_val * if theta_i > 0 { x[theta_i - 1] } else { 1.0 };
    }

    sigmoid(result)
}

// gradient_descent_step makes one step of gradient descent with provided parameters
pub fn gradient_descent_step<const N_FEATURES: usize, F: Fn([f64; N_FEATURES]) -> f64>(
    theta: [f64; N_FEATURES + 1],
    learning_rate: f64,
    xs: &[[f64; N_FEATURES]],
    ys: &[f64],
    cost_function: F,
) -> [f64; N_FEATURES + 1] {
    assert!(xs.len() > 0);
    assert!(ys.len() > 0);
    assert_eq!(xs.len(), ys.len());

    let mut diff = [0.0; N_FEATURES + 1];

    for theta_i in 0..theta.len() {
        for (point_i, x) in xs.iter().enumerate() {
            let y = ys[point_i];

            diff[theta_i] += (cost_function(*x) - y) * if theta_i > 0 { x[theta_i - 1] } else { 1.0 };
        }

        // divide by number of points
        diff[theta_i] /= ys.len() as f64;
        diff[theta_i] *= learning_rate;
    }

    let mut result = [0.0; N_FEATURES + 1];

    for (theta_i, theta_val) in theta.iter().enumerate() {
        result[theta_i] = theta_val - diff[theta_i];
    }

    result
}
