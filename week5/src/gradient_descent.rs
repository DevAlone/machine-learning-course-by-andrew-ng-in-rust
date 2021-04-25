use nalgebra::DMatrix;
use rand::prelude::ThreadRng;
use rand::Rng;
use std::convert::TryInto;

// get_cost_and_gradient takes thetas, xs, ys and regularization rate and calculates the cost
// and the gradient
// N_LAYERS is the total number of layers(input + hidden + output layers)
// INPUT_LAYER_SIZE is the number of neurons in the input layer
// OUTPUT_LAYER_SIZE is the number of neurons in the output layer
pub fn get_cost_and_gradient<
    const N_LAYERS: usize,
    const INPUT_LAYER_SIZE: usize,
    const OUTPUT_LAYER_SIZE: usize,
>(
    thetas: &[Vec<Vec<f64>>; N_LAYERS - 1],
    xs: &[[f64; INPUT_LAYER_SIZE]],
    ys: &[[bool; OUTPUT_LAYER_SIZE]],
    regularization_rate: f64,
) -> (f64, [Vec<Vec<f64>>; N_LAYERS - 1]) {
    // TODO: rewrite using nalgebra instead of raw arrays

    let number_of_examples = xs.len();
    let mut cost = 0.0;
    let mut theta_gradients: Vec<DMatrix<f64>> = Vec::new();
    for theta in thetas.iter() {
        theta_gradients.push(DMatrix::from_element(theta.len(), theta[0].len(), 0.0));
    }

    for (example_number, x) in xs.iter().enumerate() {
        let mut activations_sigmoid_gradient: Vec<Vec<f64>> = Vec::new();
        let mut activations: Vec<Vec<_>> = std::iter::repeat(vec![]).take(N_LAYERS).collect();
        activations[0] = x.to_vec();
        activations[0].insert(0, 1.0);

        for i in 1..activations.len() {
            let previous_activation = &activations[i - 1];
            let previous_activation =
                DMatrix::from_row_slice(previous_activation.len(), 1, previous_activation);

            let theta = vec_of_vec_to_matrix(&thetas[i - 1]);

            let mut current_activation: DMatrix<f64> = theta * previous_activation;
            let activation_sigmoid_gradient: Vec<f64> =
                current_activation.clone().insert_row(0, 1.0).data.into();
            let activation_sigmoid_gradient = sigmoid_gradient(&activation_sigmoid_gradient);

            activations_sigmoid_gradient.push(activation_sigmoid_gradient);

            current_activation.apply(|x| sigmoid(x));
            current_activation = current_activation.insert_row(0, 1.0);
            activations[i] = current_activation.data.into();
        }

        let output_layer_activations = &activations[N_LAYERS - 1];

        let y = ys[example_number];

        for (i, y) in y.iter().enumerate() {
            let y = *y;

            // ignore the bias unit so +1
            let activation = output_layer_activations[i + 1];
            let val = if y { activation } else { 1.0 - activation };
            assert_ne!(val, 0.0, "can't be zero!");

            cost -= val.ln();
        }

        let mut errors: Vec<Vec<_>> = std::iter::repeat(vec![]).take(N_LAYERS - 1).collect();
        let last_err_index = errors.len() - 1;
        errors[last_err_index] = output_layer_activations.clone();

        let last_layer_error = &mut errors[last_err_index];
        last_layer_error.remove(0);
        for (i, y) in y.iter().enumerate() {
            let y = *y;

            last_layer_error[i] -= if y { 1.0 } else { 0.0 };
        }

        for i in (0..errors.len() - 1).rev() {
            let next_layer_error = &errors[i + 1];
            let next_layer_error: DMatrix<f64> =
                DMatrix::from_row_slice(next_layer_error.len(), 1, next_layer_error);

            let curr_theta = vec_of_vec_to_matrix(&thetas[i + 1]);

            let sigmoid_prev_theta_mul_x = &activations_sigmoid_gradient[i];
            let sigmoid_prev_theta_mul_x = DMatrix::from_row_slice(
                sigmoid_prev_theta_mul_x.len(),
                1,
                sigmoid_prev_theta_mul_x,
            );
            let current_error: DMatrix<f64> = (curr_theta.transpose() * next_layer_error)
                .component_mul(&sigmoid_prev_theta_mul_x)
                .remove_row(0);

            errors[i] = current_error.data.into();
        }

        for (i, error) in errors.iter().enumerate() {
            let activation = &activations[i];
            theta_gradients[i] += DMatrix::from_row_slice(error.len(), 1, &error)
                * DMatrix::from_row_slice(activation.len(), 1, &activation).transpose();
        }
    }

    cost /= number_of_examples as f64;
    for theta_gradient in theta_gradients.iter_mut() {
        *theta_gradient /= number_of_examples as f64;
    }

    let mut regularization_penalty = 0.0;

    for theta in thetas {
        for theta_row in theta {
            // skip the first item which is bias
            for weight in theta_row.iter().skip(1) {
                let weight = *weight;
                regularization_penalty += weight * weight;
            }
        }
    }

    regularization_penalty =
        (regularization_rate * regularization_penalty) / (2.0 * number_of_examples as f64);

    cost += regularization_penalty;

    // regularize theta_gradients
    for (i, theta_gradient) in theta_gradients.iter_mut().enumerate() {
        let theta = vec_of_vec_to_matrix(&thetas[i]);
        let theta_slice = theta.slice((0, 1), (theta.nrows(), theta.ncols() - 1));
        let theta_slice = theta_slice.insert_column(0, 0.0);
        *theta_gradient += (regularization_rate / number_of_examples as f64) * theta_slice;
    }

    let mut theta_gradients_vec = Vec::new();
    for theta_gradient in &theta_gradients {
        theta_gradients_vec.push(matrix_to_vec_of_vec(theta_gradient));
    }

    (cost, theta_gradients_vec.try_into().unwrap())
}

pub fn sigmoid(v: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-v))
}

pub fn sigmoid_gradient(v: &[f64]) -> Vec<f64> {
    v.iter()
        .map(|x| {
            let s = sigmoid(*x);
            s * (1.0 - s)
        })
        .collect()
}

pub fn get_random_weights<const N_LAYERS: usize>(
    layers_sizes: [usize; N_LAYERS],
    rng: &mut ThreadRng,
) -> [Vec<Vec<f64>>; N_LAYERS - 1] {
    let mut result: Vec<Vec<Vec<f64>>> = std::iter::repeat(Vec::new()).take(N_LAYERS - 1).collect();

    for curr_layer_index in 1..N_LAYERS {
        let curr_layer_size = layers_sizes[curr_layer_index];
        let prev_layer_size = layers_sizes[curr_layer_index - 1];

        let mut weights = Vec::new();

        for _ in 0..curr_layer_size {
            let mut row = Vec::new();
            for _ in 0..prev_layer_size {
                row.push(rng.gen_range(-1.0..1.0));
            }
            weights.push(row);
        }

        result[curr_layer_index - 1] = weights;
    }

    result.try_into().unwrap()
}

fn vec_of_vec_to_matrix(data: &Vec<Vec<f64>>) -> DMatrix<f64> {
    let n_rows = data.len();
    let n_columns = data[0].len();
    let data_flat: Vec<f64> = data.iter().flatten().map(|x| *x).collect();
    return DMatrix::from_row_slice(n_rows, n_columns, &data_flat);
}

fn matrix_to_vec_of_vec(matrix: &DMatrix<f64>) -> Vec<Vec<f64>> {
    let mut result = Vec::new();

    for row_index in 0..matrix.nrows() {
        let mut row = Vec::new();
        for column_index in 0..matrix.ncols() {
            row.push(matrix[(row_index, column_index)]);
        }
        result.push(row);
    }

    result
}

#[cfg(test)]
mod tests {
    use crate::common;
    use crate::gradient_descent::{get_cost_and_gradient, sigmoid_gradient};

    #[test]
    fn test_cost_function() {
        let expected_theta_gradients = common::load_expected_theta_gradients();

        let (theta1, theta2, xs, ys) = common::load_demo_data();
        let theta = [theta1, theta2];

        let (cost, grad) = get_cost_and_gradient::<3, 400, 10>(&theta, &xs, &ys, 0.0);

        assert!(relative_eq!(cost, 0.287629, epsilon = 0.0001));
        assert_relative_eq_gradients(&grad, &expected_theta_gradients);

        // with regularization
        let (cost, _) = get_cost_and_gradient::<3, 400, 10>(&theta, &xs, &ys, 1.0);

        assert!(relative_eq!(cost, 0.383770, epsilon = 0.0001));
    }

    fn assert_relative_eq_gradients<const N: usize>(
        actual: &[Vec<Vec<f64>>; N],
        expected: &[Vec<Vec<f64>>; N],
    ) {
        for i in 0..N {
            let actual_theta_grad = &actual[i];
            let expected_theta_grad = &expected[i];

            assert_eq!(actual_theta_grad.len(), expected_theta_grad.len());

            for (row1, row2) in actual_theta_grad.iter().zip(expected_theta_grad) {
                assert_eq!(row1.len(), row2.len());

                for (weight1, weight2) in row1.iter().zip(row2) {
                    let is_equal = relative_eq!(weight1, weight2, epsilon = 0.0001);
                    if !is_equal {
                        println!("left = {}, right = {}", weight1, weight2);
                        assert!(false);
                    }
                }
            }
        }
    }

    #[test]
    fn test_nn_1() {
        // just some random NN

        let theta1 = vec![
            vec![0.0, 0.5, 0.2],
            vec![0.0, 0.3, 0.1],
            vec![0.0, 0.9, 0.11],
        ];
        let theta2 = vec![vec![0.0, 0.1, 0.2, 0.99], vec![0.0, 0.05, 0.5, 0.2]];
        let thetas = [theta1, theta2];

        let xs = [[0.1, 0.0], [0.2, 0.5], [1.0, 0.2]];
        let ys = [[false, true], [true, false], [false, true]];

        let (cost, gradient) = get_cost_and_gradient::<3, 2, 2>(&thetas, &xs, &ys, 0.0);

        assert!(relative_eq!(cost, 1.5452, epsilon = 0.001));

        assert_relative_eq_gradients(
            &gradient,
            &[
                vec![
                    vec![0.007577493, 0.004330289, 0.000678956],
                    vec![0.009409984, -0.000866898, 0.008883779],
                    vec![0.073167619, 0.043740677, 0.000016839],
                ],
                vec![
                    vec![0.346998, 0.201915, 0.190871, 0.222917],
                    vec![-0.063920, -0.040649, -0.037689, -0.051031],
                ],
            ],
        );

        let (cost, gradient) = get_cost_and_gradient::<3, 2, 2>(&thetas, &xs, &ys, 1.0);

        assert!(relative_eq!(cost, 1.967837, epsilon = 0.001));

        assert_relative_eq_gradients(
            &gradient,
            &[
                vec![
                    vec![0.0075775, 0.1709970, 0.0673456],
                    vec![0.0094100, 0.0991331, 0.0422171],
                    vec![0.0731676, 0.3437407, 0.0366835],
                ],
                vec![
                    vec![0.346998, 0.235248, 0.257537, 0.552917],
                    vec![-0.063920, -0.023982, 0.128977, 0.015636],
                ],
            ],
        );
    }

    #[test]
    fn test_sigmoid_gradient() {
        let gradient = sigmoid_gradient(&[-1.0, -0.5, 0.0, 0.5, 1.0]);
        let expected_gradient = [0.196612, 0.235004, 0.250000, 0.235004, 0.196612];
        assert_eq!(gradient.len(), expected_gradient.len());
        for (got, expected) in gradient.iter().zip(&expected_gradient) {
            assert!(relative_eq!(*got, *expected, epsilon = 0.001));
        }
    }
}
