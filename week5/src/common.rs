use matfile::{MatFile, NumericData};
use std::error::Error;
use std::fs::File;
use std::path::Path;

pub const INPUT_LAYER_SIZE: usize = 400;
pub const OUTPUT_LAYER_SIZE: usize = 10;

pub fn load_demo_data() -> (
    Vec<Vec<f64>>,
    Vec<Vec<f64>>,
    Vec<[f64; INPUT_LAYER_SIZE]>,
    Vec<[bool; OUTPUT_LAYER_SIZE]>,
) {
    load_demo_data_impl().unwrap()
}

fn load_demo_data_impl() -> Result<
    (
        Vec<Vec<f64>>,
        Vec<Vec<f64>>,
        Vec<[f64; INPUT_LAYER_SIZE]>,
        Vec<[bool; OUTPUT_LAYER_SIZE]>,
    ),
    Box<dyn Error>,
> {
    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("resources/test");

    let ex4_weights_file = data_dir.join("ex4weights.mat");
    let ex4_weights = MatFile::parse(File::open(ex4_weights_file)?)?;
    let theta1 = convert_theta(&get_array_from_mat_file::<401>(&ex4_weights, "Theta1"));
    let theta2 = convert_theta(&get_array_from_mat_file::<26>(&ex4_weights, "Theta2"));

    let ex4data1 = MatFile::parse(File::open(data_dir.join("ex4data1.mat"))?)?;
    let xs = get_array_from_mat_file::<400>(&ex4data1, "X");
    let ys = convert_ys::<10>(&get_array_from_mat_file::<1>(&ex4data1, "y"));

    Ok((theta1, theta2, xs, ys))
}

pub fn load_expected_theta_gradients() -> [Vec<Vec<f64>>; 2] {
    load_expected_theta_gradients_impl().unwrap()
}

fn load_expected_theta_gradients_impl() -> Result<[Vec<Vec<f64>>; 2], Box<dyn Error>> {
    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("resources/test");

    let ex4_weights_file = data_dir.join("Theta_grad.mat");
    let ex4_weights = MatFile::parse(File::open(ex4_weights_file)?)?;
    let theta1_grad = convert_theta(&get_array_from_mat_file::<401>(&ex4_weights, "Theta1_grad"));
    let theta2_grad = convert_theta(&get_array_from_mat_file::<26>(&ex4_weights, "Theta2_grad"));

    return Ok([theta1_grad, theta2_grad]);
}

fn get_array_from_mat_file<const N_COLUMNS: usize>(
    mat_file: &MatFile,
    array_name: &str,
) -> Vec<[f64; N_COLUMNS]> {
    let array = mat_file.find_by_name(array_name).unwrap().data().clone();

    if let NumericData::Double { real, .. } = array {
        let mut res: Vec<[f64; N_COLUMNS]> = Vec::new();

        let n_rows = real.len() / N_COLUMNS;
        for r in 0..n_rows {
            let mut row = [0.0; N_COLUMNS];
            for c in 0..N_COLUMNS {
                // the data is written column by column(for some reason)
                row[c] = real[c * n_rows + r];
            }
            res.push(row);
        }

        return res;
    }

    panic!("array has different type: {:?}", array);
}

fn convert_ys<const N: usize>(ys: &[[f64; 1]]) -> Vec<[bool; N]> {
    ys.iter()
        .map(|y| {
            // let x = if x[0] < 10.0 { x[0] } else { 0.0 };
            let y = y[0] as usize - 1;
            let mut item = [false; N];
            item[y] = true;
            item
        })
        .collect()
}

fn convert_theta<const N: usize>(theta: &[[f64; N]]) -> Vec<Vec<f64>> {
    theta.iter().map(|x| x.to_vec()).collect()
}
