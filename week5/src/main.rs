#![feature(adt_const_params, generic_const_exprs)]

#[macro_use]
extern crate approx;

mod common;
mod gradient_descent;

fn main() {
    // let mut rng = rand::thread_rng();
    // let weights = gradient_descent::get_random_weights([2, 3, 2, 4], &mut rng);
    //
    // println!("{:?}", weights);

    // let (theta1, theta2, xs, ys) = common::load_demo_data();
    //
    // let (cost, _) = gradient_descent::get_cost_and_gradient::<3, 400, 10>(
    //     &[theta1, theta2],
    //     [HIDDEN_LAYER_SIZE],
    //     &xs,
    //     &ys,
    //     1.0,
    // );
    //
    // println!("cost is {}", cost);
    //
    // assert!(relative_eq!(cost, 0.383770, epsilon = 0.0001));

    let theta1 = vec![
        vec![0.0, 0.5, 0.2],
        vec![0.0, 0.3, 0.1],
        vec![0.0, 0.9, 0.11],
    ];

    let theta2 = vec![vec![0.0, 0.1, 0.2, 0.99], vec![0.0, 0.05, 0.5, 0.2]];

    let xs = [[0.1, 0.0], [0.2, 0.5], [1.0, 0.2]];

    let ys = [[false, true], [true, false], [false, true]];

    let (cost, gradient) =
        gradient_descent::get_cost_and_gradient::<3, 2, 2>(&[theta1, theta2], &xs, &ys, 1.0);

    println!("cost is {}", cost);
    for item in &gradient {
        println!("gradient is {:?}", item);
    }
    //
    // assert!(relative_eq!(cost, 1.5452, epsilon = 0.001));
}
