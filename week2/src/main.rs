#![feature(adt_const_params, generic_const_exprs)]
#![allow(incomplete_features)]

mod app_data;
mod constants;
mod demo_data;
mod learning;
mod visualizer_2_features;

use crate::app_data::AppData;
use crate::constants::*;
use crate::demo_data::DemoData;
use crate::learning::learning_thread;
use druid::widget::*;
use druid::*;
use rand;
use rand::Rng;
use std::sync::{Arc, Mutex};
use std::thread;
use visualizer_2_features::Visualizer2Features;
use helpers::canvas::Canvas;
use std::time::Duration;

fn main() {
    let mut xs = Vec::new();
    let mut ys = Vec::new();

    let mut rng = rand::thread_rng();
    // generate a ray with a small random shifts for each point
    for i in 0..DEFAULT_NUMBER_OF_POINTS {
        let i = i as f64 / DEFAULT_NUMBER_OF_POINTS as f64 * MAX_VALUE;
        let mut r = || rng.gen_range(-MAX_VALUE / 10.0..MAX_VALUE / 10.0);

        xs.push([i + r(), i + r()]);
        ys.push(i + r());
    }

    let data = Arc::new(Mutex::new(DemoData {
        xs,
        ys,
        theta: [
            DEFAULT_THETA_VALUE,
            DEFAULT_THETA_VALUE,
            DEFAULT_THETA_VALUE,
        ],
    }));

    let app_data = AppData {
        new_point_x: String::new(),
        new_point_z: String::new(),
        new_point_y: String::new(),
    };

    let thread_data = data.clone();
    thread::spawn(move || learning_thread(thread_data));

    let window = WindowDesc::new(get_ui_builder(data))
        .window_size(Size::new(WINDOW_WIDTH, WINDOW_HEIGHT))
        .title(
            LocalizedString::new("Linear Regression Demo")
                .with_placeholder("linear-regression-demo"),
        );

    AppLauncher::with_window(window)
        .use_simple_logger()
        .launch(app_data)
        .expect("launch failed");
}

fn get_ui_builder(data: Arc<Mutex<DemoData>>) -> impl Fn() -> Flex<AppData> {
    move || {
        let data_copy = data.clone();
        Flex::<AppData>::column()
            .with_child(Canvas::new(
                Duration::from_millis(REFRESH_PERIOD),
                Box::new(Visualizer2Features::new(data.clone())),
            ))
            .with_child(
                Flex::row()
                    .with_child(
                        TextBox::new()
                            .with_placeholder("x")
                            .lens(AppData::new_point_x),
                    )
                    .with_child(
                        TextBox::new()
                            .with_placeholder("z")
                            .lens(AppData::new_point_z),
                    )
                    .with_child(
                        TextBox::new()
                            .with_placeholder("y")
                            .lens(AppData::new_point_y),
                    )
                    .with_child(Button::new("Add Point").on_click(
                        move |_ctx: &mut EventCtx, app_data: &mut AppData, _env: &Env| {
                            match app_data.parse_new_point() {
                                Ok(new_point) => {
                                    data_copy.lock().unwrap().add_point(new_point);
                                }
                                Err(_) => println!("invalid point"),
                            }
                        },
                    )),
            )
    }
}
