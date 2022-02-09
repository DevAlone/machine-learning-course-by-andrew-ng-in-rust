#![feature(adt_const_params, generic_const_exprs)]
#![allow(incomplete_features)]

mod app_data;
mod constants;
mod demo_data;
mod learning;
mod logistic_regression_visualizer_1_feature;
mod logistic_regression_visualizer_2_features;

use crate::app_data::AppData;
use crate::constants::*;
use crate::demo_data::{DemoData1Feature, DemoData2Features};
use crate::learning::learning_thread;
use druid::widget::*;
use druid::*;
use std::sync::{Arc, Mutex};
use std::thread;
use logistic_regression_visualizer_1_feature::Visualizer1Feature;
use helpers::canvas::Canvas;
use std::time::Duration;
use crate::logistic_regression_visualizer_2_features::Visualizer2Features;

fn main() {
    let data_1_feature = Arc::new(Mutex::new(DemoData1Feature {
        xs: vec![[1.0], [2.0], [3.0], [7.0], [8.0], [9.0]],
        ys: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        theta: [DEFAULT_THETA_VALUE, DEFAULT_THETA_VALUE],
    }));
    let data_2_features = Arc::new(Mutex::new(DemoData2Features {
        xs: vec![[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [7.0, 7.0], [8.0, 8.0], [9.0, 9.0]],
        ys: vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        theta: [DEFAULT_THETA_VALUE, DEFAULT_THETA_VALUE, DEFAULT_THETA_VALUE],
    }));

    let app_data = AppData {
        one_feature_new_point_x: String::new(),
        one_feature_new_point_y: String::new(),
        two_features_new_point_x: String::new(),
        two_features_new_point_z: String::new(),
        two_features_new_point_y: String::new(),
    };

    let thread_data_1_feature = data_1_feature.clone();
    thread::spawn(move || learning_thread(thread_data_1_feature));

    let thread_data_2_features = data_2_features.clone();
    thread::spawn(move || learning_thread(thread_data_2_features));

    let window = WindowDesc::new(get_ui_builder(data_1_feature, data_2_features))
        .window_size(Size::new(WINDOW_WIDTH, WINDOW_HEIGHT))
        .title(
            LocalizedString::new("Logistic Regression Demo")
                .with_placeholder("logistic-regression-demo"),
        );

    AppLauncher::with_window(window)
        .use_simple_logger()
        .launch(app_data)
        .expect("launch failed");
}

fn get_ui_builder(
    data_1_feature: Arc<Mutex<DemoData1Feature>>, data_2_features: Arc<Mutex<DemoData2Features>>,
) -> impl Fn() -> Flex<AppData> {
    move || {
        Flex::<AppData>::column()
            .with_child(Canvas::<AppData>::new(
                Duration::from_millis(REFRESH_PERIOD),
                Box::new(Visualizer1Feature::new(data_1_feature.clone())),
            ))
            .with_child(build_add_point_2_params_widget(
                AppData::one_feature_new_point_x, "x".to_string(),
                AppData::one_feature_new_point_y, "y".to_string(),
                data_1_feature.clone(),
            ))
            .with_child(Canvas::<AppData>::new(
                Duration::from_millis(REFRESH_PERIOD),
                Box::new(Visualizer2Features::new(data_2_features.clone())),
            ))
            .with_child(build_add_point_3_params_widget(
                AppData::two_features_new_point_x, "x".to_string(),
                AppData::two_features_new_point_z, "z".to_string(),
                AppData::two_features_new_point_y, "y".to_string(),
                data_2_features.clone(),
            ))
    }
}

fn build_add_point_2_params_widget<
    L1: 'static + Lens<AppData, String>,
    L2: 'static + Lens<AppData, String>,
>(
    field_1: L1,
    field_1_name: String,
    field_2: L2,
    field_2_name: String,
    data: Arc<Mutex<DemoData1Feature>>,
) -> impl Widget<AppData> {
    Flex::row()
        .with_child(
            TextBox::new()
                .with_placeholder(field_1_name)
                .lens(field_1),
        )
        .with_child(
            TextBox::new()
                .with_placeholder(field_2_name)
                .lens(field_2),
        )
        .with_child(Button::new("Add Point").on_click(
            move |_ctx: &mut EventCtx, app_data: &mut AppData, _env: &Env| {
                match app_data.parse_one_feature_new_point() {
                    Ok(new_point) => {
                        data.lock().unwrap().add_point(new_point);
                    }
                    Err(_) => println!("invalid point"),
                }
            },
        ))
}

fn build_add_point_3_params_widget<
    L1: 'static + Lens<AppData, String>,
    L2: 'static + Lens<AppData, String>,
    L3: 'static + Lens<AppData, String>,
>(
    field_1: L1,
    field_1_name: String,
    field_2: L2,
    field_2_name: String,
    field_3: L3,
    field_3_name: String,
    data: Arc<Mutex<DemoData2Features>>,
) -> impl Widget<AppData> {
    Flex::row()
        .with_child(
            TextBox::new()
                .with_placeholder(field_1_name)
                .lens(field_1),
        )
        .with_child(
            TextBox::new()
                .with_placeholder(field_2_name)
                .lens(field_2),
        )
        .with_child(
            TextBox::new()
                .with_placeholder(field_3_name)
                .lens(field_3),
        )
        .with_child(Button::new("Add Point").on_click(
            move |_ctx: &mut EventCtx, app_data: &mut AppData, _env: &Env| {
                match app_data.parse_two_features_new_point() {
                    Ok(new_point) => {
                        data.lock().unwrap().add_point(new_point);
                    }
                    Err(_) => println!("invalid point"),
                }
            },
        ))
}
