mod visualizer;
mod constants;
mod demo_data;
mod gradient_descent;

use druid::{AppLauncher, LocalizedString, WindowDesc};
use std::sync::{Arc, Mutex};
use std::thread;
use visualizer::Visualizer;
use crate::demo_data::DemoData;
use crate::gradient_descent::learning_thread;
use crate::constants::*;


fn main() {
    let data = Arc::new(Mutex::new(DemoData {
        points: vec![(1.0, 1.0), (2.0, 2.0)],
        theta0: DEFAULT_THETA0,
        theta1: DEFAULT_THETA1,
    }));

    let thread_data = data.clone();
    thread::spawn(move || learning_thread(thread_data));

    let window = WindowDesc::new(get_ui_builder(data))
        .title(LocalizedString::new("Linear Regression Demo")
            .with_placeholder("linear-regression-demo"));

    AppLauncher::with_window(window)
        .use_simple_logger()
        .launch(())
        .expect("launch failed");
}

fn get_ui_builder(data: Arc<Mutex<DemoData>>) -> impl Fn() -> Visualizer {
    move || Visualizer::new(data.clone())
}