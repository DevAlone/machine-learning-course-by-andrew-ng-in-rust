use plotters::prelude::{GREEN, RGBColor};

pub const MAX_VALUE: f64 = 10.0;
// since it's logistic regression, Y should be between 0 and 1
pub const MAX_Y_VALUE: f64 = 1.0;

pub const LEARNING_RATE: f64 = 0.01;
pub const GRADIENT_UPDATE_PERIOD: usize = 10;
pub const GRADIENT_STEPS_PER_UPDATE: usize = 10;
pub const DEFAULT_THETA_VALUE: f64 = 0.0;

pub const REFRESH_PERIOD: u64 = 10;
pub const WINDOW_WIDTH: f64 = 800.0;
pub const WINDOW_HEIGHT: f64 = 800.0;
pub const PLOT_DEFAULT_PITCH: f64 = 0.45;
pub const PLOT_DEFAULT_YAW: f64 = -0.90;
pub const POINT_SIZE: i32 = 2;
pub const ONE_FEATURE_VISUALIZER_HEIGHT: f64 = WINDOW_WIDTH / 3.0;
pub const NEW_POINT_COLOR: RGBColor = GREEN;
pub const FONT: (&str, i32) = ("sans-serif", 20);
pub const CHART_MARGIN: i32 = 5;
pub const LEGEND_SIZE: i32 = 30;
pub const SIGMOID_PLOTTING_PRECISION: f64 = 10.0;
pub const DEFAULT_SCALE: f64 = 0.7;
pub const SURFACE_PRECISION: f64 = 2.0;
