use druid::widget::prelude::*;
use std::sync::{Arc, Mutex};
use plotters::prelude::*;
use crate::demo_data::DemoData1Feature;
use crate::constants::*;
use std::error::Error;
use crate::app_data::AppData;
use helpers::canvas::Drawer;

pub struct Visualizer1Feature {
    data: Arc<Mutex<DemoData1Feature>>,
}

impl Visualizer1Feature {
    pub fn new(data: Arc<Mutex<DemoData1Feature>>) -> Visualizer1Feature {
        Visualizer1Feature { data }
    }
}

impl Drawer<AppData> for Visualizer1Feature {
    fn draw_demo_data(&self, buf: &mut [u8], width: usize, height: usize, app_data: &AppData)
                      -> Result<(), Box<dyn Error>> {
        let root = BitMapBackend::with_buffer(
            buf, (width as u32, height as u32),
        ).into_drawing_area();

        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("1 feature", FONT.into_font())
            .margin(CHART_MARGIN)
            .x_label_area_size(LEGEND_SIZE)
            .y_label_area_size(LEGEND_SIZE)
            .build_cartesian_2d(0f64..MAX_VALUE, 0f64..MAX_Y_VALUE)?;

        chart.configure_mesh().draw()?;
        // plot the points
        let data = self.data.lock().unwrap();
        chart.draw_series(
            data.xs.iter().enumerate().map(
                |(i, x)| Circle::new(
                    (x[0], data.ys[i]),
                    POINT_SIZE,
                    if helpers::math::logistic_regression_predict(data.theta, *x) >= 0.5 {
                        BLUE.filled()
                    } else {
                        RED.filled()
                    },
                ),
            ),
        )?;

        // plot the new point (if there's any)
        if let Ok(new_point) = app_data.parse_one_feature_new_point() {
            chart.draw_series(
                [(new_point[0], new_point[1])]
                    .iter()
                    .map(|(x, y)| {
                        Circle::new(
                            (*x, *y), POINT_SIZE, NEW_POINT_COLOR.filled(),
                        )
                    }),
            )?;
        }

        // horizontal line at Y = 0.5
        chart.draw_series(
            LineSeries::new(
                [(0.0, 0.5), (MAX_VALUE, 0.5)].iter().map(|x| *x),
                BLACK.filled(),
            ),
        )?;

        // predicting sigmoid
        chart.draw_series(
            LineSeries::new(
                (0..(MAX_VALUE * SIGMOID_PLOTTING_PRECISION) as usize)
                    .map(|x| x as f64 / SIGMOID_PLOTTING_PRECISION)
                    .map(|x| (x, helpers::math::logistic_regression_predict(data.theta, [x]))),
                GREEN.filled(),
            ),
        )?;

        Ok(())
    }

    fn get_size(&self) -> Size {
        Size::new(WINDOW_WIDTH, ONE_FEATURE_VISUALIZER_HEIGHT)
    }
}
