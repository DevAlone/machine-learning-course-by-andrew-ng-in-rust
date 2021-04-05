use druid::widget::prelude::*;
use druid::{MouseEvent, Vec2};
use std::sync::{Arc, Mutex};
use plotters::prelude::*;
use crate::demo_data::DemoData2Features;
use crate::constants::*;
use std::error::Error;
use crate::app_data::AppData;
use helpers::canvas::Drawer;
use plotters::coord::types::RangedCoordf64;
use plotters::coord::cartesian::Cartesian3d;

const WIDGET_WIDTH: f64 = WINDOW_WIDTH;
const WIDGET_HEIGHT: f64 = WIDGET_WIDTH / 2.0;

pub struct Visualizer2Features {
    data: Arc<Mutex<DemoData2Features>>,
    pitch: f64,
    yaw: f64,
}

impl Visualizer2Features {
    pub fn new(data: Arc<Mutex<DemoData2Features>>) -> Visualizer2Features {
        Visualizer2Features {
            data,
            pitch: PLOT_DEFAULT_PITCH,
            yaw: PLOT_DEFAULT_YAW,
        }
    }
}

impl Drawer<AppData> for Visualizer2Features {
    fn draw_demo_data(&self, buf: &mut [u8], width: usize, height: usize, app_data: &AppData)
                      -> Result<(), Box<dyn Error>> {
        let root = BitMapBackend::with_buffer(
            buf, (width as u32, height as u32),
        ).into_drawing_area();

        root.fill(&WHITE)?;

        let root = root.split_horizontally((WIDGET_WIDTH / 2.0) as u32);
        let mut left_chart = ChartBuilder::on(&root.0)
            .caption("2 features 2d representation", FONT.into_font())
            .margin(CHART_MARGIN)
            .x_label_area_size(LEGEND_SIZE)
            .y_label_area_size(LEGEND_SIZE)
            .build_cartesian_2d(0f64..MAX_VALUE, 0f64..MAX_VALUE)?;
        left_chart.configure_mesh().draw()?;

        self.draw_2d(&mut left_chart, &app_data)?;

        let mut right_chart = ChartBuilder::on(&root.1)
            .caption("2 features 3d representation", FONT.into_font())
            .margin(CHART_MARGIN)
            .x_label_area_size(LEGEND_SIZE)
            .y_label_area_size(LEGEND_SIZE)
            .build_cartesian_3d(
                0f64..MAX_VALUE, 0f64..MAX_Y_VALUE, 0f64..MAX_VALUE,
            )?;
        right_chart.with_projection(|mut p| {
            p.pitch = self.pitch;
            p.yaw = self.yaw;
            p.scale = DEFAULT_SCALE;
            p.into_matrix()
        });
        right_chart.configure_axes().draw()?;

        self.draw_3d(&mut right_chart, &app_data)?;

        Ok(())
    }

    fn get_size(&self) -> Size {
        Size::new(WIDGET_WIDTH, WIDGET_HEIGHT)
    }

    fn handle_mouse_move(&mut self, ctx: &EventCtx, event: &MouseEvent, diff: Vec2) {
        let size = ctx.size();
        if event.buttons.has_left() {
            self.yaw += diff.x / size.width;
            self.pitch -= diff.y / size.height;
        }
    }
}

fn get_point_color(theta: [f64; 3], x: [f64; 2]) -> RGBColor {
    if helpers::math::logistic_regression_predict(theta, x) >= 0.5 {
        BLUE
    } else {
        RED
    }
}

impl Visualizer2Features {
    fn draw_2d(
        &self,
        chart: &mut ChartContext<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
        app_data: &AppData,
    ) -> Result<(), Box<dyn Error>>
    {
        let data = self.data.lock().unwrap();
        let theta = &data.theta;
        let xs = &data.xs;
        let ys = &data.ys;

        // plot the points
        chart.draw_series(
            xs.iter().enumerate().map(
                |(i, x)| Circle::new(
                    (x[0], x[1]),
                    if ys[i] < 0.5 { POINT_SIZE } else { POINT_SIZE * 2 },
                    get_point_color(*theta, *x).filled(),
                ),
            ),
        )?;

        // plot the new point (if there's any)
        if let Ok(new_point) = app_data.parse_two_features_new_point() {
            chart.draw_series(
                [new_point]
                    .iter()
                    .map(|[x, z, y]| {
                        Circle::new(
                            (*x, *z),
                            if *y < 0.5 { POINT_SIZE } else { POINT_SIZE * 2 },
                            NEW_POINT_COLOR.filled(),
                        )
                    }),
            )?;
        }

        Ok(())
    }


    fn draw_3d(
        &self,
        chart: &mut ChartContext<
            BitMapBackend,
            Cartesian3d<RangedCoordf64, RangedCoordf64, RangedCoordf64>,
        >,
        app_data: &AppData,
    ) -> Result<(), Box<dyn Error>>
    {
        let data = self.data.lock().unwrap();
        let theta = &data.theta;
        let xs = &data.xs;
        let ys = &data.ys;

        // plot the points
        helpers::plotters::three_d::plot_points(
            chart, *theta, xs, ys, POINT_SIZE, get_point_color,
        )?;

        // plot the new point (if there's any)
        if let Ok(new_point) = app_data.parse_two_features_new_point() {
            helpers::plotters::three_d::plot_new_point(
                chart, new_point, POINT_SIZE, NEW_POINT_COLOR,
            )?;
        }

        // plot a surface representing our prediction function
        helpers::plotters::three_d::plot_surface(chart, |x: f64, z: f64| -> f64 {
            helpers::math::logistic_regression_predict(*theta, [x, z])
        }, (MAX_VALUE * SURFACE_PRECISION) as usize, MAX_VALUE, BLUE)?;

        Ok(())
    }
}