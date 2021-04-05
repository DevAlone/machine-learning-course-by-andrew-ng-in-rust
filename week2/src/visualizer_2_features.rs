use druid::widget::prelude::*;
use druid::{MouseEvent, Vec2};
use std::sync::{Arc, Mutex};
use plotters::prelude::*;
use crate::demo_data::DemoData;
use crate::constants::*;
use std::error::Error;
use crate::app_data::AppData;
use helpers::canvas::Drawer;

const WIDGET_WIDTH: f64 = WINDOW_WIDTH;
const WIDGET_HEIGHT: f64 = WIDGET_WIDTH / 2.0;

// Visualizer2Features visualizes linear regression with 2 features
pub struct Visualizer2Features {
    data: Arc<Mutex<DemoData>>,
    pitch: f64,
    yaw: f64,
}

impl Visualizer2Features {
    pub fn new(data: Arc<Mutex<DemoData>>) -> Visualizer2Features {
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

        let mut chart = ChartBuilder::on(&root)
            .caption("Left click and drag to move camera", FONT)
            .build_cartesian_3d(
                0.0..MAX_VALUE, 0.0..MAX_VALUE, 0.0..MAX_VALUE,
            )?;
        chart.with_projection(|mut p| {
            p.pitch = self.pitch;
            p.yaw = self.yaw;
            p.scale = DEFAULT_SCALE;
            p.into_matrix() // build the projection matrix
        });
        chart.configure_axes().draw()?;

        let data = self.data.lock().unwrap();
        let theta = &data.theta;
        let xs = &data.xs;
        let ys = &data.ys;

        // plot the points
        helpers::plotters::three_d::plot_points(
            &mut chart, *theta, xs, ys, POINT_SIZE, |_, _| GREEN,
        )?;

        // plot the new point (if there's any)
        if let Ok(new_point) = app_data.parse_new_point() {
            helpers::plotters::three_d::plot_new_point(
                &mut chart, new_point, POINT_SIZE, RED,
            )?;
        }

        // plot a surface representing our prediction function
        helpers::plotters::three_d::plot_surface(&mut chart, |x: f64, z: f64| -> f64 {
            theta[0] + theta[1] * x + theta[2] * z
        }, MAX_VALUE as usize, MAX_VALUE, BLUE)?;

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
