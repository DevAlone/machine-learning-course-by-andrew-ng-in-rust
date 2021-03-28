use druid::piet::{ImageFormat, InterpolationMode};
use druid::widget::prelude::*;
use druid::{Point, Rect, TimerToken};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use plotters::prelude::*;
use crate::demo_data::DemoData;
use crate::constants::*;
use std::error::Error;
use crate::app_data::AppData;

// Visualizer2Features visualizes linear regression with 2 features
pub struct Visualizer2Features {
    timer_id: TimerToken,
    data: Arc<Mutex<DemoData>>,
    pitch: f64,
    yaw: f64,
    last_mouse_position: Point,
}

impl Visualizer2Features {
    pub fn new(data: Arc<Mutex<DemoData>>) -> Visualizer2Features {
        {
            let data = data.lock().unwrap();
            assert_eq!(data.theta.len(), 3);
            // TODO: more validation?
        }

        Visualizer2Features {
            timer_id: TimerToken::INVALID,
            data,
            pitch: PLOT_DEFAULT_PITCH,
            yaw: PLOT_DEFAULT_YAW,
            last_mouse_position: Point::ORIGIN,
        }
    }
}

impl Widget<AppData> for Visualizer2Features {
    fn event(&mut self, ctx: &mut EventCtx, event: &Event, _data: &mut AppData, _env: &Env) {
        match event {
            Event::WindowConnected => {
                self.timer_id = ctx.request_timer(Duration::from_millis(REFRESH_PERIOD as u64));
            }
            Event::Timer(id) => {
                if *id == self.timer_id {
                    ctx.request_paint();
                    self.timer_id = ctx.request_timer(Duration::from_millis(REFRESH_PERIOD as u64));
                }
            }
            Event::MouseDown(event) => {
                self.last_mouse_position = event.pos;
            }
            Event::MouseMove(event) => {
                let size = ctx.size();
                let pos = event.pos;
                let diff = self.last_mouse_position - pos;

                if event.buttons.has_left() {
                    self.yaw += diff.x / size.width;
                    self.pitch -= diff.y / size.height;
                }

                self.last_mouse_position = event.pos;
            }
            _ => (),
        }
    }

    fn lifecycle(&mut self, _ctx: &mut LifeCycleCtx, _event: &LifeCycle, _data: &AppData, _env: &Env) {}

    fn update(&mut self, _ctx: &mut UpdateCtx, _old_data: &AppData, _data: &AppData, _env: &Env) {}

    fn layout(
        &mut self,
        _layout_ctx: &mut LayoutCtx,
        bc: &BoxConstraints,
        _data: &AppData,
        _env: &Env,
    ) -> Size {
        let mut size = bc.max();
        if size.width.is_infinite() && !size.height.is_infinite() {
            size.width = size.height;
        } else if size.height.is_infinite() && !size.width.is_infinite() {
            size.height = size.width;
        } else if size.width.is_infinite() && size.height.is_infinite() {
            return Size::new(WINDOW_WIDTH, WINDOW_WIDTH);
        }

        size
    }

    // The paint method gets called last, after an event flow.
    // It goes event -> update -> layout -> paint, and each method can influence the next.
    // Basically, anything that changes the appearance of a widget causes a paint.
    fn paint(&mut self, ctx: &mut PaintCtx, app_data: &AppData, _env: &Env) {
        let size = ctx.size();
        let width = size.width as usize;
        let height = size.height as usize;
        let mut buf = vec![0u8; width * height * 3];

        {
            self.draw_demo_data(&mut buf, width, height, app_data).unwrap();
        }

        let image = ctx
            .make_image(width, height, &buf, ImageFormat::Rgb)
            .unwrap();
        ctx.draw_image(
            &image,
            Rect::from_origin_size(Point::ORIGIN, size),
            InterpolationMode::Bilinear,
        );
    }
}

impl Visualizer2Features {
    fn draw_demo_data(&self, buf: &mut [u8], width: usize, height: usize, app_data: &AppData) -> Result<(), Box<dyn Error>> {
        let root = BitMapBackend::with_buffer(
            buf, (width as u32, height as u32),
        ).into_drawing_area();

        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Left click and drag to move camera", ("sans-serif", 20))
            .build_cartesian_3d(
                0.0..MAX_VALUE, 0.0..MAX_VALUE, 0.0..MAX_VALUE,
            )?;
        chart.with_projection(|mut p| {
            p.pitch = self.pitch;
            p.yaw = self.yaw;
            p.scale = 0.7;
            p.into_matrix() // build the projection matrix
        });

        chart.configure_axes().draw()?;

        let data = self.data.lock().unwrap();
        let theta = &data.theta;
        let xs = &data.xs;
        let ys = &data.ys;

        // plot the points
        chart.draw_series(
            xs
                .iter()
                .enumerate()
                .map(|(i, x_vec)| {
                    let x = x_vec[0];
                    let z = x_vec[1];
                    let y = ys[i];

                    Circle::new((x, y, z), 2, GREEN.filled())
                }),
        )?;

        // plot the new point (if there's any)
        if let Ok(new_point) = app_data.parse_new_point() {
            chart.draw_series(
                vec![(new_point.0, new_point.1, new_point.2)]
                    .iter()
                    .map(|(x, y, z)| {
                        Circle::new((*x, *y, *z), 2, RED.filled())
                    }),
            )?;
        }

        // plot a surface representing our prediction function
        let surface_function = |x: f64, z: f64| -> f64 {
            theta[0] + theta[1] * x + theta[2] * z
        };

        chart.draw_series(
            (0..=MAX_VALUE as usize).map(|x| std::iter::repeat(x).zip(
                0..=MAX_VALUE as usize
            )).flatten()
                .map(|(x, z)| {
                    let x = x as f64;
                    let z = z as f64;

                    let to_polygon_point = |x, z| (x, surface_function(x, z), z);

                    Polygon::new(vec![
                        to_polygon_point(x, z),
                        to_polygon_point(x + 1.0, z),
                        to_polygon_point(x + 1.0, z + 1.0),
                        to_polygon_point(x, z + 1.0),
                    ], &BLUE.mix(0.3))
                }),
        ).unwrap();

        Ok(())
    }
}
