use druid::piet::{ImageFormat, InterpolationMode};
use druid::widget::prelude::*;
use druid::{Point, Rect, TimerToken};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use plotters::prelude::*;
use crate::demo_data::DemoData;
use crate::constants::{REFRESH_PERIOD, MAX_VALUE, UI_PLOT_TOP_RIGHT_MARGIN, UI_LEGEND_SIZE};
use std::error::Error;

pub struct Visualizer {
    timer_id: TimerToken,
    data: Arc<Mutex<DemoData>>,
}

impl Visualizer {
    pub fn new(data: Arc<Mutex<DemoData>>) -> Visualizer {
        Visualizer {
            timer_id: TimerToken::INVALID,
            data,
        }
    }
}

impl Widget<()> for Visualizer {
    fn event(&mut self, ctx: &mut EventCtx, event: &Event, _data: &mut (), _env: &Env) {
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
                // calculate the point inside the plot

                let size = ctx.size();
                let plot_size = Size::new(
                    size.width - UI_LEGEND_SIZE as f64 - UI_PLOT_TOP_RIGHT_MARGIN as f64,
                    size.height - UI_LEGEND_SIZE as f64 - UI_PLOT_TOP_RIGHT_MARGIN as f64,
                );
                let pos = Point::new(
                    (event.pos.x - UI_LEGEND_SIZE as f64).clamp(0.0, plot_size.width),
                    (event.pos.y - UI_LEGEND_SIZE as f64).clamp(0.0, plot_size.height),
                );

                let point = (
                    (pos.x / plot_size.width * MAX_VALUE).clamp(0.0, MAX_VALUE),
                    ((plot_size.height - pos.y) / plot_size.height * MAX_VALUE).clamp(0.0, MAX_VALUE),
                );

                self.data.lock().unwrap().add_point(point);
            }
            _ => (),
        }
    }

    fn lifecycle(&mut self, _ctx: &mut LifeCycleCtx, _event: &LifeCycle, _data: &(), _env: &Env) {}

    fn update(&mut self, _ctx: &mut UpdateCtx, _old_data: &(), _data: &(), _env: &Env) {}

    fn layout(
        &mut self,
        _layout_ctx: &mut LayoutCtx,
        bc: &BoxConstraints,
        _data: &(),
        _env: &Env,
    ) -> Size {
        // BoxConstraints are passed by the parent widget.
        // This method can return any Size within those constraints:
        // bc.constrain(my_size)
        //
        // To check if a dimension is infinite or not (e.g. scrolling):
        // bc.is_width_bounded() / bc.is_height_bounded()
        bc.max()
    }

    // The paint method gets called last, after an event flow.
    // It goes event -> update -> layout -> paint, and each method can influence the next.
    // Basically, anything that changes the appearance of a widget causes a paint.
    fn paint(&mut self, ctx: &mut PaintCtx, _data: &(), _env: &Env) {
        let size = ctx.size();
        let width = size.width as usize;
        let height = size.height as usize;
        let mut buf = vec![0u8; width * height * 3];

        {
            self.draw_demo_data(&mut buf, width, height).unwrap();
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

impl Visualizer {
    fn draw_demo_data(&self, buf: &mut [u8], width: usize, height: usize) -> Result<(), Box<dyn Error>> {
        let root = BitMapBackend::with_buffer(
            buf, (width as u32, height as u32),
        ).into_drawing_area();

        root.fill(&WHITE)?;

        let root = root.margin(
            UI_PLOT_TOP_RIGHT_MARGIN, 0, 0, UI_PLOT_TOP_RIGHT_MARGIN,
        );
        let mut plot_ctx = ChartBuilder::on(&root)
            .x_label_area_size(UI_LEGEND_SIZE)
            .y_label_area_size(UI_LEGEND_SIZE)
            .build_cartesian_2d(0f64..MAX_VALUE, 0f64..MAX_VALUE)?;

        plot_ctx
            .configure_mesh()
            .draw()?;

        let data = self.data.lock().unwrap();

        plot_ctx.draw_series(
            data.points
                .iter()
                .map(|(x, y)| Circle::new((*x, *y), 2, GREEN.filled()))
        )?;

        let theta0 = data.theta0;
        let theta1 = data.theta1;

        if theta1 != 0.0 {
            // plotting library doesn't support values outside of drawing region
            // so we need to do these tricks
            let clamp_point = |point: (f64, f64)| -> (f64, f64){
                if point.1 < 0.0 {
                    return (-theta0 / theta1, 0.0);
                } else if point.1 > MAX_VALUE {
                    return ((MAX_VALUE - theta0) / theta1, MAX_VALUE);
                }

                point
            };

            let point1 = clamp_point((0.0, theta0));
            let point2 = clamp_point((MAX_VALUE, theta0 + theta1 * MAX_VALUE));

            plot_ctx.draw_series(
                LineSeries::new(
                    vec![point1, point2],
                    RED.filled(),
                )
            )?;
        }

        Ok(())
    }
}