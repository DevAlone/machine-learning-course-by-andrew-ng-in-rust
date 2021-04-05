use druid::piet::{ImageFormat, InterpolationMode};
use druid::widget::prelude::*;
use druid::{Point, Rect, TimerToken, MouseEvent, Vec2};
use std::time::Duration;
use std::error::Error;

pub struct Canvas<T> {
    timer_id: TimerToken,
    last_mouse_position: Point,
    refresh_period: Duration,
    drawer: Box<dyn Drawer<T>>,
}

pub trait Drawer<T> {
    fn draw_demo_data(
        &self, buf: &mut [u8], width: usize, height: usize, app_data: &T,
    ) -> Result<(), Box<dyn Error>>;
    fn get_size(&self) -> Size;
    fn handle_mouse_move(&mut self, _ctx: &EventCtx, _event: &MouseEvent, _diff: Vec2) {}
}

impl<T> Canvas<T> {
    pub fn new(
        refresh_period: Duration,
        drawer: Box<dyn Drawer<T>>,
    ) -> Canvas<T> {
        Canvas {
            timer_id: TimerToken::INVALID,
            last_mouse_position: Point::ORIGIN,
            refresh_period,
            drawer,
        }
    }
}

impl<T> Widget<T> for Canvas<T> {
    fn event(&mut self, ctx: &mut EventCtx, event: &Event, _data: &mut T, _env: &Env) {
        match event {
            Event::WindowConnected => {
                self.timer_id = ctx.request_timer(self.refresh_period);
            }
            Event::Timer(id) => {
                if *id == self.timer_id {
                    ctx.request_paint();
                    self.timer_id = ctx.request_timer(self.refresh_period);
                }
            }
            Event::MouseDown(event) => {
                self.last_mouse_position = event.pos;
            }
            Event::MouseMove(event) => {
                let pos = event.pos;
                let diff = self.last_mouse_position - pos;

                self.drawer.handle_mouse_move(ctx, event, diff);
                self.last_mouse_position = event.pos;
            }
            _ => (),
        }
    }

    fn lifecycle(&mut self, _ctx: &mut LifeCycleCtx, _event: &LifeCycle, _data: &T, _env: &Env) {}

    fn update(&mut self, _ctx: &mut UpdateCtx, _old_data: &T, _data: &T, _env: &Env) {}

    fn layout(
        &mut self,
        _layout_ctx: &mut LayoutCtx,
        _bc: &BoxConstraints,
        _data: &T,
        _env: &Env,
    ) -> Size {
        self.drawer.get_size()
    }

    // The paint method gets called last, after an event flow.
    // It goes event -> update -> layout -> paint, and each method can influence the next.
    // Basically, anything that changes the appearance of a widget causes a paint.
    fn paint(&mut self, ctx: &mut PaintCtx, app_data: &T, _env: &Env) {
        let size = ctx.size();
        let width = size.width as usize;
        let height = size.height as usize;
        let mut buf = vec![0u8; width * height * 3];
        self.drawer.draw_demo_data(&mut buf, width, height, app_data).unwrap();

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
