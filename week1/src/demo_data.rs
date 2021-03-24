use crate::constants::*;

#[derive(Debug)]
pub struct DemoData {
    pub points: Vec<(f64, f64)>,
    pub theta0: f64,
    pub theta1: f64,
}

impl DemoData {
    pub fn add_point(&mut self, point: (f64, f64)) {
        self.points.push(point);
        self.theta0 = DEFAULT_THETA0;
        self.theta1 = DEFAULT_THETA1;
    }
}