#[derive(Debug)]
pub struct DemoData {
    pub xs: Vec<Vec<f64>>,
    pub ys: Vec<f64>,
    pub theta: Vec<f64>,
}

impl DemoData {
    pub fn add_point(&mut self, point: &[f64]) {
        self.xs.push(point[0..point.len() - 1].to_vec());
        self.ys.push(point[point.len() - 1]);
    }
}