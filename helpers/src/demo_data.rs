#[derive(Debug)]
pub struct RegressionDemoDataNFeatures<const N_FEATURES: usize> where [(); N_FEATURES + 1]: {
    pub xs: Vec<[f64; N_FEATURES]>,
    pub ys: Vec<f64>,
    pub theta: [f64; N_FEATURES + 1],
}

pub type RegressionDemoData1Feature = RegressionDemoDataNFeatures<1>;
pub type RegressionDemoData2Features = RegressionDemoDataNFeatures<2>;

impl RegressionDemoData1Feature {
    pub fn add_point(&mut self, point: [f64; 2]) {
        self.xs.push([point[0]]);
        self.ys.push(point[1]);
    }
}

impl RegressionDemoData2Features {
    pub fn add_point(&mut self, point: [f64; 3]) {
        self.xs.push([point[0], point[1]]);
        self.ys.push(point[2]);
    }
}