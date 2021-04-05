use druid::*;
use crate::constants::*;
use std::error::Error;

#[derive(Clone, Data, Lens, Debug)]
pub struct AppData {
    pub new_point_x: String,
    pub new_point_y: String,
    pub new_point_z: String,
}

impl AppData {
    pub fn parse_new_point(&self) -> Result<[f64; 3], Box<dyn Error>> {
        let new_x = self.new_point_x.parse::<f64>()?.clamp(0.0, MAX_VALUE);
        let new_z = self.new_point_z.parse::<f64>()?.clamp(0.0, MAX_VALUE);
        let new_y = self.new_point_y.parse::<f64>()?.clamp(0.0, MAX_VALUE);

        Ok([new_x, new_z, new_y])
    }
}