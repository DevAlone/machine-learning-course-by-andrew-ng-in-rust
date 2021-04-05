use druid::*;
use crate::constants::*;
use std::error::Error;

#[derive(Clone, Data, Lens, Debug)]
pub struct AppData {
    pub one_feature_new_point_x: String,
    pub one_feature_new_point_y: String,
    pub two_features_new_point_x: String,
    pub two_features_new_point_z: String,
    pub two_features_new_point_y: String,
}

impl AppData {
    pub fn parse_one_feature_new_point(&self) -> Result<[f64; 2], Box<dyn Error>> {
        let new_x = self.one_feature_new_point_x.parse::<f64>()?.clamp(0.0, MAX_VALUE);
        let new_y = self.one_feature_new_point_y.parse::<f64>()?.clamp(0.0, MAX_Y_VALUE);

        Ok([new_x, new_y])
    }
    pub fn parse_two_features_new_point(&self) -> Result<[f64; 3], Box<dyn Error>> {
        let new_x = self.two_features_new_point_x.parse::<f64>()?.clamp(0.0, MAX_VALUE);
        let new_z = self.two_features_new_point_z.parse::<f64>()?.clamp(0.0, MAX_VALUE);
        let new_y = self.two_features_new_point_y.parse::<f64>()?.clamp(0.0, MAX_Y_VALUE);

        Ok([new_x, new_z, new_y])
    }
}