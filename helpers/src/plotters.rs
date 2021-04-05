pub mod three_d {
    use plotters::prelude::*;
    use plotters::coord::cartesian::Cartesian3d;
    use plotters::coord::types::RangedCoordf64;
    use std::error::Error;

    pub fn plot_surface<F: Fn(f64, f64) -> f64>(
        chart: &mut ChartContext<
            BitMapBackend,
            Cartesian3d<RangedCoordf64, RangedCoordf64, RangedCoordf64>,
        >,
        surface_function: F,
        number_of_points: usize,
        max_value: f64,
        color: RGBColor,
    ) -> Result<(), Box<dyn Error>> {
        let scale_factor = max_value as f64 / number_of_points as f64;

        chart.draw_series(
            (0..=number_of_points)
                .map(|x| std::iter::repeat(x).zip(0..=number_of_points))
                .flatten()
                .map(|(x, z)| {
                    let x = x as f64 * scale_factor;
                    let z = z as f64 * scale_factor;

                    let to_polygon_point = |x, z| (x, surface_function(x, z), z);

                    Polygon::new(vec![
                        to_polygon_point(x, z),
                        to_polygon_point(x + scale_factor, z),
                        to_polygon_point(x + scale_factor, z + scale_factor),
                        to_polygon_point(x, z + scale_factor),
                    ], &color.mix(0.3))
                }),
        )?;

        Ok(())
    }

    pub fn plot_points<F: Fn([f64; 3], [f64; 2]) -> RGBColor>(
        chart: &mut ChartContext<
            BitMapBackend,
            Cartesian3d<RangedCoordf64, RangedCoordf64, RangedCoordf64>,
        >,
        theta: [f64; 3],
        xs: &[[f64; 2]],
        ys: &[f64],
        point_size: i32,
        get_point_color_fn: F,
    ) -> Result<(), Box<dyn Error>> {
        chart.draw_series(
            xs
                .iter()
                .enumerate()
                .map(|(i, x_vec)| {
                    let x = x_vec[0];
                    let z = x_vec[1];
                    let y = ys[i];

                    Circle::new(
                        (x, y, z),
                        point_size,
                        get_point_color_fn(theta, *x_vec).filled(),
                    )
                })
        )?;

        Ok(())
    }

    pub fn plot_new_point(
        chart: &mut ChartContext<
            BitMapBackend,
            Cartesian3d<RangedCoordf64, RangedCoordf64, RangedCoordf64>,
        >,
        new_point: [f64; 3],
        point_size: i32,
        point_color: RGBColor,
    ) -> Result<(), Box<dyn Error>> {
        chart.draw_series(
            [new_point]
                .iter()
                .map(|[x, z, y]| {
                    Circle::new(
                        (*x, *y, *z),
                        point_size,
                        point_color.filled(),
                    )
                }),
        )?;

        Ok(())
    }
}
