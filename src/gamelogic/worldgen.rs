use noise::{NoiseFn, Perlin, Seedable};

use crate::gamelogic::content::blocks;
use crate::systems::worldgen::ChunkGenerator;
use crate::world::chunk::{Chunk, ChunkPos};

#[derive(Clone)]
pub struct Noise {
    /// The frequency of the underlying noise.
    pub frequency: f32,
    /// Each additional octave adds the same noise at double the frequency and half the value.
    pub octaves: i32,
    /// Use spline points to map the noise value to custom values.
    /// If the noise value is between two spline point x values, the y value of those to points will
    /// be linearly interpolated depending on the where the noise value lies between the two
    /// x values.
    /// If x is less or greater than the lowest or highest point, that point's value is used without
    /// interpolation.
    pub spline_points: Vec<SplinePoint>,
}

#[derive(Copy, Clone)]
pub struct SplinePoint {
    /// The input value for the curve. Must be between [-1;1].
    pub x: f32,
    /// The value between which is interpolated.
    pub y: f32,
}

impl Noise {
    fn get(&self, perlin: &Perlin, x: f64, z: f64) -> f64 {
        let v = self.get_noise_value(perlin, x, z);
        Self::interpolate_spline_points(&self.spline_points, v)
    }

    fn get_noise_value(&self, perlin: &Perlin, x: f64, z: f64) -> f64 {
        let mut f = self.frequency as f64;
        let mut a = 1.0;

        let mut v = 0.0;
        for _ in 0..self.octaves {
            v += perlin.get([x * f + 0.5, z * f + 0.5]) * a;
            f *= 2.0;
            a *= 0.5;
        }

        v
    }

    fn interpolate_spline_points(points: &[SplinePoint], x: f64) -> f64 {
        if points.is_empty() {
            return 0.0;
        }

        let rhs = points.iter().position(|p| (p.x as f64) > x);
        if rhs.is_none() {
            return points.last().unwrap().y as f64;
        }
        let rhs = rhs.unwrap();
        if rhs == 0 {
            return points.first().unwrap().y as f64;
        }

        let lhs = points.get(rhs - 1).unwrap();
        let rhs = points.get(rhs).unwrap();

        let v_start = lhs.y as f64;
        let v_diff = (rhs.y - lhs.y) as f64;
        let factor = (x as f32 - lhs.x) / (rhs.x - lhs.x);
        v_start + v_diff * factor as f64
    }
}

#[cfg(test)]
mod noise_tests {
    use noise::{Perlin, Seedable};

    use crate::assert_float_eq;
    use crate::gamelogic::worldgen::{Noise, SplinePoint};

    /// Tests that noise value is correctly calculated and interpolated using the spline points.
    #[test]
    fn get() {
        let noise = Noise {
            frequency: 2.0,
            octaves: 3,
            spline_points: vec![SplinePoint { x: -1.0, y: 0.0 }, SplinePoint { x: 1.0, y: 1.0 }],
        };
        let perlin = Perlin::new().set_seed(0);

        assert_float_eq!(noise.get(&perlin, 0.0, 0.0), 0.5);
        assert_float_eq!(noise.get(&perlin, 1.0, 0.0), 0.234834);
        assert_float_eq!(noise.get(&perlin, 0.0, 1.0), 0.676776);
        assert_float_eq!(noise.get(&perlin, 1.0, 1.0), 0.411611);
    }

    /// Tests if all common edge cases work correctly and if a normal use case produces the
    /// expected results.
    #[test]
    fn interpolate_spline_points() {
        // no spline points
        let points = vec![];
        assert_eq!(Noise::interpolate_spline_points(&points, 0.0), 0.0);

        // only higher point
        let points = vec![SplinePoint { x: 0.5, y: 1.0 }];
        assert_eq!(Noise::interpolate_spline_points(&points, 0.25), 1.0);

        // only lower point
        let points = vec![SplinePoint { x: 0.5, y: 1.0 }];
        assert_eq!(Noise::interpolate_spline_points(&points, 0.75), 1.0);

        // interpolation between multiple points
        let points = vec![
            SplinePoint { x: 0.0, y: 1.0 },
            SplinePoint { x: 0.5, y: 2.0 },
            SplinePoint { x: 1.0, y: 3.0 },
        ];
        assert_eq!(Noise::interpolate_spline_points(&points, -0.5), 1.0);
        assert_eq!(Noise::interpolate_spline_points(&points, 0.0), 1.0);
        assert_eq!(Noise::interpolate_spline_points(&points, 0.25), 1.5);
        assert_eq!(Noise::interpolate_spline_points(&points, 0.5), 2.0);
        assert_eq!(Noise::interpolate_spline_points(&points, 0.75), 2.5);
        assert_eq!(Noise::interpolate_spline_points(&points, 1.0), 3.0);
        assert_eq!(Noise::interpolate_spline_points(&points, 1.5), 3.0);
    }
}

pub struct Generator {
    perlin: Perlin,
    cfg: Config,
}

#[derive(Clone)]
pub struct Config {
    /// Defines y level until which water is placed.
    pub sea_level: i32,
    /// Defines the noise responsible for determining how far inland a point is.
    /// -1 = sea, 1 = far inland
    pub continentalness: Noise,
    /// Defines how much the point is affected by erosion, which in turn determines how
    /// mountainous it is.
    /// -1 = netherlands, 1 = tibet
    pub erosion: Noise,
}

impl Generator {
    pub fn new(seed: u32, cfg: Config) -> Generator {
        let perlin = Perlin::new();
        let perlin = perlin.set_seed(seed);

        Generator { perlin, cfg }
    }

    fn get_height_at(&self, pos: &ChunkPos, x: i32, z: i32) -> i32 {
        let noise_x = pos.x as f64 * 32.0 + x as f64;
        let noise_z = pos.z as f64 * 32.0 + z as f64;

        let height = self.cfg.continentalness.get(&self.perlin, noise_x, noise_z);
        let height = height + self.cfg.erosion.get(&self.perlin, noise_x, noise_z);

        height as i32
    }
}

// TODO calculate and cache heightmap per xz coord. get external notification form chunk loader when cache can be discarded

impl ChunkGenerator for Generator {
    fn is_interested_in(&self, pos: &ChunkPos) -> bool {
        let mut min_y = i32::MAX;
        let mut max_y = i32::MIN;

        // TODO do not run this twice
        for z in 0..32 {
            for x in 0..32 {
                let height = self.get_height_at(pos, x, z);
                min_y = min_y.min(height);
                max_y = max_y.max(height);
            }
        }

        min_y <= (pos.y + 1) * 32 && max_y >= pos.y * 32
    }

    fn generate_chunk(&self, chunk: &mut Chunk) {
        for z in 0..32 {
            for x in 0..32 {
                let height = self.get_height_at(&chunk.pos, x, z);

                for y in 0..32 {
                    if chunk.pos.y * 32 + y <= height {
                        let mut block = blocks::STONE;
                        if chunk.pos.y * 32 + y >= height - 3 { block = blocks::DIRT; }
                        if chunk.pos.y * 32 + y >= height { block = blocks::GRASS; }

                        chunk.set_block(x as u32, y as u32, z as u32, block);
                        continue;
                    }
                }
            }
        }
    }
}
