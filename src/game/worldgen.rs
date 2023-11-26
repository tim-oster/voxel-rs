use noise::{NoiseFn, Perlin, Seedable};

use crate::game::gameplay::blocks;
use crate::systems::worldgen::ChunkGenerator;
use crate::world::chunk::Chunk;

#[derive(Clone)]
pub struct Noise {
    pub frequency: f32,
    pub octaves: i32,
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
        self.interpolate_spline_points(v)
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

    fn interpolate_spline_points(&self, x: f64) -> f64 {
        if self.spline_points.is_empty() {
            return 0.0;
        }

        let rhs = self.spline_points.iter().position(|p| (p.x as f64) > x);
        if rhs.is_none() {
            return self.spline_points.last().unwrap().y as f64;
        }
        let rhs = rhs.unwrap();
        if rhs == 0 {
            return self.spline_points.first().unwrap().y as f64;
        }

        let lhs = self.spline_points.get(rhs - 1).unwrap();
        let rhs = self.spline_points.get(rhs).unwrap();

        let v_start = lhs.y as f64;
        let v_diff = (rhs.y - lhs.y) as f64;
        let factor = (x as f32 - lhs.x) / (rhs.x - lhs.x);
        v_start + v_diff * factor as f64
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
}

impl ChunkGenerator for Generator {
    fn generate_chunk(&self, chunk: &mut Chunk) {
        for z in 0..32 {
            for x in 0..32 {
                let noise_x = chunk.pos.x as f64 * 32.0 + x as f64;
                let noise_z = chunk.pos.z as f64 * 32.0 + z as f64;

                let height = self.cfg.continentalness.get(&self.perlin, noise_x, noise_z);
                let height = height + self.cfg.erosion.get(&self.perlin, noise_x, noise_z);
                let height = height as i32;

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
