use noise::{NoiseFn, Perlin, Seedable};

use crate::chunk::Chunk;
use crate::world::world::ChunkPos;

pub struct Generator {
    perlin: Perlin,
    octaves: Vec<Octave>,
}

const BASE_HEIGHT: i32 = 70;

#[derive(Copy, Clone)]
pub struct Octave {
    pub frequency: f32,
    pub amplitude: f32,
}

impl Generator {
    pub fn new(seed: u32, octaves: Vec<Octave>) -> Generator {
        let perlin = Perlin::new();
        let perlin = perlin.set_seed(seed);
        Generator { perlin, octaves }
    }

    pub fn generate(&self, pos: ChunkPos) -> Chunk {
        let mut chunk = Chunk::new();

        for z in 0..32 {
            for x in 0..32 {
                let noise_x = pos.x as f64 * 32.0 + x as f64;
                let noise_z = pos.z as f64 * 32.0 + z as f64;

                let mut height = BASE_HEIGHT as f64;
                for octave in &self.octaves {
                    let f = octave.frequency as f64;
                    height += self.perlin.get([noise_x * f + 0.5, noise_z * f + 0.5]) * octave.amplitude as f64;
                }
                let height = height as i32;

                for y in 0..32 {
                    if pos.y * 32 + y <= height {
                        // TODO use constants for block ids
                        let mut block = 3; // stone
                        if pos.y * 32 + y >= height - 3 { block = 2; } // dirt
                        if pos.y * 32 + y >= height { block = 1; } // grass

                        chunk.set_block(x as u32, y as u32, z as u32, block);
                        continue;
                    }
                }
            }
        }

        chunk
    }
}
