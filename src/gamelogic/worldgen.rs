use std::collections::VecDeque;
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Duration;

use noise::{NoiseFn, Perlin};
use rustc_hash::{FxHashMap, FxHashSet};

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
            v += perlin.get([x.mul_add(f, 0.5), z.mul_add(f, 0.5)]) * a;
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
        v_diff.mul_add(factor as f64, v_start)
    }
}

#[cfg(test)]
mod noise_tests {
    use noise::Perlin;

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
        let perlin = Perlin::new(0);

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

/// Generator implements a world generator that uses cfg to generate a perlin noise heightmap and fills chunks with
/// blocks accordingly.
pub struct Generator {
    cfg: Config,
    perlin: Perlin,
    cache: RwLock<GeneratorCache>,
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

struct GeneratorCache {
    /// columns caches the terrain heightmap per (x, z) chunk column to reuse the noise calculation for every chunk
    /// along the y-axis for the same column.
    columns: FxHashMap<(i32, i32), Arc<ChunkColumn>>,
    /// inflight is the set of columns that are currently calculated.
    inflight: FxHashSet<(i32, i32)>,
    /// keys_ordered is a list of generated chunk column keys, where the first element is the oldest.
    keys_ordered: VecDeque<(i32, i32)>,
}

pub struct ChunkColumn {
    pub min_y: i32,
    pub max_y: i32,
    pub height_map: [i16; 32 * 32],
}

impl ChunkColumn {
    fn contains_chunk(&self, chunk_y: i32) -> bool {
        self.min_y <= (chunk_y + 1) * 32 && self.max_y >= chunk_y * 32
    }
}

impl Generator {
    pub fn new(seed: u32, cfg: Config) -> Self {
        Self {
            cfg,
            perlin: Perlin::new(seed),
            cache: RwLock::new(GeneratorCache {
                columns: FxHashMap::default(),
                inflight: FxHashSet::default(),
                keys_ordered: VecDeque::new(),
            }),
        }
    }

    fn get_height_at(&self, x: i32, z: i32) -> i32 {
        let noise_x = x as f64;
        let noise_z = z as f64;

        let height = self.cfg.continentalness.get(&self.perlin, noise_x, noise_z);
        let height = height + self.cfg.erosion.get(&self.perlin, noise_x, noise_z);

        height as i32
    }

    fn get_or_generate_chunk_column(&self, col_x: i32, col_z: i32) -> Arc<ChunkColumn> {
        // fast path
        let column = {
            let cache = self.cache.read().unwrap();
            cache.columns.get(&(col_x, col_z)).cloned()
        };
        if let Some(column) = column {
            return column;
        }

        // slow path
        'retry: loop {
            let mut cache = self.cache.write().unwrap();
            if let Some(column) = cache.columns.get(&(col_x, col_z)) {
                // check if column already exists - in case of racing threads that tried to acquire the write lock
                return Arc::clone(column);
            }

            // check if chunk column generation is already inflight
            if cache.inflight.contains(&(col_x, col_z)) {
                drop(cache); // release write lock - spin loop only needs to reacquire a read lock

                loop {
                    // acquire read lock and check if column is set
                    let (column, still_inflight) = {
                        let cache = self.cache.read().unwrap();
                        let column = cache.columns.get(&(col_x, col_z)).cloned();
                        let still_inflight = cache.inflight.contains(&(col_x, col_z));
                        (column, still_inflight)
                    };
                    if let Some(column) = column {
                        return column;
                    }

                    if !still_inflight {
                        // This can occur if the spin lock is active for too long and another thread has reached the
                        // threshold of columns to keep in the cache and removes the cached column before the spin loop
                        // can pick it up. Since the write lock is no longer acquired, jump back to the beginning of the
                        // slow path and try to acquire the write lock again and repeat the process.
                        continue 'retry;
                    }

                    // if not, sleep and retry
                    thread::sleep(Duration::from_millis(5));
                }
            }

            // mark column generation as inflight
            cache.inflight.insert((col_x, col_z));
            cache.keys_ordered.push_back((col_x, col_z));
            drop(cache); // release write lock for column generation

            let column = self.generate_chunk_column(col_x, col_z);
            let column = Arc::new(column);

            let mut cache = self.cache.write().unwrap();
            cache.columns.insert((col_x, col_z), Arc::clone(&column));
            cache.inflight.remove(&(col_x, col_z));

            // clean up old columns
            if cache.keys_ordered.len() > 500 {
                let key = cache.keys_ordered.pop_front().unwrap();
                cache.columns.remove(&key);
            }

            return column;
        }
    }

    fn generate_chunk_column(&self, col_x: i32, col_z: i32) -> ChunkColumn {
        let mut min_y = i32::MAX;
        let mut max_y = i32::MIN;
        let mut height_map = [0; 32 * 32];

        for z in 0..32 {
            for x in 0..32 {
                let y = self.get_height_at(col_x * 32 + x, col_z * 32 + z);
                min_y = min_y.min(y);
                max_y = max_y.max(y);
                height_map[(z * 32 + x) as usize] = y as i16;
            }
        }

        ChunkColumn { min_y, max_y, height_map }
    }
}

impl ChunkGenerator for Generator {
    fn is_interested_in(&self, pos: &ChunkPos) -> bool {
        let col = self.get_or_generate_chunk_column(pos.x, pos.z);
        col.contains_chunk(pos.y)
    }

    fn generate_chunk(&self, chunk: &mut Chunk) {
        let col = self.get_or_generate_chunk_column(chunk.pos.x, chunk.pos.z);

        let chunk_y = chunk.pos.y * 32;
        chunk.fill_with(|x, y, z| {
            let height = col.height_map[(z * 32 + x) as usize] as i32;
            let height = (height - chunk_y).min(31);

            let y = y as i32;
            if y <= height {
                let block = if y >= height {
                    blocks::GRASS
                } else if y >= height - 3 {
                    blocks::DIRT
                } else {
                    blocks::STONE
                };
                return Some(block);
            }

            None
        });
    }
}

#[cfg(test)]
mod benches {
    use test::Bencher;

    use crate::gamelogic::worldgen;
    use crate::gamelogic::worldgen::{Generator, Noise, SplinePoint};
    use crate::systems::worldgen::ChunkGenerator;
    use crate::world::chunk::{Chunk, ChunkPos, ChunkStorageAllocator};

    /// Benchmarks how quickly an average chunk can be filled with block information.
    ///
    /// Result history (measured on AMD Ryzen 9 7950X 32 Threads)
    /// - naive for loop:               253700 ns/iter (+/- 1478)
    /// - using `fill_with` iterator:   53381 ns/iter (+/- 1458)
    //noinspection DuplicatedCode
    #[bench]
    fn some_bench(b: &mut Bencher) {
        // default worldgen configuration for testing
        let cfg = worldgen::Config {
            sea_level: 70,
            continentalness: Noise {
                frequency: 0.001,
                octaves: 3,
                spline_points: vec![
                    SplinePoint { x: -1.0, y: 20.0 },
                    SplinePoint { x: 0.4, y: 50.0 },
                    SplinePoint { x: 0.6, y: 70.0 },
                    SplinePoint { x: 0.8, y: 120.0 },
                    SplinePoint { x: 0.9, y: 190.0 },
                    SplinePoint { x: 1.0, y: 200.0 },
                ],
            },
            erosion: Noise {
                frequency: 0.01,
                octaves: 4,
                spline_points: vec![
                    SplinePoint { x: -1.0, y: -10.0 },
                    SplinePoint { x: 1.0, y: 4.0 },
                ],
            },
        };
        let gen = Generator::new(1, cfg);
        let alloc = ChunkStorageAllocator::new();
        let pos = ChunkPos::new(-1, 1, 5); // default, non-empty chunk

        let _profiler = dhat::Profiler::builder().testing().build();

        // iter code is similar to crate::systems::worldgen::Generator::enqueue_chunk
        b.iter(|| {
            if !gen.is_interested_in(&pos) {
                return None;
            }

            let mut chunk = Chunk::new(pos, 5, alloc.allocate());
            gen.generate_chunk(&mut chunk);

            Some(chunk)
        });

        let stats = dhat::HeapStats::get();
        println!("{:?}", stats);
    }
}
