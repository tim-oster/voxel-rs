use cgmath::{Point3, Vector3};

use crate::graphics::macros::{AlignedPoint3, AlignedVec3};
use crate::graphics::svo::CoordSpace;

#[derive(Debug, PartialEq)]
pub struct PickerBatch {
    rays: Vec<Ray>,
    aabbs: Vec<AABB>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PickerTask {
    pub max_dst: f32,
    pub pos: AlignedPoint3<f32>,
    pub dir: AlignedVec3<f32>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PickerResult {
    /// dst is the distance to the hit object.
    pub dst: f32,
    /// inside_voxel is true, if the ray origin is within a voxel itself.
    pub inside_voxel: bool,
    /// pos in world space, where the ray hit something.
    pub pos: AlignedPoint3<f32>,
    /// normal is the normal direction where the ray hit.
    pub normal: AlignedVec3<f32>,
}

/// PickerBatch keeps tracks of Rays and AABBs, serializes and deserializes them when raycast
/// again a SVO.
impl PickerBatch {
    pub fn new() -> PickerBatch {
        PickerBatch { rays: Vec::new(), aabbs: Vec::new() }
    }

    pub fn add_ray(&mut self, pos: Point3<f32>, dir: Vector3<f32>, max_dst: f32) {
        self.rays.push(Ray { pos, dir, max_dst });
    }

    pub fn add_aabb(&mut self, aabb: AABB) {
        self.aabbs.push(aabb);
    }

    /// serialize_tasks transforms all tasks on this batch into actual PickerTasks and writes them
    /// to the given task buffer.
    pub(crate) fn serialize_tasks(&self, tasks: &mut [PickerTask], cs: Option<CoordSpace>) -> usize {
        let mut offset = 0;

        for task in &self.rays {
            let mut pos = task.pos;
            if let Some(cs) = cs {
                pos = cs.cnv_into_space(pos);
            }
            tasks[offset] = PickerTask {
                max_dst: task.max_dst,
                pos: AlignedPoint3(pos),
                dir: AlignedVec3(task.dir),
            };
            offset += 1;
        }

        for aabb in &self.aabbs {
            for mut task in aabb.generate_picker_tasks() {
                if let Some(cs) = cs {
                    task.pos = AlignedPoint3(cs.cnv_into_space(task.pos.0));
                }
                tasks[offset] = task;
                offset += 1;
            }
        }

        offset
    }

    /// deserialize_results reads all results from the given result buffer and parses the results
    /// for all jobs on this batch.
    pub(crate) fn deserialize_results(&self, results: &[PickerResult], cs: Option<CoordSpace>) -> PickerBatchResult {
        let mut offset = 0;
        let mut batch_result = PickerBatchResult { rays: Vec::new(), aabbs: Vec::new() };

        for _ in &self.rays {
            let mut result = results[offset];
            offset += 1;

            if let Some(cs) = cs {
                result.pos = AlignedPoint3(cs.cnv_out_of_space(result.pos.0));
            }

            batch_result.rays.push(RayResult {
                dst: result.dst,
                inside_block: result.inside_voxel,
                pos: result.pos.0,
                normal: result.normal.0,
            });
        }

        for aabb in &self.aabbs {
            let (result, consumed) = aabb.parse_picker_results(&results[offset..]);
            batch_result.aabbs.push(result);
            offset += consumed;
        }

        batch_result
    }
}

#[derive(Debug, PartialEq)]
pub struct PickerBatchResult {
    pub rays: Vec<RayResult>,
    pub aabbs: Vec<AABBResult>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray {
    pos: Point3<f32>,
    dir: Vector3<f32>,
    max_dst: f32,
}

/// RayResult represent a ray intersection with a voxel. Only if dst != -1.0, are any of the other
/// fields valid. If a ray is cast from within a voxel, no intersection is returned for the voxel
/// from within the cast originates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayResult {
    pub dst: f32,
    pub inside_block: bool,
    pub pos: Point3<f32>,
    pub normal: Vector3<f32>,
}

impl RayResult {
    pub fn did_hit(&self) -> bool {
        self.dst != -1.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AABB {
    pub pos: Point3<f32>,
    pub offset: Vector3<f32>,
    pub extents: Vector3<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AABBResult {
    pub neg: Vector3<f32>,
    pub pos: Vector3<f32>,
}

impl Default for AABBResult {
    fn default() -> Self {
        AABBResult {
            neg: Vector3::new(-1.0, -1.0, -1.0),
            pos: Vector3::new(-1.0, -1.0, -1.0),
        }
    }
}

impl AABB {
    pub fn new(pos: Point3<f32>, offset: Vector3<f32>, extents: Vector3<f32>) -> AABB {
        AABB { pos, offset, extents }
    }

    fn generate_picker_tasks(&self) -> Vec<PickerTask> {
        let blocks_per_axis = vec![
            self.extents.x.ceil() as i32,
            self.extents.y.ceil() as i32,
            self.extents.z.ceil() as i32,
        ];
        let step_size_per_axis = vec![
            self.extents.x / blocks_per_axis[0] as f32,
            self.extents.y / blocks_per_axis[1] as f32,
            self.extents.z / blocks_per_axis[2] as f32,
        ];

        let mut tasks = Vec::new();

        // go through all block points across the AABB
        for x in 0..=blocks_per_axis[0] {
            for y in 0..=blocks_per_axis[1] {
                for z in 0..=blocks_per_axis[2] {
                    // At each point, up to three rays have to be cast: 3 on corners, 2 on sides,
                    // and 1 on faces. Iterate through all axes.
                    for (i, v) in vec![x, y, z].into_iter().enumerate() {
                        // if the block position of the current axis is at neither start nor end,
                        // skip it
                        if v != 0 && v != blocks_per_axis[i] {
                            continue;
                        }

                        // dir returns the normal direction for the current point per axis,
                        // where index of 0=x, 1=y, 2=z
                        let dir = |index: i32| {
                            if index == i as i32 {
                                if v == 0 { return -1.0; }
                                return 1.0;
                            }
                            0.0
                        };

                        let point = Vector3::new(
                            x as f32 * step_size_per_axis[0],
                            y as f32 * step_size_per_axis[1],
                            z as f32 * step_size_per_axis[2],
                        );
                        tasks.push(PickerTask {
                            max_dst: 10.0,
                            pos: AlignedPoint3(self.pos + self.offset + point),
                            dir: AlignedVec3(Vector3::new(dir(0), dir(1), dir(2))),
                        });
                    }
                }
            }
        }
        tasks
    }

    fn parse_picker_results(&self, data: &[PickerResult]) -> (AABBResult, usize) {
        let blocks_per_axis = vec![
            self.extents.x.ceil() as i32,
            self.extents.y.ceil() as i32,
            self.extents.z.ceil() as i32,
        ];

        let mut result = AABBResult::default();
        let mut references = vec![
            &mut result.pos.x, &mut result.neg.x,
            &mut result.pos.y, &mut result.neg.y,
            &mut result.pos.z, &mut result.neg.z,
        ];

        let mut res_index: usize = 0;

        // Go through all possible block points across the AABB using the same logic as in
        // `generate_picker_tasks` and keep the shortest distance from every hit per axis
        // in positive and negative direction.
        for x in 0..=blocks_per_axis[0] {
            for y in 0..=blocks_per_axis[1] {
                for z in 0..=blocks_per_axis[2] {
                    for (i, v) in vec![x, y, z].into_iter().enumerate() {
                        if v != 0 && v != blocks_per_axis[i] {
                            continue;
                        }

                        let dst = data[res_index].dst;
                        res_index += 1;
                        if dst == -1.0 {
                            continue;
                        }

                        let ref_index = i * 2 + if v == 0 { 1 } else { 0 };
                        if *references[ref_index] == -1.0 {
                            *references[ref_index] = dst;
                        } else {
                            *references[ref_index] = references[ref_index].min(dst);
                        }
                    }
                }
            }
        }
        (result, res_index)
    }
}

#[cfg(test)]
mod tests {
    use cgmath::{Point3, Vector3};

    use crate::graphics::macros::{AlignedPoint3, AlignedVec3};
    use crate::graphics::svo_picker::{AABB, AABBResult, PickerBatch, PickerBatchResult, PickerResult, PickerTask, RayResult};

    /// Tests if task serialization works as expected.
    #[test]
    fn picker_batch_serialization() {
        let mut batch = PickerBatch::new();
        batch.add_ray(Point3::new(1.0, 0.0, 1.0), Vector3::new(0.0, 1.0, 0.0), 20.0);
        batch.add_ray(Point3::new(2.0, 0.0, 2.0), Vector3::new(1.0, 0.0, 0.0), 40.0);
        batch.add_aabb(AABB {
            pos: Point3::new(0.5, 0.0, 0.5),
            offset: Vector3::new(-0.5, 0.0, -0.5),
            extents: Vector3::new(1.0, 1.0, 1.0),
        });
        batch.add_aabb(AABB {
            pos: Point3::new(0.0, 0.0, 0.0),
            offset: Vector3::new(0.0, 0.0, 0.0),
            extents: Vector3::new(1.5, 1.5, 1.5),
        });

        let default_task = PickerTask { max_dst: 0.0, pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.0)), dir: AlignedVec3(Vector3::new(0.0, 0.0, 0.0)) };
        let mut buffer = vec![default_task; 100];
        let tasks = batch.serialize_tasks(&mut buffer, None);

        // [2 rays] + [1 unit size aabb * ( 3 rays per corner * 8 corners )] + [1 irregular aabb * ( 3 rays per corner * 8 corners + 2 rays per half side * 4 halves per axis * 3 axis + 1 ray per face * 6 face )]
        // [ 2 ] + [ 24 ] + [ 54 ] = 80
        assert_eq!(tasks, 80);
        assert_eq!(buffer[..tasks], vec![
            // rays
            PickerTask { max_dst: 20.0, pos: AlignedPoint3(Point3::new(1.0, 0.0, 1.0)), dir: AlignedVec3(Vector3::new(0.0, 1.0, 0.0)) },
            PickerTask { max_dst: 40.0, pos: AlignedPoint3(Point3::new(2.0, 0.0, 2.0)), dir: AlignedVec3(Vector3::new(1.0, 0.0, 0.0)) },
            // aabb 1
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.0)), dir: AlignedVec3(Vector3::new(-1.0, 0.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.0)), dir: AlignedVec3(Vector3::new(0.0, -1.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.0)), dir: AlignedVec3(Vector3::new(0.0, 0.0, -1.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.0, 0.0, 1.0)), dir: AlignedVec3(Vector3::new(-1.0, 0.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.0, 0.0, 1.0)), dir: AlignedVec3(Vector3::new(0.0, -1.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.0, 0.0, 1.0)), dir: AlignedVec3(Vector3::new(0.0, 0.0, 1.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.0, 1.0, 0.0)), dir: AlignedVec3(Vector3::new(-1.0, 0.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.0, 1.0, 0.0)), dir: AlignedVec3(Vector3::new(0.0, 1.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.0, 1.0, 0.0)), dir: AlignedVec3(Vector3::new(0.0, 0.0, -1.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.0, 1.0, 1.0)), dir: AlignedVec3(Vector3::new(-1.0, 0.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.0, 1.0, 1.0)), dir: AlignedVec3(Vector3::new(0.0, 1.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.0, 1.0, 1.0)), dir: AlignedVec3(Vector3::new(0.0, 0.0, 1.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(1.0, 0.0, 0.0)), dir: AlignedVec3(Vector3::new(1.0, 0.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(1.0, 0.0, 0.0)), dir: AlignedVec3(Vector3::new(0.0, -1.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(1.0, 0.0, 0.0)), dir: AlignedVec3(Vector3::new(0.0, 0.0, -1.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(1.0, 0.0, 1.0)), dir: AlignedVec3(Vector3::new(1.0, 0.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(1.0, 0.0, 1.0)), dir: AlignedVec3(Vector3::new(0.0, -1.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(1.0, 0.0, 1.0)), dir: AlignedVec3(Vector3::new(0.0, 0.0, 1.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(1.0, 1.0, 0.0)), dir: AlignedVec3(Vector3::new(1.0, 0.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(1.0, 1.0, 0.0)), dir: AlignedVec3(Vector3::new(0.0, 1.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(1.0, 1.0, 0.0)), dir: AlignedVec3(Vector3::new(0.0, 0.0, -1.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(1.0, 1.0, 1.0)), dir: AlignedVec3(Vector3::new(1.0, 0.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(1.0, 1.0, 1.0)), dir: AlignedVec3(Vector3::new(0.0, 1.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(1.0, 1.0, 1.0)), dir: AlignedVec3(Vector3::new(0.0, 0.0, 1.0)) },
            // aabb 2
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.0)), dir: AlignedVec3(Vector3::new(-1.0, 0.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.0)), dir: AlignedVec3(Vector3::new(0.0, -1.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.0)), dir: AlignedVec3(Vector3::new(0.0, 0.0, -1.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.75)), dir: AlignedVec3(Vector3::new(-1.0, 0.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.75)), dir: AlignedVec3(Vector3::new(0.0, -1.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.0, 0.0, 1.5)), dir: AlignedVec3(Vector3::new(-1.0, 0.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.0, 0.0, 1.5)), dir: AlignedVec3(Vector3::new(0.0, -1.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.0, 0.0, 1.5)), dir: AlignedVec3(Vector3::new(0.0, 0.0, 1.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.0, 0.75, 0.0)), dir: AlignedVec3(Vector3::new(-1.0, 0.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.0, 0.75, 0.0)), dir: AlignedVec3(Vector3::new(0.0, 0.0, -1.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.0, 0.75, 0.75)), dir: AlignedVec3(Vector3::new(-1.0, 0.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.0, 0.75, 1.5)), dir: AlignedVec3(Vector3::new(-1.0, 0.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.0, 0.75, 1.5)), dir: AlignedVec3(Vector3::new(0.0, 0.0, 1.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.0, 1.5, 0.0)), dir: AlignedVec3(Vector3::new(-1.0, 0.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.0, 1.5, 0.0)), dir: AlignedVec3(Vector3::new(0.0, 1.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.0, 1.5, 0.0)), dir: AlignedVec3(Vector3::new(0.0, 0.0, -1.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.0, 1.5, 0.75)), dir: AlignedVec3(Vector3::new(-1.0, 0.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.0, 1.5, 0.75)), dir: AlignedVec3(Vector3::new(0.0, 1.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.0, 1.5, 1.5)), dir: AlignedVec3(Vector3::new(-1.0, 0.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.0, 1.5, 1.5)), dir: AlignedVec3(Vector3::new(0.0, 1.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.0, 1.5, 1.5)), dir: AlignedVec3(Vector3::new(0.0, 0.0, 1.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.75, 0.0, 0.0)), dir: AlignedVec3(Vector3::new(0.0, -1.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.75, 0.0, 0.0)), dir: AlignedVec3(Vector3::new(0.0, 0.0, -1.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.75, 0.0, 0.75)), dir: AlignedVec3(Vector3::new(0.0, -1.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.75, 0.0, 1.5)), dir: AlignedVec3(Vector3::new(0.0, -1.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.75, 0.0, 1.5)), dir: AlignedVec3(Vector3::new(0.0, 0.0, 1.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.75, 0.75, 0.0)), dir: AlignedVec3(Vector3::new(0.0, 0.0, -1.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.75, 0.75, 1.5)), dir: AlignedVec3(Vector3::new(0.0, 0.0, 1.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.75, 1.5, 0.0)), dir: AlignedVec3(Vector3::new(0.0, 1.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.75, 1.5, 0.0)), dir: AlignedVec3(Vector3::new(0.0, 0.0, -1.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.75, 1.5, 0.75)), dir: AlignedVec3(Vector3::new(0.0, 1.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.75, 1.5, 1.5)), dir: AlignedVec3(Vector3::new(0.0, 1.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(0.75, 1.5, 1.5)), dir: AlignedVec3(Vector3::new(0.0, 0.0, 1.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(1.5, 0.0, 0.0)), dir: AlignedVec3(Vector3::new(1.0, 0.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(1.5, 0.0, 0.0)), dir: AlignedVec3(Vector3::new(0.0, -1.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(1.5, 0.0, 0.0)), dir: AlignedVec3(Vector3::new(0.0, 0.0, -1.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(1.5, 0.0, 0.75)), dir: AlignedVec3(Vector3::new(1.0, 0.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(1.5, 0.0, 0.75)), dir: AlignedVec3(Vector3::new(0.0, -1.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(1.5, 0.0, 1.5)), dir: AlignedVec3(Vector3::new(1.0, 0.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(1.5, 0.0, 1.5)), dir: AlignedVec3(Vector3::new(0.0, -1.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(1.5, 0.0, 1.5)), dir: AlignedVec3(Vector3::new(0.0, 0.0, 1.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(1.5, 0.75, 0.0)), dir: AlignedVec3(Vector3::new(1.0, 0.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(1.5, 0.75, 0.0)), dir: AlignedVec3(Vector3::new(0.0, 0.0, -1.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(1.5, 0.75, 0.75)), dir: AlignedVec3(Vector3::new(1.0, 0.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(1.5, 0.75, 1.5)), dir: AlignedVec3(Vector3::new(1.0, 0.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(1.5, 0.75, 1.5)), dir: AlignedVec3(Vector3::new(0.0, 0.0, 1.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(1.5, 1.5, 0.0)), dir: AlignedVec3(Vector3::new(1.0, 0.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(1.5, 1.5, 0.0)), dir: AlignedVec3(Vector3::new(0.0, 1.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(1.5, 1.5, 0.0)), dir: AlignedVec3(Vector3::new(0.0, 0.0, -1.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(1.5, 1.5, 0.75)), dir: AlignedVec3(Vector3::new(1.0, 0.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(1.5, 1.5, 0.75)), dir: AlignedVec3(Vector3::new(0.0, 1.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(1.5, 1.5, 1.5)), dir: AlignedVec3(Vector3::new(1.0, 0.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(1.5, 1.5, 1.5)), dir: AlignedVec3(Vector3::new(0.0, 1.0, 0.0)) },
            PickerTask { max_dst: 10.0, pos: AlignedPoint3(Point3::new(1.5, 1.5, 1.5)), dir: AlignedVec3(Vector3::new(0.0, 0.0, 1.0)) },
        ]);
    }

    /// Tests if task deserialization works as expected.
    #[test]
    fn picker_batch_deserialization() {
        let mut batch = PickerBatch::new();
        batch.add_ray(Point3::new(0.0, 0.0, 0.0), Vector3::new(-1.0, 0.0, 0.0), 20.0);
        batch.add_ray(Point3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0), 20.0);
        batch.add_aabb(AABB {
            pos: Point3::new(0.5, 0.0, 0.5),
            offset: Vector3::new(-0.5, 0.0, -0.5),
            extents: Vector3::new(1.0, 1.0, 1.0),
        });
        batch.add_aabb(AABB {
            pos: Point3::new(0.0, 0.0, 0.0),
            offset: Vector3::new(0.0, 0.0, 0.0),
            extents: Vector3::new(1.5, 1.5, 1.5),
        });

        let buffer = vec![
            // rays
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.0)), normal: AlignedVec3(Vector3::new(0.0, 0.0, 0.0)) },
            PickerResult { dst: 10.0, inside_voxel: true, pos: AlignedPoint3(Point3::new(-1.0, 0.0, 0.0)), normal: AlignedVec3(Vector3::new(10.0, 0.0, 0.0)) },
            // aabb 1
            PickerResult { dst: 8.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.0)), normal: AlignedVec3(Vector3::new(-1.0, 0.0, 0.0)) },
            PickerResult { dst: 8.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.0)), normal: AlignedVec3(Vector3::new(0.0, -1.0, 0.0)) },
            PickerResult { dst: 8.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.0)), normal: AlignedVec3(Vector3::new(0.0, 0.0, -1.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.0)), normal: AlignedVec3(Vector3::new(-1.0, 0.0, 0.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.0)), normal: AlignedVec3(Vector3::new(0.0, -1.0, 0.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.0)), normal: AlignedVec3(Vector3::new(0.0, 0.0, 1.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.0)), normal: AlignedVec3(Vector3::new(-1.0, 0.0, 0.0)) },
            PickerResult { dst: 4.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.0)), normal: AlignedVec3(Vector3::new(0.0, 1.0, 0.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.0)), normal: AlignedVec3(Vector3::new(0.0, 0.0, -1.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.0)), normal: AlignedVec3(Vector3::new(-1.0, 0.0, 0.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.0)), normal: AlignedVec3(Vector3::new(0.0, 1.0, 0.0)) },
            PickerResult { dst: 4.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.0)), normal: AlignedVec3(Vector3::new(0.0, 0.0, 1.0)) },
            PickerResult { dst: 4.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.0)), normal: AlignedVec3(Vector3::new(1.0, 0.0, 0.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.0)), normal: AlignedVec3(Vector3::new(0.0, -1.0, 0.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.0)), normal: AlignedVec3(Vector3::new(0.0, 0.0, -1.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.0)), normal: AlignedVec3(Vector3::new(1.0, 0.0, 0.0)) },
            PickerResult { dst: 7.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.0)), normal: AlignedVec3(Vector3::new(0.0, -1.0, 0.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.0)), normal: AlignedVec3(Vector3::new(0.0, 0.0, 1.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.0)), normal: AlignedVec3(Vector3::new(1.0, 0.0, 0.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.0)), normal: AlignedVec3(Vector3::new(0.0, 1.0, 0.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.0)), normal: AlignedVec3(Vector3::new(0.0, 0.0, -1.0)) },
            PickerResult { dst: 2.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.0)), normal: AlignedVec3(Vector3::new(1.0, 0.0, 0.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.0)), normal: AlignedVec3(Vector3::new(0.0, 1.0, 0.0)) },
            PickerResult { dst: 1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.0)), normal: AlignedVec3(Vector3::new(0.0, 0.0, 1.0)) },
            // aabb 2
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.0)), normal: AlignedVec3(Vector3::new(-1.0, 0.0, 0.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.0)), normal: AlignedVec3(Vector3::new(0.0, -1.0, 0.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.0)), normal: AlignedVec3(Vector3::new(0.0, 0.0, -1.0)) },
            PickerResult { dst: 9.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.75)), normal: AlignedVec3(Vector3::new(-1.0, 0.0, 0.0)) },
            PickerResult { dst: 8.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 0.0, 0.75)), normal: AlignedVec3(Vector3::new(0.0, -1.0, 0.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 0.0, 1.5)), normal: AlignedVec3(Vector3::new(-1.0, 0.0, 0.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 0.0, 1.5)), normal: AlignedVec3(Vector3::new(0.0, -1.0, 0.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 0.0, 1.5)), normal: AlignedVec3(Vector3::new(0.0, 0.0, 1.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 0.75, 0.0)), normal: AlignedVec3(Vector3::new(-1.0, 0.0, 0.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 0.75, 0.0)), normal: AlignedVec3(Vector3::new(0.0, 0.0, -1.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 0.75, 0.75)), normal: AlignedVec3(Vector3::new(-1.0, 0.0, 0.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 0.75, 1.5)), normal: AlignedVec3(Vector3::new(-1.0, 0.0, 0.0)) },
            PickerResult { dst: 5.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 0.75, 1.5)), normal: AlignedVec3(Vector3::new(0.0, 0.0, 1.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 1.5, 0.0)), normal: AlignedVec3(Vector3::new(-1.0, 0.0, 0.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 1.5, 0.0)), normal: AlignedVec3(Vector3::new(0.0, 1.0, 0.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 1.5, 0.0)), normal: AlignedVec3(Vector3::new(0.0, 0.0, -1.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 1.5, 0.75)), normal: AlignedVec3(Vector3::new(-1.0, 0.0, 0.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 1.5, 0.75)), normal: AlignedVec3(Vector3::new(0.0, 1.0, 0.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 1.5, 1.5)), normal: AlignedVec3(Vector3::new(-1.0, 0.0, 0.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 1.5, 1.5)), normal: AlignedVec3(Vector3::new(0.0, 1.0, 0.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.0, 1.5, 1.5)), normal: AlignedVec3(Vector3::new(0.0, 0.0, 1.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.75, 0.0, 0.0)), normal: AlignedVec3(Vector3::new(0.0, -1.0, 0.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.75, 0.0, 0.0)), normal: AlignedVec3(Vector3::new(0.0, 0.0, -1.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.75, 0.0, 0.75)), normal: AlignedVec3(Vector3::new(0.0, -1.0, 0.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.75, 0.0, 1.5)), normal: AlignedVec3(Vector3::new(0.0, -1.0, 0.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.75, 0.0, 1.5)), normal: AlignedVec3(Vector3::new(0.0, 0.0, 1.0)) },
            PickerResult { dst: 7.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.75, 0.75, 0.0)), normal: AlignedVec3(Vector3::new(0.0, 0.0, -1.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.75, 0.75, 1.5)), normal: AlignedVec3(Vector3::new(0.0, 0.0, 1.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.75, 1.5, 0.0)), normal: AlignedVec3(Vector3::new(0.0, 1.0, 0.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.75, 1.5, 0.0)), normal: AlignedVec3(Vector3::new(0.0, 0.0, -1.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.75, 1.5, 0.75)), normal: AlignedVec3(Vector3::new(0.0, 1.0, 0.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.75, 1.5, 1.5)), normal: AlignedVec3(Vector3::new(0.0, 1.0, 0.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(0.75, 1.5, 1.5)), normal: AlignedVec3(Vector3::new(0.0, 0.0, 1.0)) },
            PickerResult { dst: 5.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(1.5, 0.0, 0.0)), normal: AlignedVec3(Vector3::new(1.0, 0.0, 0.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(1.5, 0.0, 0.0)), normal: AlignedVec3(Vector3::new(0.0, -1.0, 0.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(1.5, 0.0, 0.0)), normal: AlignedVec3(Vector3::new(0.0, 0.0, -1.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(1.5, 0.0, 0.75)), normal: AlignedVec3(Vector3::new(1.0, 0.0, 0.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(1.5, 0.0, 0.75)), normal: AlignedVec3(Vector3::new(0.0, -1.0, 0.0)) },
            PickerResult { dst: 5.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(1.5, 0.0, 1.5)), normal: AlignedVec3(Vector3::new(1.0, 0.0, 0.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(1.5, 0.0, 1.5)), normal: AlignedVec3(Vector3::new(0.0, -1.0, 0.0)) },
            PickerResult { dst: 3.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(1.5, 0.0, 1.5)), normal: AlignedVec3(Vector3::new(0.0, 0.0, 1.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(1.5, 0.75, 0.0)), normal: AlignedVec3(Vector3::new(1.0, 0.0, 0.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(1.5, 0.75, 0.0)), normal: AlignedVec3(Vector3::new(0.0, 0.0, -1.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(1.5, 0.75, 0.75)), normal: AlignedVec3(Vector3::new(1.0, 0.0, 0.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(1.5, 0.75, 1.5)), normal: AlignedVec3(Vector3::new(1.0, 0.0, 0.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(1.5, 0.75, 1.5)), normal: AlignedVec3(Vector3::new(0.0, 0.0, 1.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(1.5, 1.5, 0.0)), normal: AlignedVec3(Vector3::new(1.0, 0.0, 0.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(1.5, 1.5, 0.0)), normal: AlignedVec3(Vector3::new(0.0, 1.0, 0.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(1.5, 1.5, 0.0)), normal: AlignedVec3(Vector3::new(0.0, 0.0, -1.0)) },
            PickerResult { dst: 1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(1.5, 1.5, 0.75)), normal: AlignedVec3(Vector3::new(1.0, 0.0, 0.0)) },
            PickerResult { dst: 4.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(1.5, 1.5, 0.75)), normal: AlignedVec3(Vector3::new(0.0, 1.0, 0.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(1.5, 1.5, 1.5)), normal: AlignedVec3(Vector3::new(1.0, 0.0, 0.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(1.5, 1.5, 1.5)), normal: AlignedVec3(Vector3::new(0.0, 1.0, 0.0)) },
            PickerResult { dst: -1.0, inside_voxel: false, pos: AlignedPoint3(Point3::new(1.5, 1.5, 1.5)), normal: AlignedVec3(Vector3::new(0.0, 0.0, 1.0)) },
        ];

        let result = batch.deserialize_results(&buffer, None);

        assert_eq!(result, PickerBatchResult {
            rays: vec![
                RayResult { dst: -1.0, inside_block: false, pos: Point3::new(0.0, 0.0, 0.0), normal: Vector3::new(0.0, 0.0, 0.0) },
                RayResult { dst: 10.0, inside_block: true, pos: Point3::new(-1.0, 0.0, 0.0), normal: Vector3::new(10.0, 0.0, 0.0) },
            ],
            aabbs: vec![
                AABBResult { neg: Vector3::new(8.0, 7.0, 8.0), pos: Vector3::new(2.0, 4.0, 1.0) },
                AABBResult { neg: Vector3::new(9.0, 8.0, 7.0), pos: Vector3::new(1.0, 4.0, 3.0) },
            ],
        });
    }
}
