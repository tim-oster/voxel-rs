use std::cell::{Ref, RefCell};
use std::ops::Deref;
use std::rc::Rc;

use cgmath::{InnerSpace, Point3, Vector3};

use crate::graphics::macros::{AlignedPoint3, AlignedVec3};
use crate::systems::worldsvo::CoordSpace;

// TODO rename picker to something else

pub struct PickerBatch {
    rays: Vec<RayTask>,
    aabbs: Vec<AABBTask>,
}

impl PickerBatch {
    pub fn new() -> PickerBatch {
        PickerBatch {
            rays: Vec::new(),
            aabbs: Vec::new(),
        }
    }

    pub fn ray(&mut self, pos: Point3<f32>, dir: Vector3<f32>, max_dst: f32) -> RayResult<PickerResult> {
        let result = RayResult { value: Rc::new(RefCell::new(Default::default())) };
        let task = RayTask {
            pos,
            dir,
            max_dst,
            result: Rc::clone(&result.value),
        };
        self.rays.push(task);
        result
    }

    pub fn aabb(&mut self, aabb: &AABB) -> RayResult<AABBResult> {
        let result = RayResult { value: Rc::new(RefCell::new(Default::default())) };
        let task = AABBTask {
            aabb: *aabb,
            result: Rc::clone(&result.value),
        };
        self.aabbs.push(task);
        result
    }

    pub(crate) fn serialize_tasks(&self, tasks: &mut [PickerTask], cs: Option<CoordSpace>) {
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

        for task in &self.aabbs {
            for mut task in task.aabb.generate_picker_tasks() {
                if let Some(cs) = cs {
                    task.pos = AlignedPoint3(cs.cnv_into_space(task.pos.0));
                }
                tasks[offset] = task;
                offset += 1;
            }
        }
    }

    pub(crate) fn deserialize_results(&self, results: &[PickerResult], cs: Option<CoordSpace>) {
        let mut offset = 0;

        for task in &self.rays {
            *task.result.borrow_mut() = results[offset];
            offset += 1;

            if let Some(cs) = cs {
                let mut result = task.result.borrow_mut();
                // TODO find a better way to handle -1 values

                let original = result.pos.0;
                result.pos = AlignedPoint3(cs.cnv_out_of_space(result.pos.0));
                if original.x == -1.0 { result.pos.x = -1.0 };
                if original.y == -1.0 { result.pos.y = -1.0 };
                if original.z == -1.0 { result.pos.z = -1.0 };
            }
        }

        for task in &self.aabbs {
            let (result, consumed) = task.aabb.parse_results(&results[offset..]);
            *task.result.borrow_mut() = result;
            offset += consumed;
        }
    }
}

struct RayTask {
    pos: Point3<f32>,
    dir: Vector3<f32>,
    max_dst: f32,
    result: Rc<RefCell<PickerResult>>,
}

struct AABBTask {
    aabb: AABB,
    result: Rc<RefCell<AABBResult>>,
}

pub struct RayResult<T> {
    value: Rc<RefCell<T>>,
}

impl<T> RayResult<T> {
    pub fn get(&self) -> Ref<T> {
        Ref::map(self.value.deref().borrow(), |x| x)
    }
}

#[repr(C)]
pub(crate) struct PickerTask {
    pub max_dst: f32,
    pub pos: AlignedPoint3<f32>,
    pub dir: AlignedVec3<f32>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PickerResult {
    pub dst: f32,
    pub inside_block: bool,
    pub pos: AlignedPoint3<f32>,
    pub normal: AlignedVec3<f32>,
}

impl Default for PickerResult {
    fn default() -> Self {
        PickerResult {
            dst: -1.0,
            inside_block: false,
            pos: AlignedPoint3::new(0.0, 0.0, 0.0),
            normal: AlignedVec3::new(0.0, 0.0, 0.0),
        }
    }
}

impl PickerResult {
    pub fn did_hit(&self) -> bool {
        self.dst != -1.0
    }
}

#[derive(Clone, Copy)]
pub struct AABB {
    pub pos: Point3<f32>,
    pub offset: Vector3<f32>,
    pub extents: Vector3<f32>,
}

#[derive(Clone, Copy)]
pub struct AABBResult {
    pub neg: Point3<f32>,
    pub pos: Point3<f32>,
}

impl Default for AABBResult {
    fn default() -> Self {
        AABBResult {
            neg: Point3::new(-1.0, -1.0, -1.0),
            pos: Point3::new(-1.0, -1.0, -1.0),
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
        for x in 0..=blocks_per_axis[0] {
            for y in 0..=blocks_per_axis[1] {
                for z in 0..=blocks_per_axis[2] {
                    for (i, v) in vec![x, y, z].into_iter().enumerate() {
                        if v == 0 || v == blocks_per_axis[i] {
                            let dir = |index: i32| {
                                if index == i as i32 && (v == 0 || v == blocks_per_axis[i]) {
                                    if v == 0 {
                                        return -1.0;
                                    }
                                    return 1.0;
                                }
                                0.001 // TODO why does straight down not work?
                            };

                            tasks.push(PickerTask {
                                max_dst: 10.0,
                                pos: AlignedPoint3(self.pos + self.offset + Vector3::new(x as f32 * step_size_per_axis[0], y as f32 * step_size_per_axis[1], z as f32 * step_size_per_axis[2])),
                                dir: AlignedVec3(Vector3::new(dir(0), dir(1), dir(2)).normalize()),
                            });
                        }
                    }
                }
            }
        }
        tasks
    }

    fn parse_results(&self, data: &[PickerResult]) -> (AABBResult, usize) {
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
                            *references[ref_index] = if data[res_index].inside_block { 0.0 } else { dst };
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

// TODO write tests
