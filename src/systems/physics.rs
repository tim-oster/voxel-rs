use std::cell::Ref;

use cgmath::{EuclideanSpace, InnerSpace, Point3, Vector3};

use crate::graphics::svo::Svo;
use crate::graphics::svo_picker::{AABB, AABBResult, PickerBatch, PickerResult, RayResult};
use crate::world::chunk::ChunkPos;

const EPSILON: f32 = 0.005;

pub struct Entity {
    pub position: Point3<f32>,
    pub velocity: Vector3<f32>,
    pub aabb_def: AABBDef,
    pub caps: EntityCapabilities,
    state: EntityState,
}

pub struct EntityState {
    pub is_grounded: bool,
}

pub struct EntityCapabilities {
    pub wall_clip: bool,
    pub flying: bool,
    pub gravity: f32,
    // TODO move max velocity out of here?
    pub max_velocity: f32, // TODO differentiate between vertical & horizontal? use vec3?
}

impl Entity {
    pub fn new(position: Point3<f32>, aabb_def: AABBDef) -> Entity {
        Entity {
            position,
            velocity: Vector3::new(0.0, 0.0, 0.0),
            aabb_def,
            caps: EntityCapabilities {
                wall_clip: false,
                flying: false,
                gravity: 0.0,
                max_velocity: 1.0,
            },
            state: EntityState {
                is_grounded: false,
            },
        }
    }

    pub fn get_state(&self) -> &EntityState {
        &self.state
    }
}

pub struct AABBDef {
    pub offset: Vector3<f32>,
    pub extents: Vector3<f32>,
}

impl AABBDef {
    pub fn new(offset: Vector3<f32>, extents: Vector3<f32>) -> AABBDef {
        AABBDef { offset, extents }
    }
}

pub struct Physics {}

impl Physics {
    pub fn new() -> Physics {
        Physics {}
    }

    pub fn step(&self, delta_time: f32, svo: &Svo, entities: Vec<&mut Entity>) {
        let mut entities = entities;

        // TODO move to sub method
        struct RaycastData {
            aabb: AABB,
            aabb_result: RayResult<AABBResult>,
        }

        let mut raycast_batch = PickerBatch::new();
        let mut results = Vec::with_capacity(entities.len());
        for entity in entities.iter() {
            let aabb = AABB::new(entity.position, entity.aabb_def.offset, entity.aabb_def.extents);
            results.push(RaycastData {
                aabb,
                aabb_result: raycast_batch.aabb(&aabb),
            });
        }
        svo.raycast(raycast_batch);

        for (entity, data) in entities.iter_mut().zip(results.iter()) {
            let mut state = EntityState {
                is_grounded: false,
            };

            let res = data.aabb_result.get();
            if res.neg.y < 0.02 && res.neg.y != -1.0 {
                state.is_grounded = true;
            }

            // TODO use gravity variable & max velocity var instead
            const MAX_FALL_VELOCITY: f32 = 2.0;
            const ACCELERATION: f32 = 0.008;

            if !entity.caps.flying {
                entity.velocity.y -= ACCELERATION;
                entity.velocity.y = entity.velocity.y.clamp(-MAX_FALL_VELOCITY, MAX_FALL_VELOCITY);

                if !entity.caps.wall_clip {
                    // TODO merge into one method of find axis agnostic implementation
                    entity.velocity = Physics::apply_horizontal_physics(entity.velocity, &data.aabb, &res);
                }
                entity.velocity = Physics::apply_vertical_physics(entity.velocity, &data.aabb, &res);
            }

            // TODO should this be applied to falling as well or only to horizontal movement?
            let speed = entity.velocity.magnitude();
            let max_speed = entity.caps.max_velocity;
            if speed > max_speed {
                entity.velocity = entity.velocity.normalize() * max_speed;
            }

            entity.position += entity.velocity * delta_time;

            entity.velocity.x = 0.0;
            if entity.caps.flying { entity.velocity.y = 0.0; }
            entity.velocity.z = 0.0;

            entity.state = state;
        }
    }

    fn apply_horizontal_physics(speed: Vector3<f32>, aabb: &AABB, res: &Ref<AABBResult>) -> Vector3<f32> {
        let x_dot = speed.dot(Vector3::new(1.0, 0.0, 0.0));
        let z_dot = speed.dot(Vector3::new(0.0, 0.0, 1.0));
        let x_dst = if x_dot > 0.0 { res.pos.x } else { res.neg.x };
        let z_dst = if z_dot > 0.0 { res.pos.z } else { res.neg.z };

        let mut speed = speed;

        if x_dst == 0.0 {
            if x_dot > 0.0 {
                let actual = aabb.pos.x + aabb.offset.x + aabb.extents.x;
                let expected = actual.floor() - 2.0 * EPSILON;
                speed.x = expected - actual;
            } else {
                let actual = aabb.pos.x + aabb.offset.x;
                let expected = actual.ceil() + 2.0 * EPSILON;
                speed.x = expected - actual;
            }
        } else if x_dst != -1.0 && speed.x.abs() > (x_dst - EPSILON) {
            speed.x = (x_dst - 2.0 * EPSILON) * speed.x.signum();
        }

        if z_dst == 0.0 {
            if z_dot > 0.0 {
                let actual = aabb.pos.z + aabb.offset.z + aabb.extents.z;
                let expected = actual.floor() - 2.0 * EPSILON;
                speed.x = expected - actual;
            } else {
                let actual = aabb.pos.z + aabb.offset.z;
                let expected = actual.ceil() + 2.0 * EPSILON;
                speed.z = expected - actual;
            }
        } else if z_dst != -1.0 && speed.z.abs() > (z_dst - EPSILON) {
            speed.z = (z_dst - 2.0 * EPSILON) * speed.z.signum();
        }

        speed
    }

    fn apply_vertical_physics(speed: Vector3<f32>, aabb: &AABB, res: &Ref<AABBResult>) -> Vector3<f32> {
        let y_dot = speed.dot(Vector3::new(0.0, 1.0, 0.0));
        let y_dst = if y_dot > 0.0 { res.pos.y } else { res.neg.y };

        let mut speed = speed;
        if y_dst == 0.0 {
            if y_dot > 0.0 {
                let actual = aabb.pos.y + aabb.offset.y + aabb.extents.y;
                let expected = actual.floor() - 2.0 * EPSILON;
                speed.y = expected - actual;
            } else {
                let actual = aabb.pos.y + aabb.offset.y;
                let expected = actual.ceil() + 2.0 * EPSILON;
                speed.y = expected - actual;
            }
        } else if y_dst != -1.0 && speed.y.abs() > (y_dst - EPSILON) {
            speed.y = (y_dst - 2.0 * EPSILON) * speed.y.signum();
        }
        speed
    }
}
