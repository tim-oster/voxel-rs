use std::ops::DerefMut;
use cgmath::{InnerSpace, Point3, Vector3};
#[cfg(test)]
use mockall::automock;

use crate::graphics::svo::Svo;
use crate::graphics::svo_picker::{AABB, AABBResult, PickerBatch, PickerBatchResult};

const EPSILON: f32 = 0.005;

#[derive(Debug, PartialEq)]
pub struct Entity {
    pub position: Point3<f32>,
    pub velocity: Vector3<f32>,
    pub aabb_def: AABBDef,
    pub caps: EntityCapabilities,
    state: EntityState,
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct EntityState {
    pub is_grounded: bool,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct EntityCapabilities {
    pub wall_clip: bool,
    pub flying: bool,
    pub gravity: f32,
    pub max_fall_velocity: f32,
}

impl Default for EntityCapabilities {
    fn default() -> Self {
        EntityCapabilities {
            wall_clip: false,
            flying: false,
            gravity: 0.006,
            max_fall_velocity: 3.0,
        }
    }
}

impl Entity {
    pub fn new(position: Point3<f32>, aabb_def: AABBDef) -> Entity {
        Entity {
            position,
            velocity: Vector3::new(0.0, 0.0, 0.0),
            aabb_def,
            caps: EntityCapabilities::default(),
            state: EntityState::default(),
        }
    }

    pub fn get_state(&self) -> &EntityState {
        &self.state
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct AABBDef {
    pub offset: Vector3<f32>,
    pub extents: Vector3<f32>,
}

impl AABBDef {
    pub fn new(offset: Vector3<f32>, extents: Vector3<f32>) -> AABBDef {
        AABBDef { offset, extents }
    }
}

#[cfg_attr(test, automock)]
pub trait Raycaster {
    fn cast(&self, batch: PickerBatch) -> PickerBatchResult;
}

impl Raycaster for Svo {
    fn cast(&self, batch: PickerBatch) -> PickerBatchResult {
        self.raycast(batch)
    }
}

pub struct Physics {}

struct RaycastData {
    aabb: AABB,
    aabb_result: AABBResult,
}

impl Physics {
    pub fn new() -> Physics {
        Physics {}
    }

    pub fn step(&self, delta_time: f32, raycaster: &impl Raycaster, entities: &mut [&mut Entity]) {
        let results = Physics::raycast_entities(raycaster, entities);

        for i in 0..entities.len() {
            let data = &results[i];
            Physics::update_entity(entities[i], &data.aabb, &data.aabb_result, delta_time);
        }
    }

    fn raycast_entities(raycaster: &impl Raycaster, entities: &[&mut Entity]) -> Vec<RaycastData> {
        let mut batch = PickerBatch::new();
        let mut results = Vec::with_capacity(entities.len());

        for entity in entities.iter() {
            let aabb = AABB::new(entity.position, entity.aabb_def.offset, entity.aabb_def.extents);
            batch.aabb(aabb);
            results.push(RaycastData {
                aabb,
                aabb_result: AABBResult::default(),
            });
        }

        let batch_result = raycaster.cast(batch);
        for (i, result) in results.iter_mut().enumerate() {
            result.aabb_result = batch_result.aabbs[i];
        }

        results
    }

    fn update_entity(entity: &mut Entity, aabb: &AABB, result: &AABBResult, delta_time: f32) {
        let mut v = entity.velocity * delta_time;

        if !entity.caps.flying {
            v.y -= entity.caps.gravity;
            if v.y < 0.0 {
                v.y = v.y.max(-entity.caps.max_fall_velocity);
            }

            if !entity.caps.wall_clip {
                // TODO merge into one method of find axis agnostic implementation
                v = Physics::apply_horizontal_physics(v, &aabb, &result);
            }
            v = Physics::apply_vertical_physics(v, &aabb, &result);
        }

        entity.position += v;
        entity.state = EntityState {
            is_grounded: !entity.caps.flying && (result.neg.y + v.y) < 0.02 && result.neg.y != -1.0,
        };

        v.x = 0.0;
        v.z = 0.0;
        if entity.caps.flying { v.y = 0.0; }

        entity.velocity = v / delta_time;
    }

    fn apply_horizontal_physics(speed: Vector3<f32>, aabb: &AABB, res: &AABBResult) -> Vector3<f32> {
        const TWO_EPSILON: f32 = EPSILON * 2.0;

        let x_dot = speed.dot(Vector3::new(1.0, 0.0, 0.0));
        let z_dot = speed.dot(Vector3::new(0.0, 0.0, 1.0));
        let x_dst = if x_dot > 0.0 { res.pos.x } else { res.neg.x };
        let z_dst = if z_dot > 0.0 { res.pos.z } else { res.neg.z };

        let mut speed = speed;

        if x_dst != -1.0 {
            if x_dst == 0.0 { // TODO
                if x_dot > 0.0 {
                    let actual = aabb.pos.x + aabb.offset.x + aabb.extents.x;
                    let expected = actual.floor() - TWO_EPSILON;
                    speed.x = expected - actual;
                } else {
                    let actual = aabb.pos.x + aabb.offset.x;
                    let expected = actual.ceil() + TWO_EPSILON;
                    speed.x = expected - actual;
                }
            } else if speed.x.abs() > (x_dst - TWO_EPSILON) {
                speed.x = (x_dst - TWO_EPSILON) * speed.x.signum();
            }
        }

        if z_dst != -1.0 {
            if z_dst == 0.0 {//TODO
                if z_dot > 0.0 {
                    let actual = aabb.pos.z + aabb.offset.z + aabb.extents.z;
                    let expected = actual.floor() - TWO_EPSILON;
                    speed.x = expected - actual;
                } else {
                    let actual = aabb.pos.z + aabb.offset.z;
                    let expected = actual.ceil() + TWO_EPSILON;
                    speed.z = expected - actual;
                }
            } else if speed.z.abs() > (z_dst - TWO_EPSILON) {
                speed.z = (z_dst - TWO_EPSILON) * speed.z.signum();
            }
        }

        speed
    }

    fn apply_vertical_physics(speed: Vector3<f32>, aabb: &AABB, res: &AABBResult) -> Vector3<f32> {
        const TWO_EPSILON: f32 = EPSILON * 2.0;

        let y_dot = speed.dot(Vector3::new(0.0, 1.0, 0.0));
        let y_dst = if y_dot > 0.0 { res.pos.y } else { res.neg.y };

        let mut speed = speed;
        if y_dst != -1.0 {
            if y_dst == 0.0 { // TODO
                if y_dot > 0.0 {
                    let actual = aabb.pos.y + aabb.offset.y + aabb.extents.y;
                    let expected = actual.floor() - TWO_EPSILON;
                    speed.y = expected - actual;
                } else {
                    let actual = aabb.pos.y + aabb.offset.y;
                    let expected = actual.ceil() + TWO_EPSILON;
                    speed.y = expected - actual;
                }
            } else if speed.y.abs() > (y_dst - TWO_EPSILON) {
                speed.y = (y_dst - TWO_EPSILON) * speed.y.signum();
            }
        }
        speed
    }
}

#[cfg(test)]
mod test {
    use cgmath::{Point3, Vector3, Zero};
    use mockall::predicate::eq;

    use crate::graphics::svo_picker::{AABB, AABBResult, PickerBatch, PickerBatchResult};
    use crate::systems::physics::{AABBDef, Entity, EntityCapabilities, EntityState, MockRaycaster, Physics};

    /// Asserts that all entities are raycasted and updated. The test contains different entities
    /// in different positions and configurations.
    #[test]
    fn step() {
        struct EntityTestCase {
            name: &'static str,
            position: Point3<f32>,
            velocity: Option<Vector3<f32>>,
            aabb_def: Option<AABBDef>,
            caps: Option<EntityCapabilities>,

            aabb_result: AABBResult,

            expected_position: Point3<f32>,
            expected_velocity: Vector3<f32>,
            expected_state: Option<EntityState>,
        }

        let test_cases = vec![
            EntityTestCase {
                name: "falling - first time",
                position: Point3::new(0.0, 0.0, 0.0),
                velocity: None,
                aabb_def: None,
                caps: None,
                aabb_result: AABBResult { neg: Vector3::new(-1.0, 1.0, -1.0), pos: Vector3::new(-1.0, -1.0, -1.0) },
                expected_position: Point3::new(0.0, -0.008, 0.0),
                expected_velocity: Vector3::new(0.0, -0.008, 0.0),
                expected_state: None,
            },
            EntityTestCase {
                name: "falling - second time",
                position: Point3::new(0.0, -0.008, 0.0),
                velocity: Some(Vector3::new(0.0, -0.008, 0.0)),
                aabb_def: None,
                caps: None,
                aabb_result: AABBResult { neg: Vector3::new(-1.0, 1.0, -1.0), pos: Vector3::new(-1.0, -1.0, -1.0) },
                expected_position: Point3::new(0.0, -0.024, 0.0),
                expected_velocity: Vector3::new(0.0, -0.016, 0.0),
                expected_state: None,
            },
            EntityTestCase {
                name: "falling - hitting floor",
                position: Point3::new(0.0, -0.024, 0.0),
                velocity: Some(Vector3::new(0.0, -0.016, 0.0)),
                aabb_def: None,
                caps: None,
                aabb_result: AABBResult { neg: Vector3::new(-1.0, 0.010, -1.0), pos: Vector3::new(-1.0, -1.0, -1.0) },
                expected_position: Point3::new(0.0, -0.024, 0.0), // should be -0.034 but epsilon value prevents them from getting close
                expected_velocity: Vector3::new(0.0, -0.0, 0.0),
                expected_state: Some(EntityState { is_grounded: true }),
            },
            EntityTestCase {
                name: "falling - hitting floor with wall clip enabled",
                position: Point3::new(0.0, -0.024, 0.0),
                velocity: Some(Vector3::new(0.0, -0.016, 0.0)),
                aabb_def: None,
                caps: Some(EntityCapabilities { wall_clip: true, flying: false, gravity: 0.008, max_fall_velocity: 3.0 }),
                aabb_result: AABBResult { neg: Vector3::new(-1.0, 0.010, -1.0), pos: Vector3::new(-1.0, -1.0, -1.0) },
                expected_position: Point3::new(0.0, -0.024, 0.0), // should be -0.034 but epsilon value prevents them from getting close
                expected_velocity: Vector3::new(0.0, -0.0, 0.0),
                expected_state: Some(EntityState { is_grounded: true }),
            },
            EntityTestCase {
                name: "falling - max velocity",
                position: Point3::new(0.0, 0.0, 0.0),
                velocity: Some(Vector3::new(0.0, -4.0, 0.0)),
                aabb_def: None,
                caps: None,
                aabb_result: AABBResult { neg: Vector3::new(-1.0, 10.0, -1.0), pos: Vector3::new(-1.0, -1.0, -1.0) },
                expected_position: Point3::new(0.0, -3.0, 0.0),
                expected_velocity: Vector3::new(0.0, -3.0, 0.0),
                expected_state: None,
            },
            EntityTestCase {
                name: "jumping - no velocity limit",
                position: Point3::new(0.0, 0.0, 0.0),
                velocity: Some(Vector3::new(0.0, 5.0, 0.0)),
                aabb_def: None,
                caps: None,
                aabb_result: AABBResult { neg: Vector3::new(-1.0, -1.0, -1.0), pos: Vector3::new(-1.0, -1.0, -1.0) },
                expected_position: Point3::new(0.0, 4.992, 0.0),
                expected_velocity: Vector3::new(0.0, 4.992, 0.0),
                expected_state: None,
            },
            EntityTestCase {
                name: "jumping - with collision",
                position: Point3::new(0.0, 0.0, 0.0),
                velocity: Some(Vector3::new(0.0, 5.0, 0.0)),
                aabb_def: None,
                caps: None,
                aabb_result: AABBResult { neg: Vector3::new(-1.0, -1.0, -1.0), pos: Vector3::new(-1.0, 2.0, -1.0) },
                expected_position: Point3::new(0.0, 1.99, 0.0),
                expected_velocity: Vector3::new(0.0, 1.99, 0.0), // will be reset on the next iteration
                expected_state: None,
            },
            EntityTestCase {
                name: "jumping - after collision for velocity reset",
                position: Point3::new(0.0, 1.99, 0.0),
                velocity: Some(Vector3::new(0.0, 1.99, 0.0)),
                aabb_def: None,
                caps: None,
                aabb_result: AABBResult { neg: Vector3::new(-1.0, -1.0, -1.0), pos: Vector3::new(-1.0, 0.01, -1.0) },
                expected_position: Point3::new(0.0, 1.99, 0.0),
                expected_velocity: Vector3::new(0.0, 0.0, 0.0), // this is reset now
                expected_state: None,
            },
            EntityTestCase {
                name: "jumping - with collision and wall clip enabled",
                position: Point3::new(0.0, 0.0, 0.0),
                velocity: Some(Vector3::new(0.0, 5.0, 0.0)),
                aabb_def: None,
                caps: Some(EntityCapabilities { wall_clip: true, flying: false, gravity: 0.008, max_fall_velocity: 3.0 }),
                aabb_result: AABBResult { neg: Vector3::new(-1.0, -1.0, -1.0), pos: Vector3::new(-1.0, 2.0, -1.0) },
                expected_position: Point3::new(0.0, 1.99, 0.0),
                expected_velocity: Vector3::new(0.0, 1.99, 0.0), // would be reset on the next iteration
                expected_state: None,
            },
            EntityTestCase {
                name: "flying - ground state not set",
                position: Point3::new(0.0, 5.0, 0.0),
                velocity: Some(Vector3::new(3.0, -5.0, 3.0)),
                aabb_def: None,
                caps: Some(EntityCapabilities { wall_clip: false, flying: true, gravity: 0.008, max_fall_velocity: 3.0 }),
                aabb_result: AABBResult { neg: Vector3::new(-1.0, 5.0, -1.0), pos: Vector3::new(2.0, -1.0, 2.0) },
                expected_position: Point3::new(3.0, 0.0, 3.0),
                expected_velocity: Vector3::new(0.0, 0.0, 0.0),
                expected_state: Some(EntityState { is_grounded: false }),
            },
            EntityTestCase {
                name: "horizontal positive collision",
                position: Point3::new(0.0, 0.0, 0.0),
                velocity: Some(Vector3::new(2.0, 0.0, 2.0)),
                aabb_def: None,
                caps: None,
                aabb_result: AABBResult { neg: Vector3::new(-1.0, 0.0, -1.0), pos: Vector3::new(1.0, -1.0, 1.0) },
                expected_position: Point3::new(0.99, 0.01, 0.99),
                expected_velocity: Vector3::new(0.0, 0.01, 0.0),
                expected_state: Some(EntityState { is_grounded: true }),
            },
            EntityTestCase {
                name: "horizontal negative collision",
                position: Point3::new(0.0, 0.0, 0.0),
                velocity: Some(Vector3::new(-2.0, 0.0, -2.0)),
                aabb_def: None,
                caps: None,
                aabb_result: AABBResult { neg: Vector3::new(1.0, 0.0, 1.0), pos: Vector3::new(-1.0, -1.0, -1.0) },
                expected_position: Point3::new(-0.99, 0.01, -0.99),
                expected_velocity: Vector3::new(0.0, 0.01, 0.0),
                expected_state: Some(EntityState { is_grounded: true }),
            },
            EntityTestCase {
                name: "horizontal positive collision - with wall clip enabled",
                position: Point3::new(0.0, 0.0, 0.0),
                velocity: Some(Vector3::new(2.0, 0.0, 2.0)),
                aabb_def: None,
                caps: Some(EntityCapabilities { wall_clip: true, flying: false, gravity: 0.008, max_fall_velocity: 3.0 }),
                aabb_result: AABBResult { neg: Vector3::new(-1.0, 0.0, -1.0), pos: Vector3::new(1.0, -1.0, 1.0) },
                expected_position: Point3::new(2.0, 0.01, 2.0),
                expected_velocity: Vector3::new(0.0, 0.01, 0.0),
                expected_state: Some(EntityState { is_grounded: true }),
            },
        ];

        let mut entities = Vec::new();
        let mut expected_batch = PickerBatch::new();
        let mut expected_entities = Vec::new();
        let mut aabb_results = Vec::new();
        for c in &test_cases {
            let e = Entity {
                position: c.position,
                velocity: c.velocity.unwrap_or(Vector3::zero()),
                aabb_def: c.aabb_def.unwrap_or(AABBDef::new(Vector3::zero(), Vector3::new(1.0, 1.0, 1.0))),
                caps: c.caps.unwrap_or(EntityCapabilities {
                    wall_clip: false,
                    flying: false,
                    gravity: 0.008,
                    max_fall_velocity: 3.0,
                }),
                state: EntityState::default(),
            };

            expected_entities.push(Entity {
                position: c.expected_position,
                velocity: c.expected_velocity,
                aabb_def: e.aabb_def,
                caps: e.caps,
                state: c.expected_state.unwrap_or_default(),
            });

            expected_batch.aabb(AABB::new(c.position, e.aabb_def.offset, e.aabb_def.extents));
            aabb_results.push(c.aabb_result);

            entities.push(e);
        }

        let mut mock = MockRaycaster::new();
        mock.expect_cast()
            .with(eq(expected_batch))
            .times(1)
            .returning(move |_| PickerBatchResult { rays: Vec::new(), aabbs: aabb_results.clone() });

        let physics = Physics::new();
        let mut entity_refs = Vec::new();
        for e in entities.iter_mut() {
            entity_refs.push(e);
        }
        physics.step(1.0, &mock, &mut entity_refs);

        for (i, expected) in expected_entities.iter().enumerate() {
            assert_eq!(expected, &entities[i], "entity case '{}'", &test_cases[i].name);
        }
    }

    // TODO after tests have been written, try refactoring physics methods
}
