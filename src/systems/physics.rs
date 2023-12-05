use cgmath::{InnerSpace, Point3, Vector3};
#[cfg(test)]
use mockall::automock;

use crate::graphics::svo::Svo;
use crate::graphics::svo_picker::{AABB, AABBResult, PickerBatch, PickerBatchResult};

const EPSILON: f32 = 0.0005;

#[derive(Debug, PartialEq)]
pub struct Entity {
    pub position: Point3<f32>,
    pub velocity: Vector3<f32>,
    pub euler_rotation: Vector3<f32>,
    pub aabb_def: AABBDef,
    pub caps: EntityCapabilities,
    state: EntityState,
}

impl Entity {
    pub fn get_forward(&self) -> Vector3<f32> {
        Vector3::new(
            self.euler_rotation.y.cos() * self.euler_rotation.x.cos(),
            self.euler_rotation.x.sin(),
            self.euler_rotation.y.sin() * self.euler_rotation.x.cos(),
        ).normalize()
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct EntityState {
    /// If the entity is colliding in -y direction.
    pub is_grounded: bool,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct EntityCapabilities {
    /// Disables all collisions along x & z axis.
    pub wall_clip: bool,
    /// Disables gravity and all collisions.
    pub flying: bool,
    /// Constant acceleration in -y direction.
    pub gravity: f32,
    /// Cap on how fast the entity can fall in -y direction due to gravity.
    pub max_fall_velocity: f32,
}

impl Default for EntityCapabilities {
    fn default() -> Self {
        EntityCapabilities {
            wall_clip: false,
            flying: false,
            gravity: 0.8,
            max_fall_velocity: 3.0,
        }
    }
}

impl Entity {
    pub fn new(position: Point3<f32>, aabb_def: AABBDef) -> Entity {
        Entity {
            position,
            velocity: Vector3::new(0.0, 0.0, 0.0),
            euler_rotation: Vector3::new(0.0, 0.0, 0.0),
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
    fn raycast(&self, batch: PickerBatch) -> PickerBatchResult;
}

impl Raycaster for Svo {
    fn raycast(&self, batch: PickerBatch) -> PickerBatchResult {
        self.raycast(batch)
    }
}

pub struct Physics {}

#[allow(dead_code)]
struct RaycastData {
    aabb: AABB,
    aabb_result: AABBResult,
}

impl Physics {
    pub fn new() -> Physics {
        Physics {}
    }

    /// Simulates the next step for all `entities` for the given delta time. `raycaster` is used
    /// to identify collisions.
    pub fn step(&self, delta_time: f32, raycaster: &impl Raycaster, entities: Vec<&mut Entity>) {
        let results = Physics::raycast_entities(raycaster, &entities);
        let mut entities = entities;

        for i in 0..entities.len() {
            let data = &results[i];
            Physics::update_entity(entities[i], &data.aabb_result, delta_time);
        }
    }

    fn raycast_entities(raycaster: &impl Raycaster, entities: &Vec<&mut Entity>) -> Vec<RaycastData> {
        let mut batch = PickerBatch::new();
        let mut results = Vec::with_capacity(entities.len());

        for entity in entities.iter() {
            let aabb = AABB::new(entity.position, entity.aabb_def.offset, entity.aabb_def.extents);
            batch.add_aabb(aabb);
            results.push(RaycastData {
                aabb,
                aabb_result: AABBResult::default(),
            });
        }

        let batch_result = raycaster.raycast(batch);
        for (i, result) in results.iter_mut().enumerate() {
            result.aabb_result = batch_result.aabbs[i];
        }

        results
    }

    fn update_entity(entity: &mut Entity, result: &AABBResult, delta_time: f32) {
        let mut v = entity.velocity * delta_time;

        if !entity.caps.flying {
            v.y -= entity.caps.gravity * delta_time;
            if v.y < 0.0 {
                v.y = v.y.max(-entity.caps.max_fall_velocity);
            }

            if !entity.caps.wall_clip {
                v.x = Physics::apply_axial_physics(v.x, result.pos.x, result.neg.x);
                v.z = Physics::apply_axial_physics(v.z, result.pos.z, result.neg.z);
            }
            v.y = Physics::apply_axial_physics(v.y, result.pos.y, result.neg.y);
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

    fn apply_axial_physics(speed: f32, dst_pos: f32, dst_neg: f32) -> f32 {
        let dst = if speed > 0.0 { dst_pos } else { dst_neg };
        if dst == -1.0 {
            return speed;
        }
        if dst < 2.0 * EPSILON {
            return 0.0;
        }
        if speed.abs() > dst {
            return (dst - EPSILON) * speed.signum();
        }
        speed
    }
}

#[cfg(test)]
mod tests {
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
                aabb_result: AABBResult { neg: Vector3::new(-1.0, 0.01, -1.0), pos: Vector3::new(-1.0, -1.0, -1.0) },
                expected_position: Point3::new(0.0, -0.0335, 0.0), // should be -0.034 but epsilon value prevents them from getting close
                expected_velocity: Vector3::new(0.0, -0.0095, 0.0),
                expected_state: Some(EntityState { is_grounded: true }),
            },
            EntityTestCase {
                name: "falling - hitting floor with wall clip enabled",
                position: Point3::new(0.0, -0.024, 0.0),
                velocity: Some(Vector3::new(0.0, -0.016, 0.0)),
                aabb_def: None,
                caps: Some(EntityCapabilities { wall_clip: true, flying: false, gravity: 0.008, max_fall_velocity: 3.0 }),
                aabb_result: AABBResult { neg: Vector3::new(-1.0, 0.01, -1.0), pos: Vector3::new(-1.0, -1.0, -1.0) },
                expected_position: Point3::new(0.0, -0.0335, 0.0), // should be -0.034 but epsilon value prevents them from getting close
                expected_velocity: Vector3::new(0.0, -0.0095, 0.0),
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
                expected_position: Point3::new(0.0, 1.9995, 0.0),
                expected_velocity: Vector3::new(0.0, 1.9995, 0.0), // will be reset on the next iteration
                expected_state: None,
            },
            EntityTestCase {
                name: "jumping - after collision for velocity reset",
                position: Point3::new(0.0, 1.9995, 0.0),
                velocity: Some(Vector3::new(0.0, 1.9995, 0.0)),
                aabb_def: None,
                caps: None,
                aabb_result: AABBResult { neg: Vector3::new(-1.0, -1.0, -1.0), pos: Vector3::new(-1.0, 0.0005, -1.0) },
                expected_position: Point3::new(0.0, 1.9995, 0.0),
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
                expected_position: Point3::new(0.0, 1.9995, 0.0),
                expected_velocity: Vector3::new(0.0, 1.9995, 0.0), // would be reset on the next iteration
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
                expected_position: Point3::new(0.9995, 0.0, 0.9995),
                expected_velocity: Vector3::new(0.0, 0.0, 0.0),
                expected_state: Some(EntityState { is_grounded: true }),
            },
            EntityTestCase {
                name: "horizontal negative collision",
                position: Point3::new(0.0, 0.0, 0.0),
                velocity: Some(Vector3::new(-2.0, 0.0, -2.0)),
                aabb_def: None,
                caps: None,
                aabb_result: AABBResult { neg: Vector3::new(1.0, 0.0, 1.0), pos: Vector3::new(-1.0, -1.0, -1.0) },
                expected_position: Point3::new(-0.9995, 0.0, -0.9995),
                expected_velocity: Vector3::new(0.0, 0.0, 0.0),
                expected_state: Some(EntityState { is_grounded: true }),
            },
            EntityTestCase {
                name: "horizontal positive collision - with wall clip enabled",
                position: Point3::new(0.0, 0.0, 0.0),
                velocity: Some(Vector3::new(2.0, 0.0, 2.0)),
                aabb_def: None,
                caps: Some(EntityCapabilities { wall_clip: true, flying: false, gravity: 0.008, max_fall_velocity: 3.0 }),
                aabb_result: AABBResult { neg: Vector3::new(-1.0, 0.0, -1.0), pos: Vector3::new(1.0, -1.0, 1.0) },
                expected_position: Point3::new(2.0, 0.0, 2.0),
                expected_velocity: Vector3::new(0.0, 0.0, 0.0),
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
                euler_rotation: Vector3::new(0.0, 0.0, 0.0),
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
                euler_rotation: e.euler_rotation,
                aabb_def: e.aabb_def,
                caps: e.caps,
                state: c.expected_state.unwrap_or_default(),
            });

            expected_batch.add_aabb(AABB::new(c.position, e.aabb_def.offset, e.aabb_def.extents));
            aabb_results.push(c.aabb_result);

            entities.push(e);
        }

        let mut mock = MockRaycaster::new();
        mock.expect_raycast()
            .with(eq(expected_batch))
            .times(1)
            .returning(move |_| PickerBatchResult { rays: Vec::new(), aabbs: aabb_results.clone() });

        let physics = Physics::new();
        physics.step(1.0, &mock, entities.iter_mut().collect());
        for (i, expected) in expected_entities.iter().enumerate() {
            assert_eq!(expected, &entities[i], "entity case '{}'", &test_cases[i].name);
        }
    }
}
