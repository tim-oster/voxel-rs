use std::cell::RefCell;

use cgmath::{InnerSpace, Point3, Vector3};

use crate::graphics::svo::Svo;
use crate::graphics::svo_picker::{Aabb, AabbResult, PickerBatch, PickerBatchResult};

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
            gravity: 60.0,
            max_fall_velocity: 100.0,
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

pub trait Raycaster {
    fn raycast(&self, batch: &mut PickerBatch, result: &mut PickerBatchResult);
}

impl Raycaster for Svo {
    fn raycast(&self, batch: &mut PickerBatch, result: &mut PickerBatchResult) {
        self.raycast(batch, result);
    }
}

pub struct Physics {
    reusable_batch: RefCell<EntityBatch>,
}

impl Physics {
    pub fn new() -> Physics {
        Physics {
            reusable_batch: RefCell::new(EntityBatch::new()),
        }
    }

    /// Simulates the next step for `entity` for the given delta time. `raycaster` is used
    /// to identify collisions.
    pub fn step(&self, delta_time: f32, raycaster: &impl Raycaster, entity: &mut Entity) {
        let mut batch = self.reusable_batch.borrow_mut();
        batch.reset();
        batch.add_entity(entity);

        let results = batch.raycast(raycaster);
        Physics::update_entity(entity, &results[0], delta_time);
    }

    /// Simulates the next step for all `entities` for the given delta time. `raycaster` is used
    /// to identify collisions.
    pub fn step_many(&self, delta_time: f32, raycaster: &impl Raycaster, entities: &mut [Entity]) {
        let mut batch = self.reusable_batch.borrow_mut();
        batch.reset();

        for entity in &mut entities.iter_mut() {
            batch.add_entity(entity);
        }

        let results = batch.raycast(raycaster);

        for i in 0..entities.len() {
            Physics::update_entity(&mut entities[i], &results[i], delta_time);
        }
    }

    fn update_entity(entity: &mut Entity, result: &AabbResult, delta_time: f32) {
        // apply gravity
        if !entity.caps.flying {
            entity.velocity.y -= entity.caps.gravity * delta_time;
            if entity.velocity.y < 0.0 {
                entity.velocity.y = entity.velocity.y.max(-entity.caps.max_fall_velocity);
            }
        }

        let mut velocity = entity.velocity * delta_time;

        // calculate entity state with new velocity
        entity.state = EntityState {
            is_grounded: !entity.caps.flying && (result.neg.y + velocity.y) < 0.02 && result.neg.y != -1.0,
        };
        // reset gravity, if entity collides with ground already
        if entity.state.is_grounded && entity.velocity.y < 0.0 {
            entity.velocity.y = 0.0;
        }

        // constraint velocity by nearby collisions
        if !entity.caps.flying {
            if !entity.caps.wall_clip {
                velocity.x = Physics::apply_axial_physics(velocity.x, result.pos.x, result.neg.x);
                velocity.z = Physics::apply_axial_physics(velocity.z, result.pos.z, result.neg.z);
            }
            velocity.y = Physics::apply_axial_physics(velocity.y, result.pos.y, result.neg.y);
        }

        // apply velocity
        entity.position += velocity;
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

struct EntityBatch {
    batch: PickerBatch,
    result: PickerBatchResult,
}

impl EntityBatch {
    fn new() -> EntityBatch {
        EntityBatch {
            batch: PickerBatch::new(),
            result: PickerBatchResult::new(),
        }
    }

    fn reset(&mut self) {
        self.batch.reset();
        self.result.reset();
    }

    fn add_entity(&mut self, entity: &Entity) {
        let aabb = Aabb::new(entity.position, entity.aabb_def.offset, entity.aabb_def.extents);
        self.batch.add_aabb(aabb);
    }

    fn raycast(&mut self, raycaster: &impl Raycaster) -> &Vec<AabbResult> {
        raycaster.raycast(&mut self.batch, &mut self.result);
        &self.result.aabbs
    }
}

#[cfg(test)]
mod tests {
    use std::vec;

    use cgmath::{Point3, Vector3, Zero};

    use crate::graphics::svo_picker::{Aabb, AabbResult, PickerBatch, PickerBatchResult};
    use crate::systems::physics::{AABBDef, Entity, EntityCapabilities, EntityState, Physics, Raycaster};

    struct MockRaycaster {
        call: Option<(PickerBatch, Box<dyn Fn(&mut PickerBatchResult) + 'static>)>,
    }

    impl MockRaycaster {
        fn new() -> Self {
            Self {
                call: None,
            }
        }

        fn on<F: Fn(&mut PickerBatchResult) + 'static>(&mut self, expected_input: PickerBatch, f: F) {
            self.call = Some((expected_input, Box::new(f)));
        }
    }

    impl Raycaster for MockRaycaster {
        fn raycast(&self, batch: &mut PickerBatch, result: &mut PickerBatchResult) {
            assert!(self.call.is_some());

            let call = self.call.as_ref().unwrap();
            assert_eq!(batch, &call.0);
            (call.1)(result);
        }
    }

    /// Asserts that the single entity implementation works. All edge cases are covered by the test for step_many.
    #[test]
    fn step() {
        let mut e = Entity {
            position: Point3::new(0.0, 0.0, 0.0),
            velocity: Vector3::zero(),
            euler_rotation: Vector3::new(0.0, 0.0, 0.0),
            aabb_def: AABBDef::new(Vector3::zero(), Vector3::new(1.0, 1.0, 1.0)),
            caps: EntityCapabilities {
                wall_clip: false,
                flying: false,
                gravity: 0.008,
                max_fall_velocity: 3.0,
            },
            state: EntityState::default(),
        };

        let mut expected_batch = PickerBatch::new();
        expected_batch.aabbs.push(Aabb::new(e.position, e.aabb_def.offset, e.aabb_def.extents));

        let mut mock = MockRaycaster::new();
        mock.on(expected_batch, |dst| *dst = PickerBatchResult { rays: Vec::new(), aabbs: vec![AabbResult::default()] });

        let physics = Physics::new();
        physics.step(1.0, &mock, &mut e);
        assert_eq!(Entity {
            position: Point3::new(0.0, -0.008, 0.0),
            velocity: Vector3::new(0.0, -0.008, 0.0),
            euler_rotation: Vector3::new(0.0, 0.0, 0.0),
            aabb_def: AABBDef::new(Vector3::zero(), Vector3::new(1.0, 1.0, 1.0)),
            caps: e.caps,
            state: EntityState::default(),
        }, e);
    }

    /// Asserts that all entities are raycasted and updated. The test contains different entities
    /// in different positions and configurations.
    #[test]
    fn step_many() {
        struct EntityTestCase {
            name: &'static str,
            position: Point3<f32>,
            velocity: Option<Vector3<f32>>,
            aabb_def: Option<AABBDef>,
            caps: Option<EntityCapabilities>,

            aabb_result: AabbResult,

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
                aabb_result: AabbResult { neg: Vector3::new(-1.0, 1.0, -1.0), pos: Vector3::new(-1.0, -1.0, -1.0) },
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
                aabb_result: AabbResult { neg: Vector3::new(-1.0, 1.0, -1.0), pos: Vector3::new(-1.0, -1.0, -1.0) },
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
                aabb_result: AabbResult { neg: Vector3::new(-1.0, 0.01, -1.0), pos: Vector3::new(-1.0, -1.0, -1.0) },
                expected_position: Point3::new(0.0, -0.0335, 0.0), // should be -0.034 but epsilon value prevents them from getting close
                expected_velocity: Vector3::new(0.0, 0.0, 0.0),
                expected_state: Some(EntityState { is_grounded: true }),
            },
            EntityTestCase {
                name: "falling - hitting floor with wall clip enabled",
                position: Point3::new(0.0, -0.024, 0.0),
                velocity: Some(Vector3::new(0.0, -0.016, 0.0)),
                aabb_def: None,
                caps: Some(EntityCapabilities { wall_clip: true, flying: false, gravity: 0.008, max_fall_velocity: 3.0 }),
                aabb_result: AabbResult { neg: Vector3::new(-1.0, 0.01, -1.0), pos: Vector3::new(-1.0, -1.0, -1.0) },
                expected_position: Point3::new(0.0, -0.0335, 0.0), // should be -0.034 but epsilon value prevents them from getting close
                expected_velocity: Vector3::new(0.0, 0.0, 0.0),
                expected_state: Some(EntityState { is_grounded: true }),
            },
            EntityTestCase {
                name: "falling - max velocity",
                position: Point3::new(0.0, 0.0, 0.0),
                velocity: Some(Vector3::new(0.0, -4.0, 0.0)),
                aabb_def: None,
                caps: None,
                aabb_result: AabbResult { neg: Vector3::new(-1.0, 10.0, -1.0), pos: Vector3::new(-1.0, -1.0, -1.0) },
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
                aabb_result: AabbResult { neg: Vector3::new(-1.0, -1.0, -1.0), pos: Vector3::new(-1.0, -1.0, -1.0) },
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
                aabb_result: AabbResult { neg: Vector3::new(-1.0, -1.0, -1.0), pos: Vector3::new(-1.0, 2.0, -1.0) },
                expected_position: Point3::new(0.0, 1.9995, 0.0),
                expected_velocity: Vector3::new(0.0, 4.992, 0.0), // will be reset on the next iteration
                expected_state: None,
            },
            EntityTestCase {
                name: "jumping - after collision for velocity reset",
                position: Point3::new(0.0, 1.9995, 0.0),
                velocity: Some(Vector3::new(0.0, 1.9995, 0.0)),
                aabb_def: None,
                caps: None,
                aabb_result: AabbResult { neg: Vector3::new(-1.0, -1.0, -1.0), pos: Vector3::new(-1.0, 0.0005, -1.0) },
                expected_position: Point3::new(0.0, 1.9995, 0.0),
                expected_velocity: Vector3::new(0.0, 1.9915, 0.0), // this is reset now
                expected_state: None,
            },
            EntityTestCase {
                name: "jumping - with collision and wall clip enabled",
                position: Point3::new(0.0, 0.0, 0.0),
                velocity: Some(Vector3::new(0.0, 5.0, 0.0)),
                aabb_def: None,
                caps: Some(EntityCapabilities { wall_clip: true, flying: false, gravity: 0.008, max_fall_velocity: 3.0 }),
                aabb_result: AabbResult { neg: Vector3::new(-1.0, -1.0, -1.0), pos: Vector3::new(-1.0, 2.0, -1.0) },
                expected_position: Point3::new(0.0, 1.9995, 0.0),
                expected_velocity: Vector3::new(0.0, 4.992, 0.0), // would be reset on the next iteration
                expected_state: None,
            },
            EntityTestCase {
                name: "flying - ground state not set",
                position: Point3::new(0.0, 5.0, 0.0),
                velocity: Some(Vector3::new(3.0, -5.0, 3.0)),
                aabb_def: None,
                caps: Some(EntityCapabilities { wall_clip: false, flying: true, gravity: 0.008, max_fall_velocity: 3.0 }),
                aabb_result: AabbResult { neg: Vector3::new(-1.0, 5.0, -1.0), pos: Vector3::new(2.0, -1.0, 2.0) },
                expected_position: Point3::new(3.0, 0.0, 3.0),
                expected_velocity: Vector3::new(3.0, -5.0, 3.0),
                expected_state: Some(EntityState { is_grounded: false }),
            },
            EntityTestCase {
                name: "horizontal positive collision",
                position: Point3::new(0.0, 0.0, 0.0),
                velocity: Some(Vector3::new(2.0, 0.0, 2.0)),
                aabb_def: None,
                caps: None,
                aabb_result: AabbResult { neg: Vector3::new(-1.0, 0.0, -1.0), pos: Vector3::new(1.0, -1.0, 1.0) },
                expected_position: Point3::new(0.9995, 0.0, 0.9995),
                expected_velocity: Vector3::new(2.0, 0.0, 2.0),
                expected_state: Some(EntityState { is_grounded: true }),
            },
            EntityTestCase {
                name: "horizontal negative collision",
                position: Point3::new(0.0, 0.0, 0.0),
                velocity: Some(Vector3::new(-2.0, 0.0, -2.0)),
                aabb_def: None,
                caps: None,
                aabb_result: AabbResult { neg: Vector3::new(1.0, 0.0, 1.0), pos: Vector3::new(-1.0, -1.0, -1.0) },
                expected_position: Point3::new(-0.9995, 0.0, -0.9995),
                expected_velocity: Vector3::new(-2.0, 0.0, -2.0),
                expected_state: Some(EntityState { is_grounded: true }),
            },
            EntityTestCase {
                name: "horizontal positive collision - with wall clip enabled",
                position: Point3::new(0.0, 0.0, 0.0),
                velocity: Some(Vector3::new(2.0, 0.0, 2.0)),
                aabb_def: None,
                caps: Some(EntityCapabilities { wall_clip: true, flying: false, gravity: 0.008, max_fall_velocity: 3.0 }),
                aabb_result: AabbResult { neg: Vector3::new(-1.0, 0.0, -1.0), pos: Vector3::new(1.0, -1.0, 1.0) },
                expected_position: Point3::new(2.0, 0.0, 2.0),
                expected_velocity: Vector3::new(2.0, 0.0, 2.0),
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

            expected_batch.add_aabb(Aabb::new(c.position, e.aabb_def.offset, e.aabb_def.extents));
            aabb_results.push(c.aabb_result);

            entities.push(e);
        }

        let mut mock = MockRaycaster::new();
        mock.on(expected_batch, move |dst| *dst = PickerBatchResult { rays: Vec::new(), aabbs: aabb_results.clone() });

        let physics = Physics::new();
        physics.step_many(1.0, &mock, &mut entities);
        for (i, expected) in expected_entities.iter().enumerate() {
            assert_eq!(expected, &entities[i], "entity case '{}'", &test_cases[i].name);
        }
    }
}
