#[allow(dead_code)]
pub mod blocks {
    use crate::graphics::svo_registry::{Material, VoxelRegistry};
    use crate::world::chunk::BlockId;

    pub const AIR: BlockId = 0;
    pub const GRASS: BlockId = 1;
    pub const DIRT: BlockId = 2;
    pub const STONE: BlockId = 3;
    pub const STONE_BRICKS: BlockId = 4;
    pub const GLASS: BlockId = 5;

    pub fn new_registry() -> VoxelRegistry {
        let mut registry = VoxelRegistry::new();
        registry
            .add_texture("dirt", "assets/textures/dirt.png")
            .add_texture("dirt_normal", "assets/textures/dirt_n.png")
            .add_texture("grass_side", "assets/textures/grass_side.png")
            .add_texture("grass_side_normal", "assets/textures/grass_side_n.png")
            .add_texture("grass_top", "assets/textures/grass_top.png")
            .add_texture("grass_top_normal", "assets/textures/grass_top_n.png")
            .add_texture("stone", "assets/textures/stone.png")
            .add_texture("stone_normal", "assets/textures/stone_n.png")
            .add_texture("stone_bricks", "assets/textures/stone_bricks.png")
            .add_texture("stone_bricks_normal", "assets/textures/stone_bricks_n.png")
            .add_texture("glass", "assets/textures/glass.png")
            .add_material(AIR, Material::new())
            .add_material(GRASS, Material::new().specular(14.0, 0.4).top("grass_top").side("grass_side").bottom("dirt").with_normals())
            .add_material(DIRT, Material::new().specular(14.0, 0.4).all_sides("dirt").with_normals())
            .add_material(STONE, Material::new().specular(70.0, 0.4).all_sides("stone").with_normals())
            .add_material(STONE_BRICKS, Material::new().specular(70.0, 0.4).all_sides("stone_bricks").with_normals())
            .add_material(GLASS, Material::new().specular(70.0, 0.4).all_sides("glass"));
        registry
    }
}
