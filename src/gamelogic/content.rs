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
    pub const GRAVEL: BlockId = 6;
    pub const SAND: BlockId = 7;
    pub const WATER: BlockId = 8;
    pub const OAK_LOG: BlockId = 9;
    pub const OAK_LEAVES: BlockId = 10;
    pub const OAK_PLANKS: BlockId = 11;
    pub const COBBLESTONE: BlockId = 12;

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
            .add_texture("gravel", "assets/textures/gravel.png")
            .add_texture("gravel_normal", "assets/textures/gravel_n.png")
            .add_texture("sand", "assets/textures/sand.png")
            .add_texture("sand_normal", "assets/textures/sand_n.png")
            .add_texture("water", "assets/textures/water.png")
            .add_texture("oak_log", "assets/textures/oak_log.png")
            .add_texture("oak_log_normal", "assets/textures/oak_log_n.png")
            .add_texture("oak_log_top", "assets/textures/oak_log_top.png")
            .add_texture("oak_log_top_normal", "assets/textures/oak_log_top_n.png")
            .add_texture("oak_leaves", "assets/textures/oak_leaves.png")
            .add_texture("oak_planks", "assets/textures/oak_planks.png")
            .add_texture("oak_planks_normal", "assets/textures/oak_planks_n.png")
            .add_texture("cobblestone", "assets/textures/cobblestone.png")
            .add_texture("cobblestone_normal", "assets/textures/cobblestone_n.png")
            .add_material(AIR, Material::new())
            .add_material(GRASS, Material::new().specular(14.0, 0.4).top("grass_top").side("grass_side").bottom("dirt").with_normals())
            .add_material(DIRT, Material::new().specular(14.0, 0.4).all_sides("dirt").with_normals())
            .add_material(STONE, Material::new().specular(70.0, 0.4).all_sides("stone").with_normals())
            .add_material(STONE_BRICKS, Material::new().specular(70.0, 0.4).all_sides("stone_bricks").with_normals())
            .add_material(GLASS, Material::new().specular(70.0, 0.4).all_sides("glass"))
            .add_material(GRAVEL, Material::new().specular(70.0, 0.4).all_sides("gravel").with_normals())
            .add_material(SAND, Material::new().specular(70.0, 0.4).all_sides("sand").with_normals())
            .add_material(WATER, Material::new().specular(70.0, 0.4).all_sides("water"))
            .add_material(OAK_LOG, Material::new().specular(70.0, 0.4).side("oak_log").top("oak_log_top").bottom("oak_log_top").with_normals())
            .add_material(OAK_LEAVES, Material::new().specular(70.0, 0.4).all_sides("oak_leaves"))
            .add_material(OAK_PLANKS, Material::new().specular(70.0, 0.4).all_sides("oak_planks").with_normals())
            .add_material(COBBLESTONE, Material::new().specular(70.0, 0.4).all_sides("cobblestone").with_normals());
        registry
    }
}
