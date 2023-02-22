#[repr(C)]
pub struct Material {
    pub specular_pow: f32,
    pub specular_strength: f32,
    pub tex_top: i32,
    pub tex_side: i32,
    pub tex_bottom: i32,
    pub tex_top_normal: i32,
    pub tex_side_normal: i32,
    pub tex_bottom_normal: i32,
}
