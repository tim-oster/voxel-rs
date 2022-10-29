mod testing;

mod camera;
pub use camera::Camera;

mod resource;
pub use resource::Resource;

mod shader;
mod texture_array;

pub use shader::ShaderProgramBuilder;
pub use shader::ShaderProgram;
pub use shader::ShaderType;
pub use texture_array::*;
