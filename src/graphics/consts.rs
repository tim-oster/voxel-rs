// TODO should this be globally defined or done inside the render loop instead?
pub mod shader_buffer_indices {
    pub struct Index {
        index: u32,
    }

    impl Index {
        const fn new(index: u32) -> Index {
            Index { index }
        }

        pub fn get(&self) -> u32 {
            self.index
        }
    }

    pub const WORLD: Index = Index::new(0);
    pub const MATERIALS: Index = Index::new(2);
    pub const PICKER_OUT: Index = Index::new(1);
    pub const PICKER_IN: Index = Index::new(3);
    pub const DEBUG_IN: Index = Index::new(11);
    pub const DEBUG_OUT: Index = Index::new(12);
}
