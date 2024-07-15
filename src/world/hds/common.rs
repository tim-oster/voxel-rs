use crate::world::hds::octree::{LeafId, Position};

pub trait WorldSvo<T, F> {
    fn clear(&mut self);
    fn set_leaf(&mut self, pos: Position, leaf: T, serialize: bool) -> (LeafId, Option<T>);
    fn move_leaf(&mut self, leaf: LeafId, to_pos: Position) -> (LeafId, Option<T>);
    fn remove_leaf(&mut self, leaf: LeafId) -> Option<T>;
    fn get_leaf(&self, pos: Position) -> Option<&T>;
    fn serialize(&mut self);

    fn depth(&self) -> u8;
    fn size_in_bytes(&self) -> usize;
    unsafe fn write_to(&self, dst: *mut F) -> usize;
    unsafe fn write_changes_to(&mut self, dst: *mut F, dst_len: usize, reset: bool);
}
