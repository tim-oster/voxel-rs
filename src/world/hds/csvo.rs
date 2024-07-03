use std::alloc::{Allocator, Global};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::ptr;
use std::sync::Arc;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::graphics::svo::Container;
use crate::world::chunk::{BlockId, ChunkPos};
use crate::world::hds::ChunkBufferPool;
use crate::world::hds::internal::{ChunkBuffer, RangeBuffer};
use crate::world::hds::octree::{LeafId, Octant, OctantId, Octree, Position};
use crate::world::memory::{Pooled, StatsAllocator};
use crate::world::world::BorrowedChunk;

/// `OctantChange` describes if an octant was added (and where), or if it was removed.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
enum OctantChange {
    Add(u64, LeafId),
    Remove(u64),
}

pub trait Serializable {
    /// Returns a unique id for the serializable value. It is used to keep track of the serialized result when moving
    /// it around inside the SVO.
    fn unique_id(&self) -> u64;

    /// Serializes the data into the destination buffer and returns metadata about the data layout.
    fn serialize(&mut self, dst: &mut Vec<u8>, lod: u8) -> SerializationResult;
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SerializationResult {
    /// Header mask of consisting of two bits per child (00b = empty, 01b = 8, 10b = 16, 11b = 32 bits).
    pub header_mask: u16,
    /// Indicates how "deep" the serialized result is.
    /// 0 = nothing was serialized.
    /// 1 = for leaf values or child octants with no children.
    /// 2..n = for octants with other octants and/or children.
    pub depth: u8,
}

pub struct Csvo<T: Serializable, A: Allocator = Global> {
    octree: Octree<T>,
    change_set: FxHashSet<OctantChange>,

    buffer: RangeBuffer<u8, A>,
    leaf_info: FxHashMap<u64, LeafInfo>,
    root_info: Option<LeafInfo>,

    /// Reusable buffer for serializing octants data to be copied into actual [`RangeBuffer`].
    tmp_octant_buffer: Option<ChunkBuffer<u8>>,
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct LeafInfo {
    /// Offset into the final buffer where the serialized leaf data was copied to.
    buf_offset: usize,
    /// Metadata (like child & leaf flag) for the serialized data.
    serialization: SerializationResult,
}

impl<T: Serializable> Csvo<T> {
    pub fn new() -> Self {
        Self::new_in(Global)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_in(capacity, Global)
    }
}

impl<T: Serializable, A: Allocator> Csvo<T, A> {
    pub fn new_in(alloc: A) -> Self {
        Self::with_capacity_in(0, alloc)
    }

    pub fn with_capacity_in(capacity: usize, alloc: A) -> Self {
        Self {
            octree: Octree::new(),
            change_set: FxHashSet::default(),
            buffer: RangeBuffer::with_capacity_in(capacity, alloc),
            leaf_info: FxHashMap::default(),
            root_info: None,
            tmp_octant_buffer: Some(ChunkBuffer::new()),
        }
    }

    /// Clears all data from the SVO but does not free up memory.
    pub fn clear(&mut self) {
        self.octree.reset();
        self.change_set.clear();
        self.buffer.clear();
        self.leaf_info.clear();
        self.root_info = None;
    }

    /// See [`Octree::set_leaf`]. Setting `serialize` to false attempts to bypass re-serializing the leaf in case it
    /// was done already. This is useful if the leaf is moved around, but its content has not changed.
    pub fn set_leaf(&mut self, pos: Position, leaf: T, serialize: bool) -> (LeafId, Option<T>) {
        let uid = leaf.unique_id();
        let (leaf_id, prev_leaf) = self.octree.set_leaf(pos, leaf);

        if serialize || !self.leaf_info.contains_key(&uid) {
            self.change_set.insert(OctantChange::Add(uid, leaf_id));
        }

        (leaf_id, prev_leaf)
    }

    /// See [`Octree::move_leaf`].
    pub fn move_leaf(&mut self, leaf: LeafId, to_pos: Position) -> (LeafId, Option<T>) {
        let (new_leaf_id, old_value) = self.octree.move_leaf(leaf, to_pos);
        (new_leaf_id, old_value)
    }

    /// See [`Octree::remove_leaf`].
    pub fn remove_leaf(&mut self, leaf: LeafId) -> Option<T> {
        let value = self.octree.remove_leaf_by_id(leaf);
        if let Some(value) = &value {
            let uid = value.unique_id();
            self.change_set.insert(OctantChange::Remove(uid));
        }
        value
    }

    /// See [`Octree::get_leaf`].
    pub fn get_leaf(&self, pos: Position) -> Option<&T> {
        self.octree.get_leaf(pos)
    }

    // TODO
    // /// Serializes the root octant and adds/removes all changed leaves. Must be called before [`Csvo::write_to`] or
    // /// [`Csvo::write_changes_to`] for them to have any effect.
    // pub fn serialize(&mut self) {
    //     if self.octree.root.is_none() {
    //         return;
    //     }
    //
    //     // move tmp buffer into scope
    //     let mut tmp_buffer = self.tmp_octant_buffer.take().unwrap();
    //
    //     // rebuild & remove all changed leaf octants
    //     let changes = self.change_set.drain().collect::<Vec<OctantChange>>();
    //     for change in changes {
    //         match change {
    //             OctantChange::Add(id, leaf_id) => {
    //                 let child = &mut self.octree.octants[leaf_id.parent as usize].children[leaf_id.idx as usize];
    //                 let content = child.get_leaf_value_mut().unwrap();
    //                 let result = content.serialize(&mut tmp_buffer.data, 0);
    //                 if result.depth > 0 {
    //                     let offset = self.buffer.insert(id, &tmp_buffer);
    //                     tmp_buffer.reset();
    //
    //                     self.leaf_info.insert(id, LeafInfo { buf_offset: offset, serialization: result });
    //                 }
    //             }
    //
    //             OctantChange::Remove(id) => {
    //                 self.buffer.remove(id);
    //                 self.leaf_info.remove(&id);
    //             }
    //         }
    //     }
    //
    //     // rebuild root octree
    //     let result = self.serialize_root(&mut tmp_buffer);
    //     let offset = self.buffer.insert(u64::MAX, &tmp_buffer);
    //     tmp_buffer.reset();
    //     self.root_info = Some(LeafInfo { buf_offset: offset, serialization: result });
    //
    //     // return tmp buffer for reuse
    //     self.tmp_octant_buffer = Some(tmp_buffer);
    // }
    //
    // fn serialize_root(&self, dst: &mut ChunkBuffer) -> SerializationResult {
    //     let root_id = self.octree.root.unwrap();
    //
    //     serialize_octant(&self.octree, root_id, &mut dst.data, 0, &|params| {
    //         // retrieve leaf info, skip if not found
    //         let uid = params.content.unique_id();
    //         let info = self.leaf_info.get(&uid);
    //         if info.is_none() {
    //             return;
    //         }
    //
    //         // combine child & leaf mask and shift into position, depending on child index
    //         let info = info.unwrap();
    //         let mut mask = ((info.serialization.child_mask as u32) << 8) | info.serialization.leaf_mask as u32;
    //         if (params.idx % 2) != 0 {
    //             mask <<= 16;
    //         }
    //         params.dst[(params.idx / 2) as usize] |= mask;
    //
    //         // absolute buffer position of the leaf value
    //         params.dst[(4 + params.idx) as usize] = info.buf_offset as u32 + Self::PREAMBLE_LENGTH;
    //         // override accumulated depth, if octree is expanded due to leaf value
    //         params.result.depth = params.result.depth.max(info.serialization.depth + 1);
    //     })
    // }
}

impl<T: Serializable, A: Allocator> Container<u8> for Csvo<T, A> {
    fn depth(&self) -> u8 {
        if self.root_info.is_none() {
            return 0;
        }
        self.root_info.unwrap().serialization.depth
    }

    fn size_in_bytes(&self) -> usize {
        self.buffer.size_in_bytes()
    }

    /// Writes the full serialized SVO buffer to the `dst` pointer. Returns the number of elements written. Must be
    /// called after [`Csvo::serialize`].
    unsafe fn write_to(&self, dst: *mut u8) -> usize {
        // TODO

        if self.root_info.is_none() {
            return 0;
        }

        let start = dst as usize;
        let info = self.root_info.unwrap();
        // let dst = Self::write_preamble(info, dst); // TODO

        let len = self.buffer.bytes.len();
        ptr::copy(self.buffer.bytes.as_ptr(), dst, len);

        let dst = dst.add(len);
        ((dst as usize) - start) / 4
    }

    /// Writes all changes after the last reset to the given buffer. The implementation assumes that the same buffer,
    /// that was used in the initial call to [`Csvo::write_to`] and previous calls to this method, is reused. If `reset`
    /// is true, the change tracker is reset. Must be called after [`Csvo::serialize`].
    unsafe fn write_changes_to(&mut self, dst: *mut u8, dst_len: usize, reset: bool) {
        // TODO

        if self.root_info.is_none() {
            return;
        }
        if self.buffer.updated_ranges.is_empty() {
            return;
        }

        let info = self.root_info.unwrap();
        // let dst = Self::write_preamble(info, dst); // TODO

        for changed_range in &self.buffer.updated_ranges {
            let offset = changed_range.start as isize;
            let src = self.buffer.bytes.as_ptr().offset(offset);

            // For now a simple implementation suffices instead of having a mechanism that grows the target buffer,
            // as that involves doing so on the GPU. Panic instead to make it easy to spot and prefer a cheaper,
            // over-sized buffer.
            assert!(changed_range.start + changed_range.length < dst_len,
                    "dst is not large enough: len={} range_start={} range_length={}",
                    dst_len, changed_range.start, changed_range.length,
            );

            ptr::copy(src, dst.offset(offset), changed_range.length);
        }

        if reset {
            self.buffer.updated_ranges.clear();
        }
    }
}

pub struct SerializedChunk {
    pub pos: ChunkPos,
    pos_hash: u64,
    pub lod: u8,
    pub borrowed_chunk: Option<BorrowedChunk>,
    buffer: Option<Pooled<ChunkBuffer<u8, StatsAllocator>>>,
    result: SerializationResult,
}

impl SerializedChunk {
    pub fn new(chunk: BorrowedChunk, alloc: &Arc<ChunkBufferPool<u8>>) -> Self {
        let pos = chunk.pos;
        let lod = chunk.lod;

        // use hash of position as the unique id
        let mut hasher = DefaultHasher::new();
        pos.hash(&mut hasher);
        let pos_hash = hasher.finish();

        let storage = chunk.storage.as_ref().unwrap();
        let mut buffer = alloc.allocate();
        let result = Self::serialize(storage, &mut buffer.data, lod);
        let buffer = if result.depth > 0 { Some(buffer) } else { None };
        Self { pos, pos_hash, lod, borrowed_chunk: Some(chunk), buffer, result }
    }

    fn serialize<A1: Allocator, A2: Allocator>(octree: &Octree<BlockId, A1>, dst: &mut Vec<u8, A2>, lod: u8) -> SerializationResult {
        //     if octree.root.is_none() {
        //         return SerializationResult { header_mask: 0, depth: 0 };
        //     }
        //
        //     let root_id = octree.root.unwrap();
        //     serialize_octant(octree, root_id, dst, lod, &|params| {
        //         // TODO
        //
        //         // // apply leaf mask, child mask is already applied
        //         // params.result.leaf_mask |= 1 << params.idx;
        //         // // write actual value to target position
        //         // params.dst[(4 + params.idx) as usize] = *params.content;
        //         // // leaf values have a static depth of 1
        //         // params.result.depth = 1;
        //     })

        SerializationResult { header_mask: 0, depth: 0 }
    }
}

impl Serializable for SerializedChunk {
    fn unique_id(&self) -> u64 {
        self.pos_hash
    }

    /// Serializes the already serialized chunk by copying its results into the given buffer and returning the cached
    /// result.
    fn serialize(&mut self, dst: &mut Vec<u8>, _lod: u8) -> SerializationResult {
        if self.buffer.is_some() {
            let buffer = self.buffer.as_ref().unwrap();
            dst.extend(buffer.data.iter());

            // Drop the buffer so that the allocator can reuse it. A SerializedChunk only needs it's buffer for the
            // serialization to the SVO. After that, it is indexed by an absolute pointer. If the content changes
            // however, a new SerializedChunk is built and the old one discarded.
            self.buffer = None;
        }
        self.result
    }
}

enum Encoded {
    AbsolutePointer(u32),
    Leaf(u8),
}

trait Encoder {
    fn encode(&mut self) -> Encoded;
}

impl Encoder for SerializedChunk {
    fn encode(&self) -> Encoded {
        Encoded::AbsolutePointer(0) // TODO
    }
}

impl Encoder for BlockId {
    fn encode(&self) -> Encoded {
        Encoded::Leaf(*self as u8) // TODO u32 to u8
    }
}

fn serialize_octant<T: Encoder, A: Allocator>(octree: &Octree<T, A>, octant_id: OctantId, depth: u8, lod: u8) -> Vec<u8> {
    let octant = &octree.octants[octant_id as usize];

    if depth == 1 || lod == 1 {
        let mut leaf_mask = 0u8;

        for (idx, child) in octant.children.iter().enumerate() {
            if child.is_none() {
                continue;
            }

            // try to get the leaf value
            let mut content = child.get_leaf_value();
            // if NONE, find the first child if the child is an octant
            if content.is_none() && child.is_octant() {
                let child_id = child.get_octant_value().unwrap();
                content = pick_leaf_for_lod(octree, &octree.octants[child_id as usize]);
            }
            // if nothing was found, skip
            if content.is_none() {
                continue;
            }

            // TODO how to handle nested encoding?
            // match content.unwrap().encode() {
            //     Encoded::AbsolutePointer(ptr) => children.push((idx, vec![ptr])),
            //     Encoded::Leaf(_) => leaf_mask |= 1 << idx,
            // }
        }

        return vec![leaf_mask];
    }

    let mut children = Vec::new();
    for (idx, child) in octant.children.iter().enumerate() {
        if child.is_none() {
            continue;
        }
        if child.is_leaf() {
            panic!("octree leaves must be at a uniform level");
        }

        // decrease lod and calculate buffer offset before recursively serializing the child octant
        let child_id = child.get_octant_value().unwrap();
        let child_lod = if lod > 0 { lod - 1 } else { 0 };

        let buffer = serialize_octant(octree, child_id, depth - 1, child_lod);
        children.push((idx, buffer));
    }

    let mut buffer = Vec::new();
    if depth == 2 {
        // use leaf nodes

        buffer.push(0);

        for (i, (idx, data)) in children.into_iter().enumerate() {
            buffer[0] |= 1 << idx;
            buffer.extend(data);
        }
    } else if depth == 3 {
        // use pre-leaf nodes

        buffer.extend(std::iter::repeat(0).take(1 + children.len()));

        buffer[0] = 0;
        let mut running_offset = 0;

        for (i, (idx, data)) in children.into_iter().enumerate() {
            buffer[0] |= 1 << idx;
            buffer[1 + i] = running_offset;
            running_offset += data.len() as u8;
            buffer.extend(data);
        }
    } else {
        // use internal nodes

        let mut header_mask = 0u16;
        buffer.extend(std::iter::repeat(0).take(2));

        let mut running_offset = 0u32;
        let mut offsets = Vec::new();

        for (_, data) in &children {
            offsets.push(running_offset);
            running_offset += data.len() as u32;
        }
        for (i, (idx, data)) in children.into_iter().enumerate() {
            let offset_bits = offsets[i].max(1).ilog2();
            let header_tag = (offset_bits / 8 + 1) as u16;
            header_mask |= header_tag << (idx * 2);

            match header_tag {
                1 => buffer.push(offsets[i] as u8),
                2 => buffer.extend((offsets[i] as u16).to_be_bytes()),
                3 => buffer.extend(offsets[i].to_be_bytes()),
                _ => unreachable!(),
            }

            buffer.extend(data);
        }

        let header_bytes = header_mask.to_be_bytes();
        buffer[0] = header_bytes[0];
        buffer[1] = header_bytes[1];
    }

    buffer
}

/// Iterates recursively through the given octant in breadth-first order. The goal is to find the first, highest level
/// leaf value, if any. It uses a custom iteration order to check for leaves from y=1 to y=0. This results in a better
/// look in most scenarios.
fn pick_leaf_for_lod<'a, T, A: Allocator>(octree: &'a Octree<T, A>, parent: &'a Octant<T>) -> Option<&'a T> {
    const ORDER: [usize; 8] = [2, 3, 6, 7, 0, 1, 4, 5];
    for index in ORDER {
        let child = &parent.children[index];
        if !child.is_leaf() {
            continue;
        }
        let content = child.get_leaf_value();
        return content;
    }
    for index in ORDER {
        let child = &parent.children[index];
        if !child.is_octant() {
            continue;
        }

        let child_id = child.get_octant_value().unwrap();
        let child = &octree.octants[child_id as usize];
        let result = pick_leaf_for_lod(octree, child);
        if result.is_some() {
            return result;
        }
    }
    None
}

// TODO generic serialization framework?
// TODO how to look up materials?

#[cfg(test)]
mod tests {
    use crate::world::chunk::BlockId;
    use crate::world::hds::csvo::serialize_octant;
    use crate::world::hds::octree::{Octree, Position};

    #[test]
    fn serialize_octant_single_leaf() {
        let mut octree = Octree::new();
        octree.set_leaf(Position(0, 0, 0), 1 as BlockId);
        octree.expand_to(4);
        octree.compact();

        let result = serialize_octant(&octree, octree.root.unwrap(), octree.depth(), 0);
        assert_eq!(result, vec![
            0, 1, 0,    // inode
            1, 0,       // plnode
            1, 1,       // lnode
        ]);
    }

    #[test]
    fn serialize_octant_multiple_leaves() {
        let mut octree = Octree::new();
        octree.set_leaf(Position(0, 0, 0), 1 as BlockId);
        octree.set_leaf(Position(3, 3, 3), 2 as BlockId);
        octree.set_leaf(Position(5, 4, 4), 1 as BlockId);
        octree.set_leaf(Position(6, 7, 7), 2 as BlockId);
        octree.expand_to(4);
        octree.compact();

        let result = serialize_octant(&octree, octree.root.unwrap(), octree.depth(), 0);
        assert_eq!(result, vec![
            0, 1, 0,                    // inode
            1 | (1 << 7), 0, 3,         // plnode
            1 | (1 << 7), 1, 1 << 7,    // lnode
            1 | (1 << 7), 2, 1 << 6,    // lnode
        ]);
    }

    // TODO test lod
    // TODO test nested chunk serialization
}
