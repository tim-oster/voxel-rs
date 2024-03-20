use std::alloc::{Allocator, Global};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::ptr;
use std::sync::Arc;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::world::chunk::{BlockId, ChunkPos};
use crate::world::memory::{Pool, Pooled, StatsAllocator};
use crate::world::octree::{LeafId, Octant, OctantId, Octree, Position};
use crate::world::world::BorrowedChunk;

pub type ChunkBufferPool<A = StatsAllocator> = Pool<ChunkBuffer<A>, A>;

/// `ChunkBuffer` abstracts the temporary storage used for serializing octants into the SVO format.
pub struct ChunkBuffer<A: Allocator = Global> {
    data: Vec<u32, A>,
}

impl ChunkBuffer {
    pub fn new() -> Self {
        Self::new_in(Global)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_in(capacity, Global)
    }
}

impl<A: Allocator> ChunkBuffer<A> {
    pub fn new_in(alloc: A) -> Self {
        Self::with_capacity_in(0, alloc)
    }

    pub fn with_capacity_in(capacity: usize, alloc: A) -> Self {
        Self { data: Vec::with_capacity_in(capacity, alloc) }
    }

    pub fn reset(&mut self) {
        self.data.clear();
    }
}

/// `OctantChange` describes if an octant was added (and where), or if it was removed.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
enum OctantChange {
    Add(u64, LeafId),
    Remove(u64),
}

pub trait SvoSerializable {
    /// Returns a unique id for the serializable value. It is used to keep track of the serialized result when moving
    /// it around inside the SVO.
    fn unique_id(&self) -> u64;

    /// Serializes the data into the destination buffer and returns metadata about the data layout.
    fn serialize(&mut self, dst: &mut Vec<u32>, lod: u8) -> SerializationResult;
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SerializationResult {
    /// One bit per child to describe if the octant contains anything or is empty.
    pub child_mask: u8,
    /// One bit per child to describe if the octant contains a leaf value or another octant. Can only be set if the
    /// corresponding bit in `child_mask` is set as well.
    pub leaf_mask: u8,
    /// Indicates how "deep" the serialized result is.
    /// 0 = nothing was serialized.
    /// 1 = for leaf values or child octants with no children.
    /// 2..n = for octants with other octants and/or children.
    pub depth: u8,
}

/// SVO (Sparse Voxel Octree) is decorator for a normal octree that can serialize into a binary format useful for
/// space-efficient, quick traversal on the GPU. Depending on how `T` serializes, it is possible to have nested
/// octrees.
///
/// ### Terminology
///
/// - **Octree** = defined by one octant as the root of the tree
/// - **Octant** = a cube that can be subdivided into 8 equal sub-cubes/child octants, each containing more octants or
/// one leaf value
/// - **Leaf** = end nodes of the tree - they contain the actual value
/// - **Absolute Pointer** = SVOs use a linear byte buffer to encode their data. For one octant to be able to reference
/// a child octant, pointers are required. Absolute pointers contain the absolute position inside the buffer. This is
/// used at the boundaries of the root octree and leaf values in this implementation.
/// - **Relative Pointer** = in contrast to absolute pointers, relative pointers encode the target index relative to
/// their own position. This makes encoding of coherent data efficient as it is not required to keep track of all
/// absolute positions.
///
/// ### Optimisations
///
/// - On [`Svo::serialize`], the root octree is always fully serialized and all its octants have relative pointers
/// to each other. Leaves however are only serialized once. The root octree uses absolute pointers to index them. This
/// allows for efficient "leaf moving" by swapping the pointers in the root octree.
/// - Removing leaves frees up already allocated space. The implementation keeps track of that space and reuses it
/// efficiently for new leaves.
/// - [`Svo::write_to`] can be used to copy the whole serialized buffer to a target buffer. This only needs to be done
/// once, after that a call to [`Svo::write_changes_to`] with the same buffer suffices and only copies the changed
/// buffer ranges.
///
/// ### Binary format
///
/// Each octant is encoded as 12 `u32`s or 48 bytes - 16 bytes for the header and 32 bytes for the body. The header
/// contains child & leaf masks for every octant. If the child mask is set for an octant, than depending on the state
/// of the leaf mask at that bit, the body either contains a absolute/relative pointer to a child octant or to the
/// encoded leaf value.
///
/// **Example:**
/// ```
/// // octant child order [ 000, 100, 010, 011, 100, 101, 110, 111 ]
///
/// // header -------------------------------
/// [0]  00000000 00000000  00000010 00000010
/// //                     `-----------------`---> leaf | child mask (each one byte) for octant (0,0,0)
/// [1]  00000000 00000000  00000000 00000000
/// [2]  00000000 10000000  00000000 00000000
/// //  `-----------------`----------------------> leaf | child mask for octant (1,0,0)
/// [3]  00000000 00000000  00000000 00000000
/// // body ---------------------------------
/// [4]  00000000 00000000  00000000 11111111 // leaf value for octant (0,0,0) = 255
/// [5]  00000000 00000000  00000000 00000000
/// [6]  00000000 00000000  00000000 00000000
/// [7]  00000000 00000000  00000000 00000000
/// [8]  00000000 00000000  00000000 00000000
/// [9]  10000000 00000000  00000000 00000100 // relative pointer to octant (1,0,0) = 4 (+ 32nd bit)
/// [10] 00000000 00000000  00000000 00000000
/// [11] 00000000 00000000  00000000 00000000
/// ```
pub struct Svo<T: SvoSerializable, A: Allocator = Global> {
    octree: Octree<T>,
    change_set: FxHashSet<OctantChange>,

    buffer: SvoBuffer<A>,
    leaf_info: FxHashMap<u64, LeafInfo>,
    root_info: Option<LeafInfo>,

    /// Reusable buffer for serializing octants data to be copied into actual `SvoBuffer`.
    tmp_octant_buffer: Option<ChunkBuffer>,
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct LeafInfo {
    /// Offset into the final buffer where the serialized leaf data was copied to.
    buf_offset: usize,
    /// Metadata (like child & leaf flag) for the serialized data.
    serialization: SerializationResult,
}

impl<T: SvoSerializable> Svo<T> {
    pub fn new() -> Self {
        Self::new_in(Global)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_in(capacity, Global)
    }
}

impl<T: SvoSerializable, A: Allocator> Svo<T, A> {
    /// Static size of the serialized data required for wrapping the root octant into a traversable format.
    const PREAMBLE_LENGTH: u32 = 5;

    pub fn new_in(alloc: A) -> Self {
        Self::with_capacity_in(0, alloc)
    }

    pub fn with_capacity_in(capacity: usize, alloc: A) -> Self {
        Self {
            octree: Octree::new(),
            change_set: FxHashSet::default(),
            buffer: SvoBuffer::with_capacity_in(capacity, alloc),
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
    /// was done. This is useful if the leaf is moved around, but its content has not changed.
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

    /// Serializes the root octant and adds/removes all changed leaves. Must be called before [`Svo::write_to`] or
    /// [`Svo::write_changes_to`] for them to have any effect.
    pub fn serialize(&mut self) {
        if self.octree.root.is_none() {
            return;
        }

        // move tmp buffer into scope
        let mut tmp_buffer = self.tmp_octant_buffer.take().unwrap();

        // rebuild & remove all changed leaf octants
        let changes = self.change_set.drain().collect::<Vec<OctantChange>>();
        for change in changes {
            match change {
                OctantChange::Add(id, leaf_id) => {
                    let child = &mut self.octree.octants[leaf_id.parent as usize].children[leaf_id.idx as usize];
                    let content = child.get_leaf_value_mut().unwrap();
                    let result = content.serialize(&mut tmp_buffer.data, 0);
                    if result.depth > 0 {
                        let offset = self.buffer.insert(id, &tmp_buffer);
                        tmp_buffer.reset();

                        self.leaf_info.insert(id, LeafInfo { buf_offset: offset, serialization: result });
                    }
                }

                OctantChange::Remove(id) => {
                    self.buffer.remove(id);
                    self.leaf_info.remove(&id);
                }
            }
        }

        // rebuild root octree
        let result = self.serialize_root(&mut tmp_buffer);
        let offset = self.buffer.insert(u64::MAX, &tmp_buffer);
        tmp_buffer.reset();
        self.root_info = Some(LeafInfo { buf_offset: offset, serialization: result });

        // return tmp buffer for reuse
        self.tmp_octant_buffer = Some(tmp_buffer);
    }

    fn serialize_root(&self, dst: &mut ChunkBuffer) -> SerializationResult {
        let root_id = self.octree.root.unwrap();

        serialize_octant(&self.octree, root_id, &mut dst.data, 0, &|params| {
            // retrieve leaf info, skip if not found
            let uid = params.content.unique_id();
            let info = self.leaf_info.get(&uid);
            if info.is_none() {
                return;
            }

            // combine child & leaf mask and shift into position, depending on child index
            let info = info.unwrap();
            let mut mask = ((info.serialization.child_mask as u32) << 8) | info.serialization.leaf_mask as u32;
            if (params.idx % 2) != 0 {
                mask <<= 16;
            }
            params.dst[(params.idx / 2) as usize] |= mask;

            // absolute buffer position of the leaf value
            params.dst[(4 + params.idx) as usize] = info.buf_offset as u32 + Self::PREAMBLE_LENGTH;
            // override accumulated depth, if octree is expanded due to leaf value
            params.result.depth = params.result.depth.max(info.serialization.depth + 1);
        })
    }

    pub fn size_in_bytes(&self) -> usize {
        self.buffer.bytes.len() * 4
    }

    pub fn depth(&self) -> u8 {
        if self.root_info.is_none() {
            return 0;
        }
        self.root_info.unwrap().serialization.depth
    }

    /// Writes the full serialized SVO buffer to the `dst` pointer. Returns the number of elements written. Must be
    /// called after [`Svo::serialize`].
    pub unsafe fn write_to(&self, dst: *mut u32) -> usize {
        if self.root_info.is_none() {
            return 0;
        }

        let start = dst as usize;
        let info = self.root_info.unwrap();
        let dst = Self::write_preamble(info, dst);

        let len = self.buffer.bytes.len();
        ptr::copy(self.buffer.bytes.as_ptr(), dst, len);

        let dst = dst.add(len);
        ((dst as usize) - start) / 4
    }

    /// Writes all changes after the last reset to the given buffer. The implementation assumes that the same buffer,
    /// that was used in the initial call to [`Svo::write_to`] and previous calls to this method, is reused. If `reset`
    /// is true, the change tracker is reset. Must be called after [`Svo::serialize`].
    pub unsafe fn write_changes_to(&mut self, dst: *mut u32, dst_len: usize, reset: bool) {
        if self.root_info.is_none() {
            return;
        }
        if self.buffer.updated_ranges.is_empty() {
            return;
        }

        let info = self.root_info.unwrap();
        let dst = Self::write_preamble(info, dst);

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

    /// Writes a "fake" octant with the SVO root octant as its first child octant to build the entry point into
    /// the data structure.
    unsafe fn write_preamble(info: LeafInfo, dst: *mut u32) -> *mut u32 {
        dst.offset(0).write((info.serialization.child_mask as u32) << 8);
        dst.offset(1).write(0);
        dst.offset(2).write(0);
        dst.offset(3).write(0);
        // preamble is always written first, so PREAMBLE_LENGTH can be used as the absolut position of the root octree
        dst.offset(4).write(info.buf_offset as u32 + Self::PREAMBLE_LENGTH);
        // return to pointer to after preamble for next writes
        dst.offset(Self::PREAMBLE_LENGTH as isize)
    }
}

/// `SerializedChunk` is a wrapper that serializes the given chunk on creation and stores the results.
pub struct SerializedChunk {
    pub pos: ChunkPos,
    pos_hash: u64,
    pub lod: u8,
    pub borrowed_chunk: Option<BorrowedChunk>,
    buffer: Option<Pooled<ChunkBuffer<StatsAllocator>>>,
    result: SerializationResult,
}

impl SerializedChunk {
    pub fn new(chunk: BorrowedChunk, alloc: &Arc<ChunkBufferPool>) -> Self {
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

    fn serialize<A1: Allocator, A2: Allocator>(octree: &Octree<BlockId, A1>, dst: &mut Vec<u32, A2>, lod: u8) -> SerializationResult {
        if octree.root.is_none() {
            return SerializationResult { child_mask: 0, leaf_mask: 0, depth: 0 };
        }

        let root_id = octree.root.unwrap();
        serialize_octant(octree, root_id, dst, lod, &|params| {
            // apply leaf mask, child mask is already applied
            params.result.leaf_mask |= 1 << params.idx;
            // write actual value to target position
            params.dst[(4 + params.idx) as usize] = *params.content;
            // leaf values have a static depth of 1
            params.result.depth = 1;
        })
    }
}

impl SvoSerializable for SerializedChunk {
    fn unique_id(&self) -> u64 {
        self.pos_hash
    }

    /// Serializes the already serialized chunk by copying its results into the given buffer and returning the cached
    /// result.
    fn serialize(&mut self, dst: &mut Vec<u32>, _lod: u8) -> SerializationResult {
        if self.buffer.is_some() {
            let buffer = self.buffer.as_ref().unwrap();
            dst.extend(buffer.data.iter());

            // Drop the buffer so that he allocator can reuse it. A SerializedChunk only needs it's buffer for the
            // serialization to the SVO. After that, it is indexed by an absolute pointer. If the content changes
            // however, a new SerializedChunk is built and the old one discarded.
            self.buffer = None;
        }
        self.result
    }
}

struct ChildEncodeParams<'a, T> {
    /// Id of the octant containing the child to be serialized.
    parent_id: OctantId,
    /// Index of the child to be serialized inside the parent.
    idx: u8,
    /// `SerializationResult` of the parent octant. Can be modified per child.
    result: &'a mut SerializationResult,
    /// Buffer for the parent's octant data. At least 12 elements long, can be expanded if necessary.
    dst: &'a mut [u32],
    /// Reference to the actual leaf value.
    content: &'a T,
}

/// Serializes the given octant into `dst` by iterating through all children and recursively stepping into child
/// octants until no child or a leaf value is found. Every (recursive) call adds a new octant header (4 * u32 = 0.5 u32
/// per octant = 8 bit child & 8 bit leaf mask) and an octant body (8 * u32 = one u32 per child).
///
/// Iteration through the octant happens in depth-first order. If an octant contains other octants, their relative
/// position to each other is kept track of and is encoded as a relative pointer as the child value for the child
/// octant. Relative pointers have their 32nd bit set to one.
///
/// To encode a child the given encoder is called. Additionally, a level of detail can be specified. For every
/// `lod` > 0, the recursion depth is limited to that lod. If no leaf could be found until the LOD is exceeded,
/// [`pick_leaf_for_lod`] is used to find the first leaf in any octant at the last position.
fn serialize_octant<T, F, A1: Allocator, A2: Allocator>(octree: &Octree<T, A1>, octant_id: OctantId, dst: &mut Vec<u32, A2>, lod: u8, child_encoder: &F) -> SerializationResult
    where F: Fn(ChildEncodeParams<T>) {
    // keep track of the start position to determine how much data was added in this call
    let start_offset = dst.len();

    dst.reserve(12);
    dst.extend(std::iter::repeat(0).take(12));

    let mut result = SerializationResult {
        child_mask: 0,
        leaf_mask: 0,
        depth: 0,
    };

    let octant = &octree.octants[octant_id as usize];
    for (idx, child) in octant.children.iter().enumerate() {
        if child.is_none() {
            continue;
        }

        // mask all non-empty children
        result.child_mask |= 1 << idx;

        // if leaf is found or end of LOD is reached
        if child.is_leaf() || lod == 1 {
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

            let content = content.unwrap();
            child_encoder(ChildEncodeParams {
                parent_id: octant_id,
                idx: idx as u8,
                result: &mut result,
                dst: &mut dst[start_offset..],
                content,
            });
        } else {
            // decrease lod and calculate buffer offset before recursively serializing the child octant
            let child_id = child.get_octant_value().unwrap();
            let child_lod = if lod > 0 { lod - 1 } else { 0 };
            let child_offset = (dst.len() - start_offset) as u32;
            let child_result = serialize_octant(octree, child_id, dst, child_lod, child_encoder);

            // write result mask to this octant's header
            let mut mask = ((child_result.child_mask as u32) << 8) | child_result.leaf_mask as u32;
            if (idx % 2) != 0 {
                mask <<= 16;
            }
            dst[start_offset + (idx / 2)] |= mask;

            // calculate relative pointer and write to this octant's body
            let relative_ptr = child_offset - 4 - idx as u32;
            assert_eq!(relative_ptr & (1 << 31), 0, "relative pointer is too large");
            dst[start_offset + 4 + idx] = relative_ptr; // offset from pointer to start of next block
            dst[start_offset + 4 + idx] |= 1 << 31; // flag as relative pointer

            // expand depth, if octant was larger than any other before
            result.depth = result.depth.max(child_result.depth + 1);
        }
    }

    result
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

#[cfg(test)]
mod svo_tests {
    use rustc_hash::FxHashMap;

    use crate::world::chunk::{BlockId, ChunkPos};
    use crate::world::memory::{Pool, StatsAllocator};
    use crate::world::octree::{LeafId, Octree, Position};
    use crate::world::svo::{ChunkBuffer, LeafInfo, Range, SerializationResult, SerializedChunk, Svo, SvoBuffer};

    /// Tests that serializing an SVO with `SerializedChunk` values produces the expected result buffer.
    #[test]
    fn serialize() {
        let mut octree = Octree::new();
        octree.set_leaf(Position(31, 0, 0), 1 as BlockId);
        octree.set_leaf(Position(0, 31, 0), 2 as BlockId);
        octree.set_leaf(Position(0, 0, 31), 3 as BlockId);
        octree.expand_to(5);
        octree.compact();

        let alloc = Pool::new_in(Box::new(ChunkBuffer::new_in), None, StatsAllocator::new());
        let mut buffer = alloc.allocate();
        let result = SerializedChunk::serialize(&octree, &mut buffer.data, 0);
        let sc = SerializedChunk {
            pos: ChunkPos::new(1, 0, 0),
            lod: 0,
            borrowed_chunk: None,
            buffer: Some(buffer),
            result,
            pos_hash: 100,
        };

        let mut svo = Svo::new();
        svo.set_leaf(Position(1, 0, 0), sc, true);
        svo.serialize();

        assert_eq!(svo.root_info, Some(LeafInfo {
            buf_offset: 156,
            serialization: SerializationResult {
                child_mask: 2,
                leaf_mask: 0,
                depth: 6,
            },
        }));

        let preamble_length = 5;
        let expected = vec![
            // core octant header
            (2 << 8) << 16,
            4 << 8,
            16 << 8,
            0,
            // core octant body
            0, (1 << 31) | 7, (1 << 31) | (6 + 4 * 12), 0,
            (1 << 31) | (4 + 8 * 12), 0, 0, 0,

            // subtree for (1,0,0)
            // header 1
            2 << 8 << 16,
            0,
            0,
            0,
            // body 1
            0, (1 << 31) | 7, 0, 0,
            0, 0, 0, 0,
            // header 2
            2 << 8 << 16,
            0,
            0,
            0,
            // body 2
            0, (1 << 31) | 7, 0, 0,
            0, 0, 0, 0,
            // header 3
            ((2 << 8) | 2) << 16,
            0,
            0,
            0,
            // body 3
            0, (1 << 31) | 7, 0, 0,
            0, 0, 0, 0,
            // leaf header
            0,
            0,
            0,
            0,
            // leaf body
            0, 1, 0, 0,
            0, 0, 0, 0,

            // subtree for (0,1,0)
            // header 1
            0,
            4 << 8,
            0,
            0,
            // body 1
            0, 0, (1 << 31) | 6, 0,
            0, 0, 0, 0,
            // header 2
            0,
            4 << 8,
            0,
            0,
            // body 2
            0, 0, (1 << 31) | 6, 0,
            0, 0, 0, 0,
            // header 3
            0,
            4 << 8 | 4,
            0,
            0,
            // body 3
            0, 0, (1 << 31) | 6, 0,
            0, 0, 0, 0,
            // leaf header
            0,
            0,
            0,
            0,
            // leaf body
            0, 0, 2, 0,
            0, 0, 0, 0,

            // subtree for (0,0,1)
            // header 1
            0,
            0,
            16 << 8,
            0,
            // body 1
            0, 0, 0, 0,
            (1 << 31) | 4, 0, 0, 0,
            // header 2
            0,
            0,
            16 << 8,
            0,
            // body 2
            0, 0, 0, 0,
            (1 << 31) | 4, 0, 0, 0,
            // header 3
            0,
            0,
            16 << 8 | 16,
            0,
            // body 3
            0, 0, 0, 0,
            (1 << 31) | 4, 0, 0, 0,
            // leaf header
            0,
            0,
            0,
            0,
            // leaf body
            0, 0, 0, 0,
            3, 0, 0, 0,

            // outer octree, first node header
            (2 | 4 | 16) << 8 << 16,
            0,
            0,
            0,
            // outer octree, first node body
            0, preamble_length, 0, 0,
            0, 0, 0, 0,
        ];
        assert_eq!(svo.buffer, SvoBuffer {
            bytes: expected.clone(),
            free_ranges: vec![],
            updated_ranges: vec![Range { start: 0, length: 168 }],
            octant_to_range: FxHashMap::from_iter([
                (100, Range { start: 0, length: 156 }),
                (u64::MAX, Range { start: 156, length: 12 }),
            ]),
        });

        let mut buffer = Vec::new();
        buffer.resize(200, 0);
        let size = unsafe { svo.write_to(buffer.as_mut_ptr()) };
        assert_eq!(buffer[..size], [
            vec![
                // preamble
                2 << 8,
                0,
                0,
                0,
                156 + preamble_length,
            ],
            expected,
        ].concat());
    }

    /// Tests that removing and moving leaf values inside an SVO works and that data can be partially updated.
    #[test]
    fn serialize_with_remove_and_move() {
        let mut svo = Svo::new();

        // NOTE: serialize twice to avoid non-deterministic results due to random map lookup in implementation
        svo.set_leaf(Position(0, 0, 0), 10, true);
        svo.serialize();
        svo.set_leaf(Position(1, 0, 0), 20, true);
        svo.serialize();

        assert_eq!(svo.root_info, Some(LeafInfo {
            buf_offset: 1,
            serialization: SerializationResult {
                child_mask: 2 | 1,
                leaf_mask: 0,
                depth: 2,
            },
        }));

        let preamble_length = 5;
        let expected = vec![
            // value 1
            10,
            // root octant
            (((1 << 8) | 1) << 16) | ((1 << 8) | 1),
            0,
            0,
            0,
            5, 18, 0, 0, // absolute positions take preamble length into account
            0, 0, 0, 0,
            // value 2
            20,
        ];
        assert_eq!(svo.buffer, SvoBuffer {
            bytes: expected.clone(),
            free_ranges: vec![],
            updated_ranges: vec![Range { start: 0, length: 14 }],
            octant_to_range: FxHashMap::from_iter([
                (10, Range { start: 0, length: 1 }),
                (20, Range { start: 13, length: 1 }),
                (u64::MAX, Range { start: 1, length: 12 }),
            ]),
        });
        svo.buffer.updated_ranges.clear();

        let mut buffer = Vec::new();
        buffer.resize(200, 0);
        let size = unsafe { svo.write_to(buffer.as_mut_ptr()) };
        assert_eq!(buffer[..size], [
            vec![
                // preamble
                (2 | 1) << 8,
                0,
                0,
                0,
                1 + preamble_length,
            ],
            expected,
        ].concat());

        // remove and move leaves, and update buffer with only changed data
        let (new_leaf_id, old_value) = svo.move_leaf(LeafId { parent: 0, idx: 1 }, Position(1, 1, 1));
        assert_eq!(new_leaf_id, LeafId { parent: 0, idx: 7 });
        assert_eq!(old_value, None);

        let old_value = svo.remove_leaf(LeafId { parent: 0, idx: 0 });
        assert_eq!(old_value, Some(10));

        svo.serialize();

        assert_eq!(svo.root_info, Some(LeafInfo {
            buf_offset: 0,
            serialization: SerializationResult {
                child_mask: 1 << 7,
                leaf_mask: 0,
                depth: 2,
            },
        }));

        let expected = vec![
            // root octant
            0,
            0,
            0,
            ((1 << 8) | 1) << 16,
            0, 0, 0, 0,
            0, 0, 0, 18,
            0,
            // value 2
            20,
        ];
        assert_eq!(svo.buffer, SvoBuffer {
            bytes: expected.clone(),
            free_ranges: vec![Range { start: 12, length: 1 }],
            updated_ranges: vec![Range { start: 0, length: 12 }],
            octant_to_range: FxHashMap::from_iter([
                (20, Range { start: 13, length: 1 }),
                (u64::MAX, Range { start: 0, length: 12 }),
            ]),
        });

        unsafe { svo.write_changes_to(buffer.as_mut_ptr(), buffer.capacity(), true); };
        assert_eq!(buffer[..size], [
            vec![
                // preamble
                (1 << 7) << 8,
                0,
                0,
                (1 << 8) << 8 << 16,
                preamble_length,
            ],
            expected,
        ].concat());
    }

    /// Tests that all different LOD levels work correctly when serializing an SVO.
    #[test]
    fn serialize_with_lod() {
        let mut octree = Octree::new();
        octree.set_leaf(Position(31, 0, 0), 1 as BlockId);
        octree.set_leaf(Position(0, 31, 0), 2 as BlockId);
        octree.set_leaf(Position(0, 0, 31), 3 as BlockId);
        octree.expand_to(5);
        octree.compact();

        // LOD 5
        let mut buffer = Vec::new();
        let result = SerializedChunk::serialize(&octree, &mut buffer, 5);
        assert_eq!(buffer, vec![
            // core octant header
            (2 << 8) << 16,
            4 << 8,
            16 << 8,
            0,
            // core octant body
            0, (1 << 31) | 7, (1 << 31) | (6 + 4 * 12), 0,
            (1 << 31) | (4 + 8 * 12), 0, 0, 0,

            // subtree for (1,0,0)
            // header 1
            2 << 8 << 16,
            0,
            0,
            0,
            // body 1
            0, (1 << 31) | 7, 0, 0,
            0, 0, 0, 0,
            // header 2
            2 << 8 << 16,
            0,
            0,
            0,
            // body 2
            0, (1 << 31) | 7, 0, 0,
            0, 0, 0, 0,
            // header 3
            ((2 << 8) | 2) << 16,
            0,
            0,
            0,
            // body 3
            0, (1 << 31) | 7, 0, 0,
            0, 0, 0, 0,
            // leaf header
            0,
            0,
            0,
            0,
            // leaf body
            0, 1, 0, 0,
            0, 0, 0, 0,

            // subtree for (0,1,0)
            // header 1
            0,
            4 << 8,
            0,
            0,
            // body 1
            0, 0, (1 << 31) | 6, 0,
            0, 0, 0, 0,
            // header 2
            0,
            4 << 8,
            0,
            0,
            // body 2
            0, 0, (1 << 31) | 6, 0,
            0, 0, 0, 0,
            // header 3
            0,
            4 << 8 | 4,
            0,
            0,
            // body 3
            0, 0, (1 << 31) | 6, 0,
            0, 0, 0, 0,
            // leaf header
            0,
            0,
            0,
            0,
            // leaf body
            0, 0, 2, 0,
            0, 0, 0, 0,

            // subtree for (0,0,1)
            // header 1
            0,
            0,
            16 << 8,
            0,
            // body 1
            0, 0, 0, 0,
            (1 << 31) | 4, 0, 0, 0,
            // header 2
            0,
            0,
            16 << 8,
            0,
            // body 2
            0, 0, 0, 0,
            (1 << 31) | 4, 0, 0, 0,
            // header 3
            0,
            0,
            16 << 8 | 16,
            0,
            // body 3
            0, 0, 0, 0,
            (1 << 31) | 4, 0, 0, 0,
            // leaf header
            0,
            0,
            0,
            0,
            // leaf body
            0, 0, 0, 0,
            3, 0, 0, 0,
        ]);
        assert_eq!(result, SerializationResult {
            child_mask: 2 | 4 | 16,
            leaf_mask: 0,
            depth: 5,
        });

        // LOD 4
        let mut buffer = Vec::new();
        let result = SerializedChunk::serialize(&octree, &mut buffer, 4);
        assert_eq!(buffer, vec![
            // core octant header
            (2 << 8) << 16,
            4 << 8,
            16 << 8,
            0,
            // core octant body
            0, (1 << 31) | 7, (1 << 31) | (6 + 3 * 12), 0,
            (1 << 31) | (4 + 6 * 12), 0, 0, 0,

            // subtree for (1,0,0)
            // header 1
            2 << 8 << 16,
            0,
            0,
            0,
            // body 1
            0, (1 << 31) | 7, 0, 0,
            0, 0, 0, 0,
            // header 2
            ((2 << 8) | 2) << 16,
            0,
            0,
            0,
            // body 2
            0, (1 << 31) | 7, 0, 0,
            0, 0, 0, 0,
            // leaf header
            0,
            0,
            0,
            0,
            // leaf body
            0, 1, 0, 0,
            0, 0, 0, 0,

            // subtree for (0,1,0)
            // header 1
            0,
            4 << 8,
            0,
            0,
            // body 1
            0, 0, (1 << 31) | 6, 0,
            0, 0, 0, 0,
            // header 2
            0,
            4 << 8 | 4,
            0,
            0,
            // body 2
            0, 0, (1 << 31) | 6, 0,
            0, 0, 0, 0,
            // leaf header
            0,
            0,
            0,
            0,
            // leaf body
            0, 0, 2, 0,
            0, 0, 0, 0,

            // subtree for (0,0,1)
            // header 1
            0,
            0,
            16 << 8,
            0,
            // body 1
            0, 0, 0, 0,
            (1 << 31) | 4, 0, 0, 0,
            // header 2
            0,
            0,
            16 << 8 | 16,
            0,
            // body 2
            0, 0, 0, 0,
            (1 << 31) | 4, 0, 0, 0,
            // leaf header
            0,
            0,
            0,
            0,
            // leaf body
            0, 0, 0, 0,
            3, 0, 0, 0,
        ]);
        assert_eq!(result, SerializationResult {
            child_mask: 2 | 4 | 16,
            leaf_mask: 0,
            depth: 4,
        });

        // LOD 3
        let mut buffer = Vec::new();
        let result = SerializedChunk::serialize(&octree, &mut buffer, 3);
        assert_eq!(buffer, vec![
            // core octant header
            (2 << 8) << 16,
            4 << 8,
            16 << 8,
            0,
            // core octant body
            0, (1 << 31) | 7, (1 << 31) | (6 + 2 * 12), 0,
            (1 << 31) | (4 + 4 * 12), 0, 0, 0,

            // subtree for (1,0,0)
            // header 1
            ((2 << 8) | 2) << 16,
            0,
            0,
            0,
            // body 1
            0, (1 << 31) | 7, 0, 0,
            0, 0, 0, 0,
            // leaf header
            0,
            0,
            0,
            0,
            // leaf body
            0, 1, 0, 0,
            0, 0, 0, 0,

            // subtree for (0,1,0)
            // header 1
            0,
            4 << 8 | 4,
            0,
            0,
            // body 1
            0, 0, (1 << 31) | 6, 0,
            0, 0, 0, 0,
            // leaf header
            0,
            0,
            0,
            0,
            // leaf body
            0, 0, 2, 0,
            0, 0, 0, 0,

            // subtree for (0,0,1)
            // header 1
            0,
            0,
            16 << 8 | 16,
            0,
            // body 1
            0, 0, 0, 0,
            (1 << 31) | 4, 0, 0, 0,
            // leaf header
            0,
            0,
            0,
            0,
            // leaf body
            0, 0, 0, 0,
            3, 0, 0, 0,
        ]);
        assert_eq!(result, SerializationResult {
            child_mask: 2 | 4 | 16,
            leaf_mask: 0,
            depth: 3,
        });

        // LOD 2
        let mut buffer = Vec::new();
        let result = SerializedChunk::serialize(&octree, &mut buffer, 2);
        assert_eq!(buffer, vec![
            // core octant header
            ((2 << 8) | 2) << 16,
            4 << 8 | 4,
            16 << 8 | 16,
            0,
            // core octant body
            0, (1 << 31) | 7, (1 << 31) | (6 + 12), 0,
            (1 << 31) | (4 + 2 * 12), 0, 0, 0,

            // subtree for (1,0,0)
            // leaf header
            0,
            0,
            0,
            0,
            // leaf body
            0, 1, 0, 0,
            0, 0, 0, 0,

            // subtree for (0,1,0)
            // leaf header
            0,
            0,
            0,
            0,
            // leaf body
            0, 0, 2, 0,
            0, 0, 0, 0,

            // subtree for (0,0,1)
            // leaf header
            0,
            0,
            0,
            0,
            // leaf body
            0, 0, 0, 0,
            3, 0, 0, 0,
        ]);
        assert_eq!(result, SerializationResult {
            child_mask: 2 | 4 | 16,
            leaf_mask: 0,
            depth: 2,
        });

        // LOD 1
        let mut buffer = Vec::new();
        let result = SerializedChunk::serialize(&octree, &mut buffer, 1);
        assert_eq!(buffer, vec![
            // leaf header
            0,
            0,
            0,
            0,
            // leaf body
            0, 1, 2, 0,
            3, 0, 0, 0,
        ]);
        assert_eq!(result, SerializationResult {
            child_mask: 2 | 4 | 16,
            leaf_mask: 2 | 4 | 16,
            depth: 1,
        });
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Range {
    pub start: usize,
    pub length: usize,
}

/// `SvoBuffer` allows for copying data to an internal buffer and keeping track of the range the data was copied to using
/// unique ids. Those ids can be used to remove data from the buffer again.
///
/// Removing data does not free up the already allocated memory but instead marks the range inside the buffer as free.
/// If two adjacent ranges are removed, they are merged into one. Inserting into the buffer will prioritize reusing
/// memory over appending to the end of the internal buffer.
#[derive(Debug)]
struct SvoBuffer<A: Allocator> {
    bytes: Vec<u32, A>,
    free_ranges: Vec<Range>,
    updated_ranges: Vec<Range>,
    octant_to_range: FxHashMap<u64, Range>,
}

impl<A: Allocator> PartialEq for SvoBuffer<A> {
    fn eq(&self, other: &Self) -> bool {
        self.bytes == other.bytes
            && self.free_ranges == other.free_ranges
            && self.updated_ranges == other.updated_ranges
            && self.octant_to_range == other.octant_to_range
    }
}

impl<A: Allocator> SvoBuffer<A> {
    fn with_capacity_in(initial_capacity: usize, alloc: A) -> Self {
        let mut bytes = Vec::with_capacity_in(initial_capacity, alloc);
        bytes.extend(std::iter::repeat(0).take(initial_capacity));

        let mut buffer = Self {
            bytes,
            free_ranges: Vec::new(),
            updated_ranges: Vec::new(),
            octant_to_range: FxHashMap::default(),
        };
        if initial_capacity > 0 {
            buffer.free_ranges.push(Range { start: 0, length: initial_capacity });
        }
        buffer
    }

    fn clear(&mut self) {
        self.free_ranges.clear();
        self.free_ranges.push(Range { start: 0, length: self.bytes.capacity() });
        self.updated_ranges.clear();
        self.octant_to_range.clear();
    }

    /// Inserts by copying data from the given buffer to the first free range or to the end of the internal buffer.
    fn insert(&mut self, id: u64, buf: &ChunkBuffer) -> usize {
        self.remove(id);

        let mut ptr = self.bytes.len();
        let length = buf.data.len();

        // try to find the first free range that is at least as long as the buffer
        if let Some(pos) = self.free_ranges.iter().position(|x| length <= x.length) {
            let range = &mut self.free_ranges[pos];

            ptr = range.start;

            if length < range.length {
                range.start += length;
                range.length -= length;
            } else {
                self.free_ranges.remove(pos);
            }

            unsafe {
                ptr::copy(buf.data.as_ptr(), self.bytes.as_mut_ptr().add(ptr), length);
            }
        } else {
            // otherwise, extend at the end
            self.bytes.extend(buf.data.iter());
        }

        self.octant_to_range.insert(id, Range { start: ptr, length });

        self.updated_ranges.push(Range { start: ptr, length });
        Self::merge_ranges(&mut self.updated_ranges);

        ptr
    }

    /// Frees the corresponding range for the given id.
    fn remove(&mut self, id: u64) {
        let range = self.octant_to_range.remove(&id);
        if range.is_none() {
            return;
        }

        let range = range.unwrap();
        self.free_ranges.push(range);
        Self::merge_ranges(&mut self.free_ranges);
    }

    /// Orders all free ranges by start index and merges adjacent ranges into one.
    fn merge_ranges(ranges: &mut Vec<Range>) {
        // Unstable is fine here as no equivalent objects can exist. It should be slightly faster
        // and avoids heap allocations.
        ranges.sort_unstable_by(|lhs, rhs| lhs.start.cmp(&rhs.start));

        let mut i = 1;
        while i < ranges.len() {
            let rhs = ranges[i];
            let lhs = &mut ranges[i - 1];

            if rhs.start <= lhs.start + lhs.length {
                let diff = lhs.start + lhs.length - rhs.start;
                if rhs.length > diff {
                    lhs.length += rhs.length - diff;
                }
                ranges.remove(i);
            } else {
                i += 1;
            }
        }
    }
}

#[cfg(test)]
mod svo_buffer_tests {
    use std::alloc::Global;

    use rustc_hash::FxHashMap;

    use crate::world::svo::{ChunkBuffer, Range, SvoBuffer};

    /// Tests different insert & remove edge cases.
    #[test]
    fn buffer_insert_remove() {
        // create empty buffer with initial capacity
        let mut buffer = SvoBuffer::with_capacity_in(10, Global);

        assert_eq!(buffer, SvoBuffer {
            bytes: vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            free_ranges: vec![Range { start: 0, length: 10 }],
            updated_ranges: vec![],
            octant_to_range: Default::default(),
        });

        // insert data until initial capacity is full
        buffer.insert(1, &ChunkBuffer { data: vec![0, 1, 2, 3, 4] });
        buffer.insert(2, &ChunkBuffer { data: vec![5, 6] });
        buffer.insert(3, &ChunkBuffer { data: vec![7, 8, 9] });

        assert_eq!(buffer, SvoBuffer {
            bytes: vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            free_ranges: vec![],
            updated_ranges: vec![Range { start: 0, length: 10 }],
            octant_to_range: FxHashMap::from_iter([
                (1, Range { start: 0, length: 5 }),
                (2, Range { start: 5, length: 2 }),
                (3, Range { start: 7, length: 3 }),
            ]),
        });

        // exceed initial capacity
        buffer.insert(4, &ChunkBuffer { data: vec![10] });

        assert_eq!(buffer, SvoBuffer {
            bytes: vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            free_ranges: vec![],
            updated_ranges: vec![Range { start: 0, length: 11 }],
            octant_to_range: FxHashMap::from_iter([
                (1, Range { start: 0, length: 5 }),
                (2, Range { start: 5, length: 2 }),
                (3, Range { start: 7, length: 3 }),
                (4, Range { start: 10, length: 1 }),
            ]),
        });

        // replace already existing data
        buffer.insert(3, &ChunkBuffer { data: vec![11] });

        assert_eq!(buffer, SvoBuffer {
            bytes: vec![0, 1, 2, 3, 4, 5, 6, 11, 8, 9, 10],
            free_ranges: vec![Range { start: 8, length: 2 }],
            updated_ranges: vec![Range { start: 0, length: 11 }],
            octant_to_range: FxHashMap::from_iter([
                (1, Range { start: 0, length: 5 }),
                (2, Range { start: 5, length: 2 }),
                (3, Range { start: 7, length: 1 }),
                (4, Range { start: 10, length: 1 }),
            ]),
        });

        // remove existing data
        buffer.remove(2);
        buffer.remove(3);

        assert_eq!(buffer, SvoBuffer {
            bytes: vec![0, 1, 2, 3, 4, 5, 6, 11, 8, 9, 10],
            free_ranges: vec![Range { start: 5, length: 5 }],
            updated_ranges: vec![Range { start: 0, length: 11 }],
            octant_to_range: FxHashMap::from_iter([
                (1, Range { start: 0, length: 5 }),
                (4, Range { start: 10, length: 1 }),
            ]),
        });

        // insert into free space
        buffer.insert(5, &ChunkBuffer { data: vec![12, 13, 14] });

        assert_eq!(buffer, SvoBuffer {
            bytes: vec![0, 1, 2, 3, 4, 12, 13, 14, 8, 9, 10],
            free_ranges: vec![Range { start: 8, length: 2 }],
            updated_ranges: vec![Range { start: 0, length: 11 }],
            octant_to_range: FxHashMap::from_iter([
                (1, Range { start: 0, length: 5 }),
                (4, Range { start: 10, length: 1 }),
                (5, Range { start: 5, length: 3 }),
            ]),
        });

        // remove all and revert back to beginning
        buffer.remove(5);
        buffer.remove(4);
        buffer.remove(1);

        assert_eq!(buffer, SvoBuffer {
            bytes: vec![0, 1, 2, 3, 4, 12, 13, 14, 8, 9, 10],
            free_ranges: vec![Range { start: 0, length: 11 }],
            updated_ranges: vec![Range { start: 0, length: 11 }],
            octant_to_range: FxHashMap::default(),
        });
    }

    /// Tests that range merging edge cases work properly.
    #[test]
    fn merge_ranges() {
        struct TestCase {
            name: &'static str,
            input: Vec<Range>,
            expected: Vec<Range>,
        }
        for case in vec![
            TestCase {
                name: "join adjacent ranges",
                input: vec![
                    Range { start: 0, length: 1 },
                    Range { start: 1, length: 1 },
                    Range { start: 2, length: 1 },
                ],
                expected: vec![
                    Range { start: 0, length: 3 },
                ],
            },
            TestCase {
                name: "ignore non-adjacent ranges",
                input: vec![
                    Range { start: 0, length: 1 },
                    Range { start: 2, length: 1 },
                ],
                expected: vec![
                    Range { start: 0, length: 1 },
                    Range { start: 2, length: 1 },
                ],
            },
            TestCase {
                name: "remove fully contained ranges",
                input: vec![
                    Range { start: 0, length: 5 },
                    Range { start: 3, length: 1 },
                ],
                expected: vec![
                    Range { start: 0, length: 5 },
                ],
            },
            TestCase {
                name: "remove and extend contained ranges",
                input: vec![
                    Range { start: 0, length: 5 },
                    Range { start: 3, length: 5 },
                ],
                expected: vec![
                    Range { start: 0, length: 8 },
                ],
            },
            TestCase {
                name: "works in inverse order",
                input: vec![
                    Range { start: 3, length: 5 },
                    Range { start: 0, length: 5 },
                ],
                expected: vec![
                    Range { start: 0, length: 8 },
                ],
            },
        ] {
            let mut input = case.input;
            SvoBuffer::<Global>::merge_ranges(&mut input);
            assert_eq!(input, case.expected, "{}", case.name);
        }
    }
}
