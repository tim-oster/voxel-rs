use std::{mem, ptr};
use std::alloc::{Allocator, Global};
use std::hash::{DefaultHasher, Hash, Hasher};

use rustc_hash::{FxHashMap, FxHashSet};

use crate::world::chunk::{BlockId, ChunkPos};
use crate::world::hds::internal::{pick_leaf_for_lod, RangeBuffer};
use crate::world::hds::octree::{LeafId, OctantId, Octree, Position};
use crate::world::hds::WorldSvo;
use crate::world::world::BorrowedChunk;

// TODO use common root implementation?

/// `OctantChange` describes if an octant was added (and where), or if it was removed.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
enum OctantChange {
    Add(u64, LeafId),
    Remove(u64),
}

#[derive(Copy, Clone, Debug, PartialEq)]
struct LeafInfo {
    /// Offset into the final buffer where the serialized leaf data was copied to.
    buf_offset: usize,
}

pub struct Csvo<A: Allocator = Global> {
    octree: Octree<SerializedChunk>,
    change_set: FxHashSet<OctantChange>,
    child_depth: u8,

    buffer: RangeBuffer<u8, A>,
    leaf_info: FxHashMap<u64, LeafInfo>,
    root_info: Option<LeafInfo>,
}

impl Csvo {
    pub fn new() -> Self {
        Self::new_in(Global)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_in(capacity, Global)
    }
}

impl<A: Allocator> Csvo<A> {
    pub fn new_in(alloc: A) -> Self {
        Self::with_capacity_in(0, alloc)
    }

    pub fn with_capacity_in(capacity: usize, alloc: A) -> Self {
        Self {
            octree: Octree::new(),
            change_set: FxHashSet::default(),
            child_depth: 0,
            buffer: RangeBuffer::with_capacity_in(capacity, alloc),
            leaf_info: FxHashMap::default(),
            root_info: None,
        }
    }

    fn serialize_root(&self, octree: &Octree<SerializedChunk>, octant_id: OctantId, depth: u8) -> Vec<u8> {
        let octant = &octree.octants[octant_id as usize];

        let mut children = Vec::new();
        for (idx, child) in octant.children.iter().enumerate() {
            if child.is_none() {
                continue;
            }
            if depth == 1 {
                if let Some(content) = child.get_leaf_value() {
                    let leaf_info = self.leaf_info.get(&content.pos_hash).unwrap();
                    assert_eq!(leaf_info.buf_offset & (1 << 31), 0, "32 bit pointers must not have the 32nd bit set");

                    let pointer = leaf_info.buf_offset as u32 | (1 << 31);
                    children.push((idx, pointer.to_be_bytes().to_vec()));
                }
                continue;
            }
            assert!(!child.is_leaf(), "octree leaves must be at a uniform level");

            // decrease lod and calculate buffer offset before recursively serializing the child octant
            let child_id = child.get_octant_value().unwrap();
            let buffer = self.serialize_root(octree, child_id, depth - 1);
            children.push((idx, buffer));
        }

        let mut buffer = Vec::new();

        let mut header_mask = 0u16;
        buffer.extend(std::iter::repeat(0).take(2));

        if depth == 1 {
            for (i, (idx, pointer)) in children.into_iter().enumerate() {
                // append pointers as absolut pointers with 32 bits and without any data
                header_mask |= 3 << (idx * 2);
                buffer.extend(pointer);
            }
        } else {
            let mut running_offset = 0u32;
            let mut offsets = Vec::new();

            for (_, data) in &children {
                offsets.push(running_offset);
                running_offset += data.len() as u32;
            }
            for (i, (idx, _)) in children.iter().enumerate() {
                let offset_bits = offsets[i].max(1).ilog2();
                let header_tag = (offset_bits / 8 + 1) as u16;
                header_mask |= header_tag << (idx * 2);

                match header_tag {
                    1 => buffer.push(offsets[i] as u8),
                    2 => buffer.extend((offsets[i] as u16).to_be_bytes()),
                    3 => {
                        assert_eq!(offsets[i] & (1 << 31), 0, "32 bit pointers must not have the 32nd bit set");
                        buffer.extend(offsets[i].to_be_bytes());
                    }
                    _ => unreachable!(),
                }
            }
            for (_, data) in children {
                buffer.extend(data);
            }
        }

        let header_bytes = header_mask.to_be_bytes();
        buffer[0] = header_bytes[0];
        buffer[1] = header_bytes[1];

        buffer
    }
}

impl<A: Allocator> WorldSvo<SerializedChunk, u8> for Csvo<A> {
    /// Clears all data from the SVO but does not free up memory.
    fn clear(&mut self) {
        self.octree.reset();
        self.change_set.clear();
        self.child_depth = 0;
        self.buffer.clear();
        self.leaf_info.clear();
        self.root_info = None;
    }

    /// See [`Octree::set_leaf`]. Setting `serialize` to false attempts to bypass re-serializing the leaf in case it
    /// was done already. This is useful if the leaf is moved around, but its content has not changed.
    fn set_leaf(&mut self, pos: Position, leaf: SerializedChunk, serialize: bool) -> (LeafId, Option<SerializedChunk>) {
        let uid = leaf.pos_hash;
        let (leaf_id, prev_leaf) = self.octree.set_leaf(pos, leaf);

        if serialize || !self.leaf_info.contains_key(&uid) {
            self.change_set.insert(OctantChange::Add(uid, leaf_id));
        }

        (leaf_id, prev_leaf)
    }

    /// See [`Octree::move_leaf`].
    fn move_leaf(&mut self, leaf: LeafId, to_pos: Position) -> (LeafId, Option<SerializedChunk>) {
        let (new_leaf_id, old_value) = self.octree.move_leaf(leaf, to_pos);
        (new_leaf_id, old_value)
    }

    /// See [`Octree::remove_leaf`].
    fn remove_leaf(&mut self, leaf: LeafId) -> Option<SerializedChunk> {
        let value = self.octree.remove_leaf_by_id(leaf);
        if let Some(value) = &value {
            let uid = value.pos_hash;
            self.change_set.insert(OctantChange::Remove(uid));
        }
        value
    }

    /// See [`Octree::get_leaf`].
    fn get_leaf(&self, pos: Position) -> Option<&SerializedChunk> {
        self.octree.get_leaf(pos)
    }

    /// Serializes the root octant and adds/removes all changed leaves. Must be called before [`Csvo::write_to`] or
    /// [`Csvo::write_changes_to`] for them to have any effect.
    fn serialize(&mut self) {
        if self.octree.root.is_none() {
            return;
        }

        // rebuild & remove all changed leaf octants
        let changes = self.change_set.drain().collect::<Vec<OctantChange>>();
        for change in changes {
            match change {
                OctantChange::Add(id, leaf_id) => {
                    let child = &mut self.octree.octants[leaf_id.parent as usize].children[leaf_id.idx as usize];
                    let content = child.get_leaf_value_mut().unwrap();

                    // TODO change
                    assert!(!(content.depth != self.child_depth && self.child_depth != 0), "all children must have the same depth");
                    self.child_depth = content.depth;

                    if let Some(buffer) = content.buffer.take() {
                        // TODO reuse some buffer for this?

                        let materials = content.materials.take().unwrap();
                        let material_bytes = materials.len() * mem::size_of::<BlockId>();
                        let mut merged = Vec::with_capacity(1 + 4 + material_bytes + buffer.len());

                        merged.push(if content.lod != 0 { content.lod } else { content.depth });
                        merged.extend_from_slice(&(material_bytes as u32).to_be_bytes());
                        for material in materials {
                            merged.extend_from_slice(&material.to_be_bytes());
                        }
                        merged.extend(buffer);

                        let offset = self.buffer.insert(id, &merged);

                        // Drop the buffer so that the allocator can reuse it. A SerializedChunk only needs it's buffer for the
                        // serialization to the SVO. After that, it is indexed by an absolute pointer. If the content changes
                        // however, a new SerializedChunk is built and the old one discarded.
                        content.buffer = None;

                        self.leaf_info.insert(id, LeafInfo { buf_offset: offset });
                    }
                }

                OctantChange::Remove(id) => {
                    self.buffer.remove(id);
                    self.leaf_info.remove(&id);
                }
            }
        }

        // rebuild root octree
        let buffer = self.serialize_root(&self.octree, self.octree.root.unwrap(), self.octree.depth());
        let offset = self.buffer.insert(u64::MAX, &buffer);
        self.root_info = Some(LeafInfo { buf_offset: offset });
    }

    fn depth(&self) -> u8 {
        self.octree.depth() + self.child_depth
    }

    fn size_in_bytes(&self) -> usize {
        self.buffer.size_in_bytes()
    }

    /// Writes the full serialized SVO buffer to the `dst` pointer. Returns the number of elements written. Must be
    /// called after [`Csvo::serialize`].
    unsafe fn write_to(&self, dst: *mut u8) -> usize {
        if self.root_info.is_none() {
            return 0;
        }

        let start = dst as usize;
        let info = self.root_info.unwrap();

        let len = self.buffer.bytes.len();
        ptr::copy(self.buffer.bytes.as_ptr(), dst, len);

        let dst = dst.add(len);
        (dst as usize) - start
    }

    /// Writes all changes after the last reset to the given buffer. The implementation assumes that the same buffer,
    /// that was used in the initial call to [`Csvo::write_to`] and previous calls to this method, is reused. If `reset`
    /// is true, the change tracker is reset. Must be called after [`Csvo::serialize`].
    unsafe fn write_changes_to(&mut self, dst: *mut u8, dst_len: usize, reset: bool) {
        if self.root_info.is_none() {
            return;
        }
        if self.buffer.updated_ranges.is_empty() {
            return;
        }

        let info = self.root_info.unwrap();

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

#[cfg(test)]
mod csvo_tests {
    use rustc_hash::FxHashMap;

    use crate::world::chunk::{BlockId, Chunk, ChunkPos, ChunkStorage};
    use crate::world::hds::csvo::{Csvo, LeafInfo, SerializedChunk};
    use crate::world::hds::internal::{Range, RangeBuffer};
    use crate::world::hds::octree::Position;
    use crate::world::hds::WorldSvo;
    use crate::world::memory::{Pool, StatsAllocator};
    use crate::world::world::BorrowedChunk;

    #[test]
    fn serialize() {
        let alloc = Pool::new_in(Box::new(ChunkStorage::new_in), None, StatsAllocator::new());
        let mut chunk = Chunk::new(ChunkPos::new(0, 0, 0), 5, alloc.allocate());
        chunk.set_block(31, 0, 0, 1 as BlockId);
        chunk.set_block(0, 31, 0, 2 as BlockId);
        chunk.set_block(0, 0, 31, 3 as BlockId);
        chunk.storage.as_mut().unwrap().compact();
        let sc = SerializedChunk::new(BorrowedChunk::from(chunk));

        let mut esvo = Csvo::new();
        esvo.set_leaf(Position(1, 0, 0), sc, true);
        esvo.serialize();

        assert_eq!(esvo.root_info, Some(LeafInfo { buf_offset: 43 }));

        let expected = vec![
            // chunk LOD
            5,

            // chunk materials
            0, 0, 0, 12,
            0, 0, 0, 1,
            0, 0, 0, 2,
            0, 0, 0, 3,

            // chunk voxels
            0b00_00_00_01, 0b_00_01_01_00, 0, 7, 14,
            0, 0b00_00_01_00, 0,
            2, 0,
            2,
            2,
            0, 16, 0,
            4, 0,
            4,
            4,
            1, 0, 0,
            16, 0,
            16,
            16,

            // root octant
            0, 0b00_00_11_00,
            1 << 7, 0, 0, 0, // 0 with absolute pointer flag as u32 bytes
        ];
        assert_eq!(esvo.buffer, RangeBuffer {
            bytes: expected.clone(),
            free_ranges: vec![],
            updated_ranges: vec![Range { start: 0, length: 49 }],
            octant_to_range: FxHashMap::from_iter([
                (2435999049025295583, Range { start: 0, length: 43 }),
                (u64::MAX, Range { start: 43, length: 6 }),
            ]),
        });

        let mut buffer = Vec::new();
        buffer.resize(200, 0);
        let size = unsafe { esvo.write_to(buffer.as_mut_ptr()) };
        assert_eq!(buffer[..size], expected);
    }
}

// -------------------------------------------------------------------------------------------------

pub struct SerializedChunk {
    pos: ChunkPos,
    pos_hash: u64,
    depth: u8,
    lod: u8,
    borrowed_chunk: Option<BorrowedChunk>,
    buffer: Option<Vec<u8>>,
    materials: Option<Vec<BlockId>>,
}

impl SerializedChunk {
    // TODO use memory pool alloc: &Arc<ChunkBufferPool<u8>> ?
    pub fn new(chunk: BorrowedChunk) -> Self {
        // use hash of position as the unique id
        let mut hasher = DefaultHasher::new();
        chunk.pos.hash(&mut hasher);
        let pos_hash = hasher.finish();

        let storage = chunk.storage.as_ref().unwrap();

        let mut buffer = None;
        let mut materials = None;
        if let Some(root_id) = storage.root {
            let mut depth = storage.depth();
            if chunk.lod != 0 && chunk.lod < depth {
                depth -= chunk.lod;
            }

            let (b, m) = Self::serialize_octant(storage, root_id, depth);
            (buffer, materials) = (Some(b), Some(m));
        }

        Self {
            pos: chunk.pos,
            pos_hash,
            depth: storage.depth(),
            lod: chunk.lod,
            borrowed_chunk: Some(chunk),
            buffer,
            materials,
        }
    }

    pub fn serialize_octant<A: Allocator>(octree: &Octree<BlockId, A>, octant_id: OctantId, depth: u8) -> (Vec<u8>, Vec<BlockId>) {
        let octant = &octree.octants[octant_id as usize];

        if depth == 1 {
            let mut leaf_mask = 0u8;
            let mut materials = Vec::new();

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

                materials.push(*content.unwrap());
                leaf_mask |= 1 << idx;
            }

            return (vec![leaf_mask], materials);
        }

        let mut materials = Vec::new();
        let mut children = Vec::new();
        for (idx, child) in octant.children.iter().enumerate() {
            if child.is_none() {
                continue;
            }
            assert!(!child.is_leaf(), "octree leaves must be at a uniform level");

            // decrease lod and calculate buffer offset before recursively serializing the child octant
            let child_id = child.get_octant_value().unwrap();

            let (buffer, child_materials) = Self::serialize_octant(octree, child_id, depth - 1);
            children.push((idx, buffer));
            materials.extend(child_materials);
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
            for (i, (idx, _)) in children.iter().enumerate() {
                let offset_bits = offsets[i].max(1).ilog2();
                let header_tag = (offset_bits / 8 + 1) as u16;
                header_mask |= header_tag << (idx * 2);

                match header_tag {
                    1 => buffer.push(offsets[i] as u8),
                    2 => buffer.extend((offsets[i] as u16).to_be_bytes()),
                    3 => {
                        assert_eq!(offsets[i] & (1 << 31), 0, "32 bit pointers must not have the 32nd bit set");
                        buffer.extend(offsets[i].to_be_bytes());
                    }
                    _ => unreachable!(),
                }
            }
            for (_, data) in children {
                buffer.extend(data);
            }

            let header_bytes = header_mask.to_be_bytes();
            buffer[0] = header_bytes[0];
            buffer[1] = header_bytes[1];
        }

        (buffer, materials)
    }
}

#[cfg(test)]
mod serialized_chunk_tests {
    use crate::world::chunk::BlockId;
    use crate::world::hds::csvo::SerializedChunk;
    use crate::world::hds::octree::{Octree, Position};

    #[test]
    fn serialize_octant_single_leaf() {
        let mut octree = Octree::new();
        octree.set_leaf(Position(0, 0, 0), 1 as BlockId);
        octree.expand_to(4);
        octree.compact();

        let (result, materials) = SerializedChunk::serialize_octant(&octree, octree.root.unwrap(), octree.depth());
        assert_eq!(result, vec![
            0, 1, 0,    // inode
            1, 0,       // plnode
            1, 1,       // lnode
        ]);
        assert_eq!(materials, vec![1]);
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

        let (result, materials) = SerializedChunk::serialize_octant(&octree, octree.root.unwrap(), octree.depth());
        assert_eq!(result, vec![
            0, 1, 0,                    // inode
            1 | (1 << 7), 0, 3,         // plnode
            1 | (1 << 7), 1, 1 << 7,    // lnode
            1 | (1 << 7), 2, 1 << 6,    // lnode
        ]);
        assert_eq!(materials, vec![1, 2, 1, 2]);
    }

    #[test]
    fn serialize_octant_chunk() {
        let mut octree = Octree::new();
        octree.set_leaf(Position(31, 0, 0), 1 as BlockId);
        octree.set_leaf(Position(0, 31, 0), 2 as BlockId);
        octree.set_leaf(Position(0, 0, 31), 3 as BlockId);
        octree.compact();

        let (result, materials) = SerializedChunk::serialize_octant(&octree, octree.root.unwrap(), octree.depth());
        assert_eq!(result, vec![
            0b00_00_00_01, 0b_00_01_01_00, 0, 7, 14,
            0, 0b00_00_01_00, 0,
            2, 0,
            2,
            2,
            0, 0b00_01_00_00, 0,
            4, 0,
            4,
            4,
            0b00_00_00_01, 0, 0,
            16, 0,
            16,
            16,
        ]);
        assert_eq!(materials, vec![1, 2, 3]);
    }

    #[test]
    fn serialize_octant_chunk_with_lod() {
        let mut octree = Octree::new();
        octree.set_leaf(Position(31, 0, 0), 1 as BlockId);
        octree.set_leaf(Position(0, 31, 0), 2 as BlockId);
        octree.set_leaf(Position(0, 0, 31), 3 as BlockId);
        octree.compact();

        let (result, materials) = SerializedChunk::serialize_octant(&octree, octree.root.unwrap(), octree.depth() - 1);
        assert_eq!(result, vec![
            0b00_00_00_01, 0b_00_01_01_00, 0, 4, 8,
            2, 0,
            2,
            2,
            4, 0,
            4,
            4,
            16, 0,
            16,
            16,
        ]);
        assert_eq!(materials, vec![1, 2, 3]);

        let (result, materials) = SerializedChunk::serialize_octant(&octree, octree.root.unwrap(), octree.depth() - 2);
        assert_eq!(result, vec![
            0b00010110, 0, 2, 4,
            2,
            2,
            4,
            4,
            16,
            16,
        ]);
        assert_eq!(materials, vec![1, 2, 3]);

        let (result, materials) = SerializedChunk::serialize_octant(&octree, octree.root.unwrap(), octree.depth() - 3);
        assert_eq!(result, vec![
            0b00010110, 2, 4, 16,
        ]);
        assert_eq!(materials, vec![1, 2, 3]);

        let (result, materials) = SerializedChunk::serialize_octant(&octree, octree.root.unwrap(), octree.depth() - 4);
        assert_eq!(result, vec![
            22,
        ]);
        assert_eq!(materials, vec![1, 2, 3]);
    }
}
