use std::collections::{HashMap, HashSet, VecDeque};
use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::world::chunk;
use crate::world::chunk::{Chunk, ChunkPos};

/// BorrowedChunk wraps around an actual chunks which temporarily got its ownership transferred
/// to the borrowed chunk. It is intended to be returned to the [`World`] by calling
/// [`World::return_chunk`] but dropping it is also valid, in case the chunk should be removed.
pub struct BorrowedChunk {
    chunk: Option<Chunk>,
    was_dropped: Arc<AtomicBool>,
}

impl BorrowedChunk {
    pub fn from(chunk: Chunk) -> BorrowedChunk {
        BorrowedChunk {
            chunk: Some(chunk),
            was_dropped: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl Drop for BorrowedChunk {
    fn drop(&mut self) {
        self.was_dropped.store(true, Ordering::Relaxed);
    }
}

impl Deref for BorrowedChunk {
    type Target = Chunk;

    fn deref(&self) -> &Self::Target {
        self.chunk.as_ref().unwrap()
    }
}

impl DerefMut for BorrowedChunk {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.chunk.as_mut().unwrap()
    }
}

pub struct BorrowedChunkRef {
    was_dropped: Arc<AtomicBool>,
}

// -------------------------------------------------------------------------------------------------

/// World is a collection of chunks, where each chunk can be identified by its position.
pub struct World {
    chunks: HashMap<ChunkPos, Chunk>,
    changed_chunks_set: HashSet<ChunkPos>,
    changed_chunks: VecDeque<ChunkPos>,
    borrowed_chunks: HashMap<ChunkPos, BorrowedChunkRef>,
}

impl World {
    pub fn new() -> World {
        World {
            chunks: HashMap::new(),
            changed_chunks_set: HashSet::new(),
            changed_chunks: VecDeque::new(),
            borrowed_chunks: HashMap::new(),
        }
    }

    fn mark_chunk_as_changed(&mut self, pos: &ChunkPos) {
        if !self.changed_chunks_set.contains(&pos) {
            self.changed_chunks_set.insert(*pos);
            self.changed_chunks.push_back(*pos);
        }
    }

    /// Sets a chunk at the chunk's position and marks that position as changed. Any previous
    /// chunk at that position is overridden. If the chunk at that position was borrowed, it is
    /// also overridden.
    pub fn set_chunk(&mut self, chunk: Chunk) {
        let pos = chunk.pos;

        // remove it from borrowed chunks to prevent the old, borrowed chunk from being returned
        self.borrowed_chunks.remove(&pos);

        self.chunks.insert(pos, chunk);
        self.mark_chunk_as_changed(&pos);
    }

    /// Removes the chunk at the given position and marks that position as changed. If the chunk at
    /// that position was borrowed, it is also removed.
    pub fn remove_chunk(&mut self, pos: &ChunkPos) {
        // remove it from borrowed chunks to prevent a previously borrowed chunk from being returned
        self.borrowed_chunks.remove(&pos);

        self.chunks.remove(pos);
        self.mark_chunk_as_changed(&pos);
    }

    pub fn get_chunk(&self, pos: &ChunkPos) -> Option<&Chunk> {
        self.chunks.get(pos)
    }

    /// Returns a mutable reference to the chunk at position, if it exists. This marks the position
    /// as changed, even if the chunk is not modified by the caller.
    pub fn get_chunk_mut(&mut self, pos: &ChunkPos) -> Option<&mut Chunk> {
        self.mark_chunk_as_changed(pos);
        self.chunks.get_mut(pos)
    }

    /// Transfers the chunk's ownership from the world to the caller. Use [`World::return_chunk`]
    /// to return the ownership. Returns `None` if the chunk does not exist or was already borrowed.
    pub fn borrow_chunk(&mut self, pos: &ChunkPos) -> Option<BorrowedChunk> {
        let chunk = self.chunks.remove(pos);
        if chunk.is_none() {
            return None;
        }

        let borrowed = BorrowedChunk::from(chunk.unwrap());
        self.borrowed_chunks.insert(*pos, BorrowedChunkRef {
            was_dropped: borrowed.was_dropped.clone(),
        });
        Some(borrowed)
    }

    /// Returns a previously borrowed chunk. This does not mark the position as changed. If the
    /// position was modified in during the borrow duration, nothing happens upon return.
    pub fn return_chunk(&mut self, chunk: BorrowedChunk) {
        let pos = chunk.pos;

        // if this chunk is no longer part of the borrowed chunks set, skip it as it might have been
        // overridden by now
        if !self.borrowed_chunks.contains_key(&pos) {
            return;
        }

        let mut chunk = chunk;
        self.borrowed_chunks.remove(&pos);
        self.chunks.insert(pos, chunk.chunk.take().unwrap());

        // NOTE: this must not mark the chunk as changed, otherwise a feedback loop is created
    }

    /// Returns the block id at the given position in world space. Returns [`chunk::NO_BLOCK`] if
    /// the chunk was borrowed or does not exist.
    pub fn get_block(&self, x: i32, y: i32, z: i32) -> chunk::BlockId {
        let pos = ChunkPos::from_block_pos(x, y, z);
        if let Some(chunk) = self.chunks.get(&pos) {
            return chunk.get_block((x & 31) as u32, (y & 31) as u32, (z & 31) as u32);
        }
        chunk::NO_BLOCK
    }

    /// Sets the block id at the given position in world space. Does nothing if the chunk was
    /// borrowed or does not exist. Returns true, if the block was set.
    pub fn set_block(&mut self, x: i32, y: i32, z: i32, block: chunk::BlockId) -> bool {
        let pos = ChunkPos::from_block_pos(x, y, z);
        if let Some(chunk) = self.chunks.get_mut(&pos) {
            chunk.set_block((x & 31) as u32, (y & 31) as u32, (z & 31) as u32, block);
            self.mark_chunk_as_changed(&pos);
            return true;
        }
        false
    }

    /// Returns up to limit chunk positions of chunks that have been changed.
    pub fn get_changed_chunks(&mut self, limit: u32) -> Vec<ChunkPos> {
        // clean up dropped borrowed chunk references
        self.borrowed_chunks.retain(|_, r| !r.was_dropped.load(Ordering::Relaxed));

        // compile changed chunks
        let mut changed = Vec::new();
        for _ in 0..limit {
            let pos = self.changed_chunks.pop_front();
            if pos.is_none() {
                break;
            }

            let pos = pos.unwrap();
            self.changed_chunks_set.remove(&pos);
            changed.push(pos);
        }
        changed
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashSet, VecDeque};
    use std::sync::Arc;
    use std::sync::atomic::Ordering;

    use crate::world::chunk;
    use crate::world::chunk::{Chunk, ChunkPos};
    use crate::world::memory::ChunkStorageAllocator;

    /// Tests that setting and getting blocks on world works.
    #[test]
    fn world_get_and_set_block() {
        let alloc = Arc::new(ChunkStorageAllocator::new());
        let mut world = super::World::new();
        world.set_chunk(Chunk::new(ChunkPos::new(0, 1, 2), 5, alloc.allocate()));

        let block = world.get_block(1, 33, 65);
        assert_eq!(block, super::chunk::NO_BLOCK);

        world.set_block(1, 33, 65, 99);

        let chunk = world.chunks.get(&ChunkPos { x: 0, y: 1, z: 2 }).unwrap();
        assert_eq!(chunk.get_block(1, 1, 1), 99);

        let block = world.get_block(1, 33, 65);
        assert_eq!(block, 99);
    }

    /// Tests that different chunk actions are properly registered as chunk changes.
    #[test]
    fn world_changed_chunks() {
        let alloc = Arc::new(ChunkStorageAllocator::new());
        let mut world = super::World::new();
        world.set_chunk(Chunk::new(ChunkPos::new(0, 0, 0), 5, alloc.allocate()));
        world.set_chunk(Chunk::new(ChunkPos::new(1, 0, 0), 5, alloc.allocate()));
        world.set_chunk(Chunk::new(ChunkPos::new(2, 0, 0), 5, alloc.allocate()));

        // check for initial chunk add changes
        assert_eq!(world.changed_chunks_set, HashSet::from([
            ChunkPos::new(0, 0, 0),
            ChunkPos::new(1, 0, 0),
            ChunkPos::new(2, 0, 0),
        ]));
        assert_eq!(world.changed_chunks, VecDeque::from(vec![
            ChunkPos::new(0, 0, 0),
            ChunkPos::new(1, 0, 0),
            ChunkPos::new(2, 0, 0),
        ]));
        assert_eq!(world.get_changed_chunks(10), vec![
            ChunkPos::new(0, 0, 0),
            ChunkPos::new(1, 0, 0),
            ChunkPos::new(2, 0, 0),
        ]);

        // assert that changed chunks are empty after get_changed_chunks
        assert_eq!(true, world.changed_chunks_set.is_empty());
        assert_eq!(true, world.changed_chunks.is_empty());

        // modify world
        let mut_chunk = world.get_chunk_mut(&ChunkPos::new(2, 0, 0));
        mut_chunk.unwrap().set_block(0, 0, 0, 1);

        world.remove_chunk(&ChunkPos::new(1, 0, 0));

        for i in 0..2 {
            world.set_block(i, 0, 0, 1);
        }

        // assert that changes are read in order
        assert_eq!(world.get_changed_chunks(10), vec![
            ChunkPos::new(2, 0, 0),
            ChunkPos::new(1, 0, 0),
            ChunkPos::new(0, 0, 0),
        ]);
    }

    /// Tests that chunk borrowing system works.
    #[test]
    fn borrow_chunk() {
        let alloc = Arc::new(ChunkStorageAllocator::new());
        let mut world = super::World::new();
        world.set_chunk(Chunk::new(ChunkPos::new(0, 0, 0), 5, alloc.allocate()));

        world.set_block(0, 0, 0, 1);
        world.get_changed_chunks(10); // drain changes

        // case 1 - borrow & return
        {
            let borrow = world.borrow_chunk(&ChunkPos::new(0, 0, 0));
            assert!(borrow.is_some());
            assert!(!world.borrowed_chunks.is_empty());

            assert!(world.borrow_chunk(&ChunkPos::new(0, 0, 0)).is_none());
            assert!(world.get_chunk(&ChunkPos::new(0, 0, 0)).is_none());
            assert_eq!(world.get_block(0, 0, 0), chunk::NO_BLOCK);
            assert_eq!(world.set_block(0, 0, 0, 1), false);

            world.return_chunk(borrow.unwrap());
            assert!(world.borrowed_chunks.is_empty());

            assert!(world.get_chunk(&ChunkPos::new(0, 0, 0)).is_some());
            assert_eq!(world.get_block(0, 0, 0), 1);
            assert_eq!(world.set_block(0, 0, 0, 1), true);

            world.get_changed_chunks(10); // drain changes
        }

        // case 2 - borrow & drop
        {
            let borrow = world.borrow_chunk(&ChunkPos::new(0, 0, 0));
            assert!(borrow.is_some());
            assert!(!world.borrowed_chunks.is_empty());

            drop(borrow);
            let chunk_ref = world.borrowed_chunks.get(&ChunkPos::new(0, 0, 0)).unwrap();
            assert_eq!(chunk_ref.was_dropped.load(Ordering::Relaxed), true);

            assert!(world.get_changed_chunks(10).is_empty());
            assert!(world.borrowed_chunks.is_empty());
        }

        // case 3 - borrow & remove
        {
            world.set_chunk(Chunk::new(ChunkPos::new(0, 0, 0), 5, alloc.allocate()));

            let borrow = world.borrow_chunk(&ChunkPos::new(0, 0, 0));
            assert!(borrow.is_some());
            assert!(!world.borrowed_chunks.is_empty());

            world.remove_chunk(&ChunkPos::new(0, 0, 0));
            assert!(world.get_chunk(&ChunkPos::new(0, 0, 0)).is_none());

            world.return_chunk(borrow.unwrap());
            assert!(world.get_chunk(&ChunkPos::new(0, 0, 0)).is_none());
        }

        // case 4 - borrow & override
        {
            world.set_chunk(Chunk::new(ChunkPos::new(0, 0, 0), 5, alloc.allocate()));
            world.set_block(0, 0, 0, 1);

            let borrow = world.borrow_chunk(&ChunkPos::new(0, 0, 0));
            assert!(borrow.is_some());
            assert!(!world.borrowed_chunks.is_empty());

            world.set_chunk(Chunk::new(ChunkPos::new(0, 0, 0), 5, alloc.allocate()));
            world.set_block(0, 0, 0, 2);

            world.return_chunk(borrow.unwrap());
            assert_eq!(world.get_block(0, 0, 0), 2);
        }
    }
}
