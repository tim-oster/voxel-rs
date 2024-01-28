#![feature(allocator_api)]
#![allow(dead_code, unused_variables)]

extern crate gl;
#[macro_use]
extern crate memoffset;

use crate::gamelogic::game::Game;

mod core;
mod gamelogic;
mod graphics;
mod systems;
mod world;

/// If profiling feature is set, register DHAT as the global allocator to collect metrics.
#[cfg(feature = "dhat-heap")]
#[global_allocator]
static DHAT_ALLOC: dhat::Alloc = dhat::Alloc;

#[cfg(feature = "dhat-heap")]
pub fn global_allocated_bytes() -> usize {
    0
}

/// In normal operation, register a wrapper around the System allocator to collect how much memory was allocated
/// during runtime.
#[cfg(not(feature = "dhat-heap"))]
#[global_allocator]
static STATS_ALLOC: world::memory::GlobalStatsAllocator = world::memory::GlobalStatsAllocator {
    allocated_bytes: std::sync::atomic::AtomicUsize::new(0),
};

#[cfg(not(feature = "dhat-heap"))]
pub fn global_allocated_bytes() -> usize {
    STATS_ALLOC.allocated_bytes.load(std::sync::atomic::Ordering::Acquire)
}

fn main() {
    #[cfg(feature = "dhat-heap")]
        let _profiler = dhat::Profiler::builder().trim_backtraces(Some(20)).build();

    let game = Game::new();
    game.run();
}
