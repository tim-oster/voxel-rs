#![feature(allocator_api, test)]
#![feature(pointer_is_aligned_to)]
#![feature(unchecked_shifts)]
#![allow(dead_code, unused_variables)]

#![warn(clippy::all, clippy::nursery, clippy::pedantic)]

// from nursery
#![allow(clippy::missing_const_for_fn, clippy::significant_drop_tightening)]

// from pedantic
#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap, clippy::cast_sign_loss, clippy::cast_lossless, clippy::cast_precision_loss)]
#![allow(clippy::similar_names, clippy::many_single_char_names, clippy::struct_field_names, clippy::module_name_repetitions)]
#![allow(clippy::too_many_lines)]

extern crate gl;
#[macro_use]
extern crate memoffset;
extern crate test;

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
