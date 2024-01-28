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

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() {
    #[cfg(feature = "dhat-heap")]
        let _profiler = dhat::Profiler::builder().trim_backtraces(Some(20)).build();

    let game = Game::new();
    game.run();
}
