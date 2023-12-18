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

fn main() {
    let game = Game::new();
    game.run();
}
