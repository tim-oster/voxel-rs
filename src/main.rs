extern crate gl;
#[macro_use]
extern crate memoffset;

use crate::game::game::Game;

mod core;
mod game;
mod graphics;
mod systems;
mod world;

fn main() {
    let game = Game::new();
    game.run();
}
