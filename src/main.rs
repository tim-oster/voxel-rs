#![feature(allocator_api, test)]
#![feature(pointer_is_aligned_to)]
#![feature(unchecked_shifts)]
#![feature(duration_millis_float)]
#![feature(once_cell_get_mut)]
#![allow(dead_code, unused_variables)]

#![warn(clippy::all, clippy::nursery, clippy::pedantic)]

// from nursery
#![allow(clippy::missing_const_for_fn, clippy::significant_drop_tightening)]

// from pedantic
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::cast_precision_loss
)]
#![allow(
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::struct_field_names,
    clippy::module_name_repetitions
)]
#![allow(clippy::too_many_lines)]

extern crate gl;
#[macro_use]
extern crate memoffset;
extern crate test;

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use cgmath::{Point3, Vector3};
use clap::ArgAction;
use clap::Parser;

use crate::gamelogic::benchmark;
use crate::gamelogic::game::{Game, GameArgs};

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

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Player world position
    #[arg(
        long, value_parser, num_args = 3, value_delimiter = ' ', value_names = ["x", "y", "z"],
        default_value = "-24 80 174", allow_negative_numbers = true
    )]
    pos: Vec<f32>,
    /// Player euler rotation in degrees
    #[arg(
        long, value_parser, num_args = 3, value_delimiter = ' ', value_names = ["x", "y", "z"],
        default_value = "0 -90 0", allow_negative_numbers = true
    )]
    rot: Vec<f32>,

    /// Detach input starts the engine with detached controls by default.
    #[arg(long, action = ArgAction::SetTrue, default_value = "false")]
    detach_input: bool,
    /// Render distance as chunk loading radius.
    #[arg(long, default_value = "20")]
    render_distance: u32,

    /// Vertical field of view in degrees
    #[arg(long, default_value = "72")]
    fov: f32,

    /// Defines if the shadow render pass is enabled.
    #[arg(long, action = ArgAction::Set, default_value = "true")]
    render_shadows: bool,

    /// Optional directory path to a minecraft world to load. Must be in anvil file format.
    #[arg(long)]
    mc_world: Option<String>,
}

fn main() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::builder().trim_backtraces(Some(20)).build();

    let args = Args::parse();
    let game = Game::new(GameArgs {
        mc_world: args.mc_world,
        player_pos: Point3::new(args.pos[0], args.pos[1], args.pos[2]),
        player_euler_rot: Vector3::new(args.rot[0].to_radians(), args.rot[1].to_radians(), args.rot[2].to_radians()),
        detach_input: args.detach_input,
        render_distance: args.render_distance,
        fov_y_deg: args.fov,
        render_shadows: args.render_shadows,
    });

    let closer = Arc::new(AtomicBool::new(false));

    #[cfg(windows)]
    signal_hook::flag::register(signal_hook::consts::SIGBREAK, Arc::clone(&closer)).unwrap();

    game.run(&closer);

    benchmark::print();
}
