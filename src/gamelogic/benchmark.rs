use std::cmp::Ordering;
use std::sync::OnceLock;
use std::time::Instant;

use dashmap::DashMap;
#[cfg(feature = "benchmark")]
use serde_json::{json, Map};

struct Samples<const N: usize> {
    samples: [f32; N],
    ptr: usize,
    wrapped: bool,
}

impl<const N: usize> Samples<N> {
    const fn new() -> Self {
        Self {
            samples: [0.0; N],
            ptr: 0,
            wrapped: false,
        }
    }

    fn add(&mut self, v: f32) {
        self.samples[self.ptr] = v;
        self.ptr += 1;
        if self.ptr >= N {
            self.ptr %= N;
            self.wrapped = true;
        }
    }

    fn reset(&mut self) {
        self.samples = [0.0; N];
        self.ptr = 0;
        self.wrapped = false;
    }

    fn avg(&self) -> f32 {
        let end = if self.wrapped { N } else { self.ptr };
        self.samples[..end].iter().sum::<f32>() / end as f32
    }

    fn median(&self) -> f32 {
        let end = if self.wrapped { N } else { self.ptr };

        let mut samples = self.samples.clone();
        let samples = &mut samples[..end];
        samples.sort_unstable_by(f32::total_cmp);

        if samples.is_empty() {
            return f32::NAN;
        }
        samples[samples.len() / 2]
    }
}

// -------------------------------------------------------------------------------------------------

struct Max<T: PartialOrd> {
    value: Option<T>,
}

impl<T: PartialOrd + Copy> Max<T> {
    const fn new() -> Self {
        Self { value: None }
    }

    fn set(&mut self, v: T) {
        if self.value.is_none() || self.value.partial_cmp(&Some(v)) == Some(Ordering::Less) {
            self.value = Some(v);
        }
    }
}

// -------------------------------------------------------------------------------------------------

fn get_benchmark() -> &'static mut Benchmark {
    static mut SINGLETON: OnceLock<Benchmark> = OnceLock::new();
    unsafe {
        SINGLETON.get_mut_or_init(|| Benchmark {
            fps_samples: Samples::new(),
            frame_time_samples: Samples::new(),
            svo_gpu_bytes: Max::new(),
            traces: DashMap::new(),
        })
    }
}

struct Benchmark {
    fps_samples: Samples<1000>,
    frame_time_samples: Samples<1000>,
    svo_gpu_bytes: Max<usize>,
    traces: DashMap<String, Samples<5000>>,
}

#[cfg(feature = "benchmark")]
pub fn track_fps(fps: i32, frame_time: f32) {
    if fps == 0 {
        return;
    }
    let bm = get_benchmark();
    bm.fps_samples.add(fps as f32);
    bm.frame_time_samples.add(frame_time);
}

#[cfg(not(feature = "benchmark"))]
pub fn track_fps(fps: i32, frame_time: f32) {}

#[cfg(feature = "benchmark")]
pub fn reset_fps() {
    let bm = get_benchmark();
    bm.fps_samples.reset();
    bm.frame_time_samples.reset();
}

#[cfg(not(feature = "benchmark"))]
pub fn reset_fps() {}

#[cfg(feature = "benchmark")]
pub fn track_svo_gpu_bytes(bytes: usize) {
    let bm = get_benchmark();
    bm.svo_gpu_bytes.set(bytes);
}

#[cfg(not(feature = "benchmark"))]
pub fn track_svo_gpu_bytes(bytes: usize) {}

pub struct Trace {
    name: String,
    start: Instant,
}

pub fn start_trace(name: &str) -> Trace {
    Trace {
        name: name.to_string(),
        start: Instant::now(),
    }
}

#[cfg(feature = "benchmark")]
pub fn stop_trace(trace: Trace) {
    let elapsed = trace.start.elapsed();

    let bm = get_benchmark();
    if !bm.traces.contains_key(&trace.name) {
        bm.traces.insert(trace.name.clone(), Samples::new());
    }
    bm.traces.get_mut(&trace.name).unwrap().add(elapsed.as_millis_f32());
}

#[cfg(not(feature = "benchmark"))]
pub fn stop_trace(trace: Trace) {}

#[cfg(feature = "benchmark")]
pub fn trace<T, Fn: FnOnce() -> T>(name: &str, f: Fn) -> T {
    trace_if(name, f, |_| true)
}

#[cfg(not(feature = "benchmark"))]
pub fn trace<T, Fn: FnOnce() -> T>(name: &str, f: Fn) -> T {
    f()
}

#[cfg(feature = "benchmark")]
pub fn trace_if<T, Fn: FnOnce() -> T, Cond: FnOnce(&T) -> bool>(name: &str, f: Fn, cond: Cond) -> T {
    let trace = start_trace(name);
    let result = f();

    if cond(&result) {
        stop_trace(trace);
    }

    result
}

#[cfg(not(feature = "benchmark"))]
pub fn trace_if<T, Fn: FnOnce() -> T, Cond: FnOnce(&T) -> bool>(name: &str, f: Fn, cond: Cond) -> T {
    f()
}

#[cfg(feature = "benchmark")]
pub fn print() {
    let bm = get_benchmark();

    let mut traces = Map::new();
    for t in bm.traces.iter() {
        traces.insert(t.key().clone(), json!({
            "avg": t.avg(),
            "med": t.median(),
        }));
    }

    let results = json!({
        "fps": {
            "avg": bm.fps_samples.avg(),
            "med": bm.fps_samples.median(),
        },
        "frame_time_ms": {
            "avg": bm.frame_time_samples.avg() * 1000.0,
            "med": bm.frame_time_samples.median() * 1000.0,
        },
        "svo_size_mb": bm.svo_gpu_bytes.value.unwrap_or(0) as f32 / 1024f32 / 1024f32,
        "traces": traces,
    });

    println!("benchmark: {}", serde_json::to_string(&results).unwrap());
}

#[cfg(not(feature = "benchmark"))]
pub fn print() {}
