use criterion::{black_box, criterion_group, criterion_main, Criterion};

use fasthash::{sea::Hash64, FastHash};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

static BENCH_SIZE: usize = 10_000_000;

const K: u64 = 0x517cc1b727220a95;
fn fx_hasher(start: u64, new: u64) -> u64 {
    let tmp = start.rotate_left(5) ^ new;
    tmp.wrapping_mul(K)
}

fn default_hasher(start: u64, new: u64) -> u64 {
    let mut hasher = DefaultHasher::new();
    start.hash(&mut hasher);
    new.hash(&mut hasher);
    hasher.finish()
}

fn sea_hasher(start: u64, new: u64) -> u64 {
    Hash64::hash_with_seed(new.to_ne_bytes(), (start, 0, 0, 0))
}

fn sea_hasher_v2(start: u64, new: u64) -> u64 {
    start ^ seahash::hash(&new.to_ne_bytes())
}

fn narray_hash_bench(size: usize, hash_func: fn(u64, u64) -> u64) -> u64 {
    let mut start = 0;

    for i in 1..=size {
        start = hash_func(start, i as u64);
    }

    start
}

fn narray_hash_bench_sea(size: usize) -> u64 {
    let mut hasher = seahash::SeaHasher::new();
    for i in 1..=size {
        hasher.write_u64(i as u64);
    }

    hasher.finish()
}

fn fx_hasher_bench(c: &mut Criterion) {
    c.bench_function("fx hasher", |b| {
        b.iter(|| {
            black_box(&narray_hash_bench(black_box(BENCH_SIZE), fx_hasher));
        })
    });
}

fn default_hasher_bench(c: &mut Criterion) {
    c.bench_function("default hasher", |b| {
        b.iter(|| {
            black_box(&narray_hash_bench(black_box(BENCH_SIZE), default_hasher));
        })
    });
}

fn sea_hasher_bench(c: &mut Criterion) {
    c.bench_function("sea hasher", |b| {
        b.iter(|| {
            black_box(&narray_hash_bench(black_box(BENCH_SIZE), sea_hasher));
        })
    });
}

fn sea_hasher_bench_v2(c: &mut Criterion) {
    c.bench_function("sea hasher v2", |b| {
        b.iter(|| {
            black_box(&narray_hash_bench(black_box(BENCH_SIZE), sea_hasher_v2));
        })
    });
}

fn sea_hasher_bench_v2_stream(c: &mut Criterion) {
    c.bench_function("sea hasher v2 stream", |b| {
        b.iter(|| {
            black_box(&narray_hash_bench_sea(black_box(BENCH_SIZE)));
        })
    });
}

criterion_group!(
    benches,
    sea_hasher_bench_v2,
    sea_hasher_bench_v2_stream,
    fx_hasher_bench,
    default_hasher_bench,
    sea_hasher_bench
);
criterion_main!(benches);
