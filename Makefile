heap-profile:
	cargo run --features=dhat-heap --profile=release-dhat
	start https://nnethercote.github.io/dh_view/dh_view.html

release:
	cargo build --release --features=bundle-assets --features=use-csvo --no-default-features

bench:
	cargo bench --features=dhat-heap -- --nocapture

bench_mc_esvo:
	cargo run --release --features=use-esvo --no-default-features -- --pos -644 97 120 --rot -1 165 0 --detach-input --render-distance=30 --fov=80 --mc-world="assets/worlds/benchmark"

bench_mc_csvo:
	cargo run --release --features=use-csvo --no-default-features -- --pos -644 97 120 --rot -1 165 0 --detach-input --render-distance=30 --fov=80 --mc-world="assets/worlds/benchmark"
