heap-profile:
	cargo run --features=dhat-heap --profile=release-dhat
	start https://nnethercote.github.io/dh_view/dh_view.html

release:
	cargo build --release --features=bundle-assets

bench:
	cargo bench --features=dhat-heap -- --nocapture
