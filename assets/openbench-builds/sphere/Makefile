EXE ?= nagato

.PHONY: all
all:
	cargo build --release --features sphere-search --manifest-path ../../../Cargo.toml
	cp ../../../target/release/nagato $(EXE)
