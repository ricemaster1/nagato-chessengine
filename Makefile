EXE ?= nagato

.PHONY: all
all:
	cargo build --release
	cp target/release/nagato $(EXE)
