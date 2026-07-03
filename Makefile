EXE ?= nagato

openbench:
	cargo rustc --release -p nagato --bin nagato -- -C target-cpu=native
	cp target/release/nagato $(EXE)

all: openbench

.PHONY: all openbench
