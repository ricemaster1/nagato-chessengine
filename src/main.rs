mod bitboard;
mod board;
mod moves;
mod movegen;
mod zobrist;
mod uci;
mod search;
mod eval;
mod syzygy;
mod learn;
mod datagen;
mod nnue;

fn main() {
    zobrist::init();
    movegen::init();
    nnue::init();
    if std::env::args().any(|a| a == "bench") {
        uci::bench();
        return;
    }
    uci::uci_loop();
}
