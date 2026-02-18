mod bitboard;
mod board;
mod moves;
mod movegen;
mod zobrist;
mod uci;
mod search;
mod eval;
mod learn;
mod datagen;
mod nnue;

fn main() {
    zobrist::init();
    movegen::init();
    nnue::init();
    uci::uci_loop();
}
