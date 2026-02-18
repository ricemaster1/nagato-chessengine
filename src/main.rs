mod bitboard;
mod board;
mod moves;
mod movegen;
mod zobrist;
mod uci;
mod search;
mod eval;
mod learn;
mod nnue;
mod datagen;

fn main() {
    zobrist::init();
    movegen::init();
    nnue::init();
    uci::uci_loop();
}
