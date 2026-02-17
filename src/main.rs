mod bitboard;
mod board;
mod moves;
mod movegen;
mod zobrist;
mod uci;
mod search;
mod eval;
mod learn;

fn main() {
    zobrist::init();
    movegen::init();
    uci::uci_loop();
}
