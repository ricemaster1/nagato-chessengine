mod bitboard;
mod board;
mod moves;
mod movegen;
mod zobrist;
mod uci;
mod search;
mod eval;
mod endgame;
mod syzygy;
mod learn;
mod datagen;
mod nnue;

fn main() {
    zobrist::init();
    movegen::init();
    nnue::init();

    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(|s| s.as_str()) == Some("bench") {
        uci::bench();
        return;
    }

    uci::uci_loop();
}
