use crate::bitboard::*;
use crate::moves::*;
use crate::nnue;
use crate::zobrist;

pub const WK_CASTLE: u8 = 0b0001;
pub const WQ_CASTLE: u8 = 0b0010;
pub const BK_CASTLE: u8 = 0b0100; 
pub const BQ_CASTLE: u8 = 0b1000; 
pub const ALL_CASTLES: u8 = 0b1111;

#[derive(Clone, Copy)]
pub struct UndoInfo {
    pub castling: u8,
    pub ep_square: Option<u8>,
    pub halfmove: u16,
    pub hash: u64,
    pub captured_piece: Option<Piece>,
}

#[derive(Clone)]
pub struct Board {
    pub pieces: [[Bitboard; PIECE_COUNT]; COLOR_COUNT],
    pub occupancy: [Bitboard; COLOR_COUNT],
    pub all_occupancy: Bitboard,

    pub side: Color,
    pub castling: u8,
    pub ep_square: Option<u8>,
    pub halfmove: u16,
    pub fullmove: u16,

    pub hash: u64,

    pub history: Vec<UndoInfo>,

    pub accumulator: nnue::Accumulator,
    pub acc_history: Vec<nnue::Accumulator>,
    pub accumulator_q: nnue::AccumulatorQ,
    pub acc_history_q: Vec<nnue::AccumulatorQ>,
}

pub const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

impl Board {
    pub fn empty() -> Self {
        Board {
            pieces: [[0; PIECE_COUNT]; COLOR_COUNT],
            occupancy: [0; COLOR_COUNT],
            all_occupancy: 0,
            side: Color::White,
            castling: 0,
            ep_square: None,
            halfmove: 0,
            fullmove: 1,
            hash: 0,
            history: Vec::with_capacity(256),
            accumulator: nnue::Accumulator::new(),
            acc_history: Vec::with_capacity(256),
            accumulator_q: nnue::AccumulatorQ::new(),
            acc_history_q: Vec::with_capacity(256),
        }
    }

    pub fn start_pos() -> Self {
        Self::from_fen(START_FEN).expect("Invalid start FEN")
    }

    pub fn from_fen(fen: &str) -> Result<Self, String> {
        let mut board = Board::empty();
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() < 4 {
            return Err("FEN must have at least 4 fields".into());
        }

        let mut rank: i8 = 7;
        let mut file: i8 = 0;
        for ch in parts[0].chars() {
            match ch {
                '/' => {
                    rank -= 1;
                    file = 0;
                }
                '1'..='8' => {
                    file += ch.to_digit(10).unwrap() as i8;
                }
                _ => {
                    let color = if ch.is_uppercase() { Color::White } else { Color::Black };
                    let piece = Piece::from_char(ch).ok_or(format!("Invalid piece char: {}", ch))?;
                    let sq = make_square(file as u8, rank as u8);
                    board.put_piece(piece, color, sq);
                    file += 1;
                }
            }
        }

        board.side = match parts[1] {
            "w" => Color::White,
            "b" => Color::Black,
            _ => return Err("Invalid side to move".into()),
        };

        board.castling = 0;
        if parts[2] != "-" {
            for ch in parts[2].chars() {
                match ch {
                    'K' => board.castling |= WK_CASTLE,
                    'Q' => board.castling |= WQ_CASTLE,
                    'k' => board.castling |= BK_CASTLE,
                    'q' => board.castling |= BQ_CASTLE,
                    _ => return Err(format!("Invalid castling char: {}", ch)),
                }
            }
        }

        board.ep_square = if parts[3] == "-" {
            None
        } else {
            Some(parse_square(parts[3]).ok_or("Invalid en passant square")?)
        };

        board.halfmove = if parts.len() > 4 {
            parts[4].parse().unwrap_or(0)
        } else {
            0
        };

        board.fullmove = if parts.len() > 5 {
            parts[5].parse().unwrap_or(1)
        } else {
            1
        };

        board.hash = board.compute_hash();

        if nnue::is_active() {
            let mut acc = nnue::Accumulator::new();
            nnue::refresh_accumulator(&board, &mut acc);
            board.accumulator = acc;
            let mut acc_q = nnue::AccumulatorQ::new();
            nnue::refresh_accumulator_q(&board, &mut acc_q);
            board.accumulator_q = acc_q;
        }

        Ok(board)
    }

    pub fn to_fen(&self) -> String {
        let mut fen = String::new();

        for rank in (0..8).rev() {
            let mut empty = 0;
            for file in 0..8 {
                let sq = make_square(file, rank);
                if let Some((piece, color)) = self.piece_at(sq) {
                    if empty > 0 {
                        fen.push(char::from_digit(empty, 10).unwrap());
                        empty = 0;
                    }
                    fen.push(piece.to_char(color));
                } else {
                    empty += 1;
                }
            }
            if empty > 0 {
                fen.push(char::from_digit(empty, 10).unwrap());
            }
            if rank > 0 {
                fen.push('/');
            }
        }

        fen.push(' ');
        fen.push(match self.side {
            Color::White => 'w',
            Color::Black => 'b',
        });

        fen.push(' ');
        if self.castling == 0 {
            fen.push('-');
        } else {
            if self.castling & WK_CASTLE != 0 { fen.push('K'); }
            if self.castling & WQ_CASTLE != 0 { fen.push('Q'); }
            if self.castling & BK_CASTLE != 0 { fen.push('k'); }
            if self.castling & BQ_CASTLE != 0 { fen.push('q'); }
        }

        fen.push(' ');
        match self.ep_square {
            Some(sq) => fen.push_str(square_name(sq)),
            None => fen.push('-'),
        }

        fen.push_str(&format!(" {} {}", self.halfmove, self.fullmove));

        fen
    }

    pub fn put_piece(&mut self, piece: Piece, color: Color, sq: u8) {
        let bb = square_bb(sq);
        self.pieces[color.index()][piece.index()] |= bb;
        self.occupancy[color.index()] |= bb;
        self.all_occupancy |= bb;
    }

    pub fn remove_piece(&mut self, piece: Piece, color: Color, sq: u8) {
        let bb = square_bb(sq);
        self.pieces[color.index()][piece.index()] &= !bb;
        self.occupancy[color.index()] &= !bb;
        self.all_occupancy &= !bb;
    }

    pub fn move_piece(&mut self, piece: Piece, color: Color, from: u8, to: u8) {
        let from_to = square_bb(from) | square_bb(to);
        self.pieces[color.index()][piece.index()] ^= from_to;
        self.occupancy[color.index()] ^= from_to;
        self.all_occupancy ^= from_to;
    }

    pub fn piece_at(&self, sq: u8) -> Option<(Piece, Color)> {
        let bb = square_bb(sq);
        for color_idx in 0..COLOR_COUNT {
            if self.occupancy[color_idx] & bb != 0 {
                for piece_idx in 0..PIECE_COUNT {
                    if self.pieces[color_idx][piece_idx] & bb != 0 {
                        let color = if color_idx == 0 { Color::White } else { Color::Black };
                        let piece = unsafe { std::mem::transmute(piece_idx as u8) };
                        return Some((piece, color));
                    }
                }
            }
        }
        None
    }

    pub fn piece_on(&self, sq: u8, color: Color) -> Option<Piece> {
        let bb = square_bb(sq);
        if self.occupancy[color.index()] & bb == 0 {
            return None;
        }
        for piece_idx in 0..PIECE_COUNT {
            if self.pieces[color.index()][piece_idx] & bb != 0 {
                return Some(unsafe { std::mem::transmute(piece_idx as u8) });
            }
        }
        None
    }

    pub fn king_sq(&self, color: Color) -> u8 {
        lsb(self.pieces[color.index()][Piece::King.index()])
    }

    pub fn compute_hash(&self) -> u64 {
        let keys = zobrist::keys();
        let mut h: u64 = 0;

        for color in 0..COLOR_COUNT {
            for piece in 0..PIECE_COUNT {
                let mut bb = self.pieces[color][piece];
                while bb != 0 {
                    let sq = pop_lsb(&mut bb);
                    h ^= keys.piece_keys[color][piece][sq as usize];
                }
            }
        }

        h ^= keys.castle_keys[self.castling as usize];

        if let Some(ep) = self.ep_square {
            h ^= keys.ep_keys[file_of(ep) as usize];
        }

        if matches!(self.side, Color::Black) {
            h ^= keys.side_key;
        }

        h
    }

    const CASTLE_MASK: [u8; 64] = {
        let mut mask = [ALL_CASTLES; 64];
        mask[sq::A1 as usize] &= !WQ_CASTLE;
        mask[sq::E1 as usize] &= !(WK_CASTLE | WQ_CASTLE);
        mask[sq::H1 as usize] &= !WK_CASTLE;
        mask[sq::A8 as usize] &= !BQ_CASTLE;
        mask[sq::E8 as usize] &= !(BK_CASTLE | BQ_CASTLE);
        mask[sq::H8 as usize] &= !BK_CASTLE;
        mask
    };

    pub fn make_move(&mut self, m: Move) -> bool {
        let keys = zobrist::keys();
        let nnue_active = nnue::is_active();

        if nnue_active {
            self.acc_history.push(self.accumulator.clone());
            self.acc_history_q.push(self.accumulator_q.clone());
        }

        let captured = if m.is_capture() && !m.is_en_passant() {
            self.piece_on(m.to_sq(), self.side.flip())
        } else {
            None
        };

        let undo = UndoInfo {
            castling: self.castling,
            ep_square: self.ep_square,
            halfmove: self.halfmove,
            hash: self.hash,
            captured_piece: captured,
        };
        self.history.push(undo);

        let from = m.from_sq();
        let to = m.to_sq();
        let piece = m.piece();
        let us = self.side;
        let them = us.flip();

        if let Some(ep) = self.ep_square {
            self.hash ^= keys.ep_keys[file_of(ep) as usize];
        }

        self.hash ^= keys.castle_keys[self.castling as usize];

        self.halfmove += 1;

        self.ep_square = None;

        match m.flags() {
            FLAG_QUIET => {
                self.move_piece(piece, us, from, to);
                self.hash ^= keys.piece_keys[us.index()][piece.index()][from as usize];
                self.hash ^= keys.piece_keys[us.index()][piece.index()][to as usize];

                if piece == Piece::Pawn {
                    self.halfmove = 0;
                }
            }
            FLAG_DOUBLE_PAWN => {
                self.move_piece(Piece::Pawn, us, from, to);
                self.hash ^= keys.piece_keys[us.index()][Piece::Pawn.index()][from as usize];
                self.hash ^= keys.piece_keys[us.index()][Piece::Pawn.index()][to as usize];

                let ep = match us {
                    Color::White => from + 8,
                    Color::Black => from - 8,
                };
                self.ep_square = Some(ep);
                self.hash ^= keys.ep_keys[file_of(ep) as usize];
                self.halfmove = 0;
            }
            FLAG_KING_CASTLE => {
                self.move_piece(Piece::King, us, from, to);
                self.hash ^= keys.piece_keys[us.index()][Piece::King.index()][from as usize];
                self.hash ^= keys.piece_keys[us.index()][Piece::King.index()][to as usize];
                let (rook_from, rook_to) = match us {
                    Color::White => (sq::H1, sq::F1),
                    Color::Black => (sq::H8, sq::F8),
                };
                self.move_piece(Piece::Rook, us, rook_from, rook_to);
                self.hash ^= keys.piece_keys[us.index()][Piece::Rook.index()][rook_from as usize];
                self.hash ^= keys.piece_keys[us.index()][Piece::Rook.index()][rook_to as usize];
            }
            FLAG_QUEEN_CASTLE => {
                self.move_piece(Piece::King, us, from, to);
                self.hash ^= keys.piece_keys[us.index()][Piece::King.index()][from as usize];
                self.hash ^= keys.piece_keys[us.index()][Piece::King.index()][to as usize];
                let (rook_from, rook_to) = match us {
                    Color::White => (sq::A1, sq::D1),
                    Color::Black => (sq::A8, sq::D8),
                };
                self.move_piece(Piece::Rook, us, rook_from, rook_to);
                self.hash ^= keys.piece_keys[us.index()][Piece::Rook.index()][rook_from as usize];
                self.hash ^= keys.piece_keys[us.index()][Piece::Rook.index()][rook_to as usize];
            }
            FLAG_CAPTURE => {
                let cap = captured.unwrap();
                self.remove_piece(cap, them, to);
                self.hash ^= keys.piece_keys[them.index()][cap.index()][to as usize];
                self.move_piece(piece, us, from, to);
                self.hash ^= keys.piece_keys[us.index()][piece.index()][from as usize];
                self.hash ^= keys.piece_keys[us.index()][piece.index()][to as usize];
                self.halfmove = 0;
            }
            FLAG_EP_CAPTURE => {
                let cap_sq = match us {
                    Color::White => to - 8,
                    Color::Black => to + 8,
                };
                self.remove_piece(Piece::Pawn, them, cap_sq);
                self.hash ^= keys.piece_keys[them.index()][Piece::Pawn.index()][cap_sq as usize];
                self.move_piece(Piece::Pawn, us, from, to);
                self.hash ^= keys.piece_keys[us.index()][Piece::Pawn.index()][from as usize];
                self.hash ^= keys.piece_keys[us.index()][Piece::Pawn.index()][to as usize];
                self.halfmove = 0;
            }
            _ if m.is_promotion() => {
                let promo = m.promotion_piece().unwrap();
                self.remove_piece(Piece::Pawn, us, from);
                self.hash ^= keys.piece_keys[us.index()][Piece::Pawn.index()][from as usize];

                if m.is_capture() {
                    let cap = captured.unwrap();
                    self.remove_piece(cap, them, to);
                    self.hash ^= keys.piece_keys[them.index()][cap.index()][to as usize];
                }

                self.put_piece(promo, us, to);
                self.hash ^= keys.piece_keys[us.index()][promo.index()][to as usize];
                self.halfmove = 0;
            }
            _ => unreachable!("Unknown move flag: {:04b}", m.flags()),
        }
        if nnue_active {
            let white_king_sq = self.king_sq(Color::White);
            let black_king_sq = self.king_sq(Color::Black);
            let acc = &mut self.accumulator;
            match m.flags() {
                FLAG_QUIET | FLAG_DOUBLE_PAWN => {
                    nnue::accumulator_move(acc, piece, us, from, to, white_king_sq, black_king_sq);
                }
                FLAG_KING_CASTLE => {
                    nnue::accumulator_move(acc, Piece::King, us, from, to, white_king_sq, black_king_sq);
                    let (rook_from, rook_to) = match us {
                        Color::White => (sq::H1, sq::F1),
                        Color::Black => (sq::H8, sq::F8),
                    };
                    nnue::accumulator_move(acc, Piece::Rook, us, rook_from, rook_to, white_king_sq, black_king_sq);
                }
                FLAG_QUEEN_CASTLE => {
                    nnue::accumulator_move(acc, Piece::King, us, from, to, white_king_sq, black_king_sq);
                    let (rook_from, rook_to) = match us {
                        Color::White => (sq::A1, sq::D1),
                        Color::Black => (sq::A8, sq::D8),
                    };
                    nnue::accumulator_move(acc, Piece::Rook, us, rook_from, rook_to, white_king_sq, black_king_sq);
                }
                FLAG_CAPTURE => {
                    let cap = captured.unwrap();
                    nnue::accumulator_remove(acc, cap, them, to, white_king_sq, black_king_sq);
                    nnue::accumulator_move(acc, piece, us, from, to, white_king_sq, black_king_sq);
                }
                FLAG_EP_CAPTURE => {
                    let cap_sq = match us {
                        Color::White => to - 8,
                        Color::Black => to + 8,
                    };
                    nnue::accumulator_remove(acc, Piece::Pawn, them, cap_sq, white_king_sq, black_king_sq);
                    nnue::accumulator_move(acc, Piece::Pawn, us, from, to, white_king_sq, black_king_sq);
                }
                _ if m.is_promotion() => {
                    let promo = m.promotion_piece().unwrap();
                    nnue::accumulator_remove(acc, Piece::Pawn, us, from, white_king_sq, black_king_sq);
                    if m.is_capture() {
                        let cap = captured.unwrap();
                        nnue::accumulator_remove(acc, cap, them, to, white_king_sq, black_king_sq);
                    }
                    nnue::accumulator_add(acc, promo, us, to, white_king_sq, black_king_sq);
                }
                _ => {}
            }
            let aq = &mut self.accumulator_q;
            match m.flags() {
                FLAG_QUIET | FLAG_DOUBLE_PAWN => {
                    nnue::accumulator_move_q(aq, piece, us, from, to, white_king_sq, black_king_sq);
                }
                FLAG_KING_CASTLE => {
                    nnue::accumulator_move_q(aq, Piece::King, us, from, to, white_king_sq, black_king_sq);
                    let (rook_from, rook_to) = match us {
                        Color::White => (sq::H1, sq::F1),
                        Color::Black => (sq::H8, sq::F8),
                    };
                    nnue::accumulator_move_q(aq, Piece::Rook, us, rook_from, rook_to, white_king_sq, black_king_sq);
                }
                FLAG_QUEEN_CASTLE => {
                    nnue::accumulator_move_q(aq, Piece::King, us, from, to, white_king_sq, black_king_sq);
                    let (rook_from, rook_to) = match us {
                        Color::White => (sq::A1, sq::D1),
                        Color::Black => (sq::A8, sq::D8),
                    };
                    nnue::accumulator_move_q(aq, Piece::Rook, us, rook_from, rook_to, white_king_sq, black_king_sq);
                }
                FLAG_CAPTURE => {
                    let cap = captured.unwrap();
                    nnue::accumulator_remove_q(aq, cap, them, to, white_king_sq, black_king_sq);
                    nnue::accumulator_move_q(aq, piece, us, from, to, white_king_sq, black_king_sq);
                }
                FLAG_EP_CAPTURE => {
                    let cap_sq = match us {
                        Color::White => to - 8,
                        Color::Black => to + 8,
                    };
                    nnue::accumulator_remove_q(aq, Piece::Pawn, them, cap_sq, white_king_sq, black_king_sq);
                    nnue::accumulator_move_q(aq, Piece::Pawn, us, from, to, white_king_sq, black_king_sq);
                }
                _ if m.is_promotion() => {
                    let promo = m.promotion_piece().unwrap();
                    nnue::accumulator_remove_q(aq, Piece::Pawn, us, from, white_king_sq, black_king_sq);
                    if m.is_capture() {
                        let cap = captured.unwrap();
                        nnue::accumulator_remove_q(aq, cap, them, to, white_king_sq, black_king_sq);
                    }
                    nnue::accumulator_add_q(aq, promo, us, to, white_king_sq, black_king_sq);
                }
                _ => {}
            }
        }
        self.castling &= Self::CASTLE_MASK[from as usize] & Self::CASTLE_MASK[to as usize];
        self.hash ^= keys.castle_keys[self.castling as usize];
        self.side = them;
        self.hash ^= keys.side_key;

        if us == Color::Black {
            self.fullmove += 1;
        }
        if self.is_square_attacked(self.king_sq(us), them) {
            self.unmake_move(m);
            return false;
        }

        true
    }
    pub fn unmake_move(&mut self, m: Move) {
        let undo = self.history.pop().expect("No undo info on stack");
        if nnue::is_active() {
            if let Some(prev_acc) = self.acc_history.pop() {
                self.accumulator = prev_acc;
            }
            if let Some(prev_acc_q) = self.acc_history_q.pop() {
                self.accumulator_q = prev_acc_q;
            }
        }
        self.side = self.side.flip();
        let us = self.side;
        let them = us.flip();

        if us == Color::Black {
            self.fullmove -= 1;
        }

        let from = m.from_sq();
        let to = m.to_sq();
        let piece = m.piece();

        match m.flags() {
            FLAG_QUIET | FLAG_DOUBLE_PAWN => {
                self.move_piece(piece, us, to, from);
            }
            FLAG_KING_CASTLE => {
                self.move_piece(Piece::King, us, to, from);
                let (rook_from, rook_to) = match us {
                    Color::White => (sq::H1, sq::F1),
                    Color::Black => (sq::H8, sq::F8),
                };
                self.move_piece(Piece::Rook, us, rook_to, rook_from);
            }
            FLAG_QUEEN_CASTLE => {
                self.move_piece(Piece::King, us, to, from);
                let (rook_from, rook_to) = match us {
                    Color::White => (sq::A1, sq::D1),
                    Color::Black => (sq::A8, sq::D8),
                };
                self.move_piece(Piece::Rook, us, rook_to, rook_from);
            }
            FLAG_CAPTURE => {
                self.move_piece(piece, us, to, from);
                let cap = undo.captured_piece.unwrap();
                self.put_piece(cap, them, to);
            }
            FLAG_EP_CAPTURE => {
                self.move_piece(Piece::Pawn, us, to, from);
                let cap_sq = match us {
                    Color::White => to - 8,
                    Color::Black => to + 8,
                };
                self.put_piece(Piece::Pawn, them, cap_sq);
            }
            _ if m.is_promotion() => {
                let promo = m.promotion_piece().unwrap();
                self.remove_piece(promo, us, to);
                self.put_piece(Piece::Pawn, us, from);
                if m.is_capture() {
                    let cap = undo.captured_piece.unwrap();
                    self.put_piece(cap, them, to);
                }
            }
            _ => unreachable!(),
        }

        self.castling = undo.castling;
        self.ep_square = undo.ep_square;
        self.halfmove = undo.halfmove;
        self.hash = undo.hash;
    }
    pub fn make_null_move(&mut self) {
        let keys = zobrist::keys();

        let undo = UndoInfo {
            castling: self.castling,
            ep_square: self.ep_square,
            halfmove: self.halfmove,
            hash: self.hash,
            captured_piece: None,
        };
        self.history.push(undo);

        if let Some(ep) = self.ep_square {
            self.hash ^= keys.ep_keys[file_of(ep) as usize];
        }
        self.ep_square = None;
        self.side = self.side.flip();
        self.hash ^= keys.side_key;
    }
    pub fn unmake_null_move(&mut self) {
        let undo = self.history.pop().expect("No undo info on stack");
        self.side = self.side.flip();
        self.ep_square = undo.ep_square;
        self.hash = undo.hash;
    }
    pub fn is_square_attacked(&self, sq: u8, by_color: Color) -> bool {
        use crate::movegen;

        let them = by_color.index();
        let pawn_attacks = match by_color {
            Color::White => movegen::black_pawn_attacks(square_bb(sq)),
            Color::Black => movegen::white_pawn_attacks(square_bb(sq)),
        };
        if pawn_attacks & self.pieces[them][Piece::Pawn.index()] != 0 {
            return true;
        }
        if movegen::knight_attacks(sq) & self.pieces[them][Piece::Knight.index()] != 0 {
            return true;
        }
        if movegen::king_attacks(sq) & self.pieces[them][Piece::King.index()] != 0 {
            return true;
        }
        let bishop_attacks = movegen::bishop_attacks(sq, self.all_occupancy);
        if bishop_attacks & (self.pieces[them][Piece::Bishop.index()] | self.pieces[them][Piece::Queen.index()]) != 0 {
            return true;
        }
        let rook_attacks = movegen::rook_attacks(sq, self.all_occupancy);
        if rook_attacks & (self.pieces[them][Piece::Rook.index()] | self.pieces[them][Piece::Queen.index()]) != 0 {
            return true;
        }

        false
    }
    pub fn in_check(&self) -> bool {
        self.is_square_attacked(self.king_sq(self.side), self.side.flip())
    }
    pub fn print(&self) {
        println!();
        for rank in (0..8).rev() {
            print!("  {} ", rank + 1);
            for file in 0..8 {
                let sq = make_square(file, rank);
                match self.piece_at(sq) {
                    Some((piece, color)) => print!("{} ", piece.to_char(color)),
                    None => print!(". "),
                }
            }
            println!();
        }
        println!("    a b c d e f g h");
        println!();
        println!("  FEN: {}", self.to_fen());
        println!("  Hash: 0x{:016X}", self.hash);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_start_position() {
        crate::zobrist::init();
        let board = Board::start_pos();
        assert_eq!(board.to_fen(), START_FEN);
        assert_eq!(popcount(board.all_occupancy), 32);
        assert_eq!(board.king_sq(Color::White), sq::E1);
        assert_eq!(board.king_sq(Color::Black), sq::E8);
    }

    #[test]
    fn test_fen_roundtrip() {
        crate::zobrist::init();
        let fens = [
            START_FEN,
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/8/8/8/8/8/8/4K2k w - - 0 1",
        ];
        for fen in fens {
            let board = Board::from_fen(fen).unwrap();
            assert_eq!(board.to_fen(), fen, "FEN roundtrip failed for: {}", fen);
        }
    }

    #[test]
    fn test_hash_consistency() {
        crate::zobrist::init();
        let board = Board::start_pos();
        let h1 = board.hash;
        let h2 = board.compute_hash();
        assert_eq!(h1, h2);
    }
}
