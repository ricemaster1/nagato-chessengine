/// Bitboard type: a 64-bit integer where each bit represents a square.
/// Bit 0 = A1, Bit 1 = B1, ..., Bit 7 = H1, Bit 8 = A2, ..., Bit 63 = H8
pub type Bitboard = u64;

// ============================================================
// Square indices (rank-file mapping: index = rank*8 + file)
// ============================================================
#[rustfmt::skip]
pub const SQUARE_NAMES: [&str; 64] = [
    "a1","b1","c1","d1","e1","f1","g1","h1",
    "a2","b2","c2","d2","e2","f2","g2","h2",
    "a3","b3","c3","d3","e3","f3","g3","h3",
    "a4","b4","c4","d4","e4","f4","g4","h4",
    "a5","b5","c5","d5","e5","f5","g5","h5",
    "a6","b6","c6","d6","e6","f6","g6","h6",
    "a7","b7","c7","d7","e7","f7","g7","h7",
    "a8","b8","c8","d8","e8","f8","g8","h8",
];

// Named square constants
#[rustfmt::skip]
pub mod sq {
    pub const A1: u8 = 0;  pub const B1: u8 = 1;  pub const C1: u8 = 2;  pub const D1: u8 = 3;
    pub const E1: u8 = 4;  pub const F1: u8 = 5;  pub const G1: u8 = 6;  pub const H1: u8 = 7;
    pub const A2: u8 = 8;  pub const B2: u8 = 9;  pub const C2: u8 = 10; pub const D2: u8 = 11;
    pub const E2: u8 = 12; pub const F2: u8 = 13; pub const G2: u8 = 14; pub const H2: u8 = 15;
    pub const A3: u8 = 16; pub const B3: u8 = 17; pub const C3: u8 = 18; pub const D3: u8 = 19;
    pub const E3: u8 = 20; pub const F3: u8 = 21; pub const G3: u8 = 22; pub const H3: u8 = 23;
    pub const A4: u8 = 24; pub const B4: u8 = 25; pub const C4: u8 = 26; pub const D4: u8 = 27;
    pub const E4: u8 = 28; pub const F4: u8 = 29; pub const G4: u8 = 30; pub const H4: u8 = 31;
    pub const A5: u8 = 32; pub const B5: u8 = 33; pub const C5: u8 = 34; pub const D5: u8 = 35;
    pub const E5: u8 = 36; pub const F5: u8 = 37; pub const G5: u8 = 38; pub const H5: u8 = 39;
    pub const A6: u8 = 40; pub const B6: u8 = 41; pub const C6: u8 = 42; pub const D6: u8 = 43;
    pub const E6: u8 = 44; pub const F6: u8 = 45; pub const G6: u8 = 46; pub const H6: u8 = 47;
    pub const A7: u8 = 48; pub const B7: u8 = 49; pub const C7: u8 = 50; pub const D7: u8 = 51;
    pub const E7: u8 = 52; pub const F7: u8 = 53; pub const G7: u8 = 54; pub const H7: u8 = 55;
    pub const A8: u8 = 56; pub const B8: u8 = 57; pub const C8: u8 = 58; pub const D8: u8 = 59;
    pub const E8: u8 = 60; pub const F8: u8 = 61; pub const G8: u8 = 62; pub const H8: u8 = 63;
}

// ============================================================
// File and Rank masks
// ============================================================
pub const FILE_A: Bitboard = 0x0101010101010101;
pub const FILE_B: Bitboard = FILE_A << 1;
pub const FILE_C: Bitboard = FILE_A << 2;
pub const FILE_D: Bitboard = FILE_A << 3;
pub const FILE_E: Bitboard = FILE_A << 4;
pub const FILE_F: Bitboard = FILE_A << 5;
pub const FILE_G: Bitboard = FILE_A << 6;
pub const FILE_H: Bitboard = FILE_A << 7;

pub const FILES: [Bitboard; 8] = [FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H];

pub const RANK_1: Bitboard = 0xFF;
pub const RANK_2: Bitboard = RANK_1 << 8;
pub const RANK_3: Bitboard = RANK_1 << 16;
pub const RANK_4: Bitboard = RANK_1 << 24;
pub const RANK_5: Bitboard = RANK_1 << 32;
pub const RANK_6: Bitboard = RANK_1 << 40;
pub const RANK_7: Bitboard = RANK_1 << 48;
pub const RANK_8: Bitboard = RANK_1 << 56;

pub const RANKS: [Bitboard; 8] = [RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8];

pub const NOT_FILE_A: Bitboard = !FILE_A;
pub const NOT_FILE_H: Bitboard = !FILE_H;
pub const NOT_FILE_AB: Bitboard = !(FILE_A | FILE_B);
pub const NOT_FILE_GH: Bitboard = !(FILE_G | FILE_H);

// ============================================================
// Piece and Color types
// ============================================================
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    #[inline]
    pub fn flip(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }

    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Piece {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
}

impl Piece {
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    pub fn from_char(c: char) -> Option<Piece> {
        match c.to_ascii_lowercase() {
            'p' => Some(Piece::Pawn),
            'n' => Some(Piece::Knight),
            'b' => Some(Piece::Bishop),
            'r' => Some(Piece::Rook),
            'q' => Some(Piece::Queen),
            'k' => Some(Piece::King),
            _ => None,
        }
    }

    pub fn to_char(self, color: Color) -> char {
        let c = match self {
            Piece::Pawn => 'p',
            Piece::Knight => 'n',
            Piece::Bishop => 'b',
            Piece::Rook => 'r',
            Piece::Queen => 'q',
            Piece::King => 'k',
        };
        match color {
            Color::White => c.to_ascii_uppercase(),
            Color::Black => c,
        }
    }
}

pub const PIECE_COUNT: usize = 6;
pub const COLOR_COUNT: usize = 2;

// ============================================================
// Bitboard utility functions
// ============================================================

/// Set a bit at the given square index
#[inline]
pub fn set_bit(bb: Bitboard, sq: u8) -> Bitboard {
    bb | (1u64 << sq)
}

/// Clear a bit at the given square index
#[inline]
pub fn clear_bit(bb: Bitboard, sq: u8) -> Bitboard {
    bb & !(1u64 << sq)
}

/// Toggle a bit at the given square index
#[inline]
pub fn toggle_bit(bb: Bitboard, sq: u8) -> Bitboard {
    bb ^ (1u64 << sq)
}

/// Check if a bit is set
#[inline]
pub fn get_bit(bb: Bitboard, sq: u8) -> bool {
    (bb >> sq) & 1 != 0
}

/// Bit mask for a single square
#[inline]
pub fn square_bb(sq: u8) -> Bitboard {
    1u64 << sq
}

/// Get the file (0-7) of a square
#[inline]
pub fn file_of(sq: u8) -> u8 {
    sq & 7
}

/// Get the rank (0-7) of a square
#[inline]
pub fn rank_of(sq: u8) -> u8 {
    sq >> 3
}

/// Make a square index from file and rank
#[inline]
pub fn make_square(file: u8, rank: u8) -> u8 {
    rank * 8 + file
}

/// Population count (number of set bits)
#[inline]
pub fn popcount(bb: Bitboard) -> u32 {
    bb.count_ones()
}

/// Least significant bit index
#[inline]
pub fn lsb(bb: Bitboard) -> u8 {
    debug_assert!(bb != 0);
    bb.trailing_zeros() as u8
}

/// Pop the least significant bit, returning its index
#[inline]
pub fn pop_lsb(bb: &mut Bitboard) -> u8 {
    let sq = lsb(*bb);
    *bb &= *bb - 1;
    sq
}

/// Shift a bitboard north (towards rank 8)
#[inline]
pub fn north(bb: Bitboard) -> Bitboard {
    bb << 8
}

/// Shift a bitboard south (towards rank 1)
#[inline]
pub fn south(bb: Bitboard) -> Bitboard {
    bb >> 8
}

/// Shift east (towards H file)
#[inline]
pub fn east(bb: Bitboard) -> Bitboard {
    (bb << 1) & NOT_FILE_A
}

/// Shift west (towards A file)
#[inline]
pub fn west(bb: Bitboard) -> Bitboard {
    (bb >> 1) & NOT_FILE_H
}

#[inline]
pub fn north_east(bb: Bitboard) -> Bitboard {
    (bb << 9) & NOT_FILE_A
}

#[inline]
pub fn north_west(bb: Bitboard) -> Bitboard {
    (bb << 7) & NOT_FILE_H
}

#[inline]
pub fn south_east(bb: Bitboard) -> Bitboard {
    (bb >> 7) & NOT_FILE_A
}

#[inline]
pub fn south_west(bb: Bitboard) -> Bitboard {
    (bb >> 9) & NOT_FILE_H
}

/// Pretty-print a bitboard for debugging
pub fn print_bitboard(bb: Bitboard) {
    println!();
    for rank in (0..8).rev() {
        print!("  {} ", rank + 1);
        for file in 0..8 {
            let sq = make_square(file, rank);
            if get_bit(bb, sq) {
                print!("1 ");
            } else {
                print!(". ");
            }
        }
        println!();
    }
    println!("    a b c d e f g h");
    println!("    Hex: 0x{:016X}", bb);
}

/// Parse square name (e.g., "e4") to square index
pub fn parse_square(s: &str) -> Option<u8> {
    let bytes = s.as_bytes();
    if bytes.len() != 2 {
        return None;
    }
    let file = bytes[0].wrapping_sub(b'a');
    let rank = bytes[1].wrapping_sub(b'1');
    if file < 8 && rank < 8 {
        Some(make_square(file, rank))
    } else {
        None
    }
}

/// Square index to string name
pub fn square_name(sq: u8) -> &'static str {
    SQUARE_NAMES[sq as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_square_basics() {
        assert_eq!(file_of(sq::E4), 4);
        assert_eq!(rank_of(sq::E4), 3);
        assert_eq!(make_square(4, 3), sq::E4);
    }

    #[test]
    fn test_bit_operations() {
        let mut bb: Bitboard = 0;
        bb = set_bit(bb, sq::E4);
        assert!(get_bit(bb, sq::E4));
        assert!(!get_bit(bb, sq::D4));
        assert_eq!(popcount(bb), 1);
        assert_eq!(lsb(bb), sq::E4);

        bb = set_bit(bb, sq::A1);
        assert_eq!(popcount(bb), 2);
        assert_eq!(pop_lsb(&mut bb), sq::A1);
        assert_eq!(popcount(bb), 1);
    }

    #[test]
    fn test_parse_square() {
        assert_eq!(parse_square("a1"), Some(sq::A1));
        assert_eq!(parse_square("e4"), Some(sq::E4));
        assert_eq!(parse_square("h8"), Some(sq::H8));
        assert_eq!(parse_square("z9"), None);
    }

    #[test]
    fn test_shifts() {
        let e4 = square_bb(sq::E4);
        assert_eq!(north(e4), square_bb(sq::E5));
        assert_eq!(south(e4), square_bb(sq::E3));
        assert_eq!(east(e4), square_bb(sq::F4));
        assert_eq!(west(e4), square_bb(sq::D4));
    }
}
