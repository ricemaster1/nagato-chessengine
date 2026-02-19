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
pub const ALL_SQUARES: [u8; 64] = {
    let mut arr = [0u8; 64];
    let mut sq = 0;
    while sq < 64 {
        arr[sq as usize] = sq;
        sq += 1;
    } 
};

// ============================================================
// File and Rank masks
// ============================================================
pub const FILES: [Bitboard; 8] = {
    let mut files = [0u64; 8];
    let mut f = 0;
    while f < 8 {
        files[f] = 0x0101010101010101u64 << f;
        f += 1;
    }
    files
};

pub const RANKS: [Bitboard; 8] = {
    let mut ranks = [0u64; 8];
    let mut r = 0;
    while r < 8 {
        ranks[r] = 0xFFu64 << (r * 8);
        r += 1;
    }
    ranks
};

#[inline]
pub const fn not_file(file: u8) -> Bitboard {
    !(FILES[file as usize])
}

#[inline]
pub const not_files(files_mask: Bitboard) -> Bitboard {
    !files_mask
}

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
