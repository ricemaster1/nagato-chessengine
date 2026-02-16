/// Move representation — packed into a u32 for efficiency.
///
/// Bit layout (32 bits total):
///   bits  0-5:   from square (0-63)
///   bits  6-11:  to square (0-63)
///   bits 12-15:  flags (captures, promotions, castles, etc.)
///   bits 16-19:  moved piece type (0-5)
///   bits 20-23:  captured piece type (0-5, valid only if capture flag set)
///
/// Flags encoding:
///   0000 = quiet move
///   0001 = double pawn push
///   0010 = king castle
///   0011 = queen castle
///   0100 = capture
///   0101 = en passant capture
///   1000 = knight promotion
///   1001 = bishop promotion
///   1010 = rook promotion
///   1011 = queen promotion
///   1100 = knight promotion capture
///   1101 = bishop promotion capture
///   1110 = rook promotion capture
///   1111 = queen promotion capture

use crate::bitboard::{self, Piece};

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Move(pub u32);

// Flag constants
pub const FLAG_QUIET: u32       = 0b0000;
pub const FLAG_DOUBLE_PAWN: u32 = 0b0001;
pub const FLAG_KING_CASTLE: u32 = 0b0010;
pub const FLAG_QUEEN_CASTLE: u32= 0b0011;
pub const FLAG_CAPTURE: u32     = 0b0100;
pub const FLAG_EP_CAPTURE: u32  = 0b0101;
pub const FLAG_PROMO_N: u32     = 0b1000;
pub const FLAG_PROMO_B: u32     = 0b1001;
pub const FLAG_PROMO_R: u32     = 0b1010;
pub const FLAG_PROMO_Q: u32     = 0b1011;
pub const FLAG_PROMO_CAP_N: u32 = 0b1100;
pub const FLAG_PROMO_CAP_B: u32 = 0b1101;
pub const FLAG_PROMO_CAP_R: u32 = 0b1110;
pub const FLAG_PROMO_CAP_Q: u32 = 0b1111;

pub const MOVE_NONE: Move = Move(0);

impl Move {
    #[inline]
    pub fn new(from: u8, to: u8, flags: u32, piece: Piece) -> Move {
        Move(
            (from as u32)
            | ((to as u32) << 6)
            | (flags << 12)
            | ((piece.index() as u32) << 16)
        )
    }

    #[inline]
    pub fn new_with_capture(from: u8, to: u8, flags: u32, piece: Piece, captured: Piece) -> Move {
        Move(
            (from as u32)
            | ((to as u32) << 6)
            | (flags << 12)
            | ((piece.index() as u32) << 16)
            | ((captured.index() as u32) << 20)
        )
    }

    #[inline]
    pub fn from_sq(self) -> u8 {
        (self.0 & 0x3F) as u8
    }

    #[inline]
    pub fn to_sq(self) -> u8 {
        ((self.0 >> 6) & 0x3F) as u8
    }

    #[inline]
    pub fn flags(self) -> u32 {
        (self.0 >> 12) & 0xF
    }

    #[inline]
    pub fn piece(self) -> Piece {
        unsafe { std::mem::transmute(((self.0 >> 16) & 0x7) as u8) }
    }

    #[inline]
    pub fn captured_piece(self) -> Piece {
        unsafe { std::mem::transmute(((self.0 >> 20) & 0x7) as u8) }
    }

    #[inline]
    pub fn is_capture(self) -> bool {
        self.flags() & FLAG_CAPTURE != 0
    }

    #[inline]
    pub fn is_promotion(self) -> bool {
        self.flags() & 0b1000 != 0
    }

    #[inline]
    pub fn is_castle(self) -> bool {
        let f = self.flags();
        f == FLAG_KING_CASTLE || f == FLAG_QUEEN_CASTLE
    }

    #[inline]
    pub fn is_en_passant(self) -> bool {
        self.flags() == FLAG_EP_CAPTURE
    }

    #[inline]
    pub fn is_double_pawn(self) -> bool {
        self.flags() == FLAG_DOUBLE_PAWN
    }

    #[inline]
    pub fn is_quiet(self) -> bool {
        self.flags() == FLAG_QUIET
    }

    #[inline]
    pub fn is_null(self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub fn promotion_piece(self) -> Option<Piece> {
        if !self.is_promotion() {
            return None;
        }
        match self.flags() & 0b0011 {
            0 => Some(Piece::Knight),
            1 => Some(Piece::Bishop),
            2 => Some(Piece::Rook),
            3 => Some(Piece::Queen),
            _ => unreachable!(),
        }
    }

    /// Convert to UCI string (e.g., "e2e4", "e7e8q")
    pub fn to_uci(self) -> String {
        let from = bitboard::square_name(self.from_sq());
        let to = bitboard::square_name(self.to_sq());
        let promo = match self.promotion_piece() {
            Some(Piece::Knight) => "n",
            Some(Piece::Bishop) => "b",
            Some(Piece::Rook)   => "r",
            Some(Piece::Queen)  => "q",
            _ => "",
        };
        format!("{}{}{}", from, to, promo)
    }
}

impl std::fmt::Debug for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_uci())
    }
}

impl std::fmt::Display for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_uci())
    }
}

/// A list of moves, stack-allocated for performance
pub struct MoveList {
    pub moves: [Move; 256],
    pub len: usize,
}

impl MoveList {
    pub fn new() -> Self {
        MoveList {
            moves: [MOVE_NONE; 256],
            len: 0,
        }
    }

    #[inline]
    pub fn push(&mut self, m: Move) {
        debug_assert!(self.len < 256);
        self.moves[self.len] = m;
        self.len += 1;
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn iter(&self) -> &[Move] {
        &self.moves[..self.len]
    }

    pub fn contains(&self, m: Move) -> bool {
        self.moves[..self.len].iter().any(|&mv| mv.from_sq() == m.from_sq() && mv.to_sq() == m.to_sq() && mv.flags() == m.flags())
    }
}

impl Default for MoveList {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitboard::{sq, Piece};

    #[test]
    fn test_move_encoding() {
        let m = Move::new(sq::E2, sq::E4, FLAG_DOUBLE_PAWN, Piece::Pawn);
        assert_eq!(m.from_sq(), sq::E2);
        assert_eq!(m.to_sq(), sq::E4);
        assert_eq!(m.flags(), FLAG_DOUBLE_PAWN);
        assert!(m.is_double_pawn());
        assert!(!m.is_capture());
        assert_eq!(m.to_uci(), "e2e4");
    }

    #[test]
    fn test_promotion_move() {
        let m = Move::new(sq::E7, sq::E8, FLAG_PROMO_Q, Piece::Pawn);
        assert!(m.is_promotion());
        assert_eq!(m.promotion_piece(), Some(Piece::Queen));
        assert_eq!(m.to_uci(), "e7e8q");
    }

    #[test]
    fn test_capture_move() {
        let m = Move::new_with_capture(sq::E4, sq::D5, FLAG_CAPTURE, Piece::Pawn, Piece::Pawn);
        assert!(m.is_capture());
        assert_eq!(m.captured_piece(), Piece::Pawn);
    }
}
