/// Move generation using precomputed attack tables.
/// - Knight and King: lookup tables
/// - Sliding pieces (Bishop, Rook, Queen): magic bitboards
/// - Pawns: computed directly from bitboard shifts

use crate::bitboard::*;
use crate::board::*;
use crate::moves::*;
use std::sync::OnceLock;

// ============================================================
// Precomputed attack tables
// ============================================================

struct AttackTables {
    knight: [Bitboard; 64],
    king: [Bitboard; 64],
    // Magic bitboard tables for sliding pieces
    bishop_magics: Vec<MagicEntry>,
    rook_magics: Vec<MagicEntry>,
    bishop_table: Vec<Bitboard>,
    rook_table: Vec<Bitboard>,
}

struct MagicEntry {
    mask: Bitboard,
    magic: u64,
    shift: u8,
    offset: usize,
}

static TABLES: OnceLock<AttackTables> = OnceLock::new();

/// Initialize all attack tables. Must be called once at startup.
pub fn init() {
    TABLES.get_or_init(|| {
        let knight = init_knight_attacks();
        let king = init_king_attacks();
        let (bishop_magics, bishop_table) = init_bishop_magics();
        let (rook_magics, rook_table) = init_rook_magics();
        AttackTables {
            knight,
            king,
            bishop_magics,
            rook_magics,
            bishop_table,
            rook_table,
        }
    });
}

fn tables() -> &'static AttackTables {
    TABLES.get().expect("Attack tables not initialized! Call movegen::init() first.")
}

// ============================================================
// Knight attacks
// ============================================================
fn init_knight_attacks() -> [Bitboard; 64] {
    let mut attacks = [0u64; 64];
    for sq in 0..64u8 {
        let bb = square_bb(sq);
        let mut a: Bitboard = 0;
        a |= (bb << 17) & NOT_FILE_A;    // up 2, right 1
        a |= (bb << 15) & NOT_FILE_H;    // up 2, left 1
        a |= (bb << 10) & NOT_FILE_AB;   // up 1, right 2
        a |= (bb << 6)  & NOT_FILE_GH;   // up 1, left 2
        a |= (bb >> 6)  & NOT_FILE_AB;   // down 1, right 2
        a |= (bb >> 10) & NOT_FILE_GH;   // down 1, left 2
        a |= (bb >> 15) & NOT_FILE_A;    // down 2, right 1
        a |= (bb >> 17) & NOT_FILE_H;    // down 2, left 1
        attacks[sq as usize] = a;
    }
    attacks
}

// ============================================================
// King attacks
// ============================================================
fn init_king_attacks() -> [Bitboard; 64] {
    let mut attacks = [0u64; 64];
    for sq in 0..64u8 {
        let bb = square_bb(sq);
        let mut a: Bitboard = 0;
        a |= north(bb);
        a |= south(bb);
        a |= east(bb);
        a |= west(bb);
        a |= north_east(bb);
        a |= north_west(bb);
        a |= south_east(bb);
        a |= south_west(bb);
        attacks[sq as usize] = a;
    }
    attacks
}

// ============================================================
// Magic bitboards for sliding pieces
// ============================================================

/// Relevant occupancy mask for a bishop on a given square (edges excluded)
fn bishop_mask(sq: u8) -> Bitboard {
    let mut mask: Bitboard = 0;
    let (r, f) = (rank_of(sq) as i8, file_of(sq) as i8);

    for &(dr, df) in &[(1, 1), (1, -1), (-1, 1), (-1, -1)] {
        let (mut cr, mut cf) = (r + dr, f + df);
        while cr > 0 && cr < 7 && cf > 0 && cf < 7 {
            mask |= square_bb(make_square(cf as u8, cr as u8));
            cr += dr;
            cf += df;
        }
    }
    mask
}

/// Relevant occupancy mask for a rook on a given square (edges excluded on the relevant axes)
fn rook_mask(sq: u8) -> Bitboard {
    let mut mask: Bitboard = 0;
    let (r, f) = (rank_of(sq) as i8, file_of(sq) as i8);

    // Up
    for cr in (r + 1)..7 {
        mask |= square_bb(make_square(f as u8, cr as u8));
    }
    // Down
    for cr in 1..r {
        mask |= square_bb(make_square(f as u8, cr as u8));
    }
    // Right
    for cf in (f + 1)..7 {
        mask |= square_bb(make_square(cf as u8, r as u8));
    }
    // Left
    for cf in 1..f {
        mask |= square_bb(make_square(cf as u8, r as u8));
    }
    mask
}

/// Compute bishop attacks for a given square and occupancy (used to build table)
fn bishop_attacks_slow(sq: u8, occ: Bitboard) -> Bitboard {
    let mut attacks: Bitboard = 0;
    let (r, f) = (rank_of(sq) as i8, file_of(sq) as i8);

    for &(dr, df) in &[(1, 1), (1, -1), (-1, 1), (-1, -1)] {
        let (mut cr, mut cf) = (r + dr, f + df);
        while cr >= 0 && cr <= 7 && cf >= 0 && cf <= 7 {
            let s = make_square(cf as u8, cr as u8);
            attacks |= square_bb(s);
            if occ & square_bb(s) != 0 {
                break;
            }
            cr += dr;
            cf += df;
        }
    }
    attacks
}

/// Compute rook attacks for a given square and occupancy (used to build table)
fn rook_attacks_slow(sq: u8, occ: Bitboard) -> Bitboard {
    let mut attacks: Bitboard = 0;
    let (r, f) = (rank_of(sq) as i8, file_of(sq) as i8);

    for &(dr, df) in &[(1, 0), (-1, 0), (0, 1), (0, -1)] {
        let (mut cr, mut cf) = (r + dr, f + df);
        while cr >= 0 && cr <= 7 && cf >= 0 && cf <= 7 {
            let s = make_square(cf as u8, cr as u8);
            attacks |= square_bb(s);
            if occ & square_bb(s) != 0 {
                break;
            }
            cr += dr;
            cf += df;
        }
    }
    attacks
}

/// Enumerate all subsets of a mask using the Carry-Rippler trick
fn enumerate_subsets(mask: Bitboard) -> Vec<Bitboard> {
    let mut subsets = Vec::new();
    let mut subset: Bitboard = 0;
    loop {
        subsets.push(subset);
        subset = subset.wrapping_sub(mask) & mask;
        if subset == 0 {
            break;
        }
    }
    subsets
}

/// Find magic numbers by trial. We use known good magics for speed.
/// These were found by brute-force search and are commonly known values.
#[rustfmt::skip]
const BISHOP_MAGICS_RAW: [u64; 64] = [
    0x40040844404084, 0x2004208a004208, 0x10190041080202, 0x108060845042010,
    0x581104180800210, 0x2112080446200010, 0x1080820820060210, 0x3c0808410220200,
    0x4050404440404, 0x21001420088, 0x24d0080801082102, 0x1020a0a020400,
    0x40308200402, 0x4011002100800, 0x401484104104005, 0x801010402020200,
    0x400210c3880100, 0x404022024108200, 0x810018200204102, 0x4002801a02003,
    0x85040820080400, 0x810102c808880400, 0xe900410884800, 0x8002020480840102,
    0x220200865090201, 0x2010100a02021202, 0x152048408022401, 0x20080002081110,
    0x4001001021004000, 0x800040400a011002, 0xe4004081011002, 0x1c004001012080,
    0x8004200962a00220, 0x8422100208500202, 0x2000402200300c08, 0x8646020080080080,
    0x80020a0200100808, 0x2010004880111000, 0x623000a080011400, 0x42008c0340209202,
    0x209188240001000, 0x400408a884001800, 0x110400a6080400, 0x1840060a44020800,
    0x90080104000041, 0x201011000808101, 0x1a2208080504f080, 0x8012020600211212,
    0x500861011240000, 0x180806108200800, 0x4000020e01040044, 0x300000261044000a,
    0x802241102020002, 0x20906061210001, 0x5a84841004010310, 0x4010801011c04,
    0xa010109502200, 0x4a02012000, 0x500201010098b028, 0x8040002811040900,
    0x28000010020204, 0x6000020202d0240, 0x8918844842082200, 0x4010011029020020,
];

#[rustfmt::skip]
const ROOK_MAGICS_RAW: [u64; 64] = [
    0x8a80104000800020, 0x140002000100040, 0x2801880a0017001, 0x100081001000420,
    0x200020010080420, 0x3001c0002010008, 0x8480008002000100, 0x2080088004402900,
    0x800098204000, 0x2024401000200040, 0x100802000801000, 0x120800800801000,
    0x208808088000400, 0x2802200800400, 0x2200800100020080, 0x801000060821100,
    0x80044006422000, 0x100808020004000, 0x12108a0010204200, 0x140848010000802,
    0x481828014002800, 0x8094004002004100, 0x4010040010010802, 0x20008806104,
    0x100400080208000, 0x2040002120081000, 0x21200680100081, 0x20100080080080,
    0x2000a00200410, 0x20080800400, 0x80088400100102, 0x80004600042881,
    0x4040008040800020, 0x440003000200801, 0x4200011004500, 0x188020010100100,
    0x14800401802800, 0x2080040080800200, 0x124080204001001, 0x200046502000484,
    0x480400080088020, 0x1000422010034000, 0x30200100110040, 0x100021010009,
    0x2002080100110004, 0x202008004008002, 0x20020004010100, 0x2048440040820001,
    0x101002200408200, 0x40802000401080, 0x4008142004410100, 0x2060820c0120200,
    0x1001004080100, 0x20c020080040080, 0x2935610830022400, 0x44440041009200,
    0x280001040802101, 0x2100190040002085, 0x80c0084100102001, 0x4024081001000421,
    0x20030a0244872, 0x12001008414402, 0x2006104900a0804, 0x1004081002402,
];

const BISHOP_BITS: [u8; 64] = {
    let mut bits = [0u8; 64];
    let mut sq = 0;
    while sq < 64 {
        bits[sq] = bishop_mask_const(sq as u8).count_ones() as u8;
        sq += 1;
    }
    bits
};

const ROOK_BITS: [u8; 64] = {
    let mut bits = [0u8; 64];
    let mut sq = 0;
    while sq < 64 {
        bits[sq] = rook_mask_const(sq as u8).count_ones() as u8;
        sq += 1;
    }
    bits
};

// Const versions of mask functions for compile-time evaluation
const fn bishop_mask_const(sq: u8) -> Bitboard {
    let mut mask: Bitboard = 0;
    let r = (sq >> 3) as i8;
    let f = (sq & 7) as i8;
    let dirs: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
    let mut d = 0;
    while d < 4 {
        let (dr, df) = dirs[d];
        let (mut cr, mut cf) = (r + dr, f + df);
        while cr > 0 && cr < 7 && cf > 0 && cf < 7 {
            mask |= 1u64 << (cr as u8 * 8 + cf as u8);
            cr += dr;
            cf += df;
        }
        d += 1;
    }
    mask
}

const fn rook_mask_const(sq: u8) -> Bitboard {
    let mut mask: Bitboard = 0;
    let r = (sq >> 3) as i8;
    let f = (sq & 7) as i8;
    let mut cr = r + 1;
    while cr < 7 { mask |= 1u64 << (cr as u8 * 8 + f as u8); cr += 1; }
    cr = r - 1;
    while cr > 0 { mask |= 1u64 << (cr as u8 * 8 + f as u8); cr -= 1; }
    let mut cf = f + 1;
    while cf < 7 { mask |= 1u64 << (r as u8 * 8 + cf as u8); cf += 1; }
    cf = f - 1;
    while cf > 0 { mask |= 1u64 << (r as u8 * 8 + cf as u8); cf -= 1; }
    mask
}

fn init_bishop_magics() -> (Vec<MagicEntry>, Vec<Bitboard>) {
    // Pre-allocate: total size is sum of 2^bits for each square
    let total: usize = (0..64).map(|sq| 1 << BISHOP_BITS[sq]).sum();
    let mut table = vec![0u64; total];
    let mut magics = Vec::with_capacity(64);

    let mut offset = 0;
    for sq in 0..64u8 {
        let mask = bishop_mask(sq);
        let bits = BISHOP_BITS[sq as usize];
        let shift = 64 - bits;
        let magic = BISHOP_MAGICS_RAW[sq as usize];

        let subsets = enumerate_subsets(mask);
        for occ in &subsets {
            let idx = (occ.wrapping_mul(magic) >> shift) as usize;
            table[offset + idx] = bishop_attacks_slow(sq, *occ);
        }

        magics.push(MagicEntry { mask, magic, shift, offset });
        offset += 1 << bits;
    }

    // Shrink-wrap the Vec. We pre-sized it, but let's verify.
    assert_eq!(offset, total);

    (magics, table)
}

fn init_rook_magics() -> (Vec<MagicEntry>, Vec<Bitboard>) {
    let total: usize = (0..64).map(|sq| 1 << ROOK_BITS[sq]).sum();
    let mut table = vec![0u64; total];
    let mut magics = Vec::with_capacity(64);

    let mut offset = 0;
    for sq in 0..64u8 {
        let mask = rook_mask(sq);
        let bits = ROOK_BITS[sq as usize];
        let shift = 64 - bits;
        let magic = ROOK_MAGICS_RAW[sq as usize];

        let subsets = enumerate_subsets(mask);
        for occ in &subsets {
            let idx = (occ.wrapping_mul(magic) >> shift) as usize;
            table[offset + idx] = rook_attacks_slow(sq, *occ);
        }

        magics.push(MagicEntry { mask, magic, shift, offset });
        offset += 1 << bits;
    }

    assert_eq!(offset, total);
    (magics, table)
}

// ============================================================
// Public attack lookup functions
// ============================================================

#[inline]
pub fn knight_attacks(sq: u8) -> Bitboard {
    tables().knight[sq as usize]
}

#[inline]
pub fn king_attacks(sq: u8) -> Bitboard {
    tables().king[sq as usize]
}

#[inline]
pub fn bishop_attacks(sq: u8, occ: Bitboard) -> Bitboard {
    let t = tables();
    let entry = &t.bishop_magics[sq as usize];
    let idx = ((occ & entry.mask).wrapping_mul(entry.magic) >> entry.shift) as usize;
    t.bishop_table[entry.offset + idx]
}

#[inline]
pub fn rook_attacks(sq: u8, occ: Bitboard) -> Bitboard {
    let t = tables();
    let entry = &t.rook_magics[sq as usize];
    let idx = ((occ & entry.mask).wrapping_mul(entry.magic) >> entry.shift) as usize;
    t.rook_table[entry.offset + idx]
}

#[inline]
pub fn queen_attacks(sq: u8, occ: Bitboard) -> Bitboard {
    bishop_attacks(sq, occ) | rook_attacks(sq, occ)
}

// ============================================================
// Pawn attack helpers
// ============================================================

#[inline]
pub fn white_pawn_attacks(pawns: Bitboard) -> Bitboard {
    north_east(pawns) | north_west(pawns)
}

#[inline]
pub fn black_pawn_attacks(pawns: Bitboard) -> Bitboard {
    south_east(pawns) | south_west(pawns)
}

/// Single pawn attacks from a square for a color
#[inline]
pub fn pawn_attacks(sq: u8, color: Color) -> Bitboard {
    let bb = square_bb(sq);
    match color {
        Color::White => white_pawn_attacks(bb),
        Color::Black => black_pawn_attacks(bb),
    }
}

// ============================================================
// Move generation
// ============================================================

/// Generate all pseudo-legal moves for the current position.
/// Legality is checked by make_move (verifying the king isn't in check after).
pub fn generate_moves(board: &Board, list: &mut MoveList) {
    let us = board.side;
    let them = us.flip();
    let our_occ = board.occupancy[us.index()];
    let their_occ = board.occupancy[them.index()];
    let all_occ = board.all_occupancy;
    let empty = !all_occ;

    generate_pawn_moves(board, list, us, their_occ, empty);
    generate_knight_moves(board, list, us, our_occ, their_occ);
    generate_bishop_moves(board, list, us, our_occ, their_occ, all_occ);
    generate_rook_moves(board, list, us, our_occ, their_occ, all_occ);
    generate_queen_moves(board, list, us, our_occ, their_occ, all_occ);
    generate_king_moves(board, list, us, our_occ, their_occ);
    generate_castles(board, list, us, all_occ);
}

/// Generate only capture moves (for quiescence search)
pub fn generate_captures(board: &Board, list: &mut MoveList) {
    let us = board.side;
    let them = us.flip();
    let our_occ = board.occupancy[us.index()];
    let their_occ = board.occupancy[them.index()];
    let all_occ = board.all_occupancy;

    generate_pawn_captures(board, list, us, their_occ);
    generate_piece_captures(board, list, us, our_occ, their_occ, all_occ);
}

fn generate_pawn_moves(board: &Board, list: &mut MoveList, us: Color, their_occ: Bitboard, empty: Bitboard) {
    let pawns = board.pieces[us.index()][Piece::Pawn.index()];
    let promo_rank = match us { Color::White => RANK_8, Color::Black => RANK_1 };
    let _start_rank = match us { Color::White => RANK_2, Color::Black => RANK_7 };

    // Single pushes
    let single = match us {
        Color::White => north(pawns) & empty,
        Color::Black => south(pawns) & empty,
    };

    // Non-promotion single pushes
    let mut quiet_single = single & !promo_rank;
    while quiet_single != 0 {
        let to = pop_lsb(&mut quiet_single);
        let from = match us { Color::White => to - 8, Color::Black => to + 8 };
        list.push(Move::new(from, to, FLAG_QUIET, Piece::Pawn));
    }

    // Double pushes
    let double_target = match us {
        Color::White => north(single & RANK_3) & empty,
        Color::Black => south(single & RANK_6) & empty,
    };
    let mut doubles = double_target;
    while doubles != 0 {
        let to = pop_lsb(&mut doubles);
        let from = match us { Color::White => to - 16, Color::Black => to + 16 };
        list.push(Move::new(from, to, FLAG_DOUBLE_PAWN, Piece::Pawn));
    }

    // Promotion pushes
    let mut promo = single & promo_rank;
    while promo != 0 {
        let to = pop_lsb(&mut promo);
        let from = match us { Color::White => to - 8, Color::Black => to + 8 };
        list.push(Move::new(from, to, FLAG_PROMO_Q, Piece::Pawn));
        list.push(Move::new(from, to, FLAG_PROMO_R, Piece::Pawn));
        list.push(Move::new(from, to, FLAG_PROMO_B, Piece::Pawn));
        list.push(Move::new(from, to, FLAG_PROMO_N, Piece::Pawn));
    }

    // Captures
    let (attack_left, attack_right) = match us {
        Color::White => (north_west(pawns), north_east(pawns)),
        Color::Black => (south_west(pawns), south_east(pawns)),
    };

    let (left_offset, right_offset): (i8, i8) = match us {
        Color::White => (-7, -9),
        Color::Black => (9, 7),
    };

    // Left captures (non-promotion)
    let mut left_cap = attack_left & their_occ & !promo_rank;
    while left_cap != 0 {
        let to = pop_lsb(&mut left_cap);
        let from = (to as i8 + left_offset) as u8;
        let cap = board.piece_on(to, us.flip()).unwrap();
        list.push(Move::new_with_capture(from, to, FLAG_CAPTURE, Piece::Pawn, cap));
    }

    // Right captures (non-promotion)
    let mut right_cap = attack_right & their_occ & !promo_rank;
    while right_cap != 0 {
        let to = pop_lsb(&mut right_cap);
        let from = (to as i8 + right_offset) as u8;
        let cap = board.piece_on(to, us.flip()).unwrap();
        list.push(Move::new_with_capture(from, to, FLAG_CAPTURE, Piece::Pawn, cap));
    }

    // Promotion captures
    let mut left_promo_cap = attack_left & their_occ & promo_rank;
    while left_promo_cap != 0 {
        let to = pop_lsb(&mut left_promo_cap);
        let from = (to as i8 + left_offset) as u8;
        let cap = board.piece_on(to, us.flip()).unwrap();
        list.push(Move::new_with_capture(from, to, FLAG_PROMO_CAP_Q, Piece::Pawn, cap));
        list.push(Move::new_with_capture(from, to, FLAG_PROMO_CAP_R, Piece::Pawn, cap));
        list.push(Move::new_with_capture(from, to, FLAG_PROMO_CAP_B, Piece::Pawn, cap));
        list.push(Move::new_with_capture(from, to, FLAG_PROMO_CAP_N, Piece::Pawn, cap));
    }

    let mut right_promo_cap = attack_right & their_occ & promo_rank;
    while right_promo_cap != 0 {
        let to = pop_lsb(&mut right_promo_cap);
        let from = (to as i8 + right_offset) as u8;
        let cap = board.piece_on(to, us.flip()).unwrap();
        list.push(Move::new_with_capture(from, to, FLAG_PROMO_CAP_Q, Piece::Pawn, cap));
        list.push(Move::new_with_capture(from, to, FLAG_PROMO_CAP_R, Piece::Pawn, cap));
        list.push(Move::new_with_capture(from, to, FLAG_PROMO_CAP_B, Piece::Pawn, cap));
        list.push(Move::new_with_capture(from, to, FLAG_PROMO_CAP_N, Piece::Pawn, cap));
    }

    // En passant
    if let Some(ep_sq) = board.ep_square {
        let ep_bb = square_bb(ep_sq);
        let attackers = match us {
            Color::White => {
                let left = (ep_bb >> 9) & NOT_FILE_H & pawns;
                let right = (ep_bb >> 7) & NOT_FILE_A & pawns;
                left | right
            }
            Color::Black => {
                let left = (ep_bb << 7) & NOT_FILE_H & pawns;
                let right = (ep_bb << 9) & NOT_FILE_A & pawns;
                left | right
            }
        };
        let mut att = attackers;
        while att != 0 {
            let from = pop_lsb(&mut att);
            list.push(Move::new(from, ep_sq, FLAG_EP_CAPTURE, Piece::Pawn));
        }
    }
}

fn generate_pawn_captures(board: &Board, list: &mut MoveList, us: Color, their_occ: Bitboard) {
    let pawns = board.pieces[us.index()][Piece::Pawn.index()];
    let promo_rank = match us { Color::White => RANK_8, Color::Black => RANK_1 };

    let (attack_left, attack_right) = match us {
        Color::White => (north_west(pawns), north_east(pawns)),
        Color::Black => (south_west(pawns), south_east(pawns)),
    };

    let (left_offset, right_offset): (i8, i8) = match us {
        Color::White => (-7, -9),
        Color::Black => (9, 7),
    };

    // All captures (including promotions)
    let mut left_cap = attack_left & their_occ;
    while left_cap != 0 {
        let to = pop_lsb(&mut left_cap);
        let from = (to as i8 + left_offset) as u8;
        let cap = board.piece_on(to, us.flip()).unwrap();
        if square_bb(to) & promo_rank != 0 {
            list.push(Move::new_with_capture(from, to, FLAG_PROMO_CAP_Q, Piece::Pawn, cap));
            list.push(Move::new_with_capture(from, to, FLAG_PROMO_CAP_R, Piece::Pawn, cap));
            list.push(Move::new_with_capture(from, to, FLAG_PROMO_CAP_B, Piece::Pawn, cap));
            list.push(Move::new_with_capture(from, to, FLAG_PROMO_CAP_N, Piece::Pawn, cap));
        } else {
            list.push(Move::new_with_capture(from, to, FLAG_CAPTURE, Piece::Pawn, cap));
        }
    }

    let mut right_cap = attack_right & their_occ;
    while right_cap != 0 {
        let to = pop_lsb(&mut right_cap);
        let from = (to as i8 + right_offset) as u8;
        let cap = board.piece_on(to, us.flip()).unwrap();
        if square_bb(to) & promo_rank != 0 {
            list.push(Move::new_with_capture(from, to, FLAG_PROMO_CAP_Q, Piece::Pawn, cap));
            list.push(Move::new_with_capture(from, to, FLAG_PROMO_CAP_R, Piece::Pawn, cap));
            list.push(Move::new_with_capture(from, to, FLAG_PROMO_CAP_B, Piece::Pawn, cap));
            list.push(Move::new_with_capture(from, to, FLAG_PROMO_CAP_N, Piece::Pawn, cap));
        } else {
            list.push(Move::new_with_capture(from, to, FLAG_CAPTURE, Piece::Pawn, cap));
        }
    }

    // En passant
    if let Some(ep_sq) = board.ep_square {
        let ep_bb = square_bb(ep_sq);
        let attackers = match us {
            Color::White => {
                let left = (ep_bb >> 9) & NOT_FILE_H & pawns;
                let right = (ep_bb >> 7) & NOT_FILE_A & pawns;
                left | right
            }
            Color::Black => {
                let left = (ep_bb << 7) & NOT_FILE_H & pawns;
                let right = (ep_bb << 9) & NOT_FILE_A & pawns;
                left | right
            }
        };
        let mut att = attackers;
        while att != 0 {
            let from = pop_lsb(&mut att);
            list.push(Move::new(from, ep_sq, FLAG_EP_CAPTURE, Piece::Pawn));
        }
    }

    // Promotion pushes (non-capture) are also included in quiescence
    let empty = !board.all_occupancy;
    let single = match us {
        Color::White => north(pawns) & empty,
        Color::Black => south(pawns) & empty,
    };
    let mut promo_push = single & promo_rank;
    while promo_push != 0 {
        let to = pop_lsb(&mut promo_push);
        let from = match us { Color::White => to - 8, Color::Black => to + 8 };
        list.push(Move::new(from, to, FLAG_PROMO_Q, Piece::Pawn));
        list.push(Move::new(from, to, FLAG_PROMO_R, Piece::Pawn));
        list.push(Move::new(from, to, FLAG_PROMO_B, Piece::Pawn));
        list.push(Move::new(from, to, FLAG_PROMO_N, Piece::Pawn));
    }
}

fn generate_knight_moves(board: &Board, list: &mut MoveList, us: Color, our_occ: Bitboard, their_occ: Bitboard) {
    let mut knights = board.pieces[us.index()][Piece::Knight.index()];
    while knights != 0 {
        let from = pop_lsb(&mut knights);
        let attacks = knight_attacks(from) & !our_occ;

        let mut quiet = attacks & !their_occ;
        while quiet != 0 {
            let to = pop_lsb(&mut quiet);
            list.push(Move::new(from, to, FLAG_QUIET, Piece::Knight));
        }

        let mut caps = attacks & their_occ;
        while caps != 0 {
            let to = pop_lsb(&mut caps);
            let cap = board.piece_on(to, us.flip()).unwrap();
            list.push(Move::new_with_capture(from, to, FLAG_CAPTURE, Piece::Knight, cap));
        }
    }
}

fn generate_sliding_moves(board: &Board, list: &mut MoveList, us: Color, our_occ: Bitboard, their_occ: Bitboard, piece: Piece, attacks_fn: fn(u8, Bitboard) -> Bitboard, all_occ: Bitboard) {
    let mut pieces = board.pieces[us.index()][piece.index()];
    while pieces != 0 {
        let from = pop_lsb(&mut pieces);
        let attacks = attacks_fn(from, all_occ) & !our_occ;

        let mut quiet = attacks & !their_occ;
        while quiet != 0 {
            let to = pop_lsb(&mut quiet);
            list.push(Move::new(from, to, FLAG_QUIET, piece));
        }

        let mut caps = attacks & their_occ;
        while caps != 0 {
            let to = pop_lsb(&mut caps);
            let cap = board.piece_on(to, us.flip()).unwrap();
            list.push(Move::new_with_capture(from, to, FLAG_CAPTURE, piece, cap));
        }
    }
}

fn generate_bishop_moves(board: &Board, list: &mut MoveList, us: Color, our_occ: Bitboard, their_occ: Bitboard, all_occ: Bitboard) {
    generate_sliding_moves(board, list, us, our_occ, their_occ, Piece::Bishop, bishop_attacks, all_occ);
}

fn generate_rook_moves(board: &Board, list: &mut MoveList, us: Color, our_occ: Bitboard, their_occ: Bitboard, all_occ: Bitboard) {
    generate_sliding_moves(board, list, us, our_occ, their_occ, Piece::Rook, rook_attacks, all_occ);
}

fn generate_queen_moves(board: &Board, list: &mut MoveList, us: Color, our_occ: Bitboard, their_occ: Bitboard, all_occ: Bitboard) {
    generate_sliding_moves(board, list, us, our_occ, their_occ, Piece::Queen, queen_attacks, all_occ);
}

fn generate_king_moves(board: &Board, list: &mut MoveList, us: Color, our_occ: Bitboard, their_occ: Bitboard) {
    let from = board.king_sq(us);
    let attacks = king_attacks(from) & !our_occ;

    let mut quiet = attacks & !their_occ;
    while quiet != 0 {
        let to = pop_lsb(&mut quiet);
        list.push(Move::new(from, to, FLAG_QUIET, Piece::King));
    }

    let mut caps = attacks & their_occ;
    while caps != 0 {
        let to = pop_lsb(&mut caps);
        let cap = board.piece_on(to, us.flip()).unwrap();
        list.push(Move::new_with_capture(from, to, FLAG_CAPTURE, Piece::King, cap));
    }
}

fn generate_piece_captures(board: &Board, list: &mut MoveList, us: Color, _our_occ: Bitboard, their_occ: Bitboard, all_occ: Bitboard) {
    // Knights
    let mut knights = board.pieces[us.index()][Piece::Knight.index()];
    while knights != 0 {
        let from = pop_lsb(&mut knights);
        let mut caps = knight_attacks(from) & their_occ;
        while caps != 0 {
            let to = pop_lsb(&mut caps);
            let cap = board.piece_on(to, us.flip()).unwrap();
            list.push(Move::new_with_capture(from, to, FLAG_CAPTURE, Piece::Knight, cap));
        }
    }

    // Bishops
    let mut bishops = board.pieces[us.index()][Piece::Bishop.index()];
    while bishops != 0 {
        let from = pop_lsb(&mut bishops);
        let mut caps = bishop_attacks(from, all_occ) & their_occ;
        while caps != 0 {
            let to = pop_lsb(&mut caps);
            let cap = board.piece_on(to, us.flip()).unwrap();
            list.push(Move::new_with_capture(from, to, FLAG_CAPTURE, Piece::Bishop, cap));
        }
    }

    // Rooks
    let mut rooks = board.pieces[us.index()][Piece::Rook.index()];
    while rooks != 0 {
        let from = pop_lsb(&mut rooks);
        let mut caps = rook_attacks(from, all_occ) & their_occ;
        while caps != 0 {
            let to = pop_lsb(&mut caps);
            let cap = board.piece_on(to, us.flip()).unwrap();
            list.push(Move::new_with_capture(from, to, FLAG_CAPTURE, Piece::Rook, cap));
        }
    }

    // Queens
    let mut queens = board.pieces[us.index()][Piece::Queen.index()];
    while queens != 0 {
        let from = pop_lsb(&mut queens);
        let mut caps = queen_attacks(from, all_occ) & their_occ;
        while caps != 0 {
            let to = pop_lsb(&mut caps);
            let cap = board.piece_on(to, us.flip()).unwrap();
            list.push(Move::new_with_capture(from, to, FLAG_CAPTURE, Piece::Queen, cap));
        }
    }

    // King
    let from = board.king_sq(us);
    let mut caps = king_attacks(from) & their_occ;
    while caps != 0 {
        let to = pop_lsb(&mut caps);
        let cap = board.piece_on(to, us.flip()).unwrap();
        list.push(Move::new_with_capture(from, to, FLAG_CAPTURE, Piece::King, cap));
    }
}

fn generate_castles(board: &Board, list: &mut MoveList, us: Color, all_occ: Bitboard) {
    let them = us.flip();

    match us {
        Color::White => {
            // Kingside
            if board.castling & WK_CASTLE != 0 {
                let between = square_bb(sq::F1) | square_bb(sq::G1);
                if all_occ & between == 0
                    && !board.is_square_attacked(sq::E1, them)
                    && !board.is_square_attacked(sq::F1, them)
                    && !board.is_square_attacked(sq::G1, them)
                {
                    list.push(Move::new(sq::E1, sq::G1, FLAG_KING_CASTLE, Piece::King));
                }
            }
            // Queenside
            if board.castling & WQ_CASTLE != 0 {
                let between = square_bb(sq::D1) | square_bb(sq::C1) | square_bb(sq::B1);
                if all_occ & between == 0
                    && !board.is_square_attacked(sq::E1, them)
                    && !board.is_square_attacked(sq::D1, them)
                    && !board.is_square_attacked(sq::C1, them)
                {
                    list.push(Move::new(sq::E1, sq::C1, FLAG_QUEEN_CASTLE, Piece::King));
                }
            }
        }
        Color::Black => {
            // Kingside
            if board.castling & BK_CASTLE != 0 {
                let between = square_bb(sq::F8) | square_bb(sq::G8);
                if all_occ & between == 0
                    && !board.is_square_attacked(sq::E8, them)
                    && !board.is_square_attacked(sq::F8, them)
                    && !board.is_square_attacked(sq::G8, them)
                {
                    list.push(Move::new(sq::E8, sq::G8, FLAG_KING_CASTLE, Piece::King));
                }
            }
            // Queenside
            if board.castling & BQ_CASTLE != 0 {
                let between = square_bb(sq::D8) | square_bb(sq::C8) | square_bb(sq::B8);
                if all_occ & between == 0
                    && !board.is_square_attacked(sq::E8, them)
                    && !board.is_square_attacked(sq::D8, them)
                    && !board.is_square_attacked(sq::C8, them)
                {
                    list.push(Move::new(sq::E8, sq::C8, FLAG_QUEEN_CASTLE, Piece::King));
                }
            }
        }
    }
}

/// Perft: count the number of leaf nodes at a given depth.
/// Used to verify move generation correctness.
pub fn perft(board: &mut Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }

    let mut list = MoveList::new();
    generate_moves(board, &mut list);

    let mut count: u64 = 0;
    for i in 0..list.len() {
        let m = list.moves[i];
        if board.make_move(m) {
            count += perft(board, depth - 1);
            board.unmake_move(m);
        }
    }
    count
}

/// Divide perft: shows count per root move. Useful for debugging.
pub fn perft_divide(board: &mut Board, depth: u32) -> u64 {
    let mut list = MoveList::new();
    generate_moves(board, &mut list);

    let mut total: u64 = 0;
    for i in 0..list.len() {
        let m = list.moves[i];
        if board.make_move(m) {
            let count = if depth > 1 { perft(board, depth - 1) } else { 1 };
            println!("{}: {}", m.to_uci(), count);
            total += count;
            board.unmake_move(m);
        }
    }
    println!("\nTotal: {}", total);
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;

    fn setup() {
        crate::zobrist::init();
        init();
    }

    #[test]
    fn test_start_position_moves() {
        setup();
        let board = Board::start_pos();
        let mut list = MoveList::new();
        generate_moves(&board, &mut list);
        // Starting position should have 20 legal moves
        let mut legal = 0;
        let mut board_clone = board.clone();
        for i in 0..list.len() {
            if board_clone.make_move(list.moves[i]) {
                legal += 1;
                board_clone.unmake_move(list.moves[i]);
            }
        }
        assert_eq!(legal, 20, "Starting position should have 20 legal moves");
    }

    #[test]
    fn test_perft_start_pos() {
        setup();
        let mut board = Board::start_pos();
        assert_eq!(perft(&mut board, 1), 20);
        assert_eq!(perft(&mut board, 2), 400);
        assert_eq!(perft(&mut board, 3), 8902);
        assert_eq!(perft(&mut board, 4), 197281);
        // Depth 5 takes a bit longer but is a great validation
        // assert_eq!(perft(&mut board, 5), 4865609);
    }

    #[test]
    fn test_perft_kiwipete() {
        setup();
        // Position 2: "Kiwipete" — rich in tactical elements
        let mut board = Board::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1").unwrap();
        assert_eq!(perft(&mut board, 1), 48);
        assert_eq!(perft(&mut board, 2), 2039);
        assert_eq!(perft(&mut board, 3), 97862);
    }

    #[test]
    fn test_perft_position3() {
        setup();
        let mut board = Board::from_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1").unwrap();
        assert_eq!(perft(&mut board, 1), 14);
        assert_eq!(perft(&mut board, 2), 191);
        assert_eq!(perft(&mut board, 3), 2812);
        assert_eq!(perft(&mut board, 4), 43238);
    }

    #[test]
    fn test_perft_position4() {
        setup();
        let mut board = Board::from_fen("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1").unwrap();
        assert_eq!(perft(&mut board, 1), 6);
        assert_eq!(perft(&mut board, 2), 264);
        assert_eq!(perft(&mut board, 3), 9467);
    }

    #[test]
    fn test_perft_position5() {
        setup();
        let mut board = Board::from_fen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8").unwrap();
        assert_eq!(perft(&mut board, 1), 44);
        assert_eq!(perft(&mut board, 2), 1486);
        assert_eq!(perft(&mut board, 3), 62379);
    }

    #[test]
    fn test_knight_attacks_center() {
        setup();
        let attacks = knight_attacks(sq::E4);
        assert_eq!(popcount(attacks), 8); // Knight in center attacks 8 squares
    }

    #[test]
    fn test_knight_attacks_corner() {
        setup();
        let attacks = knight_attacks(sq::A1);
        assert_eq!(popcount(attacks), 2); // Knight in corner attacks 2 squares
    }

    #[test]
    fn test_king_attacks_center() {
        setup();
        let attacks = king_attacks(sq::E4);
        assert_eq!(popcount(attacks), 8);
    }

    #[test]
    fn test_bishop_attacks_empty() {
        setup();
        let attacks = bishop_attacks(sq::E4, 0);
        // Bishop on e4 with no blockers: d3,c2,b1 + f5,g6,h7 + f3,g2,h1 + d5,c6,b7,a8
        assert_eq!(popcount(attacks), 13);
    }

    #[test]
    fn test_rook_attacks_empty() {
        setup();
        let attacks = rook_attacks(sq::E4, 0);
        // Rook on e4 with no blockers: 7+7 = 14 squares
        assert_eq!(popcount(attacks), 14);
    }
}
