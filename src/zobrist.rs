
use crate::bitboard::{PIECE_COUNT, COLOR_COUNT};
use std::sync::OnceLock;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

pub struct ZobristKeys {
    pub piece_keys: [[[u64; 64]; PIECE_COUNT]; COLOR_COUNT],
    pub castle_keys: [u64; 16],
    pub ep_keys: [u64; 8],
    pub side_key: u64,
}

static ZOBRIST: OnceLock<ZobristKeys> = OnceLock::new();

pub fn init() {
    ZOBRIST.get_or_init(|| {
        let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF_CAFE_1234);

        let mut keys = ZobristKeys {
            piece_keys: [[[0u64; 64]; PIECE_COUNT]; COLOR_COUNT],
            castle_keys: [0u64; 16],
            ep_keys: [0u64; 8],
            side_key: 0,
        };

        for color in 0..COLOR_COUNT {
            for piece in 0..PIECE_COUNT {
                for sq in 0..64 {
                    keys.piece_keys[color][piece][sq] = rng.r#gen();
                }
            }
        }

        for i in 0..16 {
            keys.castle_keys[i] = rng.r#gen();
        }

        for i in 0..8 {
            keys.ep_keys[i] = rng.r#gen();
        }

        keys.side_key = rng.r#gen();

        keys
    });
}

#[inline]
pub fn keys() -> &'static ZobristKeys {
    ZOBRIST.get().expect("Zobrist keys not initialized! Call zobrist::init() first.")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zobrist_init() {
        init();
        let k = keys();
        assert_ne!(k.side_key, 0);
        assert_ne!(k.piece_keys[0][0][0], k.piece_keys[0][0][1]);
        assert_ne!(k.piece_keys[0][0][0], k.piece_keys[1][0][0]);
    }
}
