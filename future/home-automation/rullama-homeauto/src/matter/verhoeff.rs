//! Verhoeff check-digit algorithm.
//!
//! Used by the Matter 11-digit manual pairing code (spec §5.1.4.2).

const D: [[u8; 10]; 10] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
];

const P: [[u8; 10]; 8] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
];

const INV: [u8; 10] = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9];

/// Compute the Verhoeff check digit for a slice of decimal digits.
pub fn compute(digits: &[u8]) -> u8 {
    let mut c: u8 = 0;
    for (i, &d) in digits.iter().rev().enumerate() {
        debug_assert!(d <= 9);
        c = D[c as usize][P[(i + 1) % 8][d as usize] as usize];
    }
    INV[c as usize]
}

/// Validate that the last digit of `digits` is the correct Verhoeff check digit.
pub fn validate(digits: &[u8]) -> bool {
    let mut c: u8 = 0;
    for (i, &d) in digits.iter().rev().enumerate() {
        if d > 9 {
            return false;
        }
        c = D[c as usize][P[i % 8][d as usize] as usize];
    }
    c == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_vectors() {
        // Standard published Verhoeff examples.
        assert_eq!(compute(&[2, 3, 6]), 3);
        assert_eq!(compute(&[1, 2, 3, 4, 5]), 1);
    }

    #[test]
    fn roundtrip_validate() {
        let payload = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0];
        let check = compute(&payload);
        let mut full = payload.to_vec();
        full.push(check);
        assert!(validate(&full));
    }

    #[test]
    fn detects_single_digit_error() {
        let payload = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0];
        let check = compute(&payload);
        let mut full = payload.to_vec();
        full.push(check);
        for i in 0..full.len() {
            let mut corrupted = full.clone();
            corrupted[i] = (corrupted[i] + 1) % 10;
            assert!(!validate(&corrupted), "should catch error at index {i}");
        }
    }

    #[test]
    fn detects_adjacent_transposition() {
        let payload = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0];
        let check = compute(&payload);
        let mut full = payload.to_vec();
        full.push(check);
        for i in 0..full.len() - 1 {
            if full[i] == full[i + 1] {
                continue;
            }
            let mut swapped = full.clone();
            swapped.swap(i, i + 1);
            assert!(
                !validate(&swapped),
                "should catch transposition at index {i}"
            );
        }
    }
}
