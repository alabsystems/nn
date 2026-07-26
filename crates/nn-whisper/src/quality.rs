// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Speech recognition quality metrics.
//!
//! Text-level metrics for evaluating speech-to-text transcription accuracy:
//!
//! - **WER** (Word Error Rate): standard word-level edit distance metric.
//! - **CER** (Character Error Rate): character-level edit distance, useful for
//!   languages without clear word boundaries (Chinese, Japanese, Thai).
//! - **Normalized Edit Distance**: Levenshtein distance normalized by reference length.
//! - **MER** (Match Error Rate): `(S + D + I) / (S + D + C)` — ratio of errors
//!   to alignment length.
//!
//! Audio-level quality metrics live in [`super::quality_audio`].

/// Compute Levenshtein edit distance between two slices with case-insensitive comparison.
///
/// Returns `(distance, substitutions, deletions, insertions)`.
/// The backtrace distinguishes S/D/I so callers can compute MER.
fn edit_distance_detail<T: PartialEq>(hyp: &[T], reference: &[T], case_eq: impl Fn(&T, &T) -> bool) -> (usize, usize, usize, usize) {
    let h_len = hyp.len();
    let r_len = reference.len();

    // Full DP table needed for backtrace.
    let mut dp = vec![vec![0usize; r_len + 1]; h_len + 1];
    for j in 0..=r_len {
        dp[0][j] = j;
    }
    for i in 0..=h_len {
        dp[i][0] = i;
    }
    for i in 1..=h_len {
        for j in 1..=r_len {
            let cost = if case_eq(&hyp[i - 1], &reference[j - 1]) { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j] + 1)       // deletion from reference perspective (insertion in hyp)
                .min(dp[i][j - 1] + 1)            // insertion from reference perspective (deletion from hyp)
                .min(dp[i - 1][j - 1] + cost);    // substitution or match
        }
    }

    // Backtrace to count S, D, I.
    let mut subs = 0usize;
    let mut dels = 0usize;
    let mut ins = 0usize;
    let mut i = h_len;
    let mut j = r_len;
    while i > 0 || j > 0 {
        if i > 0 && j > 0 {
            let cost = if case_eq(&hyp[i - 1], &reference[j - 1]) { 0 } else { 1 };
            if dp[i][j] == dp[i - 1][j - 1] + cost {
                if cost == 1 {
                    subs += 1;
                }
                i -= 1;
                j -= 1;
                continue;
            }
        }
        if i > 0 && dp[i][j] == dp[i - 1][j] + 1 {
            ins += 1;
            i -= 1;
        } else if j > 0 && dp[i][j] == dp[i][j - 1] + 1 {
            dels += 1;
            j -= 1;
        } else {
            // Should not happen with a valid DP table, but break to avoid infinite loop.
            break;
        }
    }

    (dp[h_len][r_len], subs, dels, ins)
}

/// Compute Word Error Rate (WER) between a hypothesis and reference transcript.
///
/// WER = (substitutions + insertions + deletions) / reference_words
///
/// Both strings are lowercased and split on whitespace before comparison.
/// Returns 0.0 for empty reference with empty hypothesis, 1.0 for empty
/// reference with non-empty hypothesis.
///
/// # Examples
///
/// ```
/// use nn_whisper::word_error_rate;
///
/// assert!((word_error_rate("the cat sat", "the cat sat") - 0.0).abs() < 1e-6);
/// assert!((word_error_rate("the cat", "the cat sat") - 1.0 / 3.0).abs() < 1e-6);
/// assert!((word_error_rate("a cat sat", "the cat sat") - 1.0 / 3.0).abs() < 1e-6);
/// ```
#[must_use]
pub fn word_error_rate(hypothesis: &str, reference: &str) -> f64 {
    let hyp_words: Vec<&str> = hypothesis.split_whitespace().collect();
    let ref_words: Vec<&str> = reference.split_whitespace().collect();

    if ref_words.is_empty() {
        return if hyp_words.is_empty() { 0.0 } else { 1.0 };
    }

    let (dist, _, _, _) = edit_distance_detail(
        &hyp_words,
        &ref_words,
        |a: &&str, b: &&str| a.eq_ignore_ascii_case(b),
    );
    dist as f64 / ref_words.len() as f64
}

/// Compute Character Error Rate (CER) between hypothesis and reference strings.
///
/// CER = (substitutions + insertions + deletions) / reference_characters
///
/// Operates at the character (Unicode scalar value) level, making it appropriate
/// for languages without clear word boundaries such as Chinese, Japanese, or Thai.
///
/// Whitespace is stripped before comparison. Returns 0.0 for both-empty,
/// 1.0 for empty reference with non-empty hypothesis.
///
/// # Examples
///
/// ```
/// use nn_whisper::character_error_rate;
///
/// assert!((character_error_rate("hello", "hello") - 0.0).abs() < 1e-6);
/// assert!((character_error_rate("hxllo", "hello") - 1.0 / 5.0).abs() < 1e-6);
/// ```
#[must_use]
pub fn character_error_rate(hypothesis: &str, reference: &str) -> f64 {
    let hyp_chars: Vec<char> = hypothesis.chars().filter(|c| !c.is_whitespace()).collect();
    let ref_chars: Vec<char> = reference.chars().filter(|c| !c.is_whitespace()).collect();

    if ref_chars.is_empty() {
        return if hyp_chars.is_empty() { 0.0 } else { 1.0 };
    }

    let (dist, _, _, _) = edit_distance_detail(
        &hyp_chars,
        &ref_chars,
        |a: &char, b: &char| a.to_lowercase().eq(b.to_lowercase()),
    );
    dist as f64 / ref_chars.len() as f64
}

/// Compute normalized edit distance between two strings.
///
/// Returns `levenshtein(hypothesis, reference) / max(len(reference), 1)` where
/// distance is computed at the character level (whitespace preserved).
///
/// Values range from 0.0 (identical) to values potentially above 1.0 when the
/// hypothesis is longer than the reference.
///
/// Returns 0.0 for both-empty, 1.0 for empty reference with non-empty hypothesis.
///
/// # Examples
///
/// ```
/// use nn_whisper::normalized_edit_distance;
///
/// assert!((normalized_edit_distance("kitten", "sitting") - 3.0 / 7.0).abs() < 1e-6);
/// assert!((normalized_edit_distance("abc", "abc") - 0.0).abs() < 1e-6);
/// ```
#[must_use]
pub fn normalized_edit_distance(hypothesis: &str, reference: &str) -> f64 {
    let hyp_chars: Vec<char> = hypothesis.chars().collect();
    let ref_chars: Vec<char> = reference.chars().collect();

    if ref_chars.is_empty() {
        return if hyp_chars.is_empty() { 0.0 } else { 1.0 };
    }

    let (dist, _, _, _) = edit_distance_detail(
        &hyp_chars,
        &ref_chars,
        |a: &char, b: &char| a == b,
    );
    dist as f64 / ref_chars.len() as f64
}

/// Compute Match Error Rate (MER) between hypothesis and reference.
///
/// MER = (S + D + I) / (S + D + C)
///
/// where S = substitutions, D = deletions, I = insertions, C = correct matches.
/// The denominator `S + D + C` equals the alignment length (total number of
/// reference tokens plus insertions that overlap with matches). Unlike WER,
/// MER is bounded to `[0.0, 1.0]`.
///
/// Operates at the word level with case-insensitive comparison.
/// Returns 0.0 for both-empty, 1.0 for empty reference with non-empty hypothesis.
///
/// # Examples
///
/// ```
/// use nn_whisper::match_error_rate;
///
/// assert!((match_error_rate("the cat sat", "the cat sat") - 0.0).abs() < 1e-6);
/// // 1 sub, 0 del, 0 ins, 2 correct => MER = 1 / (1 + 0 + 2) = 1/3
/// assert!((match_error_rate("the dog sat", "the cat sat") - 1.0 / 3.0).abs() < 1e-6);
/// ```
#[must_use]
pub fn match_error_rate(hypothesis: &str, reference: &str) -> f64 {
    let hyp_words: Vec<&str> = hypothesis.split_whitespace().collect();
    let ref_words: Vec<&str> = reference.split_whitespace().collect();

    if ref_words.is_empty() {
        return if hyp_words.is_empty() { 0.0 } else { 1.0 };
    }

    let (_, subs, dels, ins) = edit_distance_detail(
        &hyp_words,
        &ref_words,
        |a: &&str, b: &&str| a.eq_ignore_ascii_case(b),
    );

    let errors = subs + dels + ins;
    // C (correct) = ref_len - subs - dels (aligned reference tokens that matched).
    let correct = ref_words.len().saturating_sub(subs + dels);
    let denom = subs + dels + correct; // = ref_words.len() (alignment length without insertions)

    if denom == 0 {
        return if errors == 0 { 0.0 } else { 1.0 };
    }

    errors as f64 / denom as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- WER tests ----

    #[test]
    fn test_wer_identical() {
        assert!((word_error_rate("hello world", "hello world")).abs() < 1e-6);
    }

    #[test]
    fn test_wer_completely_wrong() {
        assert!((word_error_rate("foo bar", "hello world") - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_wer_one_substitution() {
        // 1 sub out of 3 reference words = 1/3
        let wer = word_error_rate("the dog sat", "the cat sat");
        assert!((wer - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_wer_one_deletion() {
        // "the sat" vs "the cat sat" = 1 deletion / 3 = 1/3
        let wer = word_error_rate("the sat", "the cat sat");
        assert!((wer - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_wer_one_insertion() {
        // "the big cat sat" vs "the cat sat" = 1 insertion / 3 = 1/3
        let wer = word_error_rate("the big cat sat", "the cat sat");
        assert!((wer - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_wer_empty_reference_empty_hypothesis() {
        assert!((word_error_rate("", "")).abs() < 1e-6);
    }

    #[test]
    fn test_wer_empty_reference_nonempty_hypothesis() {
        assert!((word_error_rate("hello", "") - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_wer_nonempty_reference_empty_hypothesis() {
        // 3 deletions / 3 = 1.0
        assert!((word_error_rate("", "the cat sat") - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_wer_case_insensitive() {
        assert!((word_error_rate("Hello World", "hello world")).abs() < 1e-6);
    }

    // ---- CER tests ----

    #[test]
    fn test_cer_identical() {
        assert!((character_error_rate("hello", "hello")).abs() < 1e-6);
    }

    #[test]
    fn test_cer_one_substitution() {
        // 1 sub / 5 chars = 0.2
        let cer = character_error_rate("hxllo", "hello");
        assert!((cer - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_cer_one_insertion() {
        // "heello" vs "hello" = 1 insertion / 5 = 0.2
        let cer = character_error_rate("heello", "hello");
        assert!((cer - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_cer_one_deletion() {
        // "hllo" vs "hello" = 1 deletion / 5 = 0.2
        let cer = character_error_rate("hllo", "hello");
        assert!((cer - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_cer_strips_whitespace() {
        // Whitespace stripped: "helloworld" vs "helloworld" = 0
        assert!((character_error_rate("hello world", "hello world")).abs() < 1e-6);
    }

    #[test]
    fn test_cer_case_insensitive() {
        assert!((character_error_rate("HELLO", "hello")).abs() < 1e-6);
    }

    #[test]
    fn test_cer_empty_both() {
        assert!((character_error_rate("", "")).abs() < 1e-6);
    }

    #[test]
    fn test_cer_empty_reference() {
        assert!((character_error_rate("abc", "") - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cer_completely_different() {
        // "xyz" vs "abc" = 3 subs / 3 = 1.0
        assert!((character_error_rate("xyz", "abc") - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cer_unicode_characters() {
        // CJK test: each character is one edit unit.
        // Reference 3 chars, hypothesis swaps 1 = 1/3.
        let cer = character_error_rate("\u{4F60}\u{597D}\u{5417}", "\u{4F60}\u{597D}\u{5440}");
        assert!((cer - 1.0 / 3.0).abs() < 1e-6);
    }

    // ---- Normalized Edit Distance tests ----

    #[test]
    fn test_ned_identical() {
        assert!((normalized_edit_distance("abc", "abc")).abs() < 1e-6);
    }

    #[test]
    fn test_ned_kitten_sitting() {
        // Classic: kitten -> sitting = 3 edits, ref len 7 => 3/7
        let ned = normalized_edit_distance("kitten", "sitting");
        assert!((ned - 3.0 / 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_ned_empty_both() {
        assert!((normalized_edit_distance("", "")).abs() < 1e-6);
    }

    #[test]
    fn test_ned_empty_reference() {
        assert!((normalized_edit_distance("abc", "") - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ned_empty_hypothesis() {
        // 3 deletions / 3 = 1.0
        assert!((normalized_edit_distance("", "abc") - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ned_case_sensitive() {
        // "ABC" vs "abc" = 3 subs / 3 = 1.0 (case-sensitive unlike CER)
        assert!((normalized_edit_distance("ABC", "abc") - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ned_preserves_whitespace() {
        // "a b" vs "ab" => 1 deletion / 2 = 0.5
        let ned = normalized_edit_distance("a b", "ab");
        assert!((ned - 0.5).abs() < 1e-6);
    }

    // ---- MER tests ----

    #[test]
    fn test_mer_identical() {
        assert!((match_error_rate("the cat sat", "the cat sat")).abs() < 1e-6);
    }

    #[test]
    fn test_mer_one_substitution() {
        // S=1, D=0, I=0, C=2 => errors=1, denom=S+D+C=1+0+2=3 => 1/3
        let mer = match_error_rate("the dog sat", "the cat sat");
        assert!((mer - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_mer_all_wrong() {
        // S=2, D=0, I=0, C=0 => errors=2, denom=2 => 1.0
        let mer = match_error_rate("foo bar", "hello world");
        assert!((mer - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_mer_empty_both() {
        assert!((match_error_rate("", "")).abs() < 1e-6);
    }

    #[test]
    fn test_mer_empty_reference() {
        assert!((match_error_rate("hello", "") - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_mer_one_insertion() {
        // "the big cat sat" vs "the cat sat": I=1, S=0, D=0, C=3
        // errors=1, denom=S+D+C=0+0+3=3 => 1/3
        let mer = match_error_rate("the big cat sat", "the cat sat");
        assert!((mer - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_mer_one_deletion() {
        // "the sat" vs "the cat sat": D=1, S=0, I=0, C=2
        // errors=1, denom=S+D+C=0+1+2=3 => 1/3
        let mer = match_error_rate("the sat", "the cat sat");
        assert!((mer - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_mer_bounded_above_one() {
        // MER should always be <= 1.0 (unlike WER which can exceed 1.0).
        let mer = match_error_rate("a b c d e f", "x");
        assert!(mer <= 1.0 + 1e-6);
    }

    // ---- edit_distance_detail internal tests ----

    #[test]
    fn test_edit_distance_detail_counts() {
        let hyp: Vec<&str> = "the dog sat on mat".split_whitespace().collect();
        let reference: Vec<&str> = "the cat sat on the mat".split_whitespace().collect();
        let (dist, subs, dels, ins) = edit_distance_detail(
            &hyp,
            &reference,
            |a: &&str, b: &&str| a.eq_ignore_ascii_case(b),
        );
        // "the dog sat on mat" vs "the cat sat on the mat"
        // Alignment: the=the, dog!=cat(S), sat=sat, on=on, _=the(D), mat=mat
        // S=1, D=1, I=0, dist=2
        assert_eq!(dist, 2);
        assert_eq!(subs, 1);
        assert_eq!(dels, 1);
        assert_eq!(ins, 0);
    }
}
