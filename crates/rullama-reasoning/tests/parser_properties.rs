//! Property-based integration tests for `plan_parser` and `output_parser`.
//!
//! These parsers are the seam between unstructured LLM output and the
//! framework's executable task/plan representation. A silent parse failure
//! degrades an agent to a no-op plan; a panic crashes the agent loop.
//! The properties here focus on:
//!
//! - **No-panic contract**: random text inputs must always return a
//!   `Result` (or an empty `Vec`) instead of panicking.
//! - **Shape invariants** on successful parses (monotonic step numbers,
//!   priority detection on known keywords, format-instruction presence).
//! - **Roundtrip** on inputs that genuinely contain the target structure.

use proptest::prelude::*;
use rullama_reasoning::output_parser::{
    JsonListParser, JsonOutputParser, OutputParser, RegexOutputParser,
};
use rullama_reasoning::plan_parser::{ParsedStep, parse_plan_steps, steps_to_tasks};
use serde::Deserialize;

// ── plan_parser ──────────────────────────────────────────────────────────

#[test]
fn numbered_lines_with_dot_and_paren_are_both_accepted() {
    let input = "1. first\n2) second\n3. third";
    let steps = parse_plan_steps(input);
    assert_eq!(steps.len(), 3);
    assert_eq!(steps[0].description, "first");
    assert_eq!(steps[1].description, "second");
    assert_eq!(steps[2].description, "third");
}

#[test]
fn step_colon_format_is_accepted() {
    let input = "Step 1: build the index\nStep 2: query it";
    let steps = parse_plan_steps(input);
    assert_eq!(steps.len(), 2);
    assert_eq!(steps[0].description, "build the index");
    assert_eq!(steps[1].description, "query it");
}

#[test]
fn indent_produces_substeps() {
    // 2-space indent maps to indent_level=1 per the parser's convention.
    let input = "1. root\n  2. child\n  3. child2\n4. next root";
    let steps = parse_plan_steps(input);
    assert_eq!(steps.len(), 4);
    assert_eq!(steps[0].indent_level, 0);
    assert_eq!(steps[1].indent_level, 1);
    assert_eq!(steps[2].indent_level, 1);
    assert_eq!(steps[3].indent_level, 0);
}

#[test]
fn priority_keywords_flag_is_priority() {
    for keyword in ["important", "IMPORTANT", "critical", "Critical", "!"] {
        let input = format!("1. do {keyword} thing");
        let steps = parse_plan_steps(&input);
        assert_eq!(steps.len(), 1);
        assert!(
            steps[0].is_priority,
            "keyword `{keyword}` should flag priority",
        );
    }
}

#[test]
fn non_priority_items_stay_low_priority() {
    let steps = parse_plan_steps("1. ordinary task");
    assert_eq!(steps.len(), 1);
    assert!(!steps[0].is_priority);
}

#[test]
fn empty_and_whitespace_input_produce_no_steps() {
    assert!(parse_plan_steps("").is_empty());
    assert!(parse_plan_steps("   \n\n  \t\n").is_empty());
}

#[test]
fn prose_without_numbers_produces_no_steps() {
    let steps = parse_plan_steps(
        "This is a narrative paragraph with no list structure. Nothing\n\
         should be extracted from it, even though it mentions step X.",
    );
    assert!(steps.is_empty());
}

#[test]
fn action_bullets_are_picked_up_but_notes_are_skipped() {
    let input = "\
- Note: this is a comment we should skip
- implement the parser module in src/lib.rs
- Warning: this bullet is also noise
- create a new integration test harness";
    let steps = parse_plan_steps(input);
    assert_eq!(
        steps.len(),
        2,
        "only the two action bullets should extract: {steps:?}",
    );
    assert!(steps[0].description.contains("implement"));
    assert!(steps[1].description.contains("create"));
}

#[test]
fn short_bullets_are_ignored() {
    // The parser requires bullets > 10 chars; short decorative bullets
    // shouldn't create bogus steps.
    let steps = parse_plan_steps("- add x\n- fix y");
    assert!(steps.is_empty());
}

#[test]
fn steps_to_tasks_preserves_step_count_and_priority() {
    let input = "1. ordinary\n2. IMPORTANT ship it\n3. cleanup";
    let steps = parse_plan_steps(input);
    let tasks = steps_to_tasks(&steps, "plan-1234567890");
    assert_eq!(tasks.len(), 3);
    // Task 2 had "IMPORTANT" keyword.
    assert_eq!(
        tasks[1].priority,
        rullama_core::TaskPriority::High,
        "important step must map to TaskPriority::High",
    );
    // ID suffix tracks step number.
    assert!(tasks[0].id.ends_with("-step-1"));
    assert!(tasks[2].id.ends_with("-step-3"));
}

// ── output_parser: JSON ──────────────────────────────────────────────────

#[derive(Debug, Deserialize, PartialEq)]
struct Review {
    sentiment: String,
    score: i32,
}

#[test]
fn json_parser_extracts_from_markdown_fence_with_language_tag() {
    let parser = JsonOutputParser::<Review>::new();
    let input = "```json\n{\"sentiment\": \"positive\", \"score\": 9}\n```";
    let out = parser.parse(input).unwrap();
    assert_eq!(
        out,
        Review {
            sentiment: "positive".into(),
            score: 9
        }
    );
}

#[test]
fn json_parser_extracts_from_markdown_fence_without_language_tag() {
    let parser = JsonOutputParser::<Review>::new();
    let input = "```\n{\"sentiment\": \"negative\", \"score\": -1}\n```";
    let out = parser.parse(input).unwrap();
    assert_eq!(out.sentiment, "negative");
}

#[test]
fn json_parser_extracts_from_surrounding_prose() {
    let parser = JsonOutputParser::<Review>::new();
    let input = "Sure, here's the JSON: {\"sentiment\":\"x\",\"score\":0} — hope it helps!";
    let out = parser.parse(input).unwrap();
    assert_eq!(out.sentiment, "x");
}

#[test]
fn json_parser_fails_cleanly_on_no_json() {
    let parser = JsonOutputParser::<Review>::new();
    let err = parser
        .parse("Sorry I cannot help with that today")
        .unwrap_err();
    // Must be an Err, not a panic; message must mention JSON so callers can log usefully.
    assert!(
        format!("{err:#}").to_lowercase().contains("json"),
        "error should mention JSON: {err:#}",
    );
}

#[test]
fn json_list_parser_parses_array() {
    let parser = JsonListParser::<Review>::new();
    let input = "[{\"sentiment\":\"a\",\"score\":1},{\"sentiment\":\"b\",\"score\":2}]";
    let out = parser.parse(input).unwrap();
    assert_eq!(out.len(), 2);
    assert_eq!(out[1].score, 2);
}

#[test]
fn format_instructions_are_non_empty_and_mention_json() {
    let p = JsonOutputParser::<Review>::new();
    let instr = p.format_instructions();
    assert!(!instr.is_empty());
    assert!(instr.to_lowercase().contains("json"));
}

// ── output_parser: Regex ─────────────────────────────────────────────────

#[test]
fn regex_parser_extracts_named_captures() {
    let p = RegexOutputParser::new(r"sentiment: (?P<s>\w+), score: (?P<v>\d+)").unwrap();
    let out = p.parse("sentiment: good, score: 7 etc").unwrap();
    assert_eq!(out["s"], "good");
    assert_eq!(out["v"], "7");
}

#[test]
fn regex_parser_fails_when_pattern_does_not_match() {
    let p = RegexOutputParser::new(r"^(?P<head>[A-Z]+)$").unwrap();
    let err = p.parse("no uppercase match here").unwrap_err();
    assert!(format!("{err:#}").to_lowercase().contains("regex"));
}

#[test]
fn regex_parser_rejects_invalid_pattern_at_construction() {
    // Unclosed group — compilation should fail.
    assert!(RegexOutputParser::new(r"(unterminated").is_err());
}

// ── Property tests ───────────────────────────────────────────────────────

proptest! {
    /// Random text never panics the plan parser, and every returned step
    /// has a strictly positive, monotonically increasing `number`.
    #[test]
    fn plan_parser_is_panic_free_and_numbers_are_monotonic(
        input in ".{0,500}",
    ) {
        let steps = parse_plan_steps(&input);
        let mut prev = 0;
        for s in &steps {
            prop_assert!(s.number > prev, "step numbers must strictly increase: {:?}", steps);
            prev = s.number;
        }
    }

    /// Numbered inputs of the form `"N. description"` always produce
    /// exactly that many steps. The parser trims surrounding whitespace
    /// from each description, so compare against the trimmed form.
    #[test]
    fn numbered_lines_extract_one_step_each(
        count in 1usize..15,
        word in "[a-z][a-z ]{3,20}",
    ) {
        let mut lines = String::new();
        for i in 1..=count {
            lines.push_str(&format!("{i}. {word}\n"));
        }
        let steps = parse_plan_steps(&lines);
        prop_assert_eq!(steps.len(), count);
        let expected = word.trim();
        for s in &steps {
            prop_assert_eq!(s.description.as_str(), expected);
        }
    }

    /// JsonOutputParser never panics on arbitrary text. It either succeeds
    /// or returns Err — both acceptable; what matters is no panic.
    #[test]
    fn json_parser_never_panics_on_arbitrary_text(text in ".{0,300}") {
        let p = JsonOutputParser::<serde_json::Value>::new();
        let _ = p.parse(&text);
    }

    /// Any well-formed `{...}` JSON object that deserialises as
    /// `serde_json::Value` roundtrips through the parser when embedded in
    /// surrounding prose.
    #[test]
    fn json_parser_extracts_embedded_objects(
        pre in "[a-z ]{0,30}",
        key in "[a-z][a-z_]{1,10}",
        val in 0i32..1000,
        post in "[a-z .]{0,30}",
    ) {
        let embedded = format!("{pre}{{\"{key}\":{val}}}{post}");
        let p = JsonOutputParser::<serde_json::Value>::new();
        let v = p.parse(&embedded).unwrap();
        prop_assert_eq!(&v[&key], &serde_json::json!(val));
    }

    /// ParsedStep invariant: indent_level is always the string's leading
    /// whitespace length divided by 2 (`indent / 2` in the implementation).
    /// Property-check via construction.
    #[test]
    fn indent_level_equals_leading_spaces_div_2(spaces in 0usize..20) {
        let indent = " ".repeat(spaces);
        let line = format!("{indent}1. task");
        let steps = parse_plan_steps(&line);
        prop_assert_eq!(steps.len(), 1);
        prop_assert_eq!(steps[0].indent_level, spaces / 2);
    }
}

// ── ParsedStep struct-shape smoke ────────────────────────────────────────

#[test]
fn parsed_step_default_shape_is_sensible() {
    // Sanity — not a real parser test, but guards accidental field-reordering.
    let s = ParsedStep {
        number: 1,
        description: "x".into(),
        indent_level: 0,
        is_priority: false,
    };
    assert_eq!(s.number, 1);
    assert!(!s.is_priority);
}
