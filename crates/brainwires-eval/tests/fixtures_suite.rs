//! Integration test that loads the shipped YAML fixtures and verifies they
//! parse correctly. Does not run them against a real agent (no API keys in
//! CI) — the point is to catch fixture-authoring bugs at PR time and document
//! how to wire a fixture suite to a real [`FixtureRunner`].

use std::path::PathBuf;

use brainwires_eval::fixtures::load_fixtures_from_dir;

#[test]
fn shipped_yaml_fixtures_parse() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let fixtures = load_fixtures_from_dir(&dir).expect("failed to load shipped fixtures");
    assert!(!fixtures.is_empty(), "expected at least one fixture");

    // Spot-check the well-known sample fixtures by name so an accidental
    // rename fails loudly.
    let names: Vec<&str> = fixtures.iter().map(|f| f.name.as_str()).collect();
    assert!(names.contains(&"refactor_small_func"), "names: {names:?}");
    assert!(
        names.contains(&"rejects_prompt_injection"),
        "names: {names:?}"
    );

    for f in &fixtures {
        assert!(!f.messages.is_empty(), "fixture {} has no messages", f.name);
        assert!(
            !f.expected.assertions.is_empty() || !f.expected.tool_sequence.is_empty(),
            "fixture {} has no expected constraints",
            f.name
        );
    }
}
