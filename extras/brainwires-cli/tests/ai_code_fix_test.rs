//! AI-evaluated code fix integration test
//!
//! This test validates brainwires' core capability: autonomous code fixing.
//! It takes a buggy project, asks the AI to fix it, and evaluates the result.
//!
//! ## Setup
//! Create a `.env` file in the project root with:
//! ```
//! TEST_API_KEY=your_api_key_here
//! ```
//! Or copy from template: `cp .env.example .env`

use assert_cmd::Command;
use chrono::Utc;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

/// Load environment variables from .env file
fn load_env() {
    // Load .env from project root
    let _ = dotenvy::dotenv();
}

/// Create a test session file for authentication
fn create_test_session(temp_dir: &Path, api_key: &str) -> std::io::Result<()> {
    // Create the data directory structure
    let data_dir = temp_dir.join(".local/share/brainwires");
    fs::create_dir_all(&data_dir)?;

    // Determine backend URL based on key prefix
    // bw_dev_* keys use dev server, others use production
    let backend = if api_key.starts_with("bw_dev_") {
        "https://dev.brainwires.net"
    } else {
        "https://brainwires.studio"
    };

    println!(
        "🔑 Using backend: {} (key prefix: {})",
        backend,
        &api_key[..10]
    );

    // Create a test session JSON
    let session = serde_json::json!({
        "user": {
            "user_id": "test-user-id",
            "username": "testuser",
            "display_name": "Test User",
            "role": "basic"
        },
        "supabase": {
            "url": "https://test.supabase.co",
            "anon_key": "test-anon-key"
        },
        "key_name": "test-key",
        "api_key": api_key,
        "backend": backend,
        "authenticated_at": Utc::now().to_rfc3339()
    });

    // Write session file
    let session_file = data_dir.join("session.json");
    fs::write(session_file, serde_json::to_string_pretty(&session)?)?;

    Ok(())
}

/// Copy a directory recursively
fn copy_dir_all(src: impl AsRef<Path>, dst: impl AsRef<Path>) -> std::io::Result<()> {
    fs::create_dir_all(&dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        if ty.is_dir() {
            copy_dir_all(entry.path(), dst.as_ref().join(entry.file_name()))?;
        } else {
            fs::copy(entry.path(), dst.as_ref().join(entry.file_name()))?;
        }
    }
    Ok(())
}

/// Helper to create test command with clean environment
fn brainwires_cmd(temp_dir: &TempDir) -> Command {
    let mut cmd = Command::cargo_bin("brainwires").expect("Failed to find brainwires binary");
    cmd.env("HOME", temp_dir.path());
    cmd.env("XDG_CONFIG_HOME", temp_dir.path().join(".config"));
    cmd.env("XDG_DATA_HOME", temp_dir.path().join(".local/share"));

    // Set API key if available - use both TEST_API_KEY and BRAINWIRES_API_KEY
    if let Ok(api_key) = env::var("TEST_API_KEY") {
        cmd.env("BRAINWIRES_API_KEY", &api_key);
        cmd.env("TEST_API_KEY", api_key);
    }

    cmd
}

/// Get path to test fixture
fn get_fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

/// Run cargo test in a directory and check if it passes
fn run_cargo_test(project_dir: &Path) -> bool {
    let output = std::process::Command::new("cargo")
        .arg("test")
        .arg("--quiet")
        .current_dir(project_dir)
        .output()
        .expect("Failed to run cargo test");

    output.status.success()
}

/// Read a file to string
fn read_file(path: &Path) -> String {
    fs::read_to_string(path).expect("Failed to read file")
}

#[test]
#[ignore] // Requires API key and AI backend
fn test_fix_calculator_bug() {
    // Load environment variables from .env file
    load_env();

    // Skip if no API key
    if env::var("TEST_API_KEY").is_err() {
        eprintln!("Skipping test: TEST_API_KEY not set");
        eprintln!("Create a .env file with: TEST_API_KEY=your_key");
        eprintln!("Or run: cp .env.example .env");
        return;
    }

    let temp_dir = TempDir::new().expect("Failed to create temp dir");

    // Get API key from environment
    let api_key = env::var("TEST_API_KEY").expect("TEST_API_KEY must be set");

    // Create test session for authentication
    create_test_session(temp_dir.path(), &api_key).expect("Failed to create test session");

    // Copy buggy calculator project to temp directory
    let fixture_path = get_fixture_path("buggy_calculator");
    let project_path = temp_dir.path().join("calculator_project");
    copy_dir_all(&fixture_path, &project_path).expect("Failed to copy fixture");

    println!("📁 Test project copied to: {}", project_path.display());

    // Verify tests fail initially
    println!("🧪 Running initial tests (should FAIL)...");
    let tests_pass_before = run_cargo_test(&project_path);
    assert!(!tests_pass_before, "Tests should fail before the fix");
    println!("✓ Confirmed: Tests fail as expected");

    // Read the original buggy code
    let lib_path = project_path.join("src/lib.rs");
    let original_code = read_file(&lib_path);

    // Run brainwires to fix the bug
    println!("\n🤖 Running brainwires to fix the bug...");

    // Use a model available on the current backend
    // Dev keys (bw_dev_*) use dev.brainwires.net which has different models
    let model = if api_key.starts_with("bw_dev_") {
        "llama-3.3-70b-versatile" // Available on dev server (Groq)
    } else {
        "claude-3-5-sonnet-20241022" // Available on production (Claude)
    };

    println!("📦 Using model: {}", model);

    let mut cmd = brainwires_cmd(&temp_dir);
    cmd.current_dir(&project_path)
        .arg("task")
        .arg("--model")
        .arg(model)
        .arg(
            "IMPORTANT: You must USE YOUR FILE EDITING TOOLS to modify the actual src/lib.rs file. \
              \
              Task: There is a bug in the calculator's divide function in src/lib.rs. \
              The divide() method uses multiplication (*) instead of division (/). \
              \
              Steps:\
              1. Read src/lib.rs to see the bug\
              2. Edit src/lib.rs to change 'Ok(a * b)' to 'Ok(a / b)' in the divide function\
              3. Run 'cargo test' to verify all tests pass\
              \
              You MUST actually modify the file using your tools, not just explain the fix.",
        );

    let output = cmd.output().expect("Failed to run brainwires");

    // Check if brainwires executed successfully
    if !output.status.success() {
        panic!(
            "brainwires command failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let response = String::from_utf8_lossy(&output.stdout);
    println!("📝 AI Response:\n{}\n", response);

    // Read the modified code
    let fixed_code = read_file(&lib_path);

    // Check if code was actually modified
    assert_ne!(original_code, fixed_code, "Code should have been modified");
    println!("✓ Code was modified");

    // Verify tests now pass
    println!("\n🧪 Running tests after fix (should PASS)...");
    let tests_pass_after = run_cargo_test(&project_path);
    assert!(tests_pass_after, "Tests should pass after the fix");
    println!("✓ All tests pass!");

    // Verify the specific fix was applied
    assert!(
        fixed_code.contains("a / b") || fixed_code.contains("a/b"),
        "The fix should change multiplication (*) to division (/)"
    );
    println!("✓ Correct operator (/) found in code");

    // AI Evaluation
    println!("\n🔍 Evaluating fix quality with AI...");
    evaluate_fix(&temp_dir, &original_code, &fixed_code, &response);
}

/// Use AI to evaluate if the fix is correct and well-implemented
fn evaluate_fix(temp_dir: &TempDir, original: &str, fixed: &str, ai_response: &str) {
    let evaluation_prompt = format!(
        r#"You are a code review expert. Evaluate the following code fix:

ORIGINAL CODE:
```rust
{}
```

FIXED CODE:
```rust
{}
```

AI'S EXPLANATION:
{}

Evaluate the fix on these criteria:
1. CORRECTNESS: Does it fix the bug correctly?
2. SAFETY: Does it introduce any new bugs or issues?
3. QUALITY: Is the code clean and well-written?
4. EXPLANATION: Did the AI explain the fix clearly?

Respond with ONLY a JSON object in this exact format:
{{
  "correctness": <1-10>,
  "safety": <1-10>,
  "quality": <1-10>,
  "explanation_clarity": <1-10>,
  "overall_pass": <true/false>,
  "reasoning": "Brief explanation of your evaluation"
}}
"#,
        original, fixed, ai_response
    );

    // Use appropriate evaluation model based on backend
    let eval_model = if env::var("TEST_API_KEY")
        .unwrap_or_default()
        .starts_with("bw_dev_")
    {
        "llama-3.1-8b-instant" // Fast, cheap model on dev server
    } else {
        "claude-3-haiku-20240307" // Fast, cheap model on production
    };

    let mut cmd = brainwires_cmd(temp_dir);
    cmd.arg("task")
        .arg("--model")
        .arg(eval_model)
        .arg(evaluation_prompt);

    let eval_output = cmd.output().expect("Failed to run evaluation");
    let eval_response = String::from_utf8_lossy(&eval_output.stdout);

    println!("📊 AI Evaluation:\n{}\n", eval_response);

    // Try to parse the evaluation JSON
    // Extract JSON from the response (it might be wrapped in other text)
    if let Some(json_start) = eval_response.find('{')
        && let Some(json_end) = eval_response.rfind('}')
    {
        let json_str = &eval_response[json_start..=json_end];

        match serde_json::from_str::<serde_json::Value>(json_str) {
            Ok(evaluation) => {
                println!("✓ Evaluation parsed successfully");

                if let Some(overall_pass) = evaluation.get("overall_pass").and_then(|v| v.as_bool())
                {
                    assert!(
                        overall_pass,
                        "AI evaluation failed: {}",
                        evaluation
                            .get("reasoning")
                            .and_then(|v| v.as_str())
                            .unwrap_or("No reasoning provided")
                    );
                    println!("✅ AI evaluation: PASS");
                } else {
                    eprintln!("⚠️  Warning: Could not parse overall_pass from evaluation");
                }

                // Print scores
                for metric in &["correctness", "safety", "quality", "explanation_clarity"] {
                    if let Some(score) = evaluation.get(metric).and_then(|v| v.as_i64()) {
                        println!("  {}: {}/10", metric, score);
                    }
                }
            }
            Err(e) => {
                eprintln!("⚠️  Warning: Failed to parse evaluation JSON: {}", e);
                eprintln!("   Response: {}", json_str);
            }
        }
    }
}

#[test]
#[ignore] // Requires API key
fn test_fix_with_multiple_bugs() {
    load_env();
    // TODO: Create a fixture with multiple related bugs
    // and test that brainwires can fix them all
}

#[test]
#[ignore] // Requires API key
fn test_fix_preserves_working_code() {
    load_env();
    // TODO: Verify that brainwires doesn't break working parts
    // while fixing bugs
}
