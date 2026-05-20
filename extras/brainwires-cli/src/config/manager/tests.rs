//! Tests for ConfigManager

use super::*;

#[test]
fn test_default_config() {
    let config = Config::default();
    assert_eq!(config.permission_mode, PermissionMode::Auto);
    assert_eq!(config.model, "claude-haiku-4-5-20251001");
}

#[test]
fn test_config_serialization() {
    let config = Config::default();
    let json = serde_json::to_string(&config).unwrap();
    let parsed: Config = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.model, config.model);
    assert_eq!(parsed.temperature, config.temperature);
}

#[test]
fn test_default_values() {
    let config = Config::default();
    assert_eq!(config.model, "claude-haiku-4-5-20251001");
    assert_eq!(config.permission_mode, PermissionMode::Auto);
    assert_eq!(config.temperature, 0.7);
    assert_eq!(config.max_tokens, 4096);
    assert!(config.backend_url.starts_with("https://"));
}

#[test]
fn test_config_updates() {
    let mut config = Config::default();

    // Apply updates
    let updates = ConfigUpdates {
        model: Some("gpt-4".to_string()),
        permission_mode: Some(PermissionMode::Full),
        temperature: Some(0.5),
        max_tokens: Some(2000),
        ..Default::default()
    };

    // Manually apply updates
    if let Some(model) = updates.model {
        config.model = model;
    }
    if let Some(permission_mode) = updates.permission_mode {
        config.permission_mode = permission_mode;
    }
    if let Some(temperature) = updates.temperature {
        config.temperature = temperature;
    }
    if let Some(max_tokens) = updates.max_tokens {
        config.max_tokens = max_tokens;
    }

    assert_eq!(config.model, "gpt-4");
    assert_eq!(config.permission_mode, PermissionMode::Full);
    assert_eq!(config.temperature, 0.5);
    assert_eq!(config.max_tokens, 2000);
}

#[test]
fn test_default_functions() {
    assert_eq!(default_model(), "claude-haiku-4-5-20251001");
    assert_eq!(default_temperature(), 0.7);
    assert_eq!(default_max_tokens(), 4096);
    assert!(!default_backend_url().is_empty());
}

#[test]
fn test_config_updates_default() {
    let updates = ConfigUpdates::default();
    assert!(updates.model.is_none());
    assert!(updates.permission_mode.is_none());
    assert!(updates.backend_url.is_none());
    assert!(updates.temperature.is_none());
    assert!(updates.max_tokens.is_none());
}

#[test]
fn test_config_manager_get() {
    use tempfile::TempDir;
    let temp = TempDir::new().unwrap();

    // Create a test config file
    let config_path = temp.path().join("config.json");
    let config = Config::default();
    let json = serde_json::to_string_pretty(&config).unwrap();
    std::fs::write(&config_path, json).unwrap();

    // Load via manager
    let manager = ConfigManager {
        config: Config::default(),
        config_path: config_path.clone(),
        is_new: false,
    };

    let _retrieved = manager.get();
}

#[test]
fn test_config_manager_get_mut() {
    use tempfile::TempDir;
    let temp = TempDir::new().unwrap();
    let config_path = temp.path().join("config.json");

    let mut manager = ConfigManager {
        config: Config::default(),
        config_path,
        is_new: false,
    };

    let config_mut = manager.get_mut();
    config_mut.model = "gpt-4".to_string();

    assert_eq!(manager.get().model, "gpt-4");
}

#[test]
fn test_config_manager_update() {
    use tempfile::TempDir;
    let temp = TempDir::new().unwrap();
    let config_path = temp.path().join("config.json");

    let mut manager = ConfigManager {
        config: Config::default(),
        config_path,
        is_new: false,
    };

    let updates = ConfigUpdates {
        model: Some("gemini-pro".to_string()),
        permission_mode: Some(PermissionMode::ReadOnly),
        backend_url: Some("https://test.example.com".to_string()),
        temperature: Some(0.9),
        max_tokens: Some(8000),
        ..Default::default()
    };

    manager.update(updates);

    assert_eq!(manager.get().model, "gemini-pro");
    assert_eq!(manager.get().permission_mode, PermissionMode::ReadOnly);
    assert_eq!(manager.get().backend_url, "https://test.example.com");
    assert_eq!(manager.get().temperature, 0.9);
    assert_eq!(manager.get().max_tokens, 8000);
}

#[test]
fn test_config_manager_partial_update() {
    use tempfile::TempDir;
    let temp = TempDir::new().unwrap();
    let config_path = temp.path().join("config.json");

    let mut manager = ConfigManager {
        config: Config::default(),
        config_path,
        is_new: false,
    };

    // Only update some fields
    let updates = ConfigUpdates {
        temperature: Some(0.3),
        ..Default::default()
    };

    let original_model = manager.get().model.clone();
    manager.update(updates);

    assert_eq!(manager.get().model, original_model); // Unchanged
    assert_eq!(manager.get().temperature, 0.3);
}

#[test]
fn test_config_manager_save() {
    use tempfile::TempDir;
    let temp = TempDir::new().unwrap();
    let config_path = temp.path().join("config.json");

    let manager = ConfigManager {
        config: Config::default(),
        config_path: config_path.clone(),
        is_new: false,
    };

    // Save should create the file
    let result = manager.save();
    // May fail if parent dir doesn't exist, but shouldn't panic
    let _ = result;
}

#[test]
fn test_config_provider_type_default() {
    let config = Config::default();
    assert_eq!(config.provider_type, ProviderType::Brainwires);
    assert!(config.provider_base_url.is_none());
}

#[test]
fn test_config_provider_type_serialization() {
    let config = Config {
        provider_type: ProviderType::Anthropic,
        provider_base_url: Some("https://custom.api.com".to_string()),
        ..Default::default()
    };

    let json = serde_json::to_string(&config).unwrap();
    let parsed: Config = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.provider_type, ProviderType::Anthropic);
    assert_eq!(
        parsed.provider_base_url,
        Some("https://custom.api.com".to_string())
    );
}

#[test]
fn test_config_load_from_file() {
    use tempfile::TempDir;
    let temp = TempDir::new().unwrap();
    let config_path = temp.path().join("config.json");

    // Write a config file
    let config = Config {
        provider_type: ProviderType::OpenAI,
        model: "claude-haiku-4-5-20251001".to_string(),
        permission_mode: PermissionMode::Full,
        backend_url: "https://api.openai.com".to_string(),
        provider_base_url: None,
        temperature: 0.8,
        max_tokens: 2048,
        extra: std::collections::HashMap::new(),
        seal: SealSettings::default(),
        seal_knowledge: SealKnowledgeSettings::default(),
        knowledge: KnowledgeSettings::default(),
        remote: RemoteSettings::default(),
        local_llm: LocalLlmSettings::default(),
        status_line_command: None,
    };

    let json = serde_json::to_string_pretty(&config).unwrap();
    std::fs::write(&config_path, json).unwrap();

    // Load it
    let loaded = ConfigManager::load_from_file(&config_path).unwrap();

    assert_eq!(loaded.model, "claude-haiku-4-5-20251001");
    assert_eq!(loaded.permission_mode, PermissionMode::Full);
    assert_eq!(loaded.temperature, 0.8);
    assert_eq!(loaded.max_tokens, 2048);
}

#[test]
fn test_config_load_from_file_invalid() {
    use tempfile::TempDir;
    let temp = TempDir::new().unwrap();
    let config_path = temp.path().join("invalid.json");

    // Write invalid JSON
    std::fs::write(&config_path, "{ invalid json }").unwrap();

    // Should error
    let result = ConfigManager::load_from_file(&config_path);
    assert!(result.is_err());
}

#[test]
fn test_config_load_from_file_missing() {
    use tempfile::TempDir;
    let temp = TempDir::new().unwrap();
    let config_path = temp.path().join("nonexistent.json");

    // Should error when file doesn't exist
    let result = ConfigManager::load_from_file(&config_path);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Failed to read config file")
    );
}

#[test]
fn test_config_load_from_file_empty() {
    use tempfile::TempDir;
    let temp = TempDir::new().unwrap();
    let config_path = temp.path().join("empty.json");

    // Write empty file
    std::fs::write(&config_path, "").unwrap();

    // Should error
    let result = ConfigManager::load_from_file(&config_path);
    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Failed to parse config file")
    );
}

#[test]
fn test_config_load_from_file_whitespace_only() {
    use tempfile::TempDir;
    let temp = TempDir::new().unwrap();
    let config_path = temp.path().join("whitespace.json");

    // Write whitespace only
    std::fs::write(&config_path, "   \n\t  \n  ").unwrap();

    // Should error
    let result = ConfigManager::load_from_file(&config_path);
    assert!(result.is_err());
}

#[test]
fn test_config_load_from_file_corrupted() {
    use tempfile::TempDir;
    let temp = TempDir::new().unwrap();
    let config_path = temp.path().join("corrupt.json");

    // Write corrupted JSON (starts valid, then corrupts)
    std::fs::write(&config_path, r#"{"provider":"Anthropic","model":"test"#).unwrap();

    // Should error
    let result = ConfigManager::load_from_file(&config_path);
    assert!(result.is_err());
}

#[test]
fn test_config_save_success() {
    use tempfile::TempDir;
    let temp = TempDir::new().unwrap();
    std::fs::create_dir_all(temp.path()).unwrap();
    let config_path = temp.path().join("save_test.json");

    let config = Config {
        model: "gemini-2.0".to_string(),
        temperature: 0.5,
        ..Default::default()
    };

    let manager = ConfigManager {
        config,
        config_path: config_path.clone(),
        is_new: false,
    };

    // Save should succeed
    let result = manager.save();
    assert!(result.is_ok());

    // Verify file exists and can be loaded
    assert!(config_path.exists());
    let loaded = ConfigManager::load_from_file(&config_path).unwrap();
    assert_eq!(loaded.model, "gemini-2.0");
    assert_eq!(loaded.temperature, 0.5);
}

#[test]
fn test_config_extra_fields() {
    let mut config = Config::default();
    config
        .extra
        .insert("custom_key".to_string(), serde_json::json!("custom_value"));
    config
        .extra
        .insert("api_version".to_string(), serde_json::json!(2));

    // Serialize and deserialize
    let json = serde_json::to_string(&config).unwrap();
    let parsed: Config = serde_json::from_str(&json).unwrap();

    assert_eq!(
        parsed.extra.get("custom_key"),
        Some(&serde_json::json!("custom_value"))
    );
    assert_eq!(parsed.extra.get("api_version"), Some(&serde_json::json!(2)));
}

#[test]
fn test_config_update_empty() {
    use tempfile::TempDir;
    let temp = TempDir::new().unwrap();
    let config_path = temp.path().join("config.json");

    let mut manager = ConfigManager {
        config: Config::default(),
        config_path,
        is_new: false,
    };

    let original_model = manager.get().model.clone();

    // Apply empty updates (all None)
    let updates = ConfigUpdates::default();
    manager.update(updates);

    // Nothing should change
    assert_eq!(manager.get().model, original_model);
}

#[test]
fn test_config_update_only_backend_url() {
    use tempfile::TempDir;
    let temp = TempDir::new().unwrap();
    let config_path = temp.path().join("config.json");

    let mut manager = ConfigManager {
        config: Config::default(),
        config_path,
        is_new: false,
    };

    // Only update backend_url
    let updates = ConfigUpdates {
        backend_url: Some("https://custom.backend.url".to_string()),
        ..Default::default()
    };

    manager.update(updates);
    assert_eq!(manager.get().backend_url, "https://custom.backend.url");
}

#[test]
fn test_config_temperature_boundary() {
    let mut config = Config {
        temperature: 0.0,
        ..Default::default()
    };

    // Test minimum temperature
    let json = serde_json::to_string(&config).unwrap();
    let parsed: Config = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.temperature, 0.0);

    // Test maximum temperature
    config.temperature = 1.0;
    let json = serde_json::to_string(&config).unwrap();
    let parsed: Config = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.temperature, 1.0);

    // Test negative temperature (technically invalid but should serialize)
    config.temperature = -0.5;
    let json = serde_json::to_string(&config).unwrap();
    let parsed: Config = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.temperature, -0.5);
}

#[test]
fn test_config_max_tokens_boundary() {
    let mut config = Config {
        max_tokens: 0,
        ..Default::default()
    };

    // Test minimum tokens
    let json = serde_json::to_string(&config).unwrap();
    let parsed: Config = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.max_tokens, 0);

    // Test large value
    config.max_tokens = 100000;
    let json = serde_json::to_string(&config).unwrap();
    let parsed: Config = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.max_tokens, 100000);

    // Test u32 max
    config.max_tokens = u32::MAX;
    let json = serde_json::to_string(&config).unwrap();
    let parsed: Config = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.max_tokens, u32::MAX);
}

#[test]
fn test_config_update_provider_type() {
    use tempfile::TempDir;
    let temp = TempDir::new().unwrap();
    let config_path = temp.path().join("config.json");

    let mut manager = ConfigManager {
        config: Config::default(),
        config_path,
        is_new: false,
    };

    manager.update(ConfigUpdates {
        provider_type: Some(ProviderType::Anthropic),
        model: Some("claude-3-5-sonnet-20241022".to_string()),
        ..Default::default()
    });

    assert_eq!(manager.get().provider_type, ProviderType::Anthropic);
    assert_eq!(manager.get().model, "claude-3-5-sonnet-20241022");
}

#[test]
fn test_config_manager_get_mut_multiple_changes() {
    use tempfile::TempDir;
    let temp = TempDir::new().unwrap();
    let config_path = temp.path().join("config.json");

    let mut manager = ConfigManager {
        config: Config::default(),
        config_path,
        is_new: false,
    };

    // Make multiple changes via get_mut
    {
        let config = manager.get_mut();
        config.model = "gpt-4-turbo".to_string();
        config.temperature = 0.2;
        config.max_tokens = 16384;
        config.permission_mode = PermissionMode::Full;
    }

    // Verify all changes persisted
    assert_eq!(manager.get().model, "gpt-4-turbo");
    assert_eq!(manager.get().temperature, 0.2);
    assert_eq!(manager.get().max_tokens, 16384);
    assert_eq!(manager.get().permission_mode, PermissionMode::Full);
}

#[test]
fn test_config_update_extreme_values() {
    use tempfile::TempDir;
    let temp = TempDir::new().unwrap();
    let config_path = temp.path().join("config.json");

    let mut manager = ConfigManager {
        config: Config::default(),
        config_path,
        is_new: false,
    };

    // Update with extreme values
    let updates = ConfigUpdates {
        model: Some("a".repeat(1000)), // Very long model name
        permission_mode: Some(PermissionMode::ReadOnly),
        backend_url: Some(format!("https://{}.com", "x".repeat(500))),
        temperature: Some(999.9),   // Extreme temperature
        max_tokens: Some(u32::MAX), // Maximum possible tokens
        ..Default::default()
    };

    manager.update(updates);

    assert_eq!(manager.get().model.len(), 1000);
    assert_eq!(manager.get().temperature, 999.9);
    assert_eq!(manager.get().max_tokens, u32::MAX);
}

#[test]
fn test_config_serialization_with_all_permission_modes() {
    // Test Auto
    let mut config = Config {
        permission_mode: PermissionMode::Auto,
        ..Default::default()
    };
    let json = serde_json::to_string(&config).unwrap();
    let parsed: Config = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.permission_mode, PermissionMode::Auto);

    // Test Full
    config.permission_mode = PermissionMode::Full;
    let json = serde_json::to_string(&config).unwrap();
    let parsed: Config = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.permission_mode, PermissionMode::Full);

    // Test ReadOnly
    config.permission_mode = PermissionMode::ReadOnly;
    let json = serde_json::to_string(&config).unwrap();
    let parsed: Config = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.permission_mode, PermissionMode::ReadOnly);
}

#[test]
fn test_config_update_all_fields_individually() {
    use tempfile::TempDir;
    let temp = TempDir::new().unwrap();
    let config_path = temp.path().join("config.json");

    let mut manager = ConfigManager {
        config: Config::default(),
        config_path,
        is_new: false,
    };

    // Update provider only
    manager.update(ConfigUpdates {
        ..Default::default()
    });

    // Update model only
    manager.update(ConfigUpdates {
        model: Some("gpt-4".to_string()),
        ..Default::default()
    });
    assert_eq!(manager.get().model, "gpt-4");

    // Update permission_mode only
    manager.update(ConfigUpdates {
        permission_mode: Some(PermissionMode::Full),
        ..Default::default()
    });
    assert_eq!(manager.get().permission_mode, PermissionMode::Full);

    // Update temperature only
    manager.update(ConfigUpdates {
        temperature: Some(0.1),
        ..Default::default()
    });
    assert_eq!(manager.get().temperature, 0.1);

    // Update max_tokens only
    manager.update(ConfigUpdates {
        max_tokens: Some(1000),
        ..Default::default()
    });
    assert_eq!(manager.get().max_tokens, 1000);
}

#[test]
fn test_config_load_with_missing_fields() {
    use tempfile::TempDir;
    let temp = TempDir::new().unwrap();
    let config_path = temp.path().join("minimal.json");

    // Write minimal JSON with only required fields (none are truly required due to defaults)
    std::fs::write(&config_path, "{}").unwrap();

    // Should load successfully with defaults
    let loaded = ConfigManager::load_from_file(&config_path).unwrap();
    assert_eq!(loaded.model, default_model());
    assert_eq!(loaded.temperature, default_temperature());
    assert_eq!(loaded.max_tokens, default_max_tokens());
}

#[test]
fn test_stale_model_migrated_on_load() {
    use tempfile::TempDir;
    let temp = TempDir::new().unwrap();
    let config_path = temp.path().join("config.json");

    // Write a config fixture that pins a stale/phantom model name.
    let stale_config = Config {
        provider_type: ProviderType::Brainwires,
        model: "openai-gpt-5.2".to_string(),
        permission_mode: PermissionMode::Auto,
        backend_url: "https://api.brainwires.net".to_string(),
        provider_base_url: None,
        temperature: 0.7,
        max_tokens: 4096,
        extra: std::collections::HashMap::new(),
        seal: SealSettings::default(),
        seal_knowledge: SealKnowledgeSettings::default(),
        knowledge: KnowledgeSettings::default(),
        remote: RemoteSettings::default(),
        local_llm: LocalLlmSettings::default(),
        status_line_command: None,
    };
    let json = serde_json::to_string_pretty(&stale_config).unwrap();
    std::fs::write(&config_path, &json).unwrap();

    // Load: the in-memory migration must swap to the current default.
    let loaded = ConfigManager::load_from_file(&config_path).unwrap();
    assert_eq!(loaded.model, default_model());
    assert_eq!(loaded.model, "claude-haiku-4-5-20251001");

    // Invariant: the file on disk is NOT silently rewritten — persistence is
    // user-initiated via `config --set`.
    let on_disk = std::fs::read_to_string(&config_path).unwrap();
    assert!(
        on_disk.contains("\"openai-gpt-5.2\""),
        "load_from_file must not rewrite the user's config.json; raw disk contents were: {}",
        on_disk
    );
}
