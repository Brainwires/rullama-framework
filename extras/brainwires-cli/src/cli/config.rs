use crate::config::{ConfigManager, ModelRegistry};
use crate::utils::logger::Logger;
use anyhow::Result;

pub async fn handle_config(
    list: bool,
    get: Option<String>,
    set: Option<Vec<String>>,
) -> Result<()> {
    let config_manager = ConfigManager::new()?;

    if list {
        println!("\nCurrent Configuration:");
        println!("{}", serde_json::to_string_pretty(config_manager.get())?);
    } else if let Some(key) = get {
        let value = serde_json::to_value(config_manager.get())?;
        if let Some(v) = value.get(&key) {
            println!("{}", v);
        } else {
            Logger::warn("Key not found");
        }
    } else if let Some(args) = set {
        // Parse both formats:
        // 1. --set key=value (single arg with =)
        // 2. --set key value (two args)
        let (key, value) = if args.len() == 1 {
            // Format: key=value
            let parts: Vec<&str> = args[0].splitn(2, '=').collect();
            if parts.len() != 2 {
                Logger::warn("Invalid format. Use: --set key=value OR --set key value");
                Logger::warn("Example: --set model=openai/gpt-oss-120b");
                Logger::warn("Example: --set model \"openai/gpt-oss-120b\"");
                return Ok(());
            }
            (parts[0].trim(), parts[1].trim())
        } else if args.len() == 2 {
            // Format: key value
            (args[0].trim(), args[1].trim())
        } else {
            Logger::warn("Invalid format. Use: --set key=value OR --set key value");
            return Ok(());
        };

        let mut config_manager = ConfigManager::new()?;
        let mut updates = crate::config::ConfigUpdates::default();

        // Match key and parse value
        match key {
            "model" => {
                // Validate model exists in backend models table
                match ModelRegistry::find_model(value).await {
                    Ok(Some(model_info)) => {
                        updates.model = Some(value.to_string());
                        config_manager.update(updates);
                        config_manager.save()?;
                        println!("✓ Model set to: {} ({})", value, model_info.name);
                    }
                    Ok(None) => {
                        Logger::warn(format!("Model '{}' not found in available models", value));
                        println!("\nRun 'brainwires models' to see available models");
                        return Ok(());
                    }
                    Err(e) => {
                        Logger::warn(format!("Failed to validate model: {}", e));
                        println!("\nNote: Saving anyway, but the model may not work");
                        updates.model = Some(value.to_string());
                        config_manager.update(updates);
                        config_manager.save()?;
                    }
                }
            }
            "provider" | "provider_type" => {
                // Both "provider" and "provider_type" are accepted — the on-disk
                // schema uses `provider_type` (see ConfigManager), but older docs
                // and muscle memory use `provider`. Aliasing them keeps the setter
                // consistent with what `config --list` emits.
                match crate::providers::ProviderType::from_str_opt(value) {
                    Some(pt) => {
                        updates.provider_type = Some(pt);
                        config_manager.update(updates);
                        config_manager.save()?;
                        println!("✓ Provider set to: {}", pt.as_str());
                    }
                    None => {
                        Logger::warn(format!(
                            "Unknown provider '{}'. Supported: anthropic, openai, google, groq, ollama, brainwires, bedrock, vertex-ai, together, fireworks, minimax",
                            value
                        ));
                        return Ok(());
                    }
                }
            }
            "backend_url" => {
                // Validate URL format and security
                match url::Url::parse(value) {
                    Ok(parsed_url) => {
                        // Require HTTPS for production backends
                        if parsed_url.scheme() != "https" && parsed_url.scheme() != "http" {
                            Logger::warn("Backend URL must use http:// or https:// scheme");
                            return Ok(());
                        }

                        // Warn if not HTTPS (allow for local development)
                        if parsed_url.scheme() == "http" {
                            let host = parsed_url.host_str().unwrap_or("");
                            let is_localhost = host == "localhost"
                                || host == "127.0.0.1"
                                || host == "::1"
                                || host.ends_with(".localhost");

                            if !is_localhost {
                                Logger::warn(
                                    "Warning: Using HTTP for non-localhost backend is insecure!",
                                );
                                Logger::warn("Consider using HTTPS for production backends.");
                            }
                        }

                        // Block private/internal IPs (prevent SSRF-like attacks)
                        let host = parsed_url.host_str().unwrap_or("");
                        if host.starts_with("10.")
                            || host.starts_with("192.168.")
                            || (host.starts_with("172.") && {
                                let parts: Vec<&str> = host.split('.').collect();
                                if parts.len() >= 2 {
                                    parts[1].parse::<u8>().is_ok_and(|n| (16..=31).contains(&n))
                                } else {
                                    false
                                }
                            })
                            || host == "169.254.169.254"
                        {
                            Logger::warn("Backend URL cannot point to private/internal networks");
                            return Ok(());
                        }

                        updates.backend_url = Some(value.to_string());
                        config_manager.update(updates);
                        config_manager.save()?;
                        println!("✓ Backend URL set to: {}", value);
                    }
                    Err(e) => {
                        Logger::warn(format!("Invalid URL format: {}", e));
                        Logger::warn("Example: https://brainwires.studio");
                        return Ok(());
                    }
                }
            }
            "temperature" => match value.parse::<f32>() {
                Ok(temp) => {
                    if !(0.0..=1.0).contains(&temp) {
                        Logger::warn("Temperature should be between 0.0 and 1.0");
                        return Ok(());
                    }
                    updates.temperature = Some(temp);
                    config_manager.update(updates);
                    config_manager.save()?;
                    println!("✓ Temperature set to: {}", temp);
                }
                Err(_) => {
                    Logger::warn("Invalid temperature value. Must be a number between 0.0 and 1.0");
                    return Ok(());
                }
            },
            "max_tokens" => match value.parse::<u32>() {
                Ok(tokens) => {
                    updates.max_tokens = Some(tokens);
                    config_manager.update(updates);
                    config_manager.save()?;
                    println!("✓ Max tokens set to: {}", tokens);
                }
                Err(_) => {
                    Logger::warn("Invalid max_tokens value. Must be a positive integer");
                    return Ok(());
                }
            },
            "permission_mode" => {
                let mode = match value {
                    "auto" => crate::types::agent::PermissionMode::Auto,
                    "full" => crate::types::agent::PermissionMode::Full,
                    "readonly" | "read_only" => crate::types::agent::PermissionMode::ReadOnly,
                    _ => {
                        Logger::warn("Invalid permission mode. Valid values: auto, full, readonly");
                        return Ok(());
                    }
                };
                updates.permission_mode = Some(mode);
                config_manager.update(updates);
                config_manager.save()?;
                println!("✓ Permission mode set to: {:?}", mode);
            }
            _ => {
                Logger::warn(format!("Unknown config key: {}", key));
                Logger::warn(
                    "Valid keys: model, provider (alias: provider_type), backend_url, temperature, max_tokens, permission_mode",
                );
                return Ok(());
            }
        }
    } else {
        Logger::warn("Please specify an action: --list, --get, or --set");
    }

    Ok(())
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_parse_set_args_with_equals() {
        // Simulate parsing "key=value" format
        let args = ["model=openai/gpt-4".to_string()];
        let parts: Vec<&str> = args[0].splitn(2, '=').collect();

        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0], "model");
        assert_eq!(parts[1], "openai/gpt-4");
    }

    #[test]
    fn test_parse_set_args_with_spaces() {
        // Simulate parsing "key value" format
        let args = ["model".to_string(), "openai/gpt-4".to_string()];

        assert_eq!(args.len(), 2);
        assert_eq!(args[0], "model");
        assert_eq!(args[1], "openai/gpt-4");
    }

    #[test]
    fn test_temperature_validation() {
        // Test valid temperatures
        assert!("0.5".parse::<f32>().is_ok());
        assert!("1.0".parse::<f32>().is_ok());
        assert!("0.0".parse::<f32>().is_ok());

        // Test invalid temperature (string)
        assert!("invalid".parse::<f32>().is_err());
    }

    #[test]
    fn test_temperature_range() {
        let temp: f32 = "0.7".parse().unwrap();
        assert!((0.0..=1.0).contains(&temp));

        let temp_low: f32 = "-0.1".parse().unwrap();
        assert!(temp_low < 0.0); // Should fail range check

        let temp_high: f32 = "1.5".parse().unwrap();
        assert!(temp_high > 1.0); // Should fail range check
    }

    #[test]
    fn test_max_tokens_validation() {
        // Test valid max_tokens
        assert!("1000".parse::<usize>().is_ok());
        assert!("4096".parse::<usize>().is_ok());

        // Test invalid max_tokens
        assert!("not_a_number".parse::<usize>().is_err());
        assert!("-100".parse::<usize>().is_err());
    }
}
