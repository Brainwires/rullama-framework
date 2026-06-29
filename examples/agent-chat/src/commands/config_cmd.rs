use anyhow::Result;

use crate::cli::{ConfigAction, ConfigArgs};
use crate::config::ChatConfig;

pub fn run(args: ConfigArgs) -> Result<()> {
    let mut config = ChatConfig::load()?;

    match args.action {
        ConfigAction::List => {
            println!("default_provider = {}", config.default_provider);
            println!("default_model    = {}", config.default_model);
            println!(
                "system_prompt    = {}",
                config.system_prompt.as_deref().unwrap_or("(none)")
            );
            println!("permission_mode  = {}", config.permission_mode);
            println!("max_tokens       = {}", config.max_tokens);
            println!("temperature      = {}", config.temperature);
        }
        ConfigAction::Get { key } => match config.get(&key) {
            Some(val) => println!("{val}"),
            None => {
                eprintln!("Unknown key: {key}");
                std::process::exit(1);
            }
        },
        ConfigAction::Set { key, value } => {
            config.set(&key, &value)?;
            println!("Set {key} = {value}");
        }
    }

    Ok(())
}
