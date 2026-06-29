use anyhow::Result;

use crate::auth::ApiKeys;
use crate::cli::{AuthAction, AuthArgs};

pub fn run(args: AuthArgs) -> Result<()> {
    let mut keys = ApiKeys::load()?;

    match args.action {
        AuthAction::Set { provider } => {
            eprint!("Enter API key for {provider}: ");
            let key = rpassword::read_password()?;
            if key.trim().is_empty() {
                anyhow::bail!("API key cannot be empty");
            }
            keys.set(&provider, key.trim().to_string())?;
            println!("API key saved for {provider}");
        }
        AuthAction::Show => {
            if keys.keys.is_empty() {
                println!("No API keys configured.");
                println!("Use: agent-chat auth set <provider>");
            } else {
                println!("Configured providers:");
                for provider in keys.keys.keys() {
                    println!("  {provider}");
                }
            }
        }
        AuthAction::Remove { provider } => {
            keys.remove(&provider)?;
            println!("Removed API key for {provider}");
        }
    }

    Ok(())
}
