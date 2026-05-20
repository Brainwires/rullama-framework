//! CLI-local helpers for the framework's `ProviderType`.
//!
//! `ProviderType` is defined in `brainwires-provider` and re-exported via
//! `crate::providers`, so we cannot add methods to it. These free functions
//! layer CLI-specific concerns on top: env-var discovery, UI summaries,
//! and the chat-capable provider subset used by the picker.

use brainwires::providers::ProviderType;

/// The subset of providers that make sense as chat/code-assistant backends.
///
/// Speech providers (ElevenLabs, Deepgram, Azure, Fish, Cartesia, Murf) are
/// excluded — they're real providers in the framework, just not useful for
/// an interactive coding CLI.
pub const CHAT_PROVIDERS: &[ProviderType] = &[
    ProviderType::Brainwires,
    ProviderType::Anthropic,
    ProviderType::OpenAI,
    ProviderType::Google,
    ProviderType::Groq,
    ProviderType::Ollama,
    ProviderType::Bedrock,
    ProviderType::VertexAI,
    ProviderType::Together,
    ProviderType::Fireworks,
    ProviderType::Anyscale,
    ProviderType::OpenAiResponses,
    ProviderType::MiniMax,
    ProviderType::Custom,
];

/// Environment variable that supplies an API key for this provider, if any.
///
/// Returns `None` for providers that don't use a single API key (Ollama,
/// Bedrock, VertexAI use different credential chains).
pub fn env_var_name(p: ProviderType) -> Option<&'static str> {
    match p {
        ProviderType::Anthropic => Some("ANTHROPIC_API_KEY"),
        ProviderType::OpenAI | ProviderType::OpenAiResponses => Some("OPENAI_API_KEY"),
        ProviderType::Google => Some("GEMINI_API_KEY"),
        ProviderType::Groq => Some("GROQ_API_KEY"),
        ProviderType::Brainwires => Some("BRAINWIRES_API_KEY"),
        ProviderType::Together => Some("TOGETHER_API_KEY"),
        ProviderType::Fireworks => Some("FIREWORKS_API_KEY"),
        ProviderType::Anyscale => Some("ANYSCALE_API_KEY"),
        ProviderType::MiniMax => Some("MINIMAX_API_KEY"),
        ProviderType::Ollama
        | ProviderType::Bedrock
        | ProviderType::VertexAI
        | ProviderType::Custom
        | ProviderType::ElevenLabs
        | ProviderType::Deepgram
        | ProviderType::Azure
        | ProviderType::Fish
        | ProviderType::Cartesia
        | ProviderType::Murf => None,
    }
}

/// Human-readable one-line summary of a provider for pickers and help text.
pub fn summary(p: ProviderType) -> &'static str {
    match p {
        ProviderType::Brainwires => "Brainwires SaaS — managed backend with built-in routing",
        ProviderType::Anthropic => "Anthropic — Claude (Sonnet, Opus, Haiku)",
        ProviderType::OpenAI => "OpenAI — GPT-5, GPT-4o, o-series",
        ProviderType::OpenAiResponses => "OpenAI Responses API — newer tool-calling surface",
        ProviderType::Google => "Google — Gemini 2.x",
        ProviderType::Groq => "Groq — fast Llama / Mixtral inference",
        ProviderType::Ollama => "Ollama — local models (no API key required)",
        ProviderType::Bedrock => "Amazon Bedrock — Claude via AWS (uses AWS credential chain)",
        ProviderType::VertexAI => "Google Vertex AI — Claude via GCP (uses ADC)",
        ProviderType::Together => "Together AI — open-weights hosted",
        ProviderType::Fireworks => "Fireworks AI — open-weights hosted",
        ProviderType::Anyscale => "Anyscale — open-weights hosted",
        ProviderType::MiniMax => "MiniMax — Chinese frontier model",
        ProviderType::Custom => "Custom — user-defined OpenAI-compatible endpoint",
        ProviderType::ElevenLabs => "ElevenLabs — text-to-speech",
        ProviderType::Deepgram => "Deepgram — speech-to-text",
        ProviderType::Azure => "Azure — speech services",
        ProviderType::Fish => "Fish Audio — speech",
        ProviderType::Cartesia => "Cartesia — speech",
        ProviderType::Murf => "Murf AI — speech",
    }
}

/// Detect a configured provider from environment variables.
///
/// Priority order: explicit `BRAINWIRES_PROVIDER` wins, otherwise we scan
/// known API-key env vars in a stable order. The first hit wins.
///
/// Returns `Some((provider, env_var_name))` so callers can log which var
/// was used.
pub fn detect_provider_from_env() -> Option<(ProviderType, &'static str)> {
    if let Ok(name) = std::env::var("BRAINWIRES_PROVIDER")
        && let Some(p) = ProviderType::from_str_opt(&name)
    {
        return Some((p, "BRAINWIRES_PROVIDER"));
    }

    // Stable priority: Brainwires SaaS first (if the user has a Brainwires
    // key in env they presumably want it), then direct providers in
    // order of popularity for coding tasks.
    let candidates: &[(ProviderType, &'static str)] = &[
        (ProviderType::Brainwires, "BRAINWIRES_API_KEY"),
        (ProviderType::Anthropic, "ANTHROPIC_API_KEY"),
        (ProviderType::OpenAI, "OPENAI_API_KEY"),
        (ProviderType::Google, "GEMINI_API_KEY"),
        (ProviderType::Google, "GOOGLE_API_KEY"),
        (ProviderType::Groq, "GROQ_API_KEY"),
        (ProviderType::Together, "TOGETHER_API_KEY"),
        (ProviderType::Fireworks, "FIREWORKS_API_KEY"),
        (ProviderType::MiniMax, "MINIMAX_API_KEY"),
    ];

    for (p, var) in candidates {
        if std::env::var(var).is_ok() {
            return Some((*p, var));
        }
    }

    // Ollama: presence of OLLAMA_HOST hints the user runs a local Ollama.
    if std::env::var("OLLAMA_HOST").is_ok() {
        return Some((ProviderType::Ollama, "OLLAMA_HOST"));
    }

    None
}

/// A human-facing hint for how to configure credentials for a provider.
///
/// Used in error messages when the active provider has no credentials.
pub fn credential_hint(p: ProviderType) -> String {
    match p {
        ProviderType::Brainwires => {
            "Run: brainwires auth login  (or set BRAINWIRES_API_KEY)".to_string()
        }
        ProviderType::Ollama => {
            "Start Ollama locally (default http://localhost:11434) or set OLLAMA_HOST.".to_string()
        }
        ProviderType::Bedrock => {
            "Configure AWS credentials: `aws configure` or set AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY."
                .to_string()
        }
        ProviderType::VertexAI => {
            "Configure GCP credentials: `gcloud auth application-default login` or set GOOGLE_APPLICATION_CREDENTIALS."
                .to_string()
        }
        other => match env_var_name(other) {
            Some(var) => format!(
                "Run: brainwires auth login --provider {}  (or set {}=…)",
                other.as_str(),
                var
            ),
            None => format!("Run: brainwires auth login --provider {}", other.as_str()),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_var_name_covers_common_providers() {
        assert_eq!(
            env_var_name(ProviderType::Anthropic),
            Some("ANTHROPIC_API_KEY")
        );
        assert_eq!(env_var_name(ProviderType::OpenAI), Some("OPENAI_API_KEY"));
        assert_eq!(env_var_name(ProviderType::Ollama), None);
        assert_eq!(env_var_name(ProviderType::Bedrock), None);
    }

    #[test]
    fn summary_is_non_empty_for_all_chat_providers() {
        for p in CHAT_PROVIDERS {
            assert!(!summary(*p).is_empty(), "missing summary for {:?}", p);
        }
    }

    #[test]
    fn credential_hint_mentions_brainwires_login_for_brainwires() {
        assert!(credential_hint(ProviderType::Brainwires).contains("brainwires auth login"));
    }

    #[test]
    fn credential_hint_mentions_env_var_for_anthropic() {
        let hint = credential_hint(ProviderType::Anthropic);
        assert!(hint.contains("--provider anthropic"));
        assert!(hint.contains("ANTHROPIC_API_KEY"));
    }
}
