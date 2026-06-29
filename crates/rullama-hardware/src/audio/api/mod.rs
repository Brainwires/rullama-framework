/// Azure Cognitive Services speech-to-text API client.
pub mod azure_stt;
/// Azure Cognitive Services text-to-speech API client.
pub mod azure_tts;
/// Cartesia text-to-speech API client.
pub mod cartesia_tts;
/// Deepgram speech-to-text API client.
pub mod deepgram_stt;
/// Deepgram text-to-speech API client.
pub mod deepgram_tts;
/// ElevenLabs speech-to-text API client.
pub mod elevenlabs_stt;
/// ElevenLabs text-to-speech API client.
pub mod elevenlabs_tts;
/// Fish Audio speech-to-text (ASR) API client.
pub mod fish_stt;
/// Fish Audio text-to-speech API client.
pub mod fish_tts;
/// Google Cloud text-to-speech API client.
pub mod google_tts;
/// Murf AI text-to-speech API client.
pub mod murf_tts;
/// OpenAI Responses API speech-to-text client.
pub mod openai_responses_stt;
/// OpenAI Responses API text-to-speech client.
pub mod openai_responses_tts;
/// OpenAI speech-to-text API client.
pub mod openai_stt;
/// OpenAI text-to-speech API client.
pub mod openai_tts;

pub use azure_stt::AzureStt;
pub use azure_tts::AzureTts;
pub use cartesia_tts::CartesiaTts;
pub use deepgram_stt::DeepgramStt;
pub use deepgram_tts::DeepgramTts;
pub use elevenlabs_stt::ElevenLabsStt;
pub use elevenlabs_tts::ElevenLabsTts;
pub use fish_stt::FishStt;
pub use fish_tts::FishTts;
pub use google_tts::GoogleTts;
pub use murf_tts::MurfTts;
pub use openai_responses_stt::OpenAiResponsesStt;
pub use openai_responses_tts::OpenAiResponsesTts;
pub use openai_stt::OpenAiStt;
pub use openai_tts::OpenAiTts;
