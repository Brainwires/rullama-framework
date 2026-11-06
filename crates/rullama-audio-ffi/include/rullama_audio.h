/*
 * rullama_audio.h — C ABI of librullama_audio_ffi (crates/rullama-audio-ffi).
 *
 * Local audio for hosts that load a shared library (e.g. Deno.dlopen):
 * microphone capture (cpal), whisper.cpp speech-to-text, WebRTC VAD silence
 * trimming and WAV playback (cpal).
 *
 * All strings are NUL-terminated UTF-8.
 * Return codes: >= 0 ok; -1 error (message via rla_last_error), -2 wrong
 * state, -3 no model loaded, -4 no audio device.
 * Every function is panic-safe and thread-safe. Capture is resampled to
 * 16 kHz mono f32 internally and capped at 120 s.
 */
#ifndef RULLAMA_AUDIO_H
#define RULLAMA_AUDIO_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RLA_OK 0
#define RLA_ERROR (-1)
#define RLA_WRONG_STATE (-2)
#define RLA_NO_MODEL (-3)
#define RLA_NO_DEVICE (-4)

#define RLA_STATE_IDLE 0
#define RLA_STATE_CAPTURING 1
#define RLA_STATE_TRANSCRIBING 2
#define RLA_STATE_DONE 3
#define RLA_STATE_ERROR 4

/* Static version string, e.g. "rullama-audio-ffi 0.12.0". Do not free. */
const char *rla_version(void);

/* Copy the last error (truncated, NUL-terminated) into buf; returns its full
 * byte length (0 if none). buf may be NULL / cap 0 to query the length. */
int32_t rla_last_error(uint8_t *buf, size_t cap);

/* Load or replace the whisper ggml model. 0 ok, -1 error. */
int32_t rla_stt_load(const char *model_path);

/* 1 if a model is loaded, 0 if not. */
int32_t rla_stt_loaded(void);

/* Open the default input device and buffer audio. 0 ok, -2 if not idle/done
 * (starting from done discards the unread transcript), -4 no device. */
int32_t rla_capture_start(void);

/* Stop capture and start transcription on a worker thread (state -> 2).
 * -2 if not capturing; -3 no model (the capture keeps running). */
int32_t rla_capture_stop(void);

/* Stop capture WITHOUT transcribing: write the captured audio (VAD-trimmed
 * like the STT path, 16 kHz mono 16-bit PCM WAV) to wav_path, state -> 0.
 * Returns the ms of audio written (>= 0), -2 if not capturing, -1 on a write
 * error. Needs no model. */
int32_t rla_capture_stop_wav(const char *wav_path);

/* Stop capture / abandon a transcription in flight (its result is dropped),
 * drop audio, state -> 0. Always 0. */
int32_t rla_cancel(void);

/* 0 idle, 1 capturing, 2 transcribing, 3 done (transcript ready),
 * 4 error (see rla_last_error). */
int32_t rla_state(void);

/* Input RMS 0..1 of the last ~100 ms while capturing, else 0. */
float rla_level(void);

/* ms of audio captured in this capture (the final length after a stop, until
 * the next start or cancel). */
uint32_t rla_captured_ms(void);

/* In state 3: copy the transcript (NUL-terminated) and return its byte
 * length; consumes it and goes to 0 only if it fit (len < cap), else leaves
 * it. -2 if not in state 3. In state 4 returns -1 and resets to 0.
 * Captures with under ~300 ms of speech give an empty transcript. */
int32_t rla_take_transcript(uint8_t *buf, size_t cap);

/* Synchronous: decode a WAV (any rate, mono/stereo, 16-bit int or 32-bit
 * float), resample to 16 kHz mono, trim silence, run whisper, copy the text
 * like rla_take_transcript. Returns the byte length, -1 error, -3 no model.
 * Does not touch capture state. */
int32_t rla_transcribe_wav(const char *wav_path, uint8_t *buf, size_t cap);

/* Play a WAV on the default output device; returns at once (0), stopping any
 * playback in progress. -1 bad file, -4 no output device. */
int32_t rla_play_wav(const char *wav_path);

/* Stop playback. Always 0. */
int32_t rla_play_stop(void);

/* 1 while playing, else 0. */
int32_t rla_playing(void);

#ifdef __cplusplus
}
#endif

#endif /* RULLAMA_AUDIO_H */
