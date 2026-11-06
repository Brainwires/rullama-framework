/*
 * rullama_audio.h — C ABI of librullama_audio_ffi (crates/rullama-audio-ffi).
 *
 * Local audio for hosts that load a shared library (e.g. Deno.dlopen):
 * microphone capture (cpal), whisper.cpp speech-to-text, WebRTC VAD silence
 * trimming and WAV playback (cpal).
 *
 * Strings passed in are NUL-terminated UTF-8 paths. Return codes: >= 0 ok;
 * -1 error (message via rla_last_error), -2 wrong state, -3 no model loaded,
 * -4 no audio device.
 *
 * Ownership: the library never returns memory for the caller to free.
 * rla_version returns a static string; all other output (error text,
 * transcripts) is copied into a caller-owned (buf, cap) buffer: at most
 * cap - 1 bytes plus a NUL, never splitting a UTF-8 character. The return
 * value is the FULL byte length without the NUL, so a result >= cap means the
 * output was truncated; buf may be NULL with cap 0 to query the length. Use the
 * returned length rather than searching for the NUL.
 *
 * Threading: every function catches Rust panics at the boundary (a panic
 * returns -1, or the neutral value for functions without an error code, and
 * sets an "internal panic" last error) and all state is behind mutexes, so
 * any function may be called from any thread (e.g. Deno "nonblocking" calls).
 * There is one process-wide engine. The last-error slot is also process-wide,
 * not per thread: read it right after the failing call, and serialize calls
 * if exact attribution matters. rla_capture_start and rla_play_wav may block
 * for up to 5 s while a device opens; rla_transcribe_wav blocks for the whole
 * transcription.
 *
 * Lifetime: there is no init function (state is created on first use). Call
 * rla_shutdown before unloading the library; after it no library thread or
 * audio stream is running (wait for any rla_transcribe_wav call on another
 * thread to return first; rla_shutdown makes it return promptly).
 *
 * Limits: a capture buffers at most 120 s (at the device rate, mono; it is
 * resampled to 16 kHz when stopped); WAV files over 256 MiB are rejected.
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

/* Copy the most recent error message from any thread (truncated,
 * NUL-terminated) into buf; returns its full byte length (0 if none). buf may
 * be NULL / cap 0 to query the length. */
int32_t rla_last_error(uint8_t *buf, size_t cap);

/* Load or replace the whisper ggml model (loaded once, reused for every
 * transcription; the path must be an existing regular file). 0 ok, -1 error. */
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

/* Stop capture (the stream is closed before this returns) or abort a
 * transcription in flight (its result is dropped), drop audio, state -> 0.
 * Always 0. */
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

/* Release everything before unloading the library: stop capture and
 * playback, abort and join the transcription worker, abort rla_transcribe_wav
 * calls in flight on other threads (they return -1), unload the model. The
 * library stays usable (load a model again). Make no other rla_* calls
 * concurrently with it. Always 0. */
int32_t rla_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* RULLAMA_AUDIO_H */
