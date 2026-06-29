namespace BrainwiresAudio;

/// <summary>
/// High-level wrapper over the UniFFI-generated FFI bindings.
/// Manages provider lifecycle and provides a clean C# API.
/// </summary>
public sealed class AudioService : IDisposable
{
    private readonly Dictionary<string, ulong> _providerHandles = new();
    private bool _disposed;

    /// <summary>
    /// Configure (or reconfigure) a provider with an API key.
    /// </summary>
    public void ConfigureProvider(string name, string apiKey, string? region = null)
    {
        if (_providerHandles.TryGetValue(name, out var oldHandle))
        {
            AudioDemoFfiMethods.DropProvider(oldHandle);
            _providerHandles.Remove(name);
        }
        var handle = AudioDemoFfiMethods.CreateProvider(name, apiKey, region);
        _providerHandles[name] = handle;
    }

    /// <summary>
    /// Check if a provider is configured.
    /// </summary>
    public bool IsProviderConfigured(string name) => _providerHandles.ContainsKey(name);

    /// <summary>
    /// List all supported providers and their capabilities.
    /// </summary>
    public static List<FfiProviderInfo> ListProviders() => AudioDemoFfiMethods.ListProviders();

    /// <summary>
    /// List available voices for a TTS provider.
    /// </summary>
    public List<FfiVoice> ListVoices(string provider)
    {
        var handle = GetHandle(provider);
        return AudioDemoFfiMethods.TtsListVoices(handle);
    }

    /// <summary>
    /// Synthesize text to audio.
    /// </summary>
    public FfiAudioBuffer Synthesize(string provider, string text, FfiTtsOptions options)
    {
        var handle = GetHandle(provider);
        return AudioDemoFfiMethods.TtsSynthesize(handle, text, options);
    }

    /// <summary>
    /// Transcribe audio to text.
    /// </summary>
    public FfiTranscript Transcribe(string provider, FfiAudioBuffer audio, FfiSttOptions options)
    {
        var handle = GetHandle(provider);
        return AudioDemoFfiMethods.SttTranscribe(handle, audio, options);
    }

    /// <summary>
    /// List audio input (microphone) devices.
    /// </summary>
    public static List<FfiAudioDevice> ListInputDevices()
        => AudioDemoFfiMethods.AudioListInputDevices();

    /// <summary>
    /// List audio output (speaker) devices.
    /// </summary>
    public static List<FfiAudioDevice> ListOutputDevices()
        => AudioDemoFfiMethods.AudioListOutputDevices();

    /// <summary>
    /// Record audio from the default microphone.
    /// </summary>
    public static FfiAudioBuffer Record(string? deviceId, double durationSecs)
        => AudioDemoFfiMethods.AudioRecord(deviceId, durationSecs);

    /// <summary>
    /// Play audio through the default speaker.
    /// </summary>
    public static void Play(string? deviceId, FfiAudioBuffer buffer)
        => AudioDemoFfiMethods.AudioPlay(deviceId, buffer);

    private ulong GetHandle(string provider)
    {
        if (_providerHandles.TryGetValue(provider, out var handle))
            return handle;
        throw new InvalidOperationException($"Provider '{provider}' is not configured. Call ConfigureProvider first.");
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var handle in _providerHandles.Values)
        {
            AudioDemoFfiMethods.DropProvider(handle);
        }
        _providerHandles.Clear();
    }
}
