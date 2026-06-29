using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using BrainwiresAudio;

namespace AudioDemo.ViewModels;

public partial class TtsViewModel : ObservableObject
{
    private readonly AudioService _audioService;
    private FfiAudioBuffer? _lastSynthesized;

    public ObservableCollection<FfiProviderInfo> Providers { get; } = new();
    public ObservableCollection<FfiVoice> Voices { get; } = new();

    [ObservableProperty] private FfiProviderInfo? _selectedProvider;
    [ObservableProperty] private FfiVoice? _selectedVoice;
    [ObservableProperty] private string _inputText = "Hello! This is a test of the Brainwires audio framework.";
    [ObservableProperty] private double _speed = 1.0;
    [ObservableProperty] private FfiOutputFormat _selectedFormat = FfiOutputFormat.Wav;
    [ObservableProperty] private bool _isProcessing;
    [ObservableProperty] private string _statusMessage = "Select a provider and configure it in Settings.";

    public FfiOutputFormat[] OutputFormats { get; } =
        [FfiOutputFormat.Wav, FfiOutputFormat.Mp3, FfiOutputFormat.Pcm, FfiOutputFormat.Opus, FfiOutputFormat.Flac];

    public TtsViewModel(AudioService audioService)
    {
        _audioService = audioService;
        foreach (var p in AudioService.ListProviders())
        {
            if (p.HasTts) Providers.Add(p);
        }
    }

    partial void OnSelectedProviderChanged(FfiProviderInfo? value)
    {
        Voices.Clear();
        SelectedVoice = null;
        if (value == null) return;

        _ = LoadVoicesAsync(value.Name);
    }

    private async Task LoadVoicesAsync(string providerName)
    {
        if (!_audioService.IsProviderConfigured(providerName))
        {
            StatusMessage = $"Provider '{providerName}' not configured. Go to Settings first.";
            return;
        }

        try
        {
            var voices = await Task.Run(() => _audioService.ListVoices(providerName));
            Voices.Clear();
            foreach (var v in voices) Voices.Add(v);
            if (Voices.Count > 0) SelectedVoice = Voices[0];
            StatusMessage = $"Loaded {voices.Count} voices.";
        }
        catch (Exception ex)
        {
            StatusMessage = $"Failed to load voices: {ex.Message}";
        }
    }

    [RelayCommand]
    private async Task Synthesize()
    {
        if (SelectedProvider == null || SelectedVoice == null || string.IsNullOrWhiteSpace(InputText))
        {
            StatusMessage = "Select a provider, voice, and enter text.";
            return;
        }

        IsProcessing = true;
        StatusMessage = "Synthesizing...";
        try
        {
            var options = new FfiTtsOptions
            {
                VoiceId = SelectedVoice.Id,
                Speed = (float)Speed,
                OutputFormat = SelectedFormat
            };

            _lastSynthesized = await Task.Run(() =>
                _audioService.Synthesize(SelectedProvider.Name, InputText, options));

            StatusMessage = $"Synthesized {_lastSynthesized.Data.Count} bytes @ {_lastSynthesized.SampleRate}Hz";
        }
        catch (Exception ex)
        {
            StatusMessage = $"Synthesis failed: {ex.Message}";
        }
        finally
        {
            IsProcessing = false;
        }
    }

    [RelayCommand]
    private async Task Play()
    {
        if (_lastSynthesized == null)
        {
            StatusMessage = "Nothing to play. Synthesize first.";
            return;
        }

        IsProcessing = true;
        StatusMessage = "Playing...";
        try
        {
            await Task.Run(() => AudioService.Play(null, _lastSynthesized));
            StatusMessage = "Playback complete.";
        }
        catch (Exception ex)
        {
            StatusMessage = $"Playback failed: {ex.Message}";
        }
        finally
        {
            IsProcessing = false;
        }
    }
}
