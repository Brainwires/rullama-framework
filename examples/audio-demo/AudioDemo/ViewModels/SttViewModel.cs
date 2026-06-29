using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using BrainwiresAudio;

namespace AudioDemo.ViewModels;

public partial class SttViewModel : ObservableObject
{
    private readonly AudioService _audioService;
    private FfiAudioBuffer? _recordedAudio;

    public ObservableCollection<FfiProviderInfo> Providers { get; } = new();

    [ObservableProperty] private FfiProviderInfo? _selectedProvider;
    [ObservableProperty] private double _recordDuration = 5.0;
    [ObservableProperty] private string _languageHint = string.Empty;
    [ObservableProperty] private bool _enableTimestamps;
    [ObservableProperty] private string _transcriptionResult = string.Empty;
    [ObservableProperty] private bool _isProcessing;
    [ObservableProperty] private string _statusMessage = "Select a provider and configure it in Settings.";
    [ObservableProperty] private bool _hasRecording;

    public ObservableCollection<FfiTranscriptSegment> Segments { get; } = new();

    public SttViewModel(AudioService audioService)
    {
        _audioService = audioService;
        foreach (var p in AudioService.ListProviders())
        {
            if (p.HasStt) Providers.Add(p);
        }
    }

    [RelayCommand]
    private async Task Record()
    {
        IsProcessing = true;
        StatusMessage = $"Recording {RecordDuration:F1}s...";
        try
        {
            _recordedAudio = await Task.Run(() => AudioService.Record(null, RecordDuration));
            HasRecording = true;
            StatusMessage = $"Recorded {_recordedAudio.Data.Count} bytes @ {_recordedAudio.SampleRate}Hz";
        }
        catch (Exception ex)
        {
            StatusMessage = $"Recording failed: {ex.Message}";
            HasRecording = false;
        }
        finally
        {
            IsProcessing = false;
        }
    }

    [RelayCommand]
    private async Task PlayRecording()
    {
        if (_recordedAudio == null)
        {
            StatusMessage = "Nothing to play. Record first.";
            return;
        }

        IsProcessing = true;
        StatusMessage = "Playing recording...";
        try
        {
            await Task.Run(() => AudioService.Play(null, _recordedAudio));
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

    [RelayCommand]
    private async Task Transcribe()
    {
        if (SelectedProvider == null)
        {
            StatusMessage = "Select a provider first.";
            return;
        }
        if (_recordedAudio == null)
        {
            StatusMessage = "No audio to transcribe. Record first.";
            return;
        }
        if (!_audioService.IsProviderConfigured(SelectedProvider.Name))
        {
            StatusMessage = $"Provider '{SelectedProvider.Name}' not configured. Go to Settings first.";
            return;
        }

        IsProcessing = true;
        StatusMessage = "Transcribing...";
        Segments.Clear();
        try
        {
            var options = new FfiSttOptions
            {
                Language = string.IsNullOrWhiteSpace(LanguageHint) ? null : LanguageHint,
                Timestamps = EnableTimestamps,
                Prompt = null
            };

            var transcript = await Task.Run(() =>
                _audioService.Transcribe(SelectedProvider.Name, _recordedAudio, options));

            TranscriptionResult = transcript.Text;
            foreach (var seg in transcript.Segments)
            {
                Segments.Add(seg);
            }

            var duration = transcript.DurationSecs.HasValue ? $" ({transcript.DurationSecs:F1}s)" : "";
            StatusMessage = $"Transcription complete{duration}. {transcript.Segments.Count} segments.";
        }
        catch (Exception ex)
        {
            StatusMessage = $"Transcription failed: {ex.Message}";
        }
        finally
        {
            IsProcessing = false;
        }
    }
}
