using CommunityToolkit.Mvvm.ComponentModel;
using BrainwiresAudio;

namespace AudioDemo.ViewModels;

public partial class MainWindowViewModel : ObservableObject
{
    private readonly AudioService _audioService = new();

    [ObservableProperty]
    private TtsViewModel _ttsTab;

    [ObservableProperty]
    private SttViewModel _sttTab;

    [ObservableProperty]
    private SettingsViewModel _settingsTab;

    public MainWindowViewModel()
    {
        _settingsTab = new SettingsViewModel(_audioService);
        _ttsTab = new TtsViewModel(_audioService);
        _sttTab = new SttViewModel(_audioService);
    }
}
