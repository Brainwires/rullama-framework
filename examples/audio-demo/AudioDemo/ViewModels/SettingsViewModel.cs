using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using BrainwiresAudio;

namespace AudioDemo.ViewModels;

public partial class ProviderSettingViewModel : ObservableObject
{
    public FfiProviderInfo Info { get; }
    private readonly AudioService _audioService;

    [ObservableProperty] private string _apiKey = string.Empty;
    [ObservableProperty] private string _region = string.Empty;
    [ObservableProperty] private string _status = "Not configured";
    [ObservableProperty] private bool _isConnected;

    public bool RequiresRegion => Info.RequiresRegion;

    public ProviderSettingViewModel(FfiProviderInfo info, AudioService audioService)
    {
        Info = info;
        _audioService = audioService;
    }

    [RelayCommand]
    private async Task TestConnection()
    {
        if (string.IsNullOrWhiteSpace(ApiKey))
        {
            Status = "API key required";
            IsConnected = false;
            return;
        }

        try
        {
            Status = "Connecting...";
            await Task.Run(() =>
            {
                _audioService.ConfigureProvider(
                    Info.Name,
                    ApiKey,
                    RequiresRegion ? Region : null);

                if (Info.HasTts)
                {
                    _audioService.ListVoices(Info.Name);
                }
            });
            Status = "Connected";
            IsConnected = true;
        }
        catch (Exception ex)
        {
            Status = $"Error: {ex.Message}";
            IsConnected = false;
        }
    }
}

public partial class SettingsViewModel : ObservableObject
{
    public ObservableCollection<ProviderSettingViewModel> Providers { get; } = new();

    public SettingsViewModel(AudioService audioService)
    {
        foreach (var info in AudioService.ListProviders())
        {
            Providers.Add(new ProviderSettingViewModel(info, audioService));
        }
    }
}
