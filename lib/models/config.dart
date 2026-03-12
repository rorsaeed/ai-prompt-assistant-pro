import 'package:json_annotation/json_annotation.dart';

part 'config.g.dart';

@JsonSerializable(explicitToJson: true)
class Config {
  @JsonKey(name: 'api_provider')
  String apiProvider;

  @JsonKey(name: 'google_api_key')
  String? googleApiKey;

  @JsonKey(name: 'last_system_prompt_name')
  String? lastSystemPromptName;

  @JsonKey(name: 'theme_mode', defaultValue: 'system')
  String themeMode;

  @JsonKey(name: 'theme_name', defaultValue: 'stitch')
  String themeName;

  Map<String, ProviderConfig> providers;

  Config({
    required this.apiProvider,
    this.googleApiKey,
    this.lastSystemPromptName,
    this.themeMode = 'system',
    this.themeName = 'stitch',
    required this.providers,
  });

  factory Config.fromJson(Map<String, dynamic> json) => _$ConfigFromJson(json);

  Map<String, dynamic> toJson() => _$ConfigToJson(this);

  factory Config.defaultConfig() {
    return Config(
      apiProvider: 'Ollama',
      googleApiKey: null,
      lastSystemPromptName: 'Default Image-to-Prompt',
      providers: {
        'Ollama': ProviderConfig(
          apiBaseUrl: 'http://localhost:11434',
          selectedModels: [],
          keepAlive: -1,
        ),
        'LM Studio': ProviderConfig(
          apiBaseUrl: 'http://localhost:1234',
          selectedModels: [],
          unloadAfterResponse: false,
        ),
        'Koboldcpp': ProviderConfig(
          apiBaseUrl: 'http://localhost:5001',
          selectedModels: [],
        ),
        'Google': ProviderConfig(
          apiBaseUrl: 'https://generativelanguage.googleapis.com',
          selectedModels: [],
        ),
      },
    );
  }
}

@JsonSerializable()
class ProviderConfig {
  @JsonKey(name: 'api_base_url')
  String apiBaseUrl;

  @JsonKey(name: 'selected_models')
  List<String> selectedModels;

  @JsonKey(name: 'keep_alive')
  int? keepAlive; // Ollama only

  @JsonKey(name: 'unload_after_response')
  bool? unloadAfterResponse; // LM Studio only

  ProviderConfig({
    required this.apiBaseUrl,
    required this.selectedModels,
    this.keepAlive,
    this.unloadAfterResponse,
  });

  factory ProviderConfig.fromJson(Map<String, dynamic> json) =>
      _$ProviderConfigFromJson(json);

  Map<String, dynamic> toJson() => _$ProviderConfigToJson(this);
}
