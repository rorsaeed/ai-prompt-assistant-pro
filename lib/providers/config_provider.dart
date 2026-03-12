import 'package:flutter/material.dart';
import '../models/config.dart';
import '../services/storage_service.dart';
import '../theme/app_theme_data.dart';
import '../theme/stitch_theme.dart';
import '../theme/theme_palettes.dart';

class ConfigProvider with ChangeNotifier {
  final StorageService _storage;
  Config _config = Config.defaultConfig();

  ConfigProvider(this._storage);

  Config get config => _config;
  String get currentProvider => _config.apiProvider;
  ProviderConfig get currentProviderConfig =>
      _config.providers[_config.apiProvider]!;

  Future<void> initialize() async {
    _config = await _storage.loadConfig();
    notifyListeners();
  }

  Future<void> updateProvider(String provider) async {
    _config.apiProvider = provider;
    await _storage.saveConfig(_config);
    notifyListeners();
  }

  Future<void> updateBaseUrl(String provider, String url) async {
    _config.providers[provider]?.apiBaseUrl = url;
    await _storage.saveConfig(_config);
    notifyListeners();
  }

  Future<void> updateApiKey(String? key) async {
    _config.googleApiKey = key;
    await _storage.saveConfig(_config);
    notifyListeners();
  }

  Future<void> updateSelectedModels(
    String provider,
    List<String> models,
  ) async {
    _config.providers[provider]?.selectedModels = models;
    await _storage.saveConfig(_config);
    notifyListeners();
  }

  Future<void> updateKeepAlive(int value) async {
    _config.providers['Ollama']?.keepAlive = value;
    await _storage.saveConfig(_config);
    notifyListeners();
  }

  Future<void> updateUnloadAfterResponse(bool value) async {
    _config.providers['LM Studio']?.unloadAfterResponse = value;
    await _storage.saveConfig(_config);
    notifyListeners();
  }

  Future<void> updateLastSystemPromptName(String name) async {
    _config.lastSystemPromptName = name;
    await _storage.saveConfig(_config);
    notifyListeners();
  }

  List<String> getSelectedModels(String provider) {
    return _config.providers[provider]?.selectedModels ?? [];
  }

  String getBaseUrl(String provider) {
    return _config.providers[provider]?.apiBaseUrl ?? '';
  }

  int? getKeepAlive() {
    return _config.providers['Ollama']?.keepAlive;
  }

  bool getUnloadAfterResponse() {
    return _config.providers['LM Studio']?.unloadAfterResponse ?? false;
  }

  // ─── Theme ───
  ThemeMode get themeMode {
    switch (_config.themeMode) {
      case 'light':
        return ThemeMode.light;
      case 'dark':
        return ThemeMode.dark;
      default:
        return ThemeMode.system;
    }
  }

  String get themeModeString => _config.themeMode;

  String get themeName => _config.themeName;

  AppThemeData get currentPalette =>
      themePalettes[_config.themeName] ?? stitchPalette;

  ThemeData get lightTheme => StitchTheme.buildLight(currentPalette);
  ThemeData get darkTheme => StitchTheme.buildDark(currentPalette);

  Future<void> updateThemeMode(String mode) async {
    _config.themeMode = mode;
    await _storage.saveConfig(_config);
    notifyListeners();
  }

  Future<void> updateThemeName(String name) async {
    _config.themeName = name;
    await _storage.saveConfig(_config);
    notifyListeners();
  }
}
