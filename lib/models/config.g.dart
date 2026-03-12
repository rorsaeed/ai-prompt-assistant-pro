// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'config.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

Config _$ConfigFromJson(Map<String, dynamic> json) => Config(
      apiProvider: json['api_provider'] as String,
      googleApiKey: json['google_api_key'] as String?,
      lastSystemPromptName: json['last_system_prompt_name'] as String?,
      themeMode: json['theme_mode'] as String? ?? 'system',
      themeName: json['theme_name'] as String? ?? 'stitch',
      providers: (json['providers'] as Map<String, dynamic>).map(
        (k, e) =>
            MapEntry(k, ProviderConfig.fromJson(e as Map<String, dynamic>)),
      ),
    );

Map<String, dynamic> _$ConfigToJson(Config instance) => <String, dynamic>{
      'api_provider': instance.apiProvider,
      'google_api_key': instance.googleApiKey,
      'last_system_prompt_name': instance.lastSystemPromptName,
      'theme_mode': instance.themeMode,
      'theme_name': instance.themeName,
      'providers': instance.providers.map((k, e) => MapEntry(k, e.toJson())),
    };

ProviderConfig _$ProviderConfigFromJson(Map<String, dynamic> json) =>
    ProviderConfig(
      apiBaseUrl: json['api_base_url'] as String,
      selectedModels: (json['selected_models'] as List<dynamic>)
          .map((e) => e as String)
          .toList(),
      keepAlive: (json['keep_alive'] as num?)?.toInt(),
      unloadAfterResponse: json['unload_after_response'] as bool?,
    );

Map<String, dynamic> _$ProviderConfigToJson(ProviderConfig instance) =>
    <String, dynamic>{
      'api_base_url': instance.apiBaseUrl,
      'selected_models': instance.selectedModels,
      'keep_alive': instance.keepAlive,
      'unload_after_response': instance.unloadAfterResponse,
    };
