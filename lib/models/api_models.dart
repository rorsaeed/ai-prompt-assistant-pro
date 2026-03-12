enum APIProvider { ollama, lmStudio, koboldcpp, google }

extension APIProviderExtension on APIProvider {
  String get displayName {
    switch (this) {
      case APIProvider.ollama:
        return 'Ollama';
      case APIProvider.lmStudio:
        return 'LM Studio';
      case APIProvider.koboldcpp:
        return 'Koboldcpp';
      case APIProvider.google:
        return 'Google';
    }
  }

  static APIProvider fromString(String value) {
    switch (value) {
      case 'Ollama':
        return APIProvider.ollama;
      case 'LM Studio':
        return APIProvider.lmStudio;
      case 'Koboldcpp':
        return APIProvider.koboldcpp;
      case 'Google':
        return APIProvider.google;
      default:
        return APIProvider.ollama;
    }
  }
}

class ModelInfo {
  final String id;
  final String name;

  ModelInfo({required this.id, required this.name});
}

class APIError {
  final String message;
  final String? provider;
  final int? statusCode;

  APIError({required this.message, this.provider, this.statusCode});

  @override
  String toString() {
    if (provider != null) {
      return '$provider error: $message';
    }
    return message;
  }
}

class GoogleFileUpload {
  final String uri;
  final String displayName;
  final String state;

  GoogleFileUpload({
    required this.uri,
    required this.displayName,
    required this.state,
  });

  factory GoogleFileUpload.fromJson(Map<String, dynamic> json) {
    return GoogleFileUpload(
      uri: json['name'] as String,
      displayName: json['displayName'] as String,
      state: json['state'] as String,
    );
  }
}
