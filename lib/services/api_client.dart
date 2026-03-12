import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:dio/dio.dart';
import 'package:mime/mime.dart';
import '../models/api_models.dart';
import '../models/message.dart';

class APIClient {
  final String provider;
  final String baseUrl;
  final String? googleApiKey;
  final int? ollamaKeepAlive;
  final bool unloadAfterResponse;

  late final Dio _dio;

  APIClient({
    required this.provider,
    required this.baseUrl,
    this.googleApiKey,
    this.ollamaKeepAlive,
    this.unloadAfterResponse = false,
  }) {
    _dio = Dio(
      BaseOptions(
        connectTimeout: const Duration(seconds: 300),
        receiveTimeout: const Duration(seconds: 300),
      ),
    );
  }

  /// Get available models from the provider
  Future<List<ModelInfo>> getAvailableModels() async {
    try {
      switch (provider) {
        case 'Ollama':
          return await _getOllamaModels();
        case 'LM Studio':
          return await _getLMStudioModels();
        case 'Koboldcpp':
          return await _getKoboldcppModels();
        case 'Google':
          return _getGoogleModels();
        default:
          throw APIError(message: 'Unknown provider: $provider');
      }
    } catch (e) {
      throw APIError(message: e.toString(), provider: provider);
    }
  }

  Future<List<ModelInfo>> _getOllamaModels() async {
    final response = await _dio.get(
      '$baseUrl/api/tags',
      options: Options(receiveTimeout: const Duration(seconds: 10)),
    );

    final models = response.data['models'] as List;
    return models.map((m) {
      final name = m['name'] as String;
      return ModelInfo(id: name, name: name);
    }).toList();
  }

  Future<List<ModelInfo>> _getLMStudioModels() async {
    final response = await _dio.get(
      '$baseUrl/v1/models',
      options: Options(receiveTimeout: const Duration(seconds: 10)),
    );

    final data = response.data['data'] as List;
    return data
        .where((m) => !(m['owned_by'] as String).contains('koboldcpp'))
        .map((m) {
      final id = m['id'] as String;
      return ModelInfo(id: id, name: id);
    }).toList();
  }

  Future<List<ModelInfo>> _getKoboldcppModels() async {
    final response = await _dio.get(
      '$baseUrl/v1/models',
      options: Options(receiveTimeout: const Duration(seconds: 10)),
    );

    final data = response.data['data'] as List;
    return data
        .where((m) => (m['owned_by'] as String).contains('koboldcpp'))
        .map((m) {
      final id = m['id'] as String;
      return ModelInfo(id: id, name: id);
    }).toList();
  }

  List<ModelInfo> _getGoogleModels() {
    return [
      ModelInfo(id: 'gemini-3-pro-preview', name: 'gemini-3-pro-preview'),
      ModelInfo(id: 'gemini-3-flash-preview', name: 'gemini-3-flash-preview'),
      ModelInfo(id: 'gemini-2.5-pro', name: 'gemini-2.5-pro'),
      ModelInfo(id: 'gemini-2.5-flash', name: 'gemini-2.5-flash'),
      ModelInfo(id: 'gemini-2.5-flash-lite', name: 'gemini-2.5-flash-lite'),
    ];
  }

  /// Generate chat response with streaming
  Stream<String> generateChatResponse({
    required String model,
    required List<Message> messages,
    List<String>? imagePaths,
    List<String>? videoPaths,
  }) async* {
    try {
      switch (provider) {
        case 'Ollama':
          yield* _generateOllama(model, messages, imagePaths);
          break;
        case 'LM Studio':
          yield* _generateLMStudio(model, messages, imagePaths);
          break;
        case 'Koboldcpp':
          yield* _generateKoboldcpp(model, messages, imagePaths);
          break;
        case 'Google':
          yield* _generateGoogle(model, messages, imagePaths, videoPaths);
          break;
        default:
          throw APIError(message: 'Unknown provider: $provider');
      }

      // Unload model if needed (LM Studio only)
      if (provider == 'LM Studio' && unloadAfterResponse) {
        await _unloadLMStudioModel();
      }
    } catch (e) {
      if (e is APIError) rethrow;
      throw APIError(message: e.toString(), provider: provider);
    }
  }

  Stream<String> _generateOllama(
    String model,
    List<Message> messages,
    List<String>? imagePaths,
  ) async* {
    final formattedMessages = await _formatMessagesOllama(messages, imagePaths);

    final payload = {
      'model': model,
      'messages': formattedMessages,
      'stream': true,
      if (ollamaKeepAlive != null) 'keep_alive': ollamaKeepAlive,
    };

    final response = await _dio.post(
      '$baseUrl/api/chat',
      data: payload,
      options: Options(
        responseType: ResponseType.stream,
        headers: {'Content-Type': 'application/json'},
      ),
    );

    final stream = response.data.stream as Stream<List<int>>;
    await for (final chunk in stream) {
      final lines = utf8.decode(chunk).split('\n');
      for (final line in lines) {
        if (line.trim().isEmpty) continue;
        try {
          final json = jsonDecode(line);
          if (json['message'] != null && json['message']['content'] != null) {
            final content = json['message']['content'] as String;
            yield content;
          }
        } catch (_) {
          continue;
        }
      }
    }
  }

  Stream<String> _generateLMStudio(
    String model,
    List<Message> messages,
    List<String>? imagePaths,
  ) async* {
    final formattedMessages = await _formatMessagesOpenAI(messages, imagePaths);

    final payload = {
      'model': model,
      'messages': formattedMessages,
      'stream': true,
      'temperature': 0.7,
    };

    final response = await _dio.post(
      '$baseUrl/v1/chat/completions',
      data: payload,
      options: Options(
        responseType: ResponseType.stream,
        headers: {'Content-Type': 'application/json'},
      ),
    );

    yield* _parseOpenAIStream(response.data.stream);
  }

  Stream<String> _generateKoboldcpp(
    String model,
    List<Message> messages,
    List<String>? imagePaths,
  ) async* {
    final formattedMessages = await _formatMessagesOpenAI(messages, imagePaths);

    final payload = {
      'model': model,
      'messages': formattedMessages,
      'stream': true,
      'temperature': 0.7,
    };

    final response = await _dio.post(
      '$baseUrl/v1/chat/completions',
      data: payload,
      options: Options(
        responseType: ResponseType.stream,
        headers: {'Content-Type': 'application/json'},
      ),
    );

    yield* _parseOpenAIStream(response.data.stream);
  }

  Stream<String> _generateGoogle(
    String model,
    List<Message> messages,
    List<String>? imagePaths,
    List<String>? videoPaths,
  ) async* {
    // Upload videos first
    final videoFiles = <Map<String, String>>[];
    if (videoPaths != null && videoPaths.isNotEmpty) {
      for (final path in videoPaths) {
        final uri = await uploadVideoToGoogle(path);
        final mimeType = _getMimeType(path);
        videoFiles.add({'uri': uri, 'mimeType': mimeType});
      }
    }

    final formattedMessages = await _formatMessagesGoogle(
      messages,
      imagePaths,
      videoFiles.isNotEmpty ? videoFiles : null,
    );

    // Retry logic for Google
    int retries = 0;
    while (retries < 3) {
      try {
        final payload = {
          'contents': formattedMessages,
          'generationConfig': {
            'temperature': 0.9,
            'topK': 1,
            'topP': 1,
            'maxOutputTokens': 8192,
          },
        };

        final url =
            'https://generativelanguage.googleapis.com/v1beta/models/$model:streamGenerateContent?key=$googleApiKey&alt=sse';

        final response = await _dio.post(
          url,
          data: payload,
          options: Options(
            responseType: ResponseType.stream,
            headers: {'Content-Type': 'application/json'},
          ),
        );

        final stream = response.data.stream as Stream<List<int>>;
        String buffer = '';
        await for (final chunk in stream) {
          buffer += utf8.decode(chunk);
          final lines = buffer.split('\n');
          // Keep the last potentially incomplete line in the buffer
          buffer = lines.removeLast();

          for (final line in lines) {
            final trimmed = line.trim();
            if (trimmed.isEmpty || !trimmed.startsWith('data: ')) continue;

            final data = trimmed.substring(6).trim();
            if (data.isEmpty) continue;

            try {
              final json = jsonDecode(data);

              // Check for content moderation
              if (json['promptFeedback'] != null &&
                  json['promptFeedback']['blockReason'] != null) {
                throw APIError(
                  message:
                      'Content blocked: ${json['promptFeedback']['blockReason']}',
                  provider: 'Google',
                );
              }

              // Extract text
              if (json['candidates'] != null && json['candidates'].isNotEmpty) {
                final candidate = json['candidates'][0];
                if (candidate['content'] != null &&
                    candidate['content']['parts'] != null) {
                  for (final part in candidate['content']['parts']) {
                    if (part['text'] != null) {
                      // Google thought models return thought parts with 'thought': true
                      if (part['thought'] == true) {
                        yield '<think>${part['text']}</think>';
                      } else {
                        yield part['text'] as String;
                      }
                    }
                  }
                }
              }
            } catch (e) {
              if (e is APIError) rethrow;
              continue;
            }
          }
        }
        return;
      } catch (e) {
        retries++;
        if (retries >= 3) rethrow;
        await Future.delayed(Duration(seconds: retries * 2));
      }
    }
  }

  Stream<String> _parseOpenAIStream(Stream<List<int>> stream) async* {
    await for (final chunk in stream) {
      final text = utf8.decode(chunk);
      final lines = text.split('\n');

      for (final line in lines) {
        if (line.trim().isEmpty) continue;
        if (!line.startsWith('data: ')) continue;

        final data = line.substring(6).trim();
        if (data == '[DONE]') continue;

        try {
          final json = jsonDecode(data);
          if (json['choices'] != null && json['choices'].isNotEmpty) {
            final delta = json['choices'][0]['delta'];
            if (delta != null && delta['content'] != null) {
              yield delta['content'] as String;
            }
          }
        } catch (_) {
          continue;
        }
      }
    }
  }

  Future<List<Map<String, dynamic>>> _formatMessagesOllama(
    List<Message> messages,
    List<String>? imagePaths,
  ) async {
    final formatted = <Map<String, dynamic>>[];

    for (final msg in messages) {
      final messageData = <String, dynamic>{
        'role': msg.role,
        'content': msg.role == 'assistant' ? msg.displayContent : msg.content,
      };

      // Add images to the last user message
      if (msg.role == 'user' &&
          msg == messages.lastWhere((m) => m.role == 'user')) {
        if (imagePaths != null && imagePaths.isNotEmpty) {
          final images = <String>[];
          for (final path in imagePaths) {
            images.add(await _encodeImage(path));
          }
          messageData['images'] = images;
        }
      }

      formatted.add(messageData);
    }

    return formatted;
  }

  Future<List<Map<String, dynamic>>> _formatMessagesOpenAI(
    List<Message> messages,
    List<String>? imagePaths,
  ) async {
    final formatted = <Map<String, dynamic>>[];

    for (final msg in messages) {
      if (msg.role == 'user' &&
          msg == messages.lastWhere((m) => m.role == 'user') &&
          imagePaths != null &&
          imagePaths.isNotEmpty) {
        // Multi-part content for images
        final parts = <Map<String, dynamic>>[];

        parts.add({'type': 'text', 'text': msg.content});

        for (final path in imagePaths) {
          final base64 = await _encodeImage(path);
          final mimeType = _getMimeType(path);
          parts.add({
            'type': 'image_url',
            'image_url': {'url': 'data:$mimeType;base64,$base64'},
          });
        }

        formatted.add({'role': msg.role, 'content': parts});
      } else {
        formatted.add({
          'role': msg.role,
          'content': msg.role == 'assistant' ? msg.displayContent : msg.content,
        });
      }
    }

    return formatted;
  }

  Future<List<Map<String, dynamic>>> _formatMessagesGoogle(
    List<Message> messages,
    List<String>? imagePaths,
    List<Map<String, String>>? videoFiles,
  ) async {
    final formatted = <Map<String, dynamic>>[];
    String? systemPrompt;

    // Extract system prompt
    for (final msg in messages) {
      if (msg.role == 'system') {
        systemPrompt = msg.content;
        break;
      }
    }

    // Process messages
    String? lastRole;
    List<Map<String, dynamic>>? currentParts;

    for (final msg in messages) {
      if (msg.role == 'system') continue;

      final role = msg.role == 'assistant' ? 'model' : 'user';

      // Merge consecutive messages with same role
      if (role == lastRole && currentParts != null) {
        currentParts.add({
          'text': msg.role == 'assistant' ? msg.displayContent : msg.content,
        });
      } else {
        if (currentParts != null) {
          formatted.add({'role': lastRole, 'parts': currentParts});
        }
        currentParts = [
          {
            'text': msg.role == 'assistant' ? msg.displayContent : msg.content,
          },
        ];
        lastRole = role;
      }
    }

    // Add last message
    if (currentParts != null) {
      // Add images to first user message
      if (lastRole == 'user' &&
          (imagePaths != null && imagePaths.isNotEmpty ||
              videoFiles != null && videoFiles.isNotEmpty)) {
        if (imagePaths != null) {
          for (final path in imagePaths) {
            final base64 = await _encodeImage(path);
            final mimeType = _getMimeType(path);
            currentParts.add({
              'inline_data': {'mime_type': mimeType, 'data': base64},
            });
          }
        }

        if (videoFiles != null) {
          for (final vf in videoFiles) {
            currentParts.add({
              'file_data': {
                'file_uri': vf['uri'],
                'mime_type': vf['mimeType'],
              },
            });
          }
        }
      }

      formatted.add({'role': lastRole, 'parts': currentParts});
    }

    // Prepend system prompt to first user message
    if (systemPrompt != null && formatted.isNotEmpty) {
      final firstUserMsg = formatted.firstWhere(
        (m) => m['role'] == 'user',
        orElse: () => {},
      );
      if (firstUserMsg.isNotEmpty) {
        final parts = firstUserMsg['parts'] as List;
        parts.insert(0, {'text': systemPrompt});
      }
    }

    return formatted;
  }

  Future<String> uploadVideoToGoogle(String videoPath) async {
    final file = File(videoPath);
    final fileName = file.uri.pathSegments.last;
    final mimeType = _getMimeType(videoPath);

    // Start resumable upload
    final startResponse = await _dio.post(
      'https://generativelanguage.googleapis.com/upload/v1beta/files?key=$googleApiKey',
      data: jsonEncode({
        'file': {'display_name': fileName},
      }),
      options: Options(
        headers: {
          'X-Goog-Upload-Protocol': 'resumable',
          'X-Goog-Upload-Command': 'start',
          'X-Goog-Upload-Header-Content-Type': mimeType,
          'Content-Type': 'application/json',
        },
      ),
    );

    final uploadUrl = startResponse.headers['x-goog-upload-url']![0];

    // Upload file
    final fileBytes = await file.readAsBytes();
    final uploadResponse = await _dio.put(
      uploadUrl,
      data: Stream.fromIterable([fileBytes]),
      options: Options(
        headers: {
          'X-Goog-Upload-Offset': '0',
          'X-Goog-Upload-Command': 'upload, finalize',
          'Content-Length': fileBytes.length.toString(),
        },
      ),
    );

    final fileData = uploadResponse.data['file'];
    final filResourceName = fileData['name'] as String;
    final fileUri = fileData['uri'] as String;

    // Poll for processing completion
    for (int i = 0; i < 150; i++) {
      await Future.delayed(const Duration(seconds: 2));

      final statusResponse = await _dio.get(
        'https://generativelanguage.googleapis.com/v1beta/$filResourceName?key=$googleApiKey',
      );

      final state = statusResponse.data['state'] as String;
      if (state == 'ACTIVE') {
        return fileUri;
      } else if (state == 'FAILED') {
        throw APIError(message: 'Video processing failed', provider: 'Google');
      }
    }

    throw APIError(message: 'Video processing timeout', provider: 'Google');
  }

  Future<void> _unloadLMStudioModel() async {
    try {
      if (Platform.isWindows) {
        await Process.run('lms', ['unload', '--all']);
      } else {
        await Process.run('/usr/local/bin/lms', ['unload', '--all']);
      }
    } catch (e) {
      // Silent fail - not critical
      // Failed to unload LM Studio model: $e
    }
  }

  static Future<String> _encodeImage(String imagePath) async {
    final file = File(imagePath);
    final bytes = await file.readAsBytes();
    return base64Encode(bytes);
  }

  static String _getMimeType(String filePath) {
    final mimeType = lookupMimeType(filePath);
    return mimeType ?? 'application/octet-stream';
  }

  /// Extracts thinking content from text containing <think>...</think> tags.
  /// Returns a record with (thinking, content) separated.
  static ({String thinking, String content}) parseThinkingContent(
      String rawText) {
    final thinkRegex = RegExp(r'<think>(.*?)</think>', dotAll: true);
    final matches = thinkRegex.allMatches(rawText);

    if (matches.isEmpty) {
      return (thinking: '', content: rawText);
    }

    final thinkingParts = <String>[];
    for (final match in matches) {
      thinkingParts.add(match.group(1)?.trim() ?? '');
    }

    // Also handle unclosed <think> tag (still streaming thinking)
    final content = rawText.replaceAll(thinkRegex, '').trim();
    final thinking = thinkingParts.join('\n');

    return (thinking: thinking, content: content);
  }

  /// Check if the text is currently inside an unclosed <think> tag (still thinking)
  static bool isCurrentlyThinking(String rawText) {
    final openCount = '<think>'.allMatches(rawText).length;
    final closeCount = '</think>'.allMatches(rawText).length;
    return openCount > closeCount;
  }
}
