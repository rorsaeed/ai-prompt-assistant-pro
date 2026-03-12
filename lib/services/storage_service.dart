import 'dart:convert';
import 'dart:io';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;
import 'package:uuid/uuid.dart';
import '../models/config.dart';
import '../models/message.dart';
import '../models/conversation.dart';
import 'database_service.dart';

class StorageService {
  static const _uuid = Uuid();

  final DatabaseService _database;

  late String _dataDir;
  late String _tempImagesDir;
  late String _tempVideosDir;

  StorageService(this._database);

  Future<void> initialize() async {
    final appDocDir = await getApplicationDocumentsDirectory();
    _dataDir = path.join(appDocDir.path, 'ai_prompt_assistant', 'data');
    _tempImagesDir = path.join(
      appDocDir.path,
      'ai_prompt_assistant',
      'temp_images',
    );
    _tempVideosDir = path.join(
      appDocDir.path,
      'ai_prompt_assistant',
      'temp_videos',
    );

    // Create directories
    await Directory(_dataDir).create(recursive: true);
    await Directory(_tempImagesDir).create(recursive: true);
    await Directory(_tempVideosDir).create(recursive: true);
  }

  // ===== Config Management =====

  Future<Config> loadConfig() async {
    final file = File(path.join(_dataDir, 'config.json'));
    if (!await file.exists()) {
      final config = Config.defaultConfig();
      await saveConfig(config);
      return config;
    }

    final json = jsonDecode(await file.readAsString());
    return Config.fromJson(json);
  }

  Future<void> saveConfig(Config config) async {
    final file = File(path.join(_dataDir, 'config.json'));
    await file.writeAsString(
      const JsonEncoder.withIndent('  ').convert(config.toJson()),
    );
  }

  // ===== System Prompts Management =====

  /// Loads custom system prompts. Returns a map of prompt name -> {content, category}.
  Future<Map<String, Map<String, String>>> loadSystemPrompts() async {
    final file = File(path.join(_dataDir, 'system_prompts.json'));
    if (!await file.exists()) {
      return {};
    }

    final json = jsonDecode(await file.readAsString());
    final result = <String, Map<String, String>>{};

    // Migration: support old flat format (name -> content string)
    for (final entry in (json as Map<String, dynamic>).entries) {
      if (entry.value is String) {
        result[entry.key] = {'content': entry.value, 'category': 'Custom'};
      } else if (entry.value is Map) {
        final map = Map<String, dynamic>.from(entry.value as Map);
        result[entry.key] = {
          'content': map['content'] as String? ?? '',
          'category': map['category'] as String? ?? 'Custom',
        };
      }
    }
    return result;
  }

  /// Loads predefined prompts organized by category.
  /// Returns a map of category -> {prompt name -> content}.
  Future<Map<String, Map<String, String>>> loadPredefinedPrompts() async {
    final jsonString = await rootBundle.loadString(
      'assets/predefined_prompts.json',
    );
    final json = jsonDecode(jsonString) as Map<String, dynamic>;
    final result = <String, Map<String, String>>{};

    for (final categoryEntry in json.entries) {
      if (categoryEntry.value is Map) {
        result[categoryEntry.key] = Map<String, String>.from(
          categoryEntry.value as Map,
        );
      }
    }
    return result;
  }

  Future<void> saveSystemPrompt(
    String name,
    String content, {
    String category = 'Custom',
  }) async {
    final prompts = await loadSystemPrompts();
    prompts[name] = {'content': content, 'category': category};

    final file = File(path.join(_dataDir, 'system_prompts.json'));
    await file.writeAsString(
      const JsonEncoder.withIndent('  ').convert(prompts),
    );
  }

  Future<void> deleteSystemPrompt(String name) async {
    final prompts = await loadSystemPrompts();
    prompts.remove(name);

    final file = File(path.join(_dataDir, 'system_prompts.json'));
    await file.writeAsString(
      const JsonEncoder.withIndent('  ').convert(prompts),
    );
  }

  // ===== Conversations Management =====

  Future<List<Conversation>> listConversations({String? folderId}) async {
    return await _database.listConversations(folderId: folderId);
  }

  Future<List<Conversation>> getAllConversations() async {
    return await _database.getAllConversations();
  }

  Future<List<Conversation>> searchConversations(String query) async {
    return await _database.searchConversations(query);
  }

  Future<List<Message>> loadConversation(String id) async {
    final conversation = await _database.loadConversation(id);
    return conversation?.messages ?? [];
  }

  Future<String> saveConversation(
    String? chatId,
    List<Message> messages,
  ) async {
    if (chatId == null || chatId.isEmpty) {
      // Create new conversation with auto-generated title
      final firstUserMsg = messages.firstWhere(
        (m) => m.role == 'user',
        orElse: () => Message(
          role: 'user',
          content: 'New Chat',
          displayContent: 'New Chat',
          id: _uuid.v4(),
        ),
      );

      final title = _sanitizeTitle(firstUserMsg.content);
      final newId = await _database.createConversation(title);
      await _database.updateConversation(newId, messages);
      return newId;
    } else {
      // Update existing conversation
      await _database.updateConversation(chatId, messages);
      return chatId;
    }
  }

  Future<void> renameConversation(String id, String newTitle) async {
    await _database.renameConversation(id, newTitle);
  }

  Future<void> deleteConversation(String id) async {
    await _database.deleteConversation(id);
  }

  Future<void> moveConversationToFolder(String id, String? folderId) async {
    await _database.moveConversationToFolder(id, folderId);
  }


  // ===== File Uploads =====

  Future<({String path, String name})> saveUploadedFile(
    String originalPath,
    String originalName,
  ) async {
    final uuid = _uuid.v4();
    final filename = '${uuid}_$originalName';
    final destPath = path.join(_tempImagesDir, filename);

    final sourceFile = File(originalPath);
    await sourceFile.copy(destPath);

    return (path: destPath, name: originalName);
  }

  Future<({String path, String name})> saveUploadedVideo(
    String originalPath,
    String originalName,
  ) async {
    final uuid = _uuid.v4();
    final filename = '${uuid}_$originalName';
    final destPath = path.join(_tempVideosDir, filename);

    final sourceFile = File(originalPath);
    await sourceFile.copy(destPath);

    return (path: destPath, name: originalName);
  }

  Future<void> deleteUploadedFile(String filePath) async {
    final file = File(filePath);
    if (await file.exists()) {
      await file.delete();
    }
  }

  // ===== Export =====

  Future<String> exportConversationTxt(List<Message> messages) async {
    final buffer = StringBuffer();

    for (final msg in messages) {
      buffer.writeln('${msg.role.toUpperCase()}:');
      buffer.writeln(msg.content);
      if (msg.model != null) {
        buffer.writeln('(Model: ${msg.model})');
      }
      buffer.writeln();
    }

    return buffer.toString();
  }

  Future<String> exportConversationJson(List<Message> messages) async {
    return const JsonEncoder.withIndent(
      '  ',
    ).convert(messages.map((m) => m.toJson()).toList());
  }

  // ===== Bulk Analysis =====

  Future<List<String>> getImageFiles(String folderPath) async {
    final dir = Directory(folderPath);
    if (!await dir.exists()) {
      return [];
    }

    final files = await dir
        .list()
        .where((f) => f is File && _isImageFile(f.path))
        .map((f) => f.path)
        .toList();

    return files;
  }

  Future<void> savePromptToTextFile(String imagePath, String prompt) async {
    final txtPath = imagePath.replaceAll(
      RegExp(r'\.(png|jpg|jpeg|webp)$'),
      '.txt',
    );
    final file = File(txtPath);
    await file.writeAsString(prompt);
  }

  bool _isImageFile(String filePath) {
    final ext = path.extension(filePath).toLowerCase();
    return ['.png', '.jpg', '.jpeg', '.webp'].contains(ext);
  }

  String _sanitizeTitle(String input) {
    // Remove non-alphanumeric except spaces, dots, underscores
    final sanitized = input.replaceAll(RegExp(r'[^\w\s\.]'), '');
    // Limit to 50 chars
    return sanitized.length > 50 ? sanitized.substring(0, 50) : sanitized;
  }

  String get tempImagesDir => _tempImagesDir;
  String get tempVideosDir => _tempVideosDir;

  /// Deletes all temporary files (uploaded images and videos).
  Future<int> deleteTempFiles() async {
    int count = 0;
    final imgDir = Directory(_tempImagesDir);
    if (await imgDir.exists()) {
      await for (final entity in imgDir.list()) {
        if (entity is File) {
          await entity.delete();
          count++;
        }
      }
    }
    final vidDir = Directory(_tempVideosDir);
    if (await vidDir.exists()) {
      await for (final entity in vidDir.list()) {
        if (entity is File) {
          await entity.delete();
          count++;
        }
      }
    }
    return count;
  }

  /// Deletes all conversations, folders, and temporary files.
  Future<void> deleteAllData() async {
    await _database.deleteAllData();
    await deleteTempFiles();
  }
}
