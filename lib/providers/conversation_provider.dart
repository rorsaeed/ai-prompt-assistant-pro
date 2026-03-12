import 'package:flutter/foundation.dart';
import 'package:uuid/uuid.dart';
import '../models/message.dart';
import '../models/conversation.dart';
import '../services/storage_service.dart';

class ConversationProvider with ChangeNotifier {
  final StorageService _storage;
  final _uuid = const Uuid();

  // Per-tab conversation state: tabType -> state
  // Tab types: 0=Chat, 1=Video, 2=Image (matches sidebar filtering)
  int _currentTabType = 0;
  final Map<int, List<Message>> _tabMessages = {0: [], 1: [], 2: []};
  final Map<int, String?> _tabChatIds = {0: null, 1: null, 2: null};
  final Map<int, String?> _tabChatTitles = {0: null, 1: null, 2: null};

  final List<({String path, String name})> _uploadedFiles = [];
  final List<({String path, String name})> _uploadedVideos = [];
  final List<({String path, String name})> _lastUploadedFiles = [];
  final List<({String path, String name})> _lastUploadedVideos = [];

  ConversationProvider(this._storage);

  List<Message> get messages => _tabMessages[_currentTabType] ?? [];
  String? get currentChatId => _tabChatIds[_currentTabType];
  String? get currentChatTitle => _tabChatTitles[_currentTabType];
  List<({String path, String name})> get uploadedFiles => _uploadedFiles;
  List<({String path, String name})> get uploadedVideos => _uploadedVideos;

  /// Switch the active tab type. This swaps which conversation is visible.
  void switchTab(int tabIndex) {
    // Map tab indices to tab types: 0=Chat, 1=Video, 2=Image, 3+=Chat
    int tabType;
    if (tabIndex == 1) {
      tabType = 1; // Video
    } else if (tabIndex == 2) {
      tabType = 2; // Image
    } else {
      tabType = 0; // Chat (and all others)
    }
    if (_currentTabType != tabType) {
      _currentTabType = tabType;
      notifyListeners();
    }
  }

  void addMessage(Message message) {
    (_tabMessages[_currentTabType] ??= []).add(message);
    notifyListeners();
  }

  void deleteMessage(int index) {
    final msgs = _tabMessages[_currentTabType] ?? [];
    if (index >= 0 && index < msgs.length) {
      msgs.removeAt(index);
      notifyListeners();
    }
  }

  void clearMessages() {
    (_tabMessages[_currentTabType] ??= []).clear();
    _tabChatIds[_currentTabType] = null;
    _tabChatTitles[_currentTabType] = null;
    notifyListeners();
  }

  Future<void> loadConversation(String id) async {
    final loadedMessages = await _storage.loadConversation(id);
    _tabMessages[_currentTabType] = loadedMessages;
    _tabChatIds[_currentTabType] = id;

    // Get conversation details to set title
    final conversations = await _storage.getAllConversations();
    final conversation = conversations.firstWhere(
      (c) => c.id == id,
      orElse: () => Conversation(
        id: id,
        title: 'Conversation',
        messages: loadedMessages,
        createdAt: DateTime.now(),
        modifiedAt: DateTime.now(),
      ),
    );
    _tabChatTitles[_currentTabType] = conversation.title;

    // Extract images/videos from loaded conversation messages for reuse
    _lastUploadedFiles.clear();
    _lastUploadedVideos.clear();
    for (final msg in loadedMessages) {
      if (msg.images != null) {
        for (final img in msg.images!) {
          if (!_lastUploadedFiles.any((f) => f.path == img.path)) {
            _lastUploadedFiles.add((path: img.path, name: img.name));
          }
        }
      }
      if (msg.videos != null) {
        for (final vid in msg.videos!) {
          if (!_lastUploadedVideos.any((v) => v.path == vid.path)) {
            _lastUploadedVideos.add((path: vid.path, name: vid.name));
          }
        }
      }
    }

    if (kDebugMode) {
      print(
          'Loading conversation: extracted ${_lastUploadedFiles.length} images, ${_lastUploadedVideos.length} videos from messages');
    }

    _uploadedFiles.clear();
    _uploadedVideos.clear();
    notifyListeners();
  }

  Future<void> autoSaveConversation() async {
    final msgs = _tabMessages[_currentTabType] ?? [];
    if (msgs.isEmpty) return;
    final savedId =
        await _storage.saveConversation(_tabChatIds[_currentTabType], msgs);
    _tabChatIds[_currentTabType] = savedId;

    // Update title if it was a new chat
    if (_tabChatTitles[_currentTabType] == null && msgs.isNotEmpty) {
      final conversations = await _storage.getAllConversations();
      final conversation = conversations.firstWhere(
        (c) => c.id == savedId,
        orElse: () => Conversation(
          id: savedId,
          title: 'New Chat',
          messages: msgs,
          createdAt: DateTime.now(),
          modifiedAt: DateTime.now(),
        ),
      );
      _tabChatTitles[_currentTabType] = conversation.title;
    }

    notifyListeners();
  }

  Future<void> startNewChat({bool keepUploads = false}) async {
    (_tabMessages[_currentTabType] ??= []).clear();
    _tabChatIds[_currentTabType] = null;
    _tabChatTitles[_currentTabType] = null;
    if (!keepUploads) {
      _uploadedFiles.clear();
      _uploadedVideos.clear();
    } else {
      // Always restore from last uploads when keeping uploads
      if (_uploadedFiles.isEmpty && _lastUploadedFiles.isNotEmpty) {
        _uploadedFiles.addAll(_lastUploadedFiles);
      }
      if (_uploadedVideos.isEmpty && _lastUploadedVideos.isNotEmpty) {
        _uploadedVideos.addAll(_lastUploadedVideos);
      }

      // Debug output
      if (kDebugMode) {
        print(
            'Keep uploads: current=${_uploadedFiles.length}, last=${_lastUploadedFiles.length}');
      }
    }
    notifyListeners();
  }

  Future<void> renameConversation(String id, String newTitle) async {
    await _storage.renameConversation(id, newTitle);
    // Update title in all tabs that have this conversation loaded
    for (final tabType in _tabChatIds.keys) {
      if (_tabChatIds[tabType] == id) {
        _tabChatTitles[tabType] = newTitle;
      }
    }
    notifyListeners();
  }

  Future<void> deleteConversation(String id) async {
    await _storage.deleteConversation(id);
    if (_tabChatIds[_currentTabType] == id) {
      await startNewChat();
    }
    notifyListeners();
  }

  Future<void> moveConversationToFolder(String id, String? folderId) async {
    await _storage.moveConversationToFolder(id, folderId);
    notifyListeners();
  }

  Future<void> addUploadedFile(String path, String name) async {
    final saved = await _storage.saveUploadedFile(path, name);
    _uploadedFiles.add(saved);
    if (kDebugMode) {
      print('Added uploaded file: $name, total=${_uploadedFiles.length}');
    }
    notifyListeners();
  }

  Future<void> addUploadedVideo(String path, String name) async {
    final saved = await _storage.saveUploadedVideo(path, name);
    _uploadedVideos.add(saved);
    notifyListeners();
  }

  Future<void> removeUploadedFile(int index) async {
    if (index >= 0 && index < _uploadedFiles.length) {
      final file = _uploadedFiles[index];
      await _storage.deleteUploadedFile(file.path);
      _uploadedFiles.removeAt(index);
      notifyListeners();
    }
  }

  Future<void> removeUploadedVideo(int index) async {
    if (index >= 0 && index < _uploadedVideos.length) {
      final video = _uploadedVideos[index];
      await _storage.deleteUploadedFile(video.path);
      _uploadedVideos.removeAt(index);
      notifyListeners();
    }
  }

  Message createUserMessage(String content, {String? displayContent}) {
    final imageAttachments = _uploadedFiles
        .map((f) => ImageAttachment(path: f.path, name: f.name))
        .toList();

    final videoAttachments = _uploadedVideos
        .map((v) => VideoAttachment(path: v.path, name: v.name))
        .toList();

    return Message(
      role: 'user',
      content: content,
      displayContent: displayContent ?? content,
      id: _uuid.v4(),
      images: imageAttachments.isNotEmpty ? imageAttachments : null,
      videos: videoAttachments.isNotEmpty ? videoAttachments : null,
    );
  }

  Message createAssistantMessage(String content, String model) {
    return Message(
      role: 'assistant',
      content: content,
      displayContent: content,
      id: _uuid.v4(),
      model: model,
    );
  }

  Message createSystemMessage(String content) {
    return Message(
      role: 'system',
      content: content,
      displayContent: content,
      id: _uuid.v4(),
    );
  }

  void clearUploadedFiles() {
    _uploadedFiles.clear();
    _uploadedVideos.clear();
    notifyListeners();
  }

  Future<void> addUserMessage(String content,
      {String? displayContent,
      String? model,
      List<ImageAttachment>? images,
      List<VideoAttachment>? videos}) async {
    final msg = createUserMessage(content, displayContent: displayContent);
    final updatedMsg = Message(
      role: msg.role,
      content: msg.content,
      displayContent: msg.displayContent,
      id: msg.id,
      images: [
        if (msg.images != null) ...msg.images!,
        if (images != null) ...images,
      ].isEmpty
          ? null
          : [
              if (msg.images != null) ...msg.images!,
              if (images != null) ...images
            ],
      videos: [
        if (msg.videos != null) ...msg.videos!,
        if (videos != null) ...videos,
      ].isEmpty
          ? null
          : [
              if (msg.videos != null) ...msg.videos!,
              if (videos != null) ...videos
            ],
      model: model,
    );
    addMessage(updatedMsg);
    await autoSaveConversation();
  }

  Future<void> addAssistantMessage(String content,
      {required String model,
      List<ImageAttachment>? images,
      String? videoPath,
      String? videoName}) async {
    final msg = createAssistantMessage(content, model);
    Message updatedMsg = msg;
    if (images != null || (videoPath != null && videoName != null)) {
      updatedMsg = Message(
        role: msg.role,
        content: msg.content,
        displayContent: msg.displayContent,
        id: msg.id,
        model: msg.model,
        images: images,
        videos: (videoPath != null && videoName != null)
            ? [VideoAttachment(path: videoPath, name: videoName)]
            : null,
      );
    }
    addMessage(updatedMsg);
    await autoSaveConversation();
  }

  String getChatTitle() {
    return _tabChatTitles[_currentTabType] ?? 'New Chat';
  }
}
