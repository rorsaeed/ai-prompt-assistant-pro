import 'dart:io';
import 'package:path/path.dart' as path;
import 'package:path_provider/path_provider.dart';
import 'package:sqflite/sqflite.dart';
import 'package:sqflite_common_ffi/sqflite_ffi.dart';
import 'package:uuid/uuid.dart';
import '../models/conversation.dart';
import '../models/message.dart';
import '../models/folder.dart';

class DatabaseService {
  static const _uuid = Uuid();
  static const _databaseName = 'conversations.db';
  static const _databaseVersion = 1;

  Database? _database;

  Future<void> initialize() async {
    // Initialize sqflite_ffi for Windows desktop support
    if (Platform.isWindows || Platform.isLinux) {
      sqfliteFfiInit();
      databaseFactory = databaseFactoryFfi;
    }

    final appDocDir = await getApplicationDocumentsDirectory();
    final dataDir = path.join(appDocDir.path, 'ai_prompt_assistant', 'data');
    await Directory(dataDir).create(recursive: true);

    final dbPath = path.join(dataDir, _databaseName);

    _database = await openDatabase(
      dbPath,
      version: _databaseVersion,
      onCreate: _onCreate,
      onUpgrade: _onUpgrade,
    );
  }

  Future<void> _onCreate(Database db, int version) async {
    // Create folders table
    await db.execute('''
      CREATE TABLE folders (
        id TEXT PRIMARY KEY,
        parent_id TEXT,
        name TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        FOREIGN KEY (parent_id) REFERENCES folders(id) ON DELETE CASCADE
      )
    ''');

    await db.execute('''
      CREATE INDEX idx_folders_parent ON folders(parent_id)
    ''');

    // Create conversations table
    await db.execute('''
      CREATE TABLE conversations (
        id TEXT PRIMARY KEY,
        folder_id TEXT,
        title TEXT NOT NULL,
        created_at INTEGER NOT NULL,
        modified_at INTEGER NOT NULL,
        FOREIGN KEY (folder_id) REFERENCES folders(id) ON DELETE SET NULL
      )
    ''');

    await db.execute('''
      CREATE INDEX idx_conversations_folder ON conversations(folder_id)
    ''');

    await db.execute('''
      CREATE INDEX idx_conversations_modified ON conversations(modified_at DESC)
    ''');

    // Create messages table
    await db.execute('''
      CREATE TABLE messages (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        display_content TEXT NOT NULL,
        model TEXT,
        sequence INTEGER NOT NULL,
        created_at INTEGER NOT NULL,
        FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
      )
    ''');

    await db.execute('''
      CREATE INDEX idx_messages_conversation ON messages(conversation_id, sequence)
    ''');

    // Create attachments table
    await db.execute('''
      CREATE TABLE attachments (
        id TEXT PRIMARY KEY,
        message_id TEXT NOT NULL,
        type TEXT NOT NULL,
        path TEXT NOT NULL,
        name TEXT NOT NULL,
        FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE
      )
    ''');

    await db.execute('''
      CREATE INDEX idx_attachments_message ON attachments(message_id)
    ''');
  }

  Future<void> _onUpgrade(Database db, int oldVersion, int newVersion) async {
    // Handle database migrations in future versions
  }

  Future<void> close() async {
    await _database?.close();
    _database = null;
  }

  Database get database {
    if (_database == null) {
      throw Exception('Database not initialized. Call initialize() first.');
    }
    return _database!;
  }

  // ===== Folder Operations =====

  Future<String> createFolder(String name, {String? parentId}) async {
    final id = _uuid.v4();
    final now = DateTime.now().millisecondsSinceEpoch;

    await database.insert('folders', {
      'id': id,
      'parent_id': parentId,
      'name': name,
      'created_at': now,
    });

    return id;
  }

  Future<List<Folder>> listFolders({String? parentId}) async {
    final List<Map<String, dynamic>> maps = await database.query(
      'folders',
      where: parentId == null ? 'parent_id IS NULL' : 'parent_id = ?',
      whereArgs: parentId == null ? null : [parentId],
      orderBy: 'name ASC',
    );

    return maps.map((map) => Folder.fromMap(map)).toList();
  }

  Future<List<Folder>> getAllFolders() async {
    final List<Map<String, dynamic>> maps = await database.query(
      'folders',
      orderBy: 'name ASC',
    );

    return maps.map((map) => Folder.fromMap(map)).toList();
  }

  Future<Folder?> getFolder(String id) async {
    final List<Map<String, dynamic>> maps = await database.query(
      'folders',
      where: 'id = ?',
      whereArgs: [id],
      limit: 1,
    );

    if (maps.isEmpty) return null;
    return Folder.fromMap(maps.first);
  }

  Future<void> renameFolder(String id, String newName) async {
    await database.update(
      'folders',
      {'name': newName},
      where: 'id = ?',
      whereArgs: [id],
    );
  }

  Future<void> deleteFolder(String id) async {
    // Cascade delete will handle conversations and nested folders
    await database.delete(
      'folders',
      where: 'id = ?',
      whereArgs: [id],
    );
  }

  Future<void> moveFolder(String id, String? newParentId) async {
    await database.update(
      'folders',
      {'parent_id': newParentId},
      where: 'id = ?',
      whereArgs: [id],
    );
  }

  // ===== Conversation Operations =====

  Future<String> createConversation(String title, {String? folderId}) async {
    final id = _uuid.v4();
    final now = DateTime.now().millisecondsSinceEpoch;

    await database.insert('conversations', {
      'id': id,
      'folder_id': folderId,
      'title': title,
      'created_at': now,
      'modified_at': now,
    });

    return id;
  }

  Future<List<Conversation>> listConversations({String? folderId}) async {
    final List<Map<String, dynamic>> maps = await database.query(
      'conversations',
      where: folderId == null ? 'folder_id IS NULL' : 'folder_id = ?',
      whereArgs: folderId == null ? null : [folderId],
      orderBy: 'modified_at DESC',
    );

    final conversations = <Conversation>[];
    for (final map in maps) {
      final messages = await getMessages(map['id']);
      conversations.add(Conversation(
        id: map['id'],
        title: map['title'],
        folderId: map['folder_id'],
        messages: messages,
        createdAt: DateTime.fromMillisecondsSinceEpoch(map['created_at']),
        modifiedAt: DateTime.fromMillisecondsSinceEpoch(map['modified_at']),
      ));
    }

    return conversations;
  }

  Future<List<Conversation>> getAllConversations() async {
    final List<Map<String, dynamic>> maps = await database.query(
      'conversations',
      orderBy: 'modified_at DESC',
    );

    final conversations = <Conversation>[];
    for (final map in maps) {
      final messages = await getMessages(map['id']);
      conversations.add(Conversation(
        id: map['id'],
        title: map['title'],
        folderId: map['folder_id'],
        messages: messages,
        createdAt: DateTime.fromMillisecondsSinceEpoch(map['created_at']),
        modifiedAt: DateTime.fromMillisecondsSinceEpoch(map['modified_at']),
      ));
    }

    return conversations;
  }

  Future<Conversation?> loadConversation(String id) async {
    final List<Map<String, dynamic>> maps = await database.query(
      'conversations',
      where: 'id = ?',
      whereArgs: [id],
      limit: 1,
    );

    if (maps.isEmpty) return null;

    final map = maps.first;
    final messages = await getMessages(id);

    return Conversation(
      id: map['id'],
      title: map['title'],
      folderId: map['folder_id'],
      messages: messages,
      createdAt: DateTime.fromMillisecondsSinceEpoch(map['created_at']),
      modifiedAt: DateTime.fromMillisecondsSinceEpoch(map['modified_at']),
    );
  }

  Future<void> updateConversation(String id, List<Message> messages) async {
    final now = DateTime.now().millisecondsSinceEpoch;

    await database.transaction((txn) async {
      // Update modified timestamp
      await txn.update(
        'conversations',
        {'modified_at': now},
        where: 'id = ?',
        whereArgs: [id],
      );

      // Delete existing attachments for this conversation's messages
      await txn.rawDelete('''
        DELETE FROM attachments WHERE message_id IN (
          SELECT id FROM messages WHERE conversation_id = ?
        )
      ''', [id]);

      // Delete existing messages
      await txn.delete(
        'messages',
        where: 'conversation_id = ?',
        whereArgs: [id],
      );

      // Insert new messages
      for (int i = 0; i < messages.length; i++) {
        final message = messages[i];
        await txn.insert('messages', {
          'id': message.id,
          'conversation_id': id,
          'role': message.role,
          'content': message.content,
          'display_content': message.displayContent,
          'model': message.model,
          'sequence': i,
          'created_at': now,
        });

        // Insert attachments
        if (message.images != null) {
          for (final image in message.images!) {
            await txn.insert('attachments', {
              'id': _uuid.v4(),
              'message_id': message.id,
              'type': 'image',
              'path': image.path,
              'name': image.name,
            });
          }
        }

        if (message.videos != null) {
          for (final video in message.videos!) {
            await txn.insert('attachments', {
              'id': _uuid.v4(),
              'message_id': message.id,
              'type': 'video',
              'path': video.path,
              'name': video.name,
            });
          }
        }
      }
    });
  }

  Future<void> renameConversation(String id, String newTitle) async {
    await database.update(
      'conversations',
      {'title': newTitle},
      where: 'id = ?',
      whereArgs: [id],
    );
  }

  Future<void> deleteConversation(String id) async {
    // Cascade delete will handle messages and attachments
    await database.delete(
      'conversations',
      where: 'id = ?',
      whereArgs: [id],
    );
  }

  Future<void> moveConversationToFolder(String id, String? folderId) async {
    await database.update(
      'conversations',
      {'folder_id': folderId},
      where: 'id = ?',
      whereArgs: [id],
    );
  }

  // ===== Message Operations =====

  Future<List<Message>> getMessages(String conversationId) async {
    final List<Map<String, dynamic>> maps = await database.query(
      'messages',
      where: 'conversation_id = ?',
      whereArgs: [conversationId],
      orderBy: 'sequence ASC',
    );

    final messages = <Message>[];
    for (final map in maps) {
      final attachments = await _getAttachments(map['id']);

      final images = attachments
          .where((a) => a['type'] == 'image')
          .map((a) => ImageAttachment(path: a['path'], name: a['name']))
          .toList();

      final videos = attachments
          .where((a) => a['type'] == 'video')
          .map((a) => VideoAttachment(path: a['path'], name: a['name']))
          .toList();

      messages.add(Message(
        id: map['id'],
        role: map['role'],
        content: map['content'],
        displayContent: map['display_content'],
        model: map['model'],
        images: images.isEmpty ? null : images,
        videos: videos.isEmpty ? null : videos,
      ));
    }

    return messages;
  }

  Future<List<Map<String, dynamic>>> _getAttachments(String messageId) async {
    return await database.query(
      'attachments',
      where: 'message_id = ?',
      whereArgs: [messageId],
    );
  }

  // ===== Search Operations =====

  Future<List<Conversation>> searchConversations(String query) async {
    if (query.isEmpty) {
      return await getAllConversations();
    }

    final searchPattern = '%$query%';

    // Search in conversation titles
    final titleResults = await database.query(
      'conversations',
      where: 'title LIKE ?',
      whereArgs: [searchPattern],
      orderBy: 'modified_at DESC',
    );

    // Search in message content
    final messageResults = await database.rawQuery('''
      SELECT DISTINCT c.* FROM conversations c
      INNER JOIN messages m ON c.id = m.conversation_id
      WHERE m.content LIKE ? OR m.display_content LIKE ?
      ORDER BY c.modified_at DESC
    ''', [searchPattern, searchPattern]);

    // Combine and deduplicate results
    final conversationIds = <String>{};
    final allResults = <Map<String, dynamic>>[];

    for (final result in [...titleResults, ...messageResults]) {
      final id = result['id'] as String;
      if (!conversationIds.contains(id)) {
        conversationIds.add(id);
        allResults.add(result);
      }
    }

    // Build conversation objects
    final conversations = <Conversation>[];
    for (final map in allResults) {
      final messages = await getMessages(map['id']);
      conversations.add(Conversation(
        id: map['id'],
        title: map['title'],
        folderId: map['folder_id'],
        messages: messages,
        createdAt: DateTime.fromMillisecondsSinceEpoch(map['created_at']),
        modifiedAt: DateTime.fromMillisecondsSinceEpoch(map['modified_at']),
      ));
    }

    return conversations;
  }

  Future<int> getConversationCount({String? folderId}) async {
    final result = await database.rawQuery(
      'SELECT COUNT(*) as count FROM conversations WHERE ${folderId == null ? 'folder_id IS NULL' : 'folder_id = ?'}',
      folderId == null ? null : [folderId],
    );

    return Sqflite.firstIntValue(result) ?? 0;
  }

  /// Deletes all conversations, messages, attachments, and folders.
  Future<void> deleteAllData() async {
    await database.transaction((txn) async {
      await txn.delete('attachments');
      await txn.delete('messages');
      await txn.delete('conversations');
      await txn.delete('folders');
    });
  }
}
