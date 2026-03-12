import 'package:flutter/foundation.dart';
import '../models/folder.dart';
import '../services/database_service.dart';

class FolderProvider with ChangeNotifier {
  final DatabaseService _database;

  List<Folder> _folders = [];
  final Set<String> _expandedFolders = {};
  String? _selectedFolderId;

  FolderProvider(this._database);

  List<Folder> get folders => _folders;
  Set<String> get expandedFolders => _expandedFolders;
  String? get selectedFolderId => _selectedFolderId;

  Future<void> loadFolders() async {
    _folders = await _database.getAllFolders();
    notifyListeners();
  }

  Future<String> createFolder(String name, {String? parentId}) async {
    final id = await _database.createFolder(name, parentId: parentId);
    await loadFolders();
    return id;
  }

  Future<void> renameFolder(String id, String newName) async {
    await _database.renameFolder(id, newName);
    await loadFolders();
  }

  Future<void> deleteFolder(String id) async {
    await _database.deleteFolder(id);
    _expandedFolders.remove(id);
    if (_selectedFolderId == id) {
      _selectedFolderId = null;
    }
    await loadFolders();
  }

  Future<void> moveFolder(String id, String? newParentId) async {
    await _database.moveFolder(id, newParentId);
    await loadFolders();
  }

  void toggleExpanded(String folderId) {
    if (_expandedFolders.contains(folderId)) {
      _expandedFolders.remove(folderId);
    } else {
      _expandedFolders.add(folderId);
    }
    notifyListeners();
  }

  void selectFolder(String? folderId) {
    _selectedFolderId = folderId;
    notifyListeners();
  }

  bool isExpanded(String folderId) {
    return _expandedFolders.contains(folderId);
  }

  List<Folder> getChildFolders(String? parentId) {
    return _folders.where((f) => f.parentId == parentId).toList();
  }

  List<Folder> getRootFolders() {
    return _folders.where((f) => f.parentId == null).toList();
  }
}
