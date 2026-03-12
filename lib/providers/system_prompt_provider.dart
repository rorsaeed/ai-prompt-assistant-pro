import 'package:flutter/foundation.dart';
import '../services/storage_service.dart';

class SystemPromptProvider with ChangeNotifier {
  final StorageService _storage;

  final Map<String, String> _allPrompts = {};
  final Map<String, String> _promptCategories = {};
  String _currentPromptName = 'Default Image-to-Prompt';
  String _currentPromptContent = '';

  SystemPromptProvider(this._storage);

  Map<String, String> get allPrompts => _allPrompts;
  Map<String, String> get promptCategories => _promptCategories;
  String get currentPromptName => _currentPromptName;
  String get currentPromptContent => _currentPromptContent;

  /// Returns prompts grouped by category.
  /// Keys are category names, values are maps of prompt name -> content.
  Map<String, Map<String, String>> get promptsByCategory {
    final result = <String, Map<String, String>>{};
    for (final entry in _allPrompts.entries) {
      final category = _promptCategories[entry.key] ?? 'Uncategorized';
      result.putIfAbsent(category, () => {});
      result[category]![entry.key] = entry.value;
    }
    return result;
  }

  /// Returns a sorted list of category names.
  List<String> get categories {
    final cats = promptsByCategory.keys.toList();
    // Put "Custom" at the end if present
    if (cats.remove('Custom')) {
      cats.add('Custom');
    }
    return cats;
  }

  /// Returns the category of the given prompt.
  String getCategoryForPrompt(String name) {
    return _promptCategories[name] ?? 'Uncategorized';
  }

  Future<void> initialize() async {
    // Load predefined prompts (categorized)
    final predefined = await _storage.loadPredefinedPrompts();
    for (final categoryEntry in predefined.entries) {
      final category = categoryEntry.key;
      for (final promptEntry in categoryEntry.value.entries) {
        _allPrompts[promptEntry.key] = promptEntry.value;
        _promptCategories[promptEntry.key] = category;
      }
    }

    // Load custom prompts (with category metadata)
    final custom = await _storage.loadSystemPrompts();
    for (final entry in custom.entries) {
      _allPrompts[entry.key] = entry.value['content'] ?? '';
      _promptCategories[entry.key] = entry.value['category'] ?? 'Custom';
    }

    notifyListeners();
  }

  Future<void> selectPrompt(String name) async {
    _currentPromptName = name;
    _currentPromptContent = _allPrompts[name] ?? '';
    notifyListeners();
  }

  Future<void> saveCustomPrompt(
    String name,
    String content, {
    String category = 'Custom',
  }) async {
    await _storage.saveSystemPrompt(name, content, category: category);
    _allPrompts[name] = content;
    _promptCategories[name] = category;
    _currentPromptName = name;
    _currentPromptContent = content;
    notifyListeners();
  }

  Future<void> deletePrompt(String name) async {
    await _storage.deleteSystemPrompt(name);
    _allPrompts.remove(name);
    _promptCategories.remove(name);
    if (_currentPromptName == name) {
      _currentPromptName = 'Default Image-to-Prompt';
      _currentPromptContent = _allPrompts[_currentPromptName] ?? '';
    }
    notifyListeners();
  }

  void updatePromptContent(String content) {
    _currentPromptContent = content;
    notifyListeners();
  }
}
