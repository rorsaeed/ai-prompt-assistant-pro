import 'package:flutter/foundation.dart';

class GenerationStateProvider with ChangeNotifier {
  bool _isGenerating = false;
  bool _isCancelled = false;

  bool get isGenerating => _isGenerating;
  bool get isCancelled => _isCancelled;

  void setGenerating(bool value) {
    _isGenerating = value;
    if (value) {
      _isCancelled = false;
    }
    notifyListeners();
  }

  void cancelGeneration() {
    _isCancelled = true;
    notifyListeners();
  }
}
