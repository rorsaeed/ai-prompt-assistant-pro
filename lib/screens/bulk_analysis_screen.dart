import 'dart:io';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:file_picker/file_picker.dart';
import 'package:uuid/uuid.dart';
import '../providers/config_provider.dart';
import '../providers/system_prompt_provider.dart';
import '../services/storage_service.dart';
import '../services/api_client.dart';
import '../models/message.dart';
import '../theme/custom_colors.dart';
import '../theme/stitch_theme.dart';

class BulkAnalysisScreen extends StatefulWidget {
  const BulkAnalysisScreen({super.key});

  @override
  State<BulkAnalysisScreen> createState() => _BulkAnalysisScreenState();
}

class _BulkAnalysisScreenState extends State<BulkAnalysisScreen> {
  final _folderPathController = TextEditingController();
  final _uuid = const Uuid();
  bool _saveToTextFile = false;
  bool _isAnalyzing = false;
  double _progress = 0.0;
  final List<({String imagePath, String prompt})> _results = [];

  @override
  void dispose() {
    _folderPathController.dispose();
    super.dispose();
  }

  Future<void> _pickFolder() async {
    final result = await FilePicker.platform.getDirectoryPath();
    if (result != null) {
      _folderPathController.text = result;
      await _detectImages();
    }
  }

  Future<void> _detectImages() async {
    if (_folderPathController.text.isEmpty) return;

    final storageService = context.read<StorageService>();
    final images = await storageService.getImageFiles(
      _folderPathController.text,
    );

    if (mounted) {
      setState(() {});
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Found ${images.length} images')));
    }
  }

  Future<void> _analyzeAll() async {
    if (_folderPathController.text.isEmpty) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('Please select a folder')));
      return;
    }

    final configProvider = context.read<ConfigProvider>();
    final selectedModels = configProvider.getSelectedModels(
      configProvider.currentProvider,
    );

    if (selectedModels.isEmpty) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Please select at least one model')),
        );
      }
      return;
    }

    setState(() {
      _isAnalyzing = true;
      _progress = 0.0;
      _results.clear();
    });

    final storageService = context.read<StorageService>();
    final promptProvider = context.read<SystemPromptProvider>();
    final images = await storageService.getImageFiles(
      _folderPathController.text,
    );

    if (images.isEmpty) {
      setState(() {
        _isAnalyzing = false;
      });
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('No images found in folder')),
        );
      }
      return;
    }

    // Use first selected model only
    final model = selectedModels.first;
    final client = APIClient(
      provider: configProvider.currentProvider,
      baseUrl: configProvider.currentProviderConfig.apiBaseUrl,
      googleApiKey: configProvider.config.googleApiKey,
      ollamaKeepAlive: configProvider.getKeepAlive(),
      unloadAfterResponse: configProvider.getUnloadAfterResponse(),
    );

    // Analyze each image
    for (int i = 0; i < images.length; i++) {
      try {
        final imagePath = images[i];
        final buffer = StringBuffer();

        // Create simple messages for analysis
        final systemMsg = Message(
          role: 'system',
          content: promptProvider.currentPromptContent,
          displayContent: promptProvider.currentPromptContent,
          id: _uuid.v4(),
        );
        final userMsg = Message(
          role: 'user',
          content: 'Describe this image in detail.',
          displayContent: 'Describe this image in detail.',
          id: _uuid.v4(),
        );

        await for (final chunk in client.generateChatResponse(
          model: model,
          messages: [systemMsg, userMsg],
          imagePaths: [imagePath],
        )) {
          buffer.write(chunk);
        }

        final prompt = buffer.toString();

        // Save to text file if enabled
        if (_saveToTextFile) {
          await storageService.savePromptToTextFile(imagePath, prompt);
        }

        // Add to results
        setState(() {
          _results.add((imagePath: imagePath, prompt: prompt));
          _progress = (i + 1) / images.length;
        });
      } catch (e) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Error analyzing ${images[i]}: $e')),
          );
        }
      }
    }

    setState(() {
      _isAnalyzing = false;
    });

    if (mounted) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('Analysis complete!')));
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colors = theme.colorScheme;

    return Padding(
      padding: const EdgeInsets.all(StitchTheme.spaceMD),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Folder Selection
          Row(
            children: [
              Expanded(
                child: TextField(
                  controller: _folderPathController,
                  decoration: const InputDecoration(
                    labelText: 'Folder Path',
                    prefixIcon: Icon(Icons.folder_rounded),
                  ),
                  readOnly: true,
                ),
              ),
              const SizedBox(width: StitchTheme.spaceSM),
              FilledButton.tonalIcon(
                onPressed: _isAnalyzing ? null : _pickFolder,
                icon: const Icon(Icons.folder_open_rounded, size: 18),
                label: const Text('Browse'),
              ),
            ],
          ),
          const SizedBox(height: StitchTheme.spaceMD),

          // Options
          Card(
            child: CheckboxListTile(
              title: const Text('Save prompts to text file'),
              subtitle: Text(
                'Saves alongside each image',
                style: theme.textTheme.bodySmall,
              ),
              secondary: Icon(
                Icons.save_rounded,
                color: colors.primary,
              ),
              value: _saveToTextFile,
              onChanged: _isAnalyzing
                  ? null
                  : (value) {
                      setState(() {
                        _saveToTextFile = value ?? false;
                      });
                    },
            ),
          ),
          const SizedBox(height: StitchTheme.spaceSM),

          // Analyze Button
          SizedBox(
            width: double.infinity,
            child: FilledButton.icon(
              onPressed: _isAnalyzing ? null : _analyzeAll,
              icon: const Icon(Icons.auto_awesome_rounded, size: 18),
              label: const Text('Analyze All Images'),
            ),
          ),
          const SizedBox(height: StitchTheme.spaceMD),

          // Progress
          if (_isAnalyzing) ...[
            Card(
              child: Padding(
                padding: const EdgeInsets.all(StitchTheme.spaceMD),
                child: Column(
                  children: [
                    Row(
                      children: [
                        SizedBox(
                          height: 18,
                          width: 18,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: colors.primary,
                          ),
                        ),
                        const SizedBox(width: StitchTheme.spaceSM),
                        Text(
                          'Analyzing... ${(_progress * 100).toStringAsFixed(0)}%',
                          style: theme.textTheme.titleSmall,
                        ),
                      ],
                    ),
                    const SizedBox(height: StitchTheme.spaceSM),
                    ClipRRect(
                      borderRadius: BorderRadius.circular(StitchTheme.radiusSM),
                      child: LinearProgressIndicator(
                        value: _progress,
                        minHeight: 6,
                      ),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: StitchTheme.spaceMD),
          ],

          // Results
          if (_results.isNotEmpty) ...[
            Row(
              children: [
                Icon(
                  Icons.check_circle_rounded,
                  size: 20,
                  color:
                      Theme.of(context).extension<CustomColors>()!.successColor,
                ),
                const SizedBox(width: StitchTheme.spaceSM),
                Text(
                  'Results (${_results.length})',
                  style: theme.textTheme.titleMedium,
                ),
              ],
            ),
            const SizedBox(height: StitchTheme.spaceSM),
            Expanded(
              child: GridView.builder(
                gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                  crossAxisCount: 3,
                  crossAxisSpacing: StitchTheme.spaceSM,
                  mainAxisSpacing: StitchTheme.spaceSM,
                  childAspectRatio: 0.7,
                ),
                itemCount: _results.length,
                itemBuilder: (context, index) {
                  final result = _results[index];
                  return Card(
                    clipBehavior: Clip.antiAlias,
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Expanded(
                          flex: 3,
                          child: Image.file(
                            File(result.imagePath),
                            width: double.infinity,
                            fit: BoxFit.cover,
                          ),
                        ),
                        Expanded(
                          flex: 2,
                          child: Padding(
                            padding: const EdgeInsets.all(StitchTheme.spaceSM),
                            child: TextField(
                              controller: TextEditingController(
                                text: result.prompt,
                              ),
                              decoration: const InputDecoration(
                                contentPadding: EdgeInsets.all(
                                  StitchTheme.spaceSM,
                                ),
                                isDense: true,
                              ),
                              maxLines: 5,
                              readOnly: true,
                              style: theme.textTheme.bodySmall,
                            ),
                          ),
                        ),
                      ],
                    ),
                  );
                },
              ),
            ),
          ] else if (!_isAnalyzing)
            Expanded(
              child: Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Container(
                      padding: const EdgeInsets.all(20),
                      decoration: BoxDecoration(
                        color: colors.primary.withValues(alpha: 0.08),
                        shape: BoxShape.circle,
                      ),
                      child: Icon(
                        Icons.photo_library_rounded,
                        size: 48,
                        color: colors.primary.withValues(alpha: 0.5),
                      ),
                    ),
                    const SizedBox(height: StitchTheme.spaceMD),
                    Text(
                      'No results yet',
                      style: theme.textTheme.titleMedium?.copyWith(
                        color: colors.onSurface.withValues(alpha: 0.7),
                      ),
                    ),
                    const SizedBox(height: StitchTheme.spaceSM),
                    Text(
                      'Select a folder and click "Analyze All Images" to start',
                      style: theme.textTheme.bodyMedium?.copyWith(
                        color: colors.onSurfaceVariant,
                      ),
                    ),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }
}
