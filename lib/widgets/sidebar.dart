import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:url_launcher/url_launcher.dart';
import 'package:share_plus/share_plus.dart';
import '../providers/config_provider.dart';
import '../providers/conversation_provider.dart';
import '../providers/generation_state_provider.dart';
import '../providers/folder_provider.dart';
import '../providers/system_prompt_provider.dart';
import '../providers/ui_state_provider.dart';
import '../services/storage_service.dart';
import '../services/api_client.dart';
import '../models/api_models.dart';
import '../models/conversation.dart';
import '../theme/stitch_theme.dart';
import 'search_bar.dart';
import 'conversation_tree.dart';
import 'folder_dialog.dart';

class Sidebar extends StatefulWidget {
  final bool asPersistentPanel;

  const Sidebar({super.key, this.asPersistentPanel = false});

  @override
  State<Sidebar> createState() => _SidebarState();
}

class _SidebarState extends State<Sidebar> {
  final _renameController = TextEditingController();
  final _baseUrlController = TextEditingController();
  final _apiKeyController = TextEditingController();

  List<Conversation>? _conversations;
  String _searchQuery = '';
  String _promptSearchQuery = '';
  List<ModelInfo>? _availableModels;
  bool _loadingModels = false;
  String? _modelError;
  String? _lastProvider;

  // Accordion: track which section is expanded
  int _expandedSection = 1;

  @override
  void initState() {
    super.initState();
    _loadConversations();
    _loadFolders();

    // Listen to conversation provider changes to refresh the list
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final configProvider = context.read<ConfigProvider>();
      _lastProvider = configProvider.currentProvider;
      _loadModels();

      context.read<ConversationProvider>().addListener(_onConversationChanged);
      context.read<FolderProvider>().addListener(_onFolderChanged);
      context.read<UiStateProvider>().addListener(_onTabChanged);
      configProvider.addListener(_onConfigChanged);
    });
  }

  @override
  void dispose() {
    // Remove listeners before disposing
    try {
      context
          .read<ConversationProvider>()
          .removeListener(_onConversationChanged);
      context.read<FolderProvider>().removeListener(_onFolderChanged);
      context.read<UiStateProvider>().removeListener(_onTabChanged);
      context.read<ConfigProvider>().removeListener(_onConfigChanged);
    } catch (e) {
      // Ignore if context is no longer valid
    }
    _renameController.dispose();
    _baseUrlController.dispose();
    _apiKeyController.dispose();
    super.dispose();
  }

  void _onConfigChanged() {
    if (!mounted) return;
    final configProvider = context.read<ConfigProvider>();
    if (configProvider.currentProvider != _lastProvider) {
      _lastProvider = configProvider.currentProvider;
      _loadModels();
    }
  }

  void _onConversationChanged() {
    // Reload conversations when the provider notifies changes
    if (mounted) {
      _loadConversations();
    }
  }

  void _onFolderChanged() {
    // Reload conversations when folders change (e.g., move operation)
    if (mounted) {
      _loadConversations();
    }
  }

  void _onTabChanged() {
    if (mounted) {
      // Reload the sidebar conversation list filtered for the active tab.
      _loadConversations();
    }
  }

  Future<void> _loadConversations() async {
    final storage = context.read<StorageService>();
    final uiState = context.read<UiStateProvider>();
    final convs = _searchQuery.isEmpty
        ? await storage.getAllConversations()
        : await storage.searchConversations(_searchQuery);

    final filtered = convs.where((c) {
      final isVeo = c.messages
          .any((m) => m.model?.toLowerCase().contains('veo') ?? false);
      final isImagen = c.messages.any((m) =>
          (m.model?.toLowerCase().contains('imagen') ?? false) ||
          m.images != null && m.images!.isNotEmpty);

      if (uiState.currentTabIndex == 1) {
        // Veo tab
        return isVeo;
      } else if (uiState.currentTabIndex == 2) {
        // Image Studio tab
        return isImagen;
      } else {
        // Chat tab
        return !isVeo && !isImagen;
      }
    }).toList();

    setState(() {
      _conversations = filtered;
    });
  }

  Future<void> _loadFolders() async {
    final folderProvider = context.read<FolderProvider>();
    await folderProvider.loadFolders();
  }

  void _onSearchChanged(String query) {
    setState(() {
      _searchQuery = query;
    });
    _loadConversations();
  }

  Future<void> _loadModels() async {
    final configProvider = context.read<ConfigProvider>();
    final provider = configProvider.currentProvider;
    final config = configProvider.currentProviderConfig;

    setState(() {
      _loadingModels = true;
      _modelError = null;
    });

    try {
      final client = APIClient(
        provider: provider,
        baseUrl: config.apiBaseUrl,
        googleApiKey: configProvider.config.googleApiKey,
      );

      final models = await client.getAvailableModels();

      if (mounted) {
        // Prune selected models that no longer exist in available models
        final availableIds = models.map((m) => m.id).toSet();
        final currentSelected = configProvider.getSelectedModels(provider);
        final pruned =
            currentSelected.where((id) => availableIds.contains(id)).toList();
        if (pruned.length != currentSelected.length) {
          await configProvider.updateSelectedModels(provider, pruned);
        }

        setState(() {
          _availableModels = models;
          _loadingModels = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _modelError = e.toString();
          _loadingModels = false;
        });
      }
    }
  }

  void _closeSidebar() {
    if (!widget.asPersistentPanel && Navigator.canPop(context)) {
      Navigator.pop(context);
    }
  }

  @override
  Widget build(BuildContext context) {
    final configProvider = context.watch<ConfigProvider>();
    final conversationProvider = context.watch<ConversationProvider>();
    final generationState = context.watch<GenerationStateProvider>();
    final storageService = context.read<StorageService>();
    final theme = Theme.of(context);
    final colors = theme.colorScheme;

    final content = Column(
      children: [
        // ─── Sidebar Header ───
        Container(
          padding: const EdgeInsets.fromLTRB(
            StitchTheme.spaceMD,
            StitchTheme.spaceLG,
            StitchTheme.spaceMD,
            StitchTheme.spaceSM,
          ),
          child: Row(
            children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: colors.primary.withValues(alpha: 0.12),
                  borderRadius: BorderRadius.circular(StitchTheme.radiusSM),
                ),
                child: Icon(
                  Icons.settings_rounded,
                  size: 20,
                  color: colors.primary,
                ),
              ),
              const SizedBox(width: StitchTheme.spaceSM),
              Text('Settings', style: theme.textTheme.titleLarge),
              const Spacer(),
              if (!widget.asPersistentPanel)
                IconButton(
                  onPressed: _closeSidebar,
                  icon: const Icon(Icons.close_rounded),
                  tooltip: 'Close',
                  iconSize: 20,
                ),
            ],
          ),
        ),

        const Divider(height: 1),

        // ─── Accordion Sections ───
        Expanded(
          child: ListView(
            padding: const EdgeInsets.symmetric(
              vertical: StitchTheme.spaceSM,
            ),
            children: [
              // ── Section 0: Conversations ──
              _SidebarSection(
                icon: Icons.chat_bubble_outline_rounded,
                title: 'Conversations',
                isExpanded: _expandedSection == 0,
                onToggle: () => setState(() {
                  _expandedSection = _expandedSection == 0 ? -1 : 0;
                }),
                child: _buildConversationsSection(
                  conversationProvider,
                  storageService,
                  colors,
                ),
              ),

              // ── Section 1: Configuration ──
              _SidebarSection(
                icon: Icons.tune_rounded,
                title: 'Configuration',
                isExpanded: _expandedSection == 1,
                onToggle: () => setState(() {
                  _expandedSection = _expandedSection == 1 ? -1 : 1;
                }),
                child: _buildConfigurationSection(
                  configProvider,
                  generationState,
                  colors,
                ),
              ),

              // ── Section 2: Export ──
              if (conversationProvider.messages.isNotEmpty)
                _SidebarSection(
                  icon: Icons.upload_file_rounded,
                  title: 'Export',
                  isExpanded: _expandedSection == 2,
                  onToggle: () => setState(() {
                    _expandedSection = _expandedSection == 2 ? -1 : 2;
                  }),
                  child: _buildExportSection(
                    conversationProvider,
                    storageService,
                    colors,
                  ),
                ),

              // ── Section 3: Data Management ──
              _SidebarSection(
                icon: Icons.delete_sweep_rounded,
                title: 'Data Management',
                isExpanded: _expandedSection == 3,
                onToggle: () => setState(() {
                  _expandedSection = _expandedSection == 3 ? -1 : 3;
                }),
                child: _buildDataManagementSection(
                  conversationProvider,
                  storageService,
                  colors,
                ),
              ),

              // ── Footer ──
              const SizedBox(height: StitchTheme.spaceMD),
              Padding(
                padding: const EdgeInsets.symmetric(
                  horizontal: StitchTheme.spaceMD,
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    TextButton.icon(
                      onPressed: () {
                        launchUrl(Uri.parse('https://github.com'));
                      },
                      icon: const Icon(Icons.language_rounded, size: 16),
                      label: const Text('Website'),
                    ),
                    Container(
                      height: 16,
                      width: 1,
                      color: theme.dividerColor,
                    ),
                    TextButton.icon(
                      onPressed: () {
                        launchUrl(Uri.parse('https://github.com'));
                      },
                      icon: const Icon(Icons.code_rounded, size: 16),
                      label: const Text('GitHub'),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: StitchTheme.spaceSM),
              Text(
                'v1.0.0',
                textAlign: TextAlign.center,
                style: theme.textTheme.bodySmall?.copyWith(
                  color: colors.onSurface.withValues(alpha: 0.4),
                  fontSize: 11,
                  letterSpacing: 0.5,
                ),
              ),
              const SizedBox(height: StitchTheme.spaceMD),
            ],
          ),
        ),
      ],
    );

    // Persistent panel mode — no Drawer wrapper
    if (widget.asPersistentPanel) {
      return Material(
        color: StitchTheme.sidebarBg(context),
        child: content,
      );
    }

    // Drawer mode
    return Drawer(
      child: content,
    );
  }

  // ═══════════════════════════════════════════════════
  // SECTION BUILDERS
  // ═══════════════════════════════════════════════════

  Widget _buildConversationsSection(
    ConversationProvider conversationProvider,
    StorageService storageService,
    ColorScheme colors,
  ) {
    final folderProvider = context.watch<FolderProvider>();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      mainAxisSize: MainAxisSize.min,
      children: [
        // New Chat and New Folder buttons
        Row(
          children: [
            Expanded(
              child: FilledButton.icon(
                onPressed: () {
                  conversationProvider.startNewChat(keepUploads: false);
                  _closeSidebar();
                },
                icon: const Icon(Icons.add_rounded, size: 18),
                label: const Text('New Chat'),
              ),
            ),
            const SizedBox(width: 4),
            SizedBox(
              width: 36,
              child: PopupMenuButton<String>(
                tooltip: 'New Chat Options',
                icon: Icon(
                  Icons.arrow_drop_down,
                  color: Theme.of(context).colorScheme.primary,
                ),
                onSelected: (value) {
                  if (value == 'keep') {
                    conversationProvider.startNewChat(keepUploads: true);
                    _closeSidebar();
                  }
                },
                itemBuilder: (context) => [
                  PopupMenuItem(
                    value: 'keep',
                    child: Row(
                      children: [
                        Icon(
                          Icons.content_copy,
                          size: 18,
                          color: Theme.of(context).colorScheme.onSurface,
                        ),
                        const SizedBox(width: 8),
                        const Text('Keep Uploads'),
                      ],
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(width: 8),
            FilledButton.tonal(
              onPressed: () {
                _showCreateFolderDialog(folderProvider);
              },
              child: const Icon(Icons.create_new_folder, size: 18),
            ),
          ],
        ),
        const SizedBox(height: StitchTheme.spaceSM),

        // Search bar
        ConversationSearchBar(
          onSearchChanged: _onSearchChanged,
          initialQuery: _searchQuery,
        ),

        // Conversation tree with fixed height
        if (_conversations != null)
          SizedBox(
            height: 400, // Fixed height for the conversation tree
            child: ConversationTree(
              conversations: _conversations!,
              searchQuery: _searchQuery.isNotEmpty ? _searchQuery : null,
            ),
          )
        else
          const SizedBox(
            height: 400,
            child: Center(
              child: CircularProgressIndicator(),
            ),
          ),
      ],
    );
  }

  void _showCreateFolderDialog(FolderProvider folderProvider) {
    showDialog(
      context: context,
      builder: (context) => FolderDialog(
        title: 'New Folder',
        onSave: (name) async {
          await folderProvider.createFolder(name);
          await _loadConversations();
        },
      ),
    );
  }

  Widget _buildConfigurationSection(
    ConfigProvider configProvider,
    GenerationStateProvider generationState,
    ColorScheme colors,
  ) {
    final promptProvider = context.watch<SystemPromptProvider>();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        // System Prompt Selection
        Text(
          'System Prompt',
          style: Theme.of(context).textTheme.labelLarge,
        ),
        const SizedBox(height: StitchTheme.spaceXS),
        TextField(
          decoration: InputDecoration(
            hintText: 'Search prompts...',
            prefixIcon: const Icon(Icons.search, size: 18),
            isDense: true,
            contentPadding: const EdgeInsets.symmetric(
              horizontal: 12,
              vertical: 8,
            ),
            border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(8),
            ),
          ),
          onChanged: (value) {
            setState(() {
              _promptSearchQuery = value.toLowerCase();
            });
          },
        ),
        const SizedBox(height: StitchTheme.spaceXS),
        Container(
          constraints: const BoxConstraints(maxHeight: 250),
          decoration: BoxDecoration(
            border: Border.all(color: colors.outline.withOpacity(0.3)),
            borderRadius: BorderRadius.circular(8),
          ),
          child: ListView(
            shrinkWrap: true,
            children: [
              for (final category in promptProvider.categories) ...[
                // Check if any prompts in this category match the search
                if (promptProvider.promptsByCategory[category]!.keys.any(
                    (name) =>
                        _promptSearchQuery.isEmpty ||
                        name.toLowerCase().contains(_promptSearchQuery))) ...[
                  Padding(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                    child: Text(
                      category,
                      style: TextStyle(
                        fontSize: 11,
                        fontWeight: FontWeight.bold,
                        color: colors.primary,
                        letterSpacing: 0.5,
                      ),
                    ),
                  ),
                  ...promptProvider.promptsByCategory[category]!.keys
                      .where((name) =>
                          _promptSearchQuery.isEmpty ||
                          name.toLowerCase().contains(_promptSearchQuery))
                      .map((name) {
                    final isSelected = promptProvider.currentPromptName == name;
                    return RadioListTile<String>(
                      dense: true,
                      title: Text(
                        name,
                        style: TextStyle(
                          fontSize: 13,
                          fontWeight:
                              isSelected ? FontWeight.w600 : FontWeight.w400,
                        ),
                      ),
                      value: name,
                      groupValue: promptProvider.currentPromptName,
                      onChanged: (value) async {
                        if (value != null) {
                          await promptProvider.selectPrompt(value);
                          await configProvider
                              .updateLastSystemPromptName(value);
                        }
                      },
                    );
                  }),
                ],
              ],
            ],
          ),
        ),
        const SizedBox(height: StitchTheme.spaceMD),
        const Divider(),
        const SizedBox(height: StitchTheme.spaceMD),

        // Provider Selection
        Text(
          'API Provider',
          style: Theme.of(context).textTheme.labelLarge,
        ),
        const SizedBox(height: StitchTheme.spaceXS),
        ...['Ollama', 'LM Studio', 'Koboldcpp', 'Google'].map((provider) {
          return RadioListTile<String>(
            title: Text(provider),
            value: provider,
            groupValue: configProvider.currentProvider,
            onChanged: generationState.isGenerating
                ? null
                : (value) {
                    if (value != null) {
                      configProvider.updateProvider(value);
                      _availableModels = null;
                    }
                  },
          );
        }),
        const SizedBox(height: StitchTheme.spaceSM),

        // Provider-specific config
        _buildProviderConfig(configProvider),
        const SizedBox(height: StitchTheme.spaceMD),

        // Model Selection
        Text(
          'Select Model(s)',
          style: Theme.of(context).textTheme.labelLarge,
        ),
        const SizedBox(height: StitchTheme.spaceSM),
        FilledButton.tonalIcon(
          onPressed: _loadingModels ? null : _loadModels,
          icon: _loadingModels
              ? SizedBox(
                  height: 16,
                  width: 16,
                  child: CircularProgressIndicator(
                    strokeWidth: 2,
                    color: colors.primary,
                  ),
                )
              : const Icon(Icons.refresh_rounded, size: 18),
          label: Text(_loadingModels ? 'Fetching...' : 'Fetch Models'),
        ),
        if (_modelError != null)
          Padding(
            padding: const EdgeInsets.only(top: StitchTheme.spaceSM),
            child: Text(
              _modelError!,
              style: TextStyle(color: colors.error, fontSize: 12),
            ),
          ),
        if (_availableModels != null) ...[
          const SizedBox(height: StitchTheme.spaceSM),
          Wrap(
            spacing: 6,
            runSpacing: 6,
            children: _availableModels!.map((model) {
              final isSelected = configProvider
                  .getSelectedModels(configProvider.currentProvider)
                  .contains(model.id);
              return FilterChip(
                label: Text(model.name),
                selected: isSelected,
                onSelected: (selected) {
                  final current = List<String>.from(
                    configProvider.getSelectedModels(
                      configProvider.currentProvider,
                    ),
                  );
                  if (selected) {
                    current.add(model.id);
                  } else {
                    current.remove(model.id);
                  }
                  configProvider.updateSelectedModels(
                    configProvider.currentProvider,
                    current,
                  );
                },
              );
            }).toList(),
          ),
        ],
      ],
    );
  }

  Widget _buildExportSection(
    ConversationProvider conversationProvider,
    StorageService storageService,
    ColorScheme colors,
  ) {
    return Row(
      children: [
        Expanded(
          child: OutlinedButton.icon(
            onPressed: () async {
              final content = await storageService.exportConversationTxt(
                conversationProvider.messages,
              );
              await Share.share(content);
            },
            icon: const Icon(Icons.text_snippet_rounded, size: 16),
            label: const Text('.txt'),
          ),
        ),
        const SizedBox(width: StitchTheme.spaceSM),
        Expanded(
          child: OutlinedButton.icon(
            onPressed: () async {
              final content = await storageService.exportConversationJson(
                conversationProvider.messages,
              );
              await Share.share(content);
            },
            icon: const Icon(Icons.data_object_rounded, size: 16),
            label: const Text('.json'),
          ),
        ),
      ],
    );
  }

  Widget _buildDataManagementSection(
    ConversationProvider conversationProvider,
    StorageService storageService,
    ColorScheme colors,
  ) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        OutlinedButton.icon(
          onPressed: () => _confirmDeleteTempFiles(storageService, colors),
          icon: Icon(Icons.cleaning_services_rounded,
              size: 16, color: colors.onSurface),
          label: const Text('Delete Temporary Files'),
        ),
        const SizedBox(height: StitchTheme.spaceSM),
        FilledButton.icon(
          onPressed: () => _confirmDeleteAll(
            conversationProvider,
            storageService,
            colors,
          ),
          icon: const Icon(Icons.delete_forever_rounded, size: 16),
          label: const Text('Delete All Data'),
          style: FilledButton.styleFrom(
            backgroundColor: colors.error,
            foregroundColor: colors.onError,
          ),
        ),
      ],
    );
  }

  Future<void> _confirmDeleteTempFiles(
    StorageService storageService,
    ColorScheme colors,
  ) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        icon: Icon(Icons.cleaning_services_rounded, color: colors.primary),
        title: const Text('Delete Temporary Files?'),
        content: const Text(
          'This will delete all uploaded images and videos stored in '
          'temporary folders. Attachments referenced in conversations '
          'will no longer be viewable.\n\n'
          'This action cannot be undone.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(context, true),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
    if (confirmed == true && mounted) {
      final count = await storageService.deleteTempFiles();
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Deleted $count temporary file(s).')),
        );
      }
    }
  }

  Future<void> _confirmDeleteAll(
    ConversationProvider conversationProvider,
    StorageService storageService,
    ColorScheme colors,
  ) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        icon: Icon(Icons.delete_forever_rounded, color: colors.error),
        title: const Text('Delete All Data?'),
        content: const Text(
          'This will permanently delete ALL conversations, folders, '
          'and temporary files (uploaded images and videos).\n\n'
          'This action cannot be undone.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.pop(context, true),
            style: FilledButton.styleFrom(
              backgroundColor: colors.error,
              foregroundColor: colors.onError,
            ),
            child: const Text('Delete Everything'),
          ),
        ],
      ),
    );
    if (confirmed == true && mounted) {
      await storageService.deleteAllData();
      conversationProvider.clearMessages();
      final folderProvider = context.read<FolderProvider>();
      await folderProvider.loadFolders();
      await _loadConversations();
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
              content: Text('All conversations and temporary files deleted.')),
        );
      }
    }
  }

  Widget _buildProviderConfig(ConfigProvider configProvider) {
    final provider = configProvider.currentProvider;

    switch (provider) {
      case 'Google':
        return TextField(
          controller: _apiKeyController
            ..text = configProvider.config.googleApiKey ?? '',
          decoration: const InputDecoration(
            labelText: 'API Key',
            prefixIcon: Icon(Icons.key_rounded),
          ),
          obscureText: true,
          onChanged: (value) {
            configProvider.updateApiKey(value.isEmpty ? null : value);
          },
        );

      case 'Ollama':
        return Column(
          children: [
            TextField(
              controller: _baseUrlController
                ..text = configProvider.getBaseUrl(provider),
              decoration: const InputDecoration(
                labelText: 'API Base URL',
                prefixIcon: Icon(Icons.link_rounded),
              ),
              onChanged: (value) {
                configProvider.updateBaseUrl(provider, value);
              },
            ),
            const SizedBox(height: StitchTheme.spaceSM),
            TextField(
              decoration: const InputDecoration(
                labelText: 'Keep Alive (-1 for default)',
                prefixIcon: Icon(Icons.timer_rounded),
              ),
              keyboardType: TextInputType.number,
              controller: TextEditingController(
                text: (configProvider.getKeepAlive() ?? -1).toString(),
              ),
              onChanged: (value) {
                final intValue = int.tryParse(value);
                if (intValue != null) {
                  configProvider.updateKeepAlive(intValue);
                }
              },
            ),
          ],
        );

      case 'LM Studio':
        return Column(
          children: [
            TextField(
              controller: _baseUrlController
                ..text = configProvider.getBaseUrl(provider),
              decoration: const InputDecoration(
                labelText: 'API Base URL',
                prefixIcon: Icon(Icons.link_rounded),
              ),
              onChanged: (value) {
                configProvider.updateBaseUrl(provider, value);
              },
            ),
            const SizedBox(height: StitchTheme.spaceSM),
            CheckboxListTile(
              title: const Text('Unload model after response'),
              value: configProvider.getUnloadAfterResponse(),
              onChanged: (value) {
                configProvider.updateUnloadAfterResponse(value ?? false);
              },
            ),
          ],
        );

      default:
        return TextField(
          controller: _baseUrlController
            ..text = configProvider.getBaseUrl(provider),
          decoration: const InputDecoration(
            labelText: 'API Base URL',
            prefixIcon: Icon(Icons.link_rounded),
          ),
          onChanged: (value) {
            configProvider.updateBaseUrl(provider, value);
          },
        );
    }
  }
}

// ═══════════════════════════════════════════════════
// Accordion Section Widget
// ═══════════════════════════════════════════════════

class _SidebarSection extends StatelessWidget {
  final IconData icon;
  final String title;
  final bool isExpanded;
  final VoidCallback onToggle;
  final Widget child;

  const _SidebarSection({
    required this.icon,
    required this.title,
    required this.isExpanded,
    required this.onToggle,
    required this.child,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colors = theme.colorScheme;

    return Padding(
      padding: const EdgeInsets.symmetric(
        horizontal: StitchTheme.spaceSM,
        vertical: 2,
      ),
      child: Column(
        children: [
          // Section Header
          Material(
            color: isExpanded
                ? colors.primary.withValues(alpha: 0.08)
                : Colors.transparent,
            borderRadius: BorderRadius.circular(StitchTheme.radiusSM),
            child: InkWell(
              borderRadius: BorderRadius.circular(StitchTheme.radiusSM),
              onTap: onToggle,
              child: Padding(
                padding: const EdgeInsets.symmetric(
                  horizontal: StitchTheme.spaceSM,
                  vertical: StitchTheme.spaceSM + 2,
                ),
                child: Row(
                  children: [
                    Icon(
                      icon,
                      size: 20,
                      color: isExpanded
                          ? colors.primary
                          : theme.textTheme.bodySmall?.color,
                    ),
                    const SizedBox(width: StitchTheme.spaceSM),
                    Expanded(
                      child: Text(
                        title,
                        style: theme.textTheme.titleMedium?.copyWith(
                          color: isExpanded
                              ? colors.primary
                              : theme.textTheme.bodyLarge?.color,
                        ),
                      ),
                    ),
                    AnimatedRotation(
                      turns: isExpanded ? 0.5 : 0,
                      duration: const Duration(milliseconds: 200),
                      child: Icon(
                        Icons.expand_more_rounded,
                        size: 20,
                        color: isExpanded
                            ? colors.primary
                            : theme.textTheme.bodySmall?.color,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),

          // Section Content
          AnimatedCrossFade(
            firstChild: Padding(
              padding: const EdgeInsets.fromLTRB(
                StitchTheme.spaceMD,
                StitchTheme.spaceSM,
                StitchTheme.spaceMD,
                StitchTheme.spaceSM,
              ),
              child: child,
            ),
            secondChild: const SizedBox.shrink(),
            crossFadeState: isExpanded
                ? CrossFadeState.showFirst
                : CrossFadeState.showSecond,
            duration: const Duration(milliseconds: 200),
            sizeCurve: Curves.easeInOut,
          ),
        ],
      ),
    );
  }
}
