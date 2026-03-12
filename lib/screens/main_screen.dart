import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/conversation_provider.dart';
import '../providers/config_provider.dart';
import '../providers/ui_state_provider.dart';
import '../theme/stitch_theme.dart';
import '../theme/theme_palettes.dart';
import '../widgets/sidebar.dart';
import 'chat_screen.dart';
import 'bulk_analysis_screen.dart';
import 'recommended_models_screen.dart';

class MainScreen extends StatefulWidget {
  const MainScreen({super.key});

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen>
    with SingleTickerProviderStateMixin {
  late TabController _tabController;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 3, vsync: this);
    _tabController.addListener(_onTabChanged);
  }

  void _onTabChanged() {
    if (!mounted) return;
    context.read<UiStateProvider>().setTabIndex(_tabController.index);
    context.read<ConversationProvider>().switchTab(_tabController.index);
  }

  @override
  void dispose() {
    _tabController.removeListener(_onTabChanged);
    _tabController.dispose();
    super.dispose();
  }

  void _cycleThemeMode(ConfigProvider configProvider) {
    const modes = ['system', 'light', 'dark'];
    final currentIndex = modes.indexOf(configProvider.themeModeString);
    final nextIndex = (currentIndex + 1) % modes.length;
    configProvider.updateThemeMode(modes[nextIndex]);
  }

  IconData _themeIcon(String mode) {
    switch (mode) {
      case 'light':
        return Icons.light_mode_rounded;
      case 'dark':
        return Icons.dark_mode_rounded;
      default:
        return Icons.brightness_auto_rounded;
    }
  }

  String _themeTooltip(String mode) {
    switch (mode) {
      case 'light':
        return 'Theme: Light (tap to switch)';
      case 'dark':
        return 'Theme: Dark (tap to switch)';
      default:
        return 'Theme: System (tap to switch)';
    }
  }

  @override
  Widget build(BuildContext context) {
    final configProvider = context.watch<ConfigProvider>();
    final screenWidth = MediaQuery.of(context).size.width;
    final isWide = screenWidth >= 1200;
    final isNarrow = screenWidth < 800;

    final tabBar = TabBar(
      controller: _tabController,
      tabs: isNarrow
          ? const [
              Tab(icon: Icon(Icons.chat_rounded), text: 'Chat'),
              Tab(icon: Icon(Icons.photo_library_rounded), text: 'Bulk'),
              Tab(icon: Icon(Icons.model_training_rounded), text: 'Models'),
            ]
          : const [
              Tab(
                icon: Icon(Icons.chat_rounded),
                text: 'Chat',
              ),
              Tab(
                icon: Icon(Icons.photo_library_rounded),
                text: 'Bulk Analysis',
              ),
              Tab(
                icon: Icon(Icons.model_training_rounded),
                text: 'Recommended Models',
              ),
            ],
    );

    final appBar = AppBar(
      automaticallyImplyLeading: !isWide,
      title: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            padding: const EdgeInsets.all(6),
            decoration: BoxDecoration(
              color:
                  Theme.of(context).colorScheme.primary.withValues(alpha: 0.12),
              borderRadius: BorderRadius.circular(StitchTheme.radiusSM),
            ),
            child: Icon(
              Icons.auto_awesome_rounded,
              size: 20,
              color: Theme.of(context).colorScheme.primary,
            ),
          ),
          const SizedBox(width: 10),
          const Text('AI Prompt Assistant'),
        ],
      ),
      actions: [
        // Theme palette picker
        PopupMenuButton<String>(
          tooltip: 'Choose theme',
          icon: Icon(
            Icons.palette_rounded,
            color: Theme.of(context).colorScheme.primary,
          ),
          onSelected: (value) => configProvider.updateThemeName(value),
          itemBuilder: (context) => themeOrder.map((id) {
            final palette = themePalettes[id]!;
            final isSelected = configProvider.themeName == id;
            return PopupMenuItem<String>(
              value: id,
              child: Row(
                children: [
                  Container(
                    width: 18,
                    height: 18,
                    decoration: BoxDecoration(
                      color: palette.seedColor,
                      shape: BoxShape.circle,
                      border: Border.all(
                        color: isSelected
                            ? Theme.of(context).colorScheme.primary
                            : Theme.of(context)
                                .colorScheme
                                .outline
                                .withValues(alpha: 0.3),
                        width: isSelected ? 2 : 1,
                      ),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Text(
                    palette.displayName,
                    style: TextStyle(
                      fontWeight:
                          isSelected ? FontWeight.w600 : FontWeight.w400,
                    ),
                  ),
                  if (isSelected) ...[
                    const Spacer(),
                    Icon(
                      Icons.check_rounded,
                      size: 18,
                      color: Theme.of(context).colorScheme.primary,
                    ),
                  ],
                ],
              ),
            );
          }).toList(),
        ),
        // Light/dark/system toggle
        IconButton(
          onPressed: () => _cycleThemeMode(configProvider),
          icon: Icon(_themeIcon(configProvider.themeModeString)),
          tooltip: _themeTooltip(configProvider.themeModeString),
        ),
        const SizedBox(width: 4),
      ],
      bottom: tabBar,
    );

    final body = TabBarView(
      controller: _tabController,
      children: const [
        ChatScreen(),
        BulkAnalysisScreen(),
        RecommendedModelsScreen(),
      ],
    );

    // Wide layout: persistent sidebar panel
    if (isWide) {
      return Row(
        children: [
          const SizedBox(
            width: StitchTheme.sidebarWidth,
            child: Sidebar(asPersistentPanel: true),
          ),
          Container(
            width: 1,
            color: Theme.of(context).dividerColor,
          ),
          Expanded(
            child: Scaffold(
              appBar: appBar,
              body: body,
            ),
          ),
        ],
      );
    }

    // Medium/Narrow layout: drawer sidebar
    return Scaffold(
      appBar: appBar,
      drawer: const Sidebar(),
      body: body,
    );
  }
}
