import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:window_manager/window_manager.dart';
import 'package:media_kit/media_kit.dart';
import 'dart:io';
import 'services/storage_service.dart';
import 'services/database_service.dart';
import 'providers/config_provider.dart';
import 'providers/conversation_provider.dart';
import 'providers/system_prompt_provider.dart';
import 'providers/generation_state_provider.dart';
import 'providers/folder_provider.dart';
import 'providers/ui_state_provider.dart';
import 'screens/main_screen.dart';
import 'screens/splash_screen.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  MediaKit.ensureInitialized();

  // Initialize window manager for desktop (fast — no heavy I/O)
  if (Platform.isWindows || Platform.isLinux || Platform.isMacOS) {
    await windowManager.ensureInitialized();

    WindowOptions windowOptions = const WindowOptions(
      minimumSize: Size(800, 600),
      title: 'AI Prompt Assistant',
      center: true,
    );
    windowManager.waitUntilReadyToShow(windowOptions, () async {
      await windowManager.maximize();
      await windowManager.show();
      await windowManager.focus();
    });
  }

  // Launch immediately with splash screen — heavy init happens in background
  runApp(const AppBootstrap());
}

/// Root widget that manages the splash → main app transition.
///
/// Shows [SplashScreen] immediately while services + providers initialize
/// in the background. Once complete, cross-fades to [MyApp].
class AppBootstrap extends StatefulWidget {
  const AppBootstrap({super.key});

  @override
  State<AppBootstrap> createState() => _AppBootstrapState();
}

class _AppBootstrapState extends State<AppBootstrap> {
  Widget? _initializedApp;

  @override
  Widget build(BuildContext context) {
    // Once initialization is done, show the real app
    if (_initializedApp != null) {
      return _initializedApp!;
    }

    // Show splash screen with dark theme while loading
    return MaterialApp(
      title: 'AI Prompt Assistant',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(useMaterial3: true).copyWith(
        scaffoldBackgroundColor: const Color(0xFF0D0B2E),
      ),
      home: SplashScreen(
        onInitialize: _initializeApp,
        onComplete: (app) {
          if (mounted) {
            setState(() => _initializedApp = app);
          }
        },
      ),
    );
  }

  /// Performs all heavy initialization and returns the fully configured main app.
  Future<Widget> _initializeApp() async {
    // Initialize database service
    final databaseService = DatabaseService();
    await databaseService.initialize();

    // Initialize storage service
    final storageService = StorageService(databaseService);
    await storageService.initialize();

    // Create providers
    final configProvider = ConfigProvider(storageService);
    await configProvider.initialize();

    final conversationProvider = ConversationProvider(storageService);

    final folderProvider = FolderProvider(databaseService);
    await folderProvider.loadFolders();

    final promptProvider = SystemPromptProvider(storageService);
    await promptProvider.initialize();

    // Set last used system prompt
    if (configProvider.config.lastSystemPromptName != null) {
      await promptProvider.selectPrompt(
        configProvider.config.lastSystemPromptName!,
      );
    } else if (promptProvider.allPrompts
        .containsKey('Default Image-to-Prompt')) {
      await promptProvider.selectPrompt('Default Image-to-Prompt');
    }

    final generationStateProvider = GenerationStateProvider();
    final uiStateProvider = UiStateProvider();

    // Return the fully initialized main app widget tree
    return MultiProvider(
      providers: [
        Provider<DatabaseService>.value(value: databaseService),
        Provider<StorageService>.value(value: storageService),
        ChangeNotifierProvider<ConfigProvider>.value(value: configProvider),
        ChangeNotifierProvider<ConversationProvider>.value(
          value: conversationProvider,
        ),
        ChangeNotifierProvider<FolderProvider>.value(
          value: folderProvider,
        ),
        ChangeNotifierProvider<SystemPromptProvider>.value(
          value: promptProvider,
        ),
        ChangeNotifierProvider<GenerationStateProvider>.value(
          value: generationStateProvider,
        ),
        ChangeNotifierProvider<UiStateProvider>.value(
          value: uiStateProvider,
        ),
      ],
      child: const MyApp(),
    );
  }
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    final configProvider = context.watch<ConfigProvider>();

    return MaterialApp(
      title: 'AI Prompt Assistant',
      debugShowCheckedModeBanner: false,
      theme: configProvider.lightTheme,
      darkTheme: configProvider.darkTheme,
      themeMode: configProvider.themeMode,
      home: const MainScreen(),
    );
  }
}
