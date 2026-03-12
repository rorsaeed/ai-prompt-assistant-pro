import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'package:media_kit/media_kit.dart';
import 'package:media_kit_video/media_kit_video.dart';
import 'package:file_picker/file_picker.dart';
import 'package:desktop_drop/desktop_drop.dart';
import 'package:cross_file/cross_file.dart';
import 'package:flutter_markdown/flutter_markdown.dart';
import '../providers/config_provider.dart';
import '../providers/conversation_provider.dart';
import '../providers/system_prompt_provider.dart';
import '../providers/generation_state_provider.dart';
import '../services/api_client.dart';
import '../theme/stitch_theme.dart';

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen>
    with AutomaticKeepAliveClientMixin {
  final _messageController = TextEditingController();
  final _scrollController = ScrollController();
  bool _isDragging = false;

  @override
  bool get wantKeepAlive => true;

  @override
  void dispose() {
    _messageController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  Future<void> _handleDroppedFiles(List<XFile> files) async {
    final conversationProvider = context.read<ConversationProvider>();
    final configProvider = context.read<ConfigProvider>();

    for (final file in files) {
      final ext = file.path.split('.').last.toLowerCase();

      if (['png', 'jpg', 'jpeg', 'webp'].contains(ext)) {
        await conversationProvider.addUploadedFile(file.path, file.name);
      } else if (['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'wmv', 'm4v']
          .contains(ext)) {
        if (configProvider.currentProvider == 'Google') {
          await conversationProvider.addUploadedVideo(file.path, file.name);
        } else {
          if (mounted) {
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(
                content: Text('Video upload only supported for Google'),
              ),
            );
          }
        }
      }
    }
  }

  Future<void> _pickImages() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['png', 'jpg', 'jpeg', 'webp'],
      allowMultiple: true,
    );

    if (result != null && mounted) {
      final conversationProvider = context.read<ConversationProvider>();
      for (final file in result.files) {
        if (file.path != null) {
          await conversationProvider.addUploadedFile(file.path!, file.name);
        }
      }
    }
  }

  Future<void> _pickVideos() async {
    final configProvider = context.read<ConfigProvider>();
    if (configProvider.currentProvider != 'Google') {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Video upload only supported for Google'),
          ),
        );
      }
      return;
    }

    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: [
        'mp4',
        'avi',
        'mov',
        'mkv',
        'webm',
        'flv',
        'wmv',
        'm4v',
      ],
      allowMultiple: true,
    );

    if (result != null && mounted) {
      final conversationProvider = context.read<ConversationProvider>();
      for (final file in result.files) {
        if (file.path != null) {
          await conversationProvider.addUploadedVideo(file.path!, file.name);
        }
      }
    }
  }

  Future<void> _sendMessage({bool analyzeOnly = false}) async {
    final conversationProvider = context.read<ConversationProvider>();
    final configProvider = context.read<ConfigProvider>();
    final generationState = context.read<GenerationStateProvider>();

    final messageText = analyzeOnly
        ? (conversationProvider.uploadedFiles.isNotEmpty ||
                conversationProvider.uploadedVideos.isNotEmpty
            ? 'Analyze'
            : '')
        : _messageController.text.trim();

    if (messageText.isEmpty &&
        conversationProvider.uploadedFiles.isEmpty &&
        conversationProvider.uploadedVideos.isEmpty) {
      return;
    }

    // Validate model selection
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

    // Create user message
    final userMessage = conversationProvider.createUserMessage(messageText);
    conversationProvider.addMessage(userMessage);

    // Clear input
    _messageController.clear();

    // Set generating state
    generationState.setGenerating(true);

    // Scroll to bottom
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });

    // Generate responses
    await _generateResponses();

    // Auto-save
    await conversationProvider.autoSaveConversation();

    // Reset generating state
    generationState.setGenerating(false);
  }

  Future<void> _generateResponses() async {
    final conversationProvider = context.read<ConversationProvider>();
    final configProvider = context.read<ConfigProvider>();
    final promptProvider = context.read<SystemPromptProvider>();

    final selectedModels = configProvider.getSelectedModels(
      configProvider.currentProvider,
    );
    final messages = conversationProvider.messages;

    // Add system prompt
    final messagesWithSystem = [
      conversationProvider.createSystemMessage(
        promptProvider.currentPromptContent,
      ),
      ...messages,
    ];

    // Extract image and video paths
    final lastUserMsg = messages.lastWhere((m) => m.role == 'user');
    final imagePaths = lastUserMsg.images?.map((img) => img.path).toList();
    final videoPaths = lastUserMsg.videos?.map((vid) => vid.path).toList();

    final generationState = context.read<GenerationStateProvider>();

    // Generate response for each model
    for (final model in selectedModels) {
      if (generationState.isCancelled) break;

      try {
        final client = APIClient(
          provider: configProvider.currentProvider,
          baseUrl: configProvider.currentProviderConfig.apiBaseUrl,
          googleApiKey: configProvider.config.googleApiKey,
          ollamaKeepAlive: configProvider.getKeepAlive(),
          unloadAfterResponse: configProvider.getUnloadAfterResponse(),
        );

        final buffer = StringBuffer();

        // Create a placeholder message
        final placeholderMsg = conversationProvider.createAssistantMessage(
          '',
          model,
        );
        conversationProvider.addMessage(placeholderMsg);
        final messageIndex = conversationProvider.messages.length - 1;

        await for (final chunk in client.generateChatResponse(
          model: model,
          messages: messagesWithSystem,
          imagePaths: imagePaths,
          videoPaths: videoPaths,
        )) {
          if (generationState.isCancelled) break;

          buffer.write(chunk);

          // Parse thinking content from the raw buffer
          final rawText = buffer.toString();
          final parsed = APIClient.parseThinkingContent(rawText);
          final isThinking = APIClient.isCurrentlyThinking(rawText);

          // If still inside a <think> block, show content so far as thinking
          String thinkingText = parsed.thinking;
          String displayText = parsed.content;

          if (isThinking) {
            // Extract partial thinking (unclosed <think> tag)
            final lastOpenIdx = rawText.lastIndexOf('<think>');
            if (lastOpenIdx != -1) {
              final partialThinking = rawText.substring(lastOpenIdx + 7).trim();
              if (thinkingText.isNotEmpty) {
                thinkingText += '\n$partialThinking';
              } else {
                thinkingText = partialThinking;
              }
            }
          }

          // Update the message
          final updatedMsg = placeholderMsg.copyWith(
            content: rawText,
            displayContent: displayText,
            thinking: thinkingText.isNotEmpty ? thinkingText : null,
          );
          conversationProvider.messages[messageIndex] = updatedMsg;

          // Trigger rebuild
          if (mounted) {
            setState(() {});
          }

          // Scroll to bottom
          WidgetsBinding.instance.addPostFrameCallback((_) {
            if (_scrollController.hasClients) {
              _scrollController.animateTo(
                _scrollController.position.maxScrollExtent,
                duration: const Duration(milliseconds: 100),
                curve: Curves.easeOut,
              );
            }
          });
        }
      } catch (e) {
        if (mounted && !generationState.isCancelled) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Error with $model: ${e.toString()}')),
          );
        }
      }
    }
  }

  Future<void> _regenerateResponse(int messageIndex) async {
    final conversationProvider = context.read<ConversationProvider>();
    final generationState = context.read<GenerationStateProvider>();

    // Remove the assistant message
    conversationProvider.deleteMessage(messageIndex);

    // Set generating state
    generationState.setGenerating(true);

    // Generate new response
    await _generateResponses();

    // Auto-save
    await conversationProvider.autoSaveConversation();

    // Reset generating state
    generationState.setGenerating(false);
  }

  @override
  Widget build(BuildContext context) {
    super.build(context);
    final conversationProvider = context.watch<ConversationProvider>();
    final generationState = context.watch<GenerationStateProvider>();
    final configProvider = context.watch<ConfigProvider>();
    final theme = Theme.of(context);
    final colors = theme.colorScheme;

    return DropTarget(
      onDragDone: (details) async {
        await _handleDroppedFiles(details.files);
        setState(() => _isDragging = false);
      },
      onDragEntered: (_) => setState(() => _isDragging = true),
      onDragExited: (_) => setState(() => _isDragging = false),
      child: Stack(
        children: [
          Column(
            children: [
              // Messages List
              Expanded(
                child: conversationProvider.messages.isEmpty
                    ? Center(
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
                                Icons.chat_bubble_outline_rounded,
                                size: 48,
                                color: colors.primary.withValues(alpha: 0.5),
                              ),
                            ),
                            const SizedBox(height: StitchTheme.spaceMD),
                            Text(
                              'Start a conversation',
                              style: theme.textTheme.titleMedium?.copyWith(
                                color: colors.onSurface.withValues(alpha: 0.7),
                              ),
                            ),
                            const SizedBox(height: StitchTheme.spaceSM),
                            Text(
                              'Upload images and send a message to begin',
                              style: theme.textTheme.bodyMedium?.copyWith(
                                color: colors.onSurfaceVariant,
                              ),
                            ),
                          ],
                        ),
                      )
                    : ListView.builder(
                        controller: _scrollController,
                        padding: const EdgeInsets.all(StitchTheme.spaceMD),
                        itemCount: conversationProvider.messages.length,
                        itemBuilder: (context, index) {
                          return _buildMessageItem(
                            conversationProvider.messages[index],
                            index,
                            generationState.isGenerating,
                          );
                        },
                      ),
              ),

              // Input Section
              if (!generationState.isGenerating) ...[
                Container(
                  decoration: BoxDecoration(
                    color: colors.surface,
                    border: Border(
                      top: BorderSide(color: colors.outlineVariant, width: 1),
                    ),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      // Image Upload Section
                      if (conversationProvider.uploadedFiles.isNotEmpty ||
                          conversationProvider.uploadedVideos.isEmpty)
                        Padding(
                          padding: const EdgeInsets.fromLTRB(
                            StitchTheme.spaceMD,
                            StitchTheme.spaceSM,
                            StitchTheme.spaceMD,
                            0,
                          ),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Row(
                                children: [
                                  Icon(
                                    Icons.image_rounded,
                                    size: 16,
                                    color: colors.onSurfaceVariant,
                                  ),
                                  const SizedBox(width: 6),
                                  Text(
                                    'Images',
                                    style: theme.textTheme.labelMedium,
                                  ),
                                  const Spacer(),
                                  TextButton.icon(
                                    onPressed: _pickImages,
                                    icon: const Icon(
                                        Icons.add_photo_alternate_rounded,
                                        size: 16),
                                    label: const Text('Add'),
                                  ),
                                ],
                              ),
                              if (conversationProvider
                                  .uploadedFiles.isNotEmpty) ...[
                                const SizedBox(height: StitchTheme.spaceXS),
                                Wrap(
                                  spacing: StitchTheme.spaceSM,
                                  runSpacing: StitchTheme.spaceSM,
                                  children: conversationProvider.uploadedFiles
                                      .asMap()
                                      .entries
                                      .map((entry) {
                                    return _buildImageThumbnail(
                                      entry.key,
                                      entry.value.path,
                                    );
                                  }).toList(),
                                ),
                              ],
                            ],
                          ),
                        ),

                      // Video Upload Section (Google only)
                      if (configProvider.currentProvider == 'Google')
                        Padding(
                          padding: const EdgeInsets.symmetric(
                            horizontal: StitchTheme.spaceMD,
                          ),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Row(
                                children: [
                                  Icon(
                                    Icons.videocam_rounded,
                                    size: 16,
                                    color: colors.onSurfaceVariant,
                                  ),
                                  const SizedBox(width: 6),
                                  Text(
                                    'Videos (Google)',
                                    style: theme.textTheme.labelMedium,
                                  ),
                                  const Spacer(),
                                  TextButton.icon(
                                    onPressed: _pickVideos,
                                    icon: const Icon(
                                        Icons.video_library_rounded,
                                        size: 16),
                                    label: const Text('Add'),
                                  ),
                                ],
                              ),
                              if (conversationProvider
                                  .uploadedVideos.isNotEmpty) ...[
                                const SizedBox(height: StitchTheme.spaceXS),
                                Wrap(
                                  spacing: StitchTheme.spaceSM,
                                  runSpacing: StitchTheme.spaceSM,
                                  children: conversationProvider.uploadedVideos
                                      .asMap()
                                      .entries
                                      .map((entry) {
                                    return _buildVideoThumbnail(
                                      entry.key,
                                      entry.value,
                                    );
                                  }).toList(),
                                ),
                              ],
                            ],
                          ),
                        ),

                      // Message Input
                      Padding(
                        padding: const EdgeInsets.all(StitchTheme.spaceSM),
                        child: Row(
                          crossAxisAlignment: CrossAxisAlignment.end,
                          children: [
                            // New Chat button with dropdown for keep uploads
                            Padding(
                              padding: const EdgeInsets.only(
                                  right: StitchTheme.spaceSM),
                              child: PopupMenuButton<bool>(
                                tooltip: 'New Chat',
                                offset: const Offset(0, -100),
                                onSelected: (keepUploads) {
                                  final convProvider =
                                      context.read<ConversationProvider>();
                                  convProvider.startNewChat(
                                      keepUploads: keepUploads);
                                },
                                itemBuilder: (context) => [
                                  const PopupMenuItem(
                                    value: false,
                                    child: ListTile(
                                      leading: Icon(Icons.add_comment_rounded),
                                      title: Text('New Chat'),
                                      dense: true,
                                      contentPadding: EdgeInsets.zero,
                                    ),
                                  ),
                                  const PopupMenuItem(
                                    value: true,
                                    child: ListTile(
                                      leading: Icon(Icons.image_rounded),
                                      title: Text('New Chat (Keep Uploads)'),
                                      dense: true,
                                      contentPadding: EdgeInsets.zero,
                                    ),
                                  ),
                                ],
                                child: Container(
                                  decoration: BoxDecoration(
                                    color: colors.surfaceContainerHighest,
                                    borderRadius: BorderRadius.circular(8),
                                  ),
                                  padding: const EdgeInsets.all(8),
                                  child: Icon(
                                    Icons.add_comment_rounded,
                                    color: colors.onSurface,
                                  ),
                                ),
                              ),
                            ),
                            if (conversationProvider.uploadedFiles.isNotEmpty ||
                                conversationProvider.uploadedVideos.isNotEmpty)
                              Padding(
                                padding: const EdgeInsets.only(
                                    right: StitchTheme.spaceSM),
                                child: FilledButton.tonalIcon(
                                  onPressed: () =>
                                      _sendMessage(analyzeOnly: true),
                                  icon: const Icon(Icons.auto_awesome_rounded,
                                      size: 18),
                                  label: Text(
                                    conversationProvider
                                            .uploadedVideos.isNotEmpty
                                        ? 'Analyze All'
                                        : 'Analyze',
                                  ),
                                ),
                              ),
                              // Removed Prompt Director button
                            Expanded(
                              child: TextField(
                                controller: _messageController,
                                decoration: InputDecoration(
                                  hintText: 'Type a message...',
                                  suffixIcon: IconButton(
                                    onPressed: () => _sendMessage(),
                                    icon: Icon(
                                      Icons.send_rounded,
                                      color: colors.primary,
                                    ),
                                    tooltip: 'Send',
                                  ),
                                ),
                                maxLines: 6,
                                minLines: 1,
                                textInputAction: TextInputAction.send,
                                onSubmitted: (_) => _sendMessage(),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ] else ...[
                Container(
                  padding: const EdgeInsets.all(StitchTheme.spaceMD),
                  decoration: BoxDecoration(
                    color: colors.surface,
                    border: Border(
                      top: BorderSide(color: colors.outlineVariant, width: 1),
                    ),
                  ),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
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
                        'Generating responses...',
                        style: theme.textTheme.bodyMedium?.copyWith(
                          color: colors.onSurfaceVariant,
                        ),
                      ),
                      const SizedBox(width: StitchTheme.spaceMD),
                      FilledButton.tonalIcon(
                        onPressed: () {
                          context
                              .read<GenerationStateProvider>()
                              .cancelGeneration();
                        },
                        icon: const Icon(Icons.stop_rounded, size: 18),
                        label: const Text('Stop'),
                        style: FilledButton.styleFrom(
                          backgroundColor: colors.errorContainer,
                          foregroundColor: colors.onErrorContainer,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ],
          ),
          if (_isDragging)
            Positioned.fill(
              child: Container(
                color: colors.primary.withOpacity(0.1),
                child: Center(
                  child: Container(
                    padding: const EdgeInsets.all(32),
                    decoration: BoxDecoration(
                      color: colors.surface,
                      borderRadius: BorderRadius.circular(16),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withOpacity(0.1),
                          blurRadius: 10,
                          spreadRadius: 2,
                        ),
                      ],
                    ),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(Icons.upload_file_rounded,
                            size: 48, color: colors.primary),
                        const SizedBox(height: 16),
                        Text(
                          'Drop images or videos here',
                          style: theme.textTheme.titleLarge?.copyWith(
                            color: colors.primary,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildMessageItem(dynamic message, int index, bool isGenerating) {
    final conversationProvider = context.read<ConversationProvider>();
    final isUser = message.role == 'user';
    final theme = Theme.of(context);
    final colors = theme.colorScheme;

    return Padding(
      padding: const EdgeInsets.only(bottom: StitchTheme.spaceMD),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Avatar
          Container(
            width: 36,
            height: 36,
            decoration: BoxDecoration(
              color: isUser
                  ? colors.tertiary.withValues(alpha: 0.2)
                  : colors.primary.withValues(alpha: 0.15),
              shape: BoxShape.circle,
            ),
            child: Icon(
              isUser ? Icons.person_rounded : Icons.smart_toy_rounded,
              color: isUser ? colors.tertiary : colors.primary,
              size: 18,
            ),
          ),
          const SizedBox(width: StitchTheme.spaceSM),

          // Message Content
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Role and Model label
                Padding(
                  padding: const EdgeInsets.only(bottom: 4),
                  child: Text(
                    isUser ? 'You' : message.model ?? 'Assistant',
                    style: theme.textTheme.labelMedium?.copyWith(
                      fontWeight: FontWeight.w600,
                      color: isUser ? colors.tertiary : colors.primary,
                    ),
                  ),
                ),

                // Thinking Section (collapsible, hidden by default)
                if (!isUser &&
                    message.thinking != null &&
                    message.thinking!.isNotEmpty)
                  _buildThinkingSection(message.thinking!, theme, colors),

                // Message Bubble
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(StitchTheme.spaceMD),
                  decoration: BoxDecoration(
                    color: isUser
                        ? colors.tertiaryContainer.withValues(alpha: 0.3)
                        : StitchTheme.surfaceContainerColor(context),
                    borderRadius: BorderRadius.only(
                      topLeft: Radius.circular(
                        isUser ? StitchTheme.radiusMD : StitchTheme.spaceXS,
                      ),
                      topRight: Radius.circular(
                        isUser ? StitchTheme.spaceXS : StitchTheme.radiusMD,
                      ),
                      bottomLeft: const Radius.circular(StitchTheme.radiusMD),
                      bottomRight: const Radius.circular(StitchTheme.radiusMD),
                    ),
                    border: Border.all(
                      color: colors.outlineVariant.withValues(alpha: 0.5),
                      width: 1,
                    ),
                  ),
                  child: message.displayContent.isEmpty &&
                          message.thinking != null &&
                          message.thinking!.isNotEmpty
                      ? Text(
                          'Thinking...',
                          style: theme.textTheme.bodyMedium?.copyWith(
                            color: colors.onSurfaceVariant,
                            fontStyle: FontStyle.italic,
                          ),
                        )
                      : _MarkdownWithCodeBlocks(
                          data: message.displayContent,
                          styleSheet: MarkdownStyleSheet(
                            p: theme.textTheme.bodyMedium,
                            code: theme.textTheme.bodySmall?.copyWith(
                              fontFamily: 'monospace',
                              backgroundColor: colors.surfaceContainerHighest,
                            ),
                            codeblockDecoration: BoxDecoration(
                              color: colors.surfaceContainerHighest,
                              borderRadius:
                                  BorderRadius.circular(StitchTheme.radiusSM),
                              border: Border.all(
                                color: colors.outlineVariant
                                    .withValues(alpha: 0.5),
                              ),
                            ),
                            blockquoteDecoration: BoxDecoration(
                              border: Border(
                                left: BorderSide(
                                  color: colors.primary.withValues(alpha: 0.5),
                                  width: 3,
                                ),
                              ),
                            ),
                            h1: theme.textTheme.headlineMedium,
                            h2: theme.textTheme.titleLarge,
                            h3: theme.textTheme.titleMedium,
                          ),
                        ),
                ),

                // Images
                if (message.images != null && message.images!.isNotEmpty) ...[
                  const SizedBox(height: StitchTheme.spaceSM),
                  Wrap(
                    spacing: StitchTheme.spaceSM,
                    runSpacing: StitchTheme.spaceSM,
                    children: message.images!.map<Widget>((img) {
                      return GestureDetector(
                        onTap: () => _showImageDialog(img.path),
                        child: Container(
                          width: 140,
                          height: 140,
                          decoration: BoxDecoration(
                            border: Border.all(
                              color: colors.outlineVariant,
                              width: 1,
                            ),
                            borderRadius:
                                BorderRadius.circular(StitchTheme.radiusSM),
                          ),
                          child: ClipRRect(
                            borderRadius:
                                BorderRadius.circular(StitchTheme.radiusSM),
                            child: Image.file(
                              File(img.path),
                              fit: BoxFit.cover,
                            ),
                          ),
                        ),
                      );
                    }).toList(),
                  ),
                ],

                // Videos
                if (message.videos != null && message.videos!.isNotEmpty) ...[
                  const SizedBox(height: StitchTheme.spaceSM),
                  Wrap(
                    spacing: StitchTheme.spaceSM,
                    runSpacing: StitchTheme.spaceSM,
                    children: message.videos!.map<Widget>((vid) {
                      return GestureDetector(
                        onTap: () => _showVideoDialog(vid.path, vid.name),
                        child: Container(
                          width: 180,
                          height: 130,
                          decoration: BoxDecoration(
                            color: colors.surfaceContainerHighest,
                            border: Border.all(
                              color: colors.outlineVariant,
                              width: 1,
                            ),
                            borderRadius:
                                BorderRadius.circular(StitchTheme.radiusSM),
                          ),
                          child: ClipRRect(
                            borderRadius:
                                BorderRadius.circular(StitchTheme.radiusSM),
                            child: Stack(
                              fit: StackFit.expand,
                              children: [
                                FutureBuilder<Widget>(
                                  future: _generateVideoThumbnail(vid.path),
                                  builder: (context, snapshot) {
                                    if (snapshot.hasData) {
                                      return snapshot.data!;
                                    }
                                    return Container(
                                      color: colors.surfaceContainerHighest,
                                      child: Icon(
                                        Icons.video_library_rounded,
                                        size: 40,
                                        color: colors.onSurfaceVariant,
                                      ),
                                    );
                                  },
                                ),
                                Center(
                                  child: Container(
                                    padding: const EdgeInsets.all(8),
                                    decoration: const BoxDecoration(
                                      color: Colors.black54,
                                      shape: BoxShape.circle,
                                    ),
                                    child: const Icon(
                                      Icons.play_arrow_rounded,
                                      color: Colors.white,
                                      size: 28,
                                    ),
                                  ),
                                ),
                                Positioned(
                                  bottom: 0,
                                  left: 0,
                                  right: 0,
                                  child: Container(
                                    padding: const EdgeInsets.symmetric(
                                      horizontal: StitchTheme.spaceXS,
                                      vertical: 2,
                                    ),
                                    color: Colors.black54,
                                    child: Text(
                                      vid.name,
                                      style:
                                          theme.textTheme.bodySmall?.copyWith(
                                        color: Colors.white,
                                        fontSize: 10,
                                      ),
                                      overflow: TextOverflow.ellipsis,
                                    ),
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),
                      );
                    }).toList(),
                  ),
                ],
              ],
            ),
          ),

          // Actions
          Column(
            children: [
              IconButton(
                icon: Icon(
                  Icons.delete_outline_rounded,
                  size: 18,
                  color: colors.onSurfaceVariant,
                ),
                onPressed: isGenerating
                    ? null
                    : () {
                        conversationProvider.deleteMessage(index);
                      },
                tooltip: 'Delete',
                visualDensity: VisualDensity.compact,
              ),
              IconButton(
                icon: Icon(
                  Icons.copy_rounded,
                  size: 18,
                  color: colors.onSurfaceVariant,
                ),
                onPressed: () {
                  Clipboard.setData(
                    ClipboardData(text: message.displayContent),
                  );
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(
                      content: Text(isUser ? 'Message copied to clipboard' : 'Response copied to clipboard'),
                      duration: const Duration(seconds: 1),
                    ),
                  );
                },
                tooltip: 'Copy',
                visualDensity: VisualDensity.compact,
              ),
              if (!isUser)
                IconButton(
                  icon: Icon(
                    Icons.refresh_rounded,
                    size: 18,
                    color: colors.onSurfaceVariant,
                  ),
                  onPressed:
                      isGenerating ? null : () => _regenerateResponse(index),
                  tooltip: 'Regenerate',
                  visualDensity: VisualDensity.compact,
                ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildThinkingSection(
      String thinking, ThemeData theme, ColorScheme colors) {
    return Padding(
      padding: const EdgeInsets.only(bottom: StitchTheme.spaceXS),
      child: Container(
        width: double.infinity,
        decoration: BoxDecoration(
          color: colors.surfaceContainerHighest.withValues(alpha: 0.4),
          borderRadius: BorderRadius.circular(StitchTheme.radiusSM),
          border: Border.all(
            color: colors.outlineVariant.withValues(alpha: 0.3),
            width: 1,
          ),
        ),
        child: Theme(
          data: theme.copyWith(dividerColor: Colors.transparent),
          child: ExpansionTile(
            tilePadding:
                const EdgeInsets.symmetric(horizontal: StitchTheme.spaceSM),
            childrenPadding: const EdgeInsets.fromLTRB(
              StitchTheme.spaceSM,
              0,
              StitchTheme.spaceSM,
              StitchTheme.spaceSM,
            ),
            initiallyExpanded: false,
            dense: true,
            leading: Icon(
              Icons.psychology_rounded,
              size: 18,
              color: colors.onSurfaceVariant.withValues(alpha: 0.7),
            ),
            title: Text(
              'Thinking',
              style: theme.textTheme.labelMedium?.copyWith(
                color: colors.onSurfaceVariant.withValues(alpha: 0.8),
                fontStyle: FontStyle.italic,
              ),
            ),
            children: [
              SizedBox(
                width: double.infinity,
                child: _MarkdownWithCodeBlocks(
                  data: thinking,
                  styleSheet: MarkdownStyleSheet(
                    p: theme.textTheme.bodySmall?.copyWith(
                      color: colors.onSurfaceVariant.withValues(alpha: 0.8),
                    ),
                    code: theme.textTheme.bodySmall?.copyWith(
                      fontFamily: 'monospace',
                      fontSize: 12,
                      backgroundColor: colors.surfaceContainerHighest,
                      color: colors.onSurfaceVariant,
                    ),
                    codeblockDecoration: BoxDecoration(
                      color: colors.surfaceContainerHighest,
                      borderRadius: BorderRadius.circular(StitchTheme.radiusSM),
                      border: Border.all(
                        color: colors.outlineVariant.withValues(alpha: 0.5),
                      ),
                    ),
                    blockquoteDecoration: BoxDecoration(
                      border: Border(
                        left: BorderSide(
                          color: colors.onSurfaceVariant.withValues(alpha: 0.3),
                          width: 2,
                        ),
                      ),
                    ),
                    h1: theme.textTheme.titleMedium?.copyWith(
                      color: colors.onSurfaceVariant,
                    ),
                    h2: theme.textTheme.titleSmall?.copyWith(
                      color: colors.onSurfaceVariant,
                    ),
                    h3: theme.textTheme.labelLarge?.copyWith(
                      color: colors.onSurfaceVariant,
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildImageThumbnail(int index, String path) {
    final conversationProvider = context.read<ConversationProvider>();
    final colors = Theme.of(context).colorScheme;

    return Stack(
      children: [
        GestureDetector(
          onTap: () => _showImageDialog(path),
          child: Container(
            width: 120,
            height: 120,
            decoration: BoxDecoration(
              border: Border.all(color: colors.outlineVariant),
              borderRadius: BorderRadius.circular(StitchTheme.radiusSM),
            ),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(StitchTheme.radiusSM),
              child: Image.file(File(path), fit: BoxFit.cover),
            ),
          ),
        ),
        Positioned(
          top: 4,
          right: 4,
          child: Material(
            color: colors.error,
            borderRadius: BorderRadius.circular(12),
            child: InkWell(
              onTap: () => conversationProvider.removeUploadedFile(index),
              borderRadius: BorderRadius.circular(12),
              child: Padding(
                padding: const EdgeInsets.all(4),
                child: Icon(
                  Icons.close_rounded,
                  size: 14,
                  color: colors.onError,
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildVideoThumbnail(int index, ({String path, String name}) video) {
    final conversationProvider = context.read<ConversationProvider>();
    final colors = Theme.of(context).colorScheme;
    final theme = Theme.of(context);

    return Stack(
      children: [
        GestureDetector(
          onTap: () => _showVideoDialog(video.path, video.name),
          child: Container(
            width: 160,
            height: 120,
            decoration: BoxDecoration(
              color: colors.surfaceContainerHighest,
              border: Border.all(color: colors.outlineVariant),
              borderRadius: BorderRadius.circular(StitchTheme.radiusSM),
            ),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(StitchTheme.radiusSM),
              child: Stack(
                fit: StackFit.expand,
                children: [
                  // Thumbnail from video
                  FutureBuilder<Widget>(
                    future: _generateVideoThumbnail(video.path),
                    builder: (context, snapshot) {
                      if (snapshot.hasData) {
                        return snapshot.data!;
                      }
                      return Container(
                        color: colors.surfaceContainerHighest,
                        child: Icon(
                          Icons.video_library_rounded,
                          size: 40,
                          color: colors.onSurfaceVariant,
                        ),
                      );
                    },
                  ),
                  // Play overlay
                  Center(
                    child: Container(
                      padding: const EdgeInsets.all(8),
                      decoration: const BoxDecoration(
                        color: Colors.black54,
                        shape: BoxShape.circle,
                      ),
                      child: const Icon(
                        Icons.play_arrow_rounded,
                        color: Colors.white,
                        size: 28,
                      ),
                    ),
                  ),
                  // File name at bottom
                  Positioned(
                    bottom: 0,
                    left: 0,
                    right: 0,
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: StitchTheme.spaceXS,
                        vertical: 2,
                      ),
                      color: Colors.black54,
                      child: Text(
                        video.name,
                        style: theme.textTheme.bodySmall?.copyWith(
                          color: Colors.white,
                          fontSize: 10,
                        ),
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
        Positioned(
          top: 4,
          right: 4,
          child: Material(
            color: colors.error,
            borderRadius: BorderRadius.circular(12),
            child: InkWell(
              onTap: () => conversationProvider.removeUploadedVideo(index),
              borderRadius: BorderRadius.circular(12),
              child: Padding(
                padding: const EdgeInsets.all(4),
                child: Icon(
                  Icons.close_rounded,
                  size: 14,
                  color: colors.onError,
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }

  Future<Widget> _generateVideoThumbnail(String videoPath) async {
    // Use media_kit to extract a frame
    final player = Player();

    // Mute before opening to prevent audio playing during thumbnail generation
    await player.setVolume(0);
    await player.open(Media(videoPath), play: false);
    // Wait for the first frame
    await player.stream.width.firstWhere((w) => w != null);
    // Seek slightly in to avoid black frames
    await player.seek(const Duration(milliseconds: 500));
    await Future.delayed(const Duration(milliseconds: 300));

    final screenshot = await player.screenshot();
    await player.dispose();

    if (screenshot != null) {
      return Image.memory(
        screenshot,
        fit: BoxFit.cover,
        gaplessPlayback: true,
      );
    }

    return const Icon(
      Icons.video_library_rounded,
      size: 40,
    );
  }

  void _showVideoDialog(String path, String name) {
    showDialog(
      context: context,
      builder: (context) => _VideoPlayerDialog(path: path, name: name),
    );
  }

  void _showImageDialog(String path) {
    showDialog(
      context: context,
      builder: (context) => Dialog(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Expanded(
              child: ClipRRect(
                borderRadius: const BorderRadius.vertical(
                  top: Radius.circular(StitchTheme.radiusLG),
                ),
                child: Image.file(File(path)),
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(StitchTheme.spaceSM),
              child: TextButton(
                onPressed: () => Navigator.pop(context),
                child: const Text('Close'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _VideoPlayerDialog extends StatefulWidget {
  final String path;
  final String name;

  const _VideoPlayerDialog({required this.path, required this.name});

  @override
  State<_VideoPlayerDialog> createState() => _VideoPlayerDialogState();
}

class _VideoPlayerDialogState extends State<_VideoPlayerDialog> {
  late final Player _player;
  late final VideoController _controller;

  @override
  void initState() {
    super.initState();
    _player = Player();
    _controller = VideoController(_player);
    _player.open(Media(widget.path));
  }

  @override
  void dispose() {
    _player.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;

    return Dialog(
      backgroundColor: colors.surface,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(StitchTheme.radiusLG),
      ),
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 700, maxHeight: 500),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // Title bar
            Padding(
              padding: const EdgeInsets.symmetric(
                horizontal: StitchTheme.spaceMD,
                vertical: StitchTheme.spaceSM,
              ),
              child: Row(
                children: [
                  Icon(Icons.video_library_rounded,
                      size: 18, color: colors.primary),
                  const SizedBox(width: StitchTheme.spaceXS),
                  Expanded(
                    child: Text(
                      widget.name,
                      style: Theme.of(context).textTheme.titleSmall,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                  IconButton(
                    icon: const Icon(Icons.close, size: 20),
                    onPressed: () => Navigator.pop(context),
                    visualDensity: VisualDensity.compact,
                  ),
                ],
              ),
            ),
            const Divider(height: 1),

            // Video with built-in controls
            Flexible(
              child: ClipRRect(
                borderRadius: const BorderRadius.only(
                  bottomLeft: Radius.circular(StitchTheme.radiusLG),
                  bottomRight: Radius.circular(StitchTheme.radiusLG),
                ),
                child: Video(
                  controller: _controller,
                  controls: MaterialVideoControls,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Regex to match fenced code blocks: ```lang\ncode\n```
final _codeBlockRegex = RegExp(
  r'```(\w*)\n(.*?)```',
  dotAll: true,
);

/// Widget that renders markdown with code blocks having a copy button.
/// Splits content at fenced code blocks and renders each segment separately
/// to avoid flutter_markdown builder assertion issues.
class _MarkdownWithCodeBlocks extends StatelessWidget {
  final String data;
  final MarkdownStyleSheet? styleSheet;

  const _MarkdownWithCodeBlocks({
    required this.data,
    this.styleSheet,
  });

  @override
  Widget build(BuildContext context) {
    final segments = _splitByCodeBlocks(data);
    final theme = Theme.of(context);
    final colors = theme.colorScheme;

    if (segments.length == 1 && segments.first.type == _SegmentType.markdown) {
      // No code blocks, render as plain markdown
      return MarkdownBody(
        data: data,
        selectable: true,
        styleSheet: styleSheet,
      );
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: segments.map((segment) {
        if (segment.type == _SegmentType.codeBlock) {
          return _CodeBlockWidget(
            code: segment.content,
            language: segment.language,
            theme: theme,
            colors: colors,
          );
        } else {
          final text = segment.content.trim();
          if (text.isEmpty) return const SizedBox.shrink();
          return MarkdownBody(
            data: text,
            selectable: true,
            styleSheet: styleSheet,
          );
        }
      }).toList(),
    );
  }

  List<_ContentSegment> _splitByCodeBlocks(String text) {
    final segments = <_ContentSegment>[];
    int lastEnd = 0;

    for (final match in _codeBlockRegex.allMatches(text)) {
      // Add markdown before this code block
      if (match.start > lastEnd) {
        segments.add(_ContentSegment(
          type: _SegmentType.markdown,
          content: text.substring(lastEnd, match.start),
        ));
      }

      // Add the code block
      segments.add(_ContentSegment(
        type: _SegmentType.codeBlock,
        content: match.group(2) ?? '',
        language: (match.group(1)?.isNotEmpty == true) ? match.group(1) : null,
      ));

      lastEnd = match.end;
    }

    // Add remaining markdown after last code block
    if (lastEnd < text.length) {
      segments.add(_ContentSegment(
        type: _SegmentType.markdown,
        content: text.substring(lastEnd),
      ));
    }

    if (segments.isEmpty) {
      segments.add(_ContentSegment(
        type: _SegmentType.markdown,
        content: text,
      ));
    }

    return segments;
  }
}

enum _SegmentType { markdown, codeBlock }

class _ContentSegment {
  final _SegmentType type;
  final String content;
  final String? language;

  _ContentSegment({
    required this.type,
    required this.content,
    this.language,
  });
}

class _CodeBlockWidget extends StatefulWidget {
  final String code;
  final String? language;
  final ThemeData theme;
  final ColorScheme colors;

  const _CodeBlockWidget({
    required this.code,
    this.language,
    required this.theme,
    required this.colors,
  });

  @override
  State<_CodeBlockWidget> createState() => _CodeBlockWidgetState();
}

class _CodeBlockWidgetState extends State<_CodeBlockWidget> {
  bool _copied = false;

  void _copyCode() {
    Clipboard.setData(ClipboardData(text: widget.code.trimRight()));
    setState(() => _copied = true);
    Future.delayed(const Duration(seconds: 2), () {
      if (mounted) setState(() => _copied = false);
    });
  }

  @override
  Widget build(BuildContext context) {
    final colors = widget.colors;
    final theme = widget.theme;

    return Container(
      margin: const EdgeInsets.symmetric(vertical: 4),
      decoration: BoxDecoration(
        color: colors.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(StitchTheme.radiusSM),
        border: Border.all(
          color: colors.outlineVariant.withValues(alpha: 0.5),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Header with language label and copy button
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            decoration: BoxDecoration(
              color: colors.surfaceContainerHighest.withValues(alpha: 0.8),
              border: Border(
                bottom: BorderSide(
                  color: colors.outlineVariant.withValues(alpha: 0.4),
                ),
              ),
              borderRadius: const BorderRadius.only(
                topLeft: Radius.circular(StitchTheme.radiusSM),
                topRight: Radius.circular(StitchTheme.radiusSM),
              ),
            ),
            child: Row(
              children: [
                if (widget.language != null && widget.language!.isNotEmpty)
                  Text(
                    widget.language!,
                    style: theme.textTheme.labelSmall?.copyWith(
                      color: colors.onSurfaceVariant.withValues(alpha: 0.7),
                      fontFamily: 'monospace',
                    ),
                  ),
                const Spacer(),
                InkWell(
                  onTap: _copyCode,
                  borderRadius: BorderRadius.circular(4),
                  child: Padding(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 6,
                      vertical: 2,
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          _copied ? Icons.check_rounded : Icons.copy_rounded,
                          size: 14,
                          color: _copied
                              ? colors.primary
                              : colors.onSurfaceVariant.withValues(alpha: 0.7),
                        ),
                        const SizedBox(width: 4),
                        Text(
                          _copied ? 'Copied' : 'Copy',
                          style: theme.textTheme.labelSmall?.copyWith(
                            color: _copied
                                ? colors.primary
                                : colors.onSurfaceVariant
                                    .withValues(alpha: 0.7),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),
          // Code content
          SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            padding: const EdgeInsets.all(12),
            child: SelectableText(
              widget.code.trimRight(),
              style: theme.textTheme.bodySmall?.copyWith(
                fontFamily: 'monospace',
                color: colors.onSurface,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
