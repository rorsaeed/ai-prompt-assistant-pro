import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../models/folder.dart';
import '../models/conversation.dart';
import '../providers/folder_provider.dart';
import '../providers/conversation_provider.dart';
import 'folder_dialog.dart';
import 'move_conversation_dialog.dart';

class ConversationTree extends StatefulWidget {
  final List<Conversation> conversations;
  final String? searchQuery;

  const ConversationTree({
    super.key,
    required this.conversations,
    this.searchQuery,
  });

  @override
  State<ConversationTree> createState() => _ConversationTreeState();
}

class _ConversationTreeState extends State<ConversationTree> {
  @override
  Widget build(BuildContext context) {
    final folderProvider = context.watch<FolderProvider>();
    final conversationProvider = context.watch<ConversationProvider>();
    final rootFolders = folderProvider.getRootFolders();
    final rootConversations =
        widget.conversations.where((c) => c.folderId == null).toList();

    return ListView(
      shrinkWrap: true,
      children: [
        // Root level conversations (no folder)
        ...rootConversations.map((conv) => _buildConversationItem(
              context,
              conv,
              conversationProvider,
              folderProvider,
              0,
            )),

        // Root folders
        ...rootFolders.map((folder) => _buildFolderTree(
              context,
              folder,
              folderProvider,
              conversationProvider,
              0,
            )),
      ],
    );
  }

  Widget _buildFolderTree(
    BuildContext context,
    Folder folder,
    FolderProvider folderProvider,
    ConversationProvider conversationProvider,
    int indent,
  ) {
    final isExpanded = folderProvider.isExpanded(folder.id);
    final childFolders = folderProvider.getChildFolders(folder.id);
    final folderConversations =
        widget.conversations.where((c) => c.folderId == folder.id).toList();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _buildFolderItem(
          context,
          folder,
          folderProvider,
          conversationProvider,
          indent,
          folderConversations.length,
        ),
        if (isExpanded) ...[
          // Child conversations
          ...folderConversations.map((conv) => _buildConversationItem(
                context,
                conv,
                conversationProvider,
                folderProvider,
                indent + 1,
              )),

          // Child folders
          ...childFolders.map((childFolder) => _buildFolderTree(
                context,
                childFolder,
                folderProvider,
                conversationProvider,
                indent + 1,
              )),
        ],
      ],
    );
  }

  Widget _buildFolderItem(
    BuildContext context,
    Folder folder,
    FolderProvider folderProvider,
    ConversationProvider conversationProvider,
    int indent,
    int conversationCount,
  ) {
    final isExpanded = folderProvider.isExpanded(folder.id);
    final theme = Theme.of(context);

    return Padding(
      padding: EdgeInsets.only(left: indent * 16.0),
      child: InkWell(
        onTap: () {
          folderProvider.toggleExpanded(folder.id);
        },
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
          child: Row(
            children: [
              Icon(
                isExpanded ? Icons.folder_open : Icons.folder,
                size: 20,
                color: theme.colorScheme.primary,
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  folder.name,
                  style: const TextStyle(
                    fontWeight: FontWeight.w600,
                    fontSize: 14,
                  ),
                ),
              ),
              Text(
                '($conversationCount)',
                style: TextStyle(
                  fontSize: 12,
                  color: theme.colorScheme.onSurface.withOpacity(0.6),
                ),
              ),
              PopupMenuButton<String>(
                icon: Icon(
                  Icons.more_vert,
                  size: 18,
                  color: theme.colorScheme.onSurface.withOpacity(0.6),
                ),
                onSelected: (value) async {
                  if (value == 'rename') {
                    _showRenameFolderDialog(context, folder, folderProvider);
                  } else if (value == 'delete') {
                    _confirmDeleteFolder(context, folder, folderProvider);
                  } else if (value == 'new_subfolder') {
                    _showCreateSubfolderDialog(context, folder, folderProvider);
                  }
                },
                itemBuilder: (context) => [
                  const PopupMenuItem(
                    value: 'new_subfolder',
                    child: Row(
                      children: [
                        Icon(Icons.create_new_folder, size: 18),
                        SizedBox(width: 8),
                        Text('New Subfolder'),
                      ],
                    ),
                  ),
                  const PopupMenuItem(
                    value: 'rename',
                    child: Row(
                      children: [
                        Icon(Icons.edit, size: 18),
                        SizedBox(width: 8),
                        Text('Rename'),
                      ],
                    ),
                  ),
                  const PopupMenuItem(
                    value: 'delete',
                    child: Row(
                      children: [
                        Icon(Icons.delete, size: 18),
                        SizedBox(width: 8),
                        Text('Delete'),
                      ],
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildConversationItem(
    BuildContext context,
    Conversation conversation,
    ConversationProvider conversationProvider,
    FolderProvider folderProvider,
    int indent,
  ) {
    final theme = Theme.of(context);
    final isSelected = conversationProvider.currentChatId == conversation.id;
    final highlightQuery = widget.searchQuery?.toLowerCase();
    final shouldHighlight = highlightQuery != null &&
        highlightQuery.isNotEmpty &&
        conversation.title.toLowerCase().contains(highlightQuery);

    return Padding(
      padding: EdgeInsets.only(left: indent * 16.0),
      child: InkWell(
        onTap: () async {
          await conversationProvider.loadConversation(conversation.id);
        },
        child: Container(
          color: isSelected
              ? theme.colorScheme.primaryContainer.withOpacity(0.3)
              : null,
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 10),
          child: Row(
            children: [
              Icon(
                Icons.chat_bubble_outline,
                size: 18,
                color: theme.colorScheme.onSurface.withOpacity(0.7),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      conversation.title,
                      style: TextStyle(
                        fontSize: 13,
                        fontWeight:
                            isSelected ? FontWeight.w600 : FontWeight.w400,
                        backgroundColor: shouldHighlight
                            ? Colors.yellow.withOpacity(0.3)
                            : null,
                      ),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                    const SizedBox(height: 2),
                    Text(
                      _formatDate(conversation.modifiedAt),
                      style: TextStyle(
                        fontSize: 11,
                        color: theme.colorScheme.onSurface.withOpacity(0.5),
                      ),
                    ),
                  ],
                ),
              ),
              PopupMenuButton<String>(
                icon: Icon(
                  Icons.more_vert,
                  size: 18,
                  color: theme.colorScheme.onSurface.withOpacity(0.6),
                ),
                onSelected: (value) async {
                  if (value == 'rename') {
                    _showRenameConversationDialog(
                        context, conversation, conversationProvider);
                  } else if (value == 'delete') {
                    _confirmDeleteConversation(
                        context, conversation, conversationProvider);
                  } else if (value == 'move') {
                    _showMoveConversationDialog(
                      context,
                      conversation,
                      conversationProvider,
                      folderProvider,
                    );
                  }
                },
                itemBuilder: (context) => [
                  const PopupMenuItem(
                    value: 'move',
                    child: Row(
                      children: [
                        Icon(Icons.drive_file_move, size: 18),
                        SizedBox(width: 8),
                        Text('Move to Folder'),
                      ],
                    ),
                  ),
                  const PopupMenuItem(
                    value: 'rename',
                    child: Row(
                      children: [
                        Icon(Icons.edit, size: 18),
                        SizedBox(width: 8),
                        Text('Rename'),
                      ],
                    ),
                  ),
                  const PopupMenuItem(
                    value: 'delete',
                    child: Row(
                      children: [
                        Icon(Icons.delete, size: 18),
                        SizedBox(width: 8),
                        Text('Delete'),
                      ],
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  String _formatDate(DateTime date) {
    final now = DateTime.now();
    final difference = now.difference(date);

    if (difference.inDays == 0) {
      return 'Today ${date.hour.toString().padLeft(2, '0')}:${date.minute.toString().padLeft(2, '0')}';
    } else if (difference.inDays == 1) {
      return 'Yesterday';
    } else if (difference.inDays < 7) {
      return '${difference.inDays} days ago';
    } else {
      return '${date.year}-${date.month.toString().padLeft(2, '0')}-${date.day.toString().padLeft(2, '0')}';
    }
  }

  void _showRenameFolderDialog(
    BuildContext context,
    Folder folder,
    FolderProvider folderProvider,
  ) {
    showDialog(
      context: context,
      builder: (context) => FolderDialog(
        title: 'Rename Folder',
        initialName: folder.name,
        onSave: (name) async {
          await folderProvider.renameFolder(folder.id, name);
        },
      ),
    );
  }

  void _showCreateSubfolderDialog(
    BuildContext context,
    Folder parentFolder,
    FolderProvider folderProvider,
  ) {
    showDialog(
      context: context,
      builder: (context) => FolderDialog(
        title: 'New Subfolder',
        onSave: (name) async {
          await folderProvider.createFolder(name, parentId: parentFolder.id);
        },
      ),
    );
  }

  void _confirmDeleteFolder(
    BuildContext context,
    Folder folder,
    FolderProvider folderProvider,
  ) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Delete Folder'),
        content: Text(
          'Are you sure you want to delete "${folder.name}"? This will delete all conversations and subfolders within it.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () async {
              Navigator.pop(context);
              await folderProvider.deleteFolder(folder.id);
            },
            style: TextButton.styleFrom(
              foregroundColor: Colors.red,
            ),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
  }

  void _showRenameConversationDialog(
    BuildContext context,
    Conversation conversation,
    ConversationProvider conversationProvider,
  ) {
    final controller = TextEditingController(text: conversation.title);

    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Rename Conversation'),
        content: TextField(
          controller: controller,
          decoration: const InputDecoration(
            labelText: 'Title',
            border: OutlineInputBorder(),
          ),
          autofocus: true,
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () async {
              if (controller.text.trim().isNotEmpty) {
                await conversationProvider.renameConversation(
                  conversation.id,
                  controller.text.trim(),
                );
                Navigator.pop(context);
              }
            },
            child: const Text('Save'),
          ),
        ],
      ),
    );
  }

  void _confirmDeleteConversation(
    BuildContext context,
    Conversation conversation,
    ConversationProvider conversationProvider,
  ) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Delete Conversation'),
        content: Text(
          'Are you sure you want to delete "${conversation.title}"?',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () async {
              Navigator.pop(context);
              await conversationProvider.deleteConversation(conversation.id);
            },
            style: TextButton.styleFrom(
              foregroundColor: Colors.red,
            ),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
  }

  void _showMoveConversationDialog(
    BuildContext context,
    Conversation conversation,
    ConversationProvider conversationProvider,
    FolderProvider folderProvider,
  ) {
    showDialog(
      context: context,
      builder: (context) => MoveConversationDialog(
        conversation: conversation,
        currentFolderId: conversation.folderId,
        onMove: (folderId) async {
          await conversationProvider.moveConversationToFolder(
            conversation.id,
            folderId,
          );
        },
      ),
    );
  }
}
