import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../models/conversation.dart';
import '../models/folder.dart';
import '../providers/folder_provider.dart';

class MoveConversationDialog extends StatefulWidget {
  final Conversation conversation;
  final String? currentFolderId;
  final Future<void> Function(String? folderId) onMove;

  const MoveConversationDialog({
    super.key,
    required this.conversation,
    required this.currentFolderId,
    required this.onMove,
  });

  @override
  State<MoveConversationDialog> createState() => _MoveConversationDialogState();
}

class _MoveConversationDialogState extends State<MoveConversationDialog> {
  String? _selectedFolderId;
  bool _isLoading = false;

  @override
  void initState() {
    super.initState();
    _selectedFolderId = widget.currentFolderId;
  }

  Future<void> _handleMove() async {
    setState(() {
      _isLoading = true;
    });

    try {
      await widget.onMove(_selectedFolderId);
      if (mounted) {
        Navigator.of(context).pop();
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Failed to move: ${e.toString()}')),
        );
      }
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final folderProvider = context.watch<FolderProvider>();
    final folders = folderProvider.folders;

    return AlertDialog(
      title: const Text('Move Conversation'),
      content: SizedBox(
        width: double.maxFinite,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Move "${widget.conversation.title}" to:',
              style: const TextStyle(fontSize: 14),
            ),
            const SizedBox(height: 16),
            Flexible(
              child: ListView(
                shrinkWrap: true,
                children: [
                  _buildFolderOption(
                    context,
                    folderId: null,
                    folderName: 'Root (No Folder)',
                    icon: Icons.home,
                  ),
                  const Divider(),
                  ...folders
                      .where((f) => f.parentId == null)
                      .map((folder) => _buildFolderTree(
                            context,
                            folder,
                            folders,
                            0,
                          )),
                ],
              ),
            ),
          ],
        ),
      ),
      actions: [
        TextButton(
          onPressed: _isLoading ? null : () => Navigator.of(context).pop(),
          child: const Text('Cancel'),
        ),
        TextButton(
          onPressed: _isLoading ? null : _handleMove,
          child: _isLoading
              ? const SizedBox(
                  width: 16,
                  height: 16,
                  child: CircularProgressIndicator(strokeWidth: 2),
                )
              : const Text('Move'),
        ),
      ],
    );
  }

  Widget _buildFolderTree(
    BuildContext context,
    Folder folder,
    List<Folder> allFolders,
    int indent,
  ) {
    final childFolders =
        allFolders.where((f) => f.parentId == folder.id).toList();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _buildFolderOption(
          context,
          folderId: folder.id,
          folderName: folder.name,
          icon: Icons.folder,
          indent: indent,
        ),
        ...childFolders.map((child) => _buildFolderTree(
              context,
              child,
              allFolders,
              indent + 1,
            )),
      ],
    );
  }

  Widget _buildFolderOption(
    BuildContext context, {
    required String? folderId,
    required String folderName,
    required IconData icon,
    int indent = 0,
  }) {
    final isSelected = _selectedFolderId == folderId;
    final theme = Theme.of(context);

    return Padding(
      padding: EdgeInsets.only(left: indent * 20.0),
      child: RadioListTile<String?>(
        value: folderId,
        groupValue: _selectedFolderId,
        onChanged: _isLoading
            ? null
            : (value) {
                setState(() {
                  _selectedFolderId = value;
                });
              },
        title: Row(
          children: [
            Icon(
              icon,
              size: 18,
              color: isSelected
                  ? theme.colorScheme.primary
                  : theme.colorScheme.onSurface.withOpacity(0.7),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: Text(
                folderName,
                style: TextStyle(
                  fontSize: 14,
                  fontWeight: isSelected ? FontWeight.w600 : FontWeight.w400,
                ),
              ),
            ),
          ],
        ),
        dense: true,
        contentPadding: EdgeInsets.zero,
      ),
    );
  }
}
