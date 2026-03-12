import 'package:json_annotation/json_annotation.dart';
import 'message.dart';

part 'conversation.g.dart';

@JsonSerializable(explicitToJson: true)
class Conversation {
  final String id;
  final String? folderId;
  final String title;
  final List<Message> messages;
  final DateTime createdAt;
  final DateTime modifiedAt;

  Conversation({
    required this.id,
    this.folderId,
    required this.title,
    required this.messages,
    required this.createdAt,
    required this.modifiedAt,
  });

  factory Conversation.fromJson(Map<String, dynamic> json) =>
      _$ConversationFromJson(json);

  Map<String, dynamic> toJson() => _$ConversationToJson(this);

  // For SQLite database operations
  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'folder_id': folderId,
      'title': title,
      'created_at': createdAt.millisecondsSinceEpoch,
      'modified_at': modifiedAt.millisecondsSinceEpoch,
    };
  }

  Conversation copyWith({
    String? id,
    String? folderId,
    String? title,
    List<Message>? messages,
    DateTime? createdAt,
    DateTime? modifiedAt,
  }) {
    return Conversation(
      id: id ?? this.id,
      folderId: folderId ?? this.folderId,
      title: title ?? this.title,
      messages: messages ?? this.messages,
      createdAt: createdAt ?? this.createdAt,
      modifiedAt: modifiedAt ?? this.modifiedAt,
    );
  }
}
