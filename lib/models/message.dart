import 'package:json_annotation/json_annotation.dart';

part 'message.g.dart';

@JsonSerializable(explicitToJson: true)
class Message {
  final String role; // 'user', 'assistant', 'system'
  final String content; // Actual prompt text
  final String displayContent; // UI display text
  final String id;
  final String? model; // Model name (for assistant messages)
  final String? thinking; // Thinking/reasoning content (for thought models)
  final List<ImageAttachment>? images; // Attached images (for user messages)
  final List<VideoAttachment>? videos; // Attached videos (for user messages)

  Message({
    required this.role,
    required this.content,
    required this.displayContent,
    required this.id,
    this.model,
    this.thinking,
    this.images,
    this.videos,
  });

  factory Message.fromJson(Map<String, dynamic> json) =>
      _$MessageFromJson(json);

  Map<String, dynamic> toJson() => _$MessageToJson(this);

  Message copyWith({
    String? role,
    String? content,
    String? displayContent,
    String? id,
    String? model,
    String? thinking,
    List<ImageAttachment>? images,
    List<VideoAttachment>? videos,
  }) {
    return Message(
      role: role ?? this.role,
      content: content ?? this.content,
      displayContent: displayContent ?? this.displayContent,
      id: id ?? this.id,
      model: model ?? this.model,
      thinking: thinking ?? this.thinking,
      images: images ?? this.images,
      videos: videos ?? this.videos,
    );
  }
}

@JsonSerializable()
class ImageAttachment {
  final String path;
  final String name;

  ImageAttachment({required this.path, required this.name});

  factory ImageAttachment.fromJson(Map<String, dynamic> json) =>
      _$ImageAttachmentFromJson(json);

  Map<String, dynamic> toJson() => _$ImageAttachmentToJson(this);
}

@JsonSerializable()
class VideoAttachment {
  final String path;
  final String name;

  VideoAttachment({required this.path, required this.name});

  factory VideoAttachment.fromJson(Map<String, dynamic> json) =>
      _$VideoAttachmentFromJson(json);

  Map<String, dynamic> toJson() => _$VideoAttachmentToJson(this);
}
