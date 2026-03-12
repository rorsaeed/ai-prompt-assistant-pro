// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'message.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

Message _$MessageFromJson(Map<String, dynamic> json) => Message(
      role: json['role'] as String,
      content: json['content'] as String,
      displayContent: json['displayContent'] as String,
      id: json['id'] as String,
      model: json['model'] as String?,
      thinking: json['thinking'] as String?,
      images: (json['images'] as List<dynamic>?)
          ?.map((e) => ImageAttachment.fromJson(e as Map<String, dynamic>))
          .toList(),
      videos: (json['videos'] as List<dynamic>?)
          ?.map((e) => VideoAttachment.fromJson(e as Map<String, dynamic>))
          .toList(),
    );

Map<String, dynamic> _$MessageToJson(Message instance) => <String, dynamic>{
      'role': instance.role,
      'content': instance.content,
      'displayContent': instance.displayContent,
      'id': instance.id,
      'model': instance.model,
      'thinking': instance.thinking,
      'images': instance.images?.map((e) => e.toJson()).toList(),
      'videos': instance.videos?.map((e) => e.toJson()).toList(),
    };

ImageAttachment _$ImageAttachmentFromJson(Map<String, dynamic> json) =>
    ImageAttachment(
      path: json['path'] as String,
      name: json['name'] as String,
    );

Map<String, dynamic> _$ImageAttachmentToJson(ImageAttachment instance) =>
    <String, dynamic>{
      'path': instance.path,
      'name': instance.name,
    };

VideoAttachment _$VideoAttachmentFromJson(Map<String, dynamic> json) =>
    VideoAttachment(
      path: json['path'] as String,
      name: json['name'] as String,
    );

Map<String, dynamic> _$VideoAttachmentToJson(VideoAttachment instance) =>
    <String, dynamic>{
      'path': instance.path,
      'name': instance.name,
    };
