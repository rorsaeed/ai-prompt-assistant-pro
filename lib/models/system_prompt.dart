import 'package:json_annotation/json_annotation.dart';

part 'system_prompt.g.dart';

@JsonSerializable()
class SystemPrompt {
  final String name;
  final String content;

  SystemPrompt({required this.name, required this.content});

  factory SystemPrompt.fromJson(Map<String, dynamic> json) =>
      _$SystemPromptFromJson(json);

  Map<String, dynamic> toJson() => _$SystemPromptToJson(this);
}
