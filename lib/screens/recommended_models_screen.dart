import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:url_launcher/url_launcher.dart';
import '../theme/stitch_theme.dart';

class RecommendedModelsScreen extends StatelessWidget {
  const RecommendedModelsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final colors = theme.colorScheme;

    return ListView(
      padding: const EdgeInsets.all(StitchTheme.spaceMD),
      children: [
        // Header
        Row(
          children: [
            Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: colors.primary.withValues(alpha: 0.1),
                borderRadius: BorderRadius.circular(StitchTheme.radiusSM),
              ),
              child: Icon(
                Icons.model_training_rounded,
                size: 24,
                color: colors.primary,
              ),
            ),
            const SizedBox(width: StitchTheme.spaceSM),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Recommended Vision Models',
                    style: theme.textTheme.headlineMedium,
                  ),
                  const SizedBox(height: 2),
                  Text(
                    'Download via LM Studio or Ollama',
                    style: theme.textTheme.bodySmall,
                  ),
                ],
              ),
            ),
          ],
        ),
        const SizedBox(height: StitchTheme.spaceLG),

        _buildModelCard(
          context,
          name: 'Llama JoyCaption Alpha One',
          vram: '12GB+',
          description:
              'Best for system prompt builder. Excellent at following detailed instructions.',
          lmStudioLink: 'https://lmstudio.ai',
          ollamaCommand: 'ollama pull llama-joycaption-alpha-one-hf-llava',
        ),

        _buildModelCard(
          context,
          name: 'Gemma 3 27B',
          vram: '24GB+',
          description:
              'High-quality vision model with excellent understanding of complex scenes.',
          lmStudioLink: 'https://lmstudio.ai',
          ollamaCommand: 'ollama pull gemma3:27b',
        ),

        _buildModelCard(
          context,
          name: 'Gemma 3 12B',
          vram: '8GB+',
          description:
              'Balanced performance and quality. Great for most use cases.',
          lmStudioLink: 'https://lmstudio.ai',
          ollamaCommand: 'ollama pull gemma3:12b',
        ),

        _buildModelCard(
          context,
          name: 'Gemma 3 4B',
          vram: '4GB+',
          description:
              'Lightweight model for lower-end hardware. Still provides good results.',
          lmStudioLink: 'https://lmstudio.ai',
          ollamaCommand: 'ollama pull gemma3:4b',
        ),

        _buildModelCard(
          context,
          name: 'Qwen2.5-VL-7B',
          vram: '8GB+',
          description:
              'Excellent at detailed image analysis and following instructions.',
          lmStudioLink: 'https://lmstudio.ai',
          ollamaCommand: 'ollama pull qwen2.5-vl:7b',
        ),

        _buildModelCard(
          context,
          name: 'LLaVA 1.6',
          vram: '8GB+',
          description:
              'Popular open-source vision model with good general-purpose capabilities.',
          lmStudioLink: 'https://lmstudio.ai',
          ollamaCommand: 'ollama pull llava:latest',
        ),

        const SizedBox(height: StitchTheme.spaceSM),

        // Google Gemini Section
        Card(
          child: Padding(
            padding: const EdgeInsets.all(StitchTheme.spaceMD),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.all(8),
                      decoration: BoxDecoration(
                        color: colors.tertiary.withValues(alpha: 0.12),
                        borderRadius: BorderRadius.circular(
                          StitchTheme.radiusSM,
                        ),
                      ),
                      child: Icon(
                        Icons.cloud_rounded,
                        size: 18,
                        color: colors.tertiary,
                      ),
                    ),
                    const SizedBox(width: StitchTheme.spaceSM),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Google Gemini Models',
                            style: theme.textTheme.titleMedium,
                          ),
                          Text(
                            'Cloud API — supports image & video analysis',
                            style: theme.textTheme.bodySmall,
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: StitchTheme.spaceMD),
                Text(
                  'Available Models:',
                  style: theme.textTheme.labelLarge,
                ),
                const SizedBox(height: StitchTheme.spaceSM),
                ...[
                  'gemini-3-pro-preview',
                  'gemini-3-flash-preview',
                  'gemini-2.5-pro',
                  'gemini-2.5-flash',
                  'gemini-2.5-flash-lite'
                ].map((model) => Padding(
                      padding: const EdgeInsets.only(bottom: 4),
                      child: Row(
                        children: [
                          Icon(
                            Icons.circle,
                            size: 6,
                            color: colors.primary,
                          ),
                          const SizedBox(width: StitchTheme.spaceSM),
                          Text(model, style: theme.textTheme.bodyMedium),
                        ],
                      ),
                    )),
                const SizedBox(height: StitchTheme.spaceMD),
                FilledButton.icon(
                  onPressed: () {
                    launchUrl(
                      Uri.parse('https://aistudio.google.com/app/api-keys'),
                    );
                  },
                  icon: const Icon(Icons.key_rounded, size: 18),
                  label: const Text('Get API Key'),
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: StitchTheme.spaceLG),
      ],
    );
  }

  Widget _buildModelCard(
    BuildContext context, {
    required String name,
    required String vram,
    required String description,
    required String lmStudioLink,
    required String ollamaCommand,
  }) {
    final theme = Theme.of(context);
    final colors = theme.colorScheme;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(StitchTheme.spaceMD),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: Text(
                    name,
                    style: theme.textTheme.titleMedium,
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: StitchTheme.spaceSM,
                    vertical: StitchTheme.spaceXS,
                  ),
                  decoration: BoxDecoration(
                    color: colors.primaryContainer,
                    borderRadius: BorderRadius.circular(StitchTheme.radiusSM),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        Icons.memory_rounded,
                        size: 14,
                        color: colors.onPrimaryContainer,
                      ),
                      const SizedBox(width: 4),
                      Text(
                        vram,
                        style: theme.textTheme.labelMedium?.copyWith(
                          color: colors.onPrimaryContainer,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: StitchTheme.spaceSM),
            Text(description, style: theme.textTheme.bodyMedium),
            const SizedBox(height: StitchTheme.spaceMD),
            Row(
              children: [
                OutlinedButton.icon(
                  onPressed: () {
                    launchUrl(Uri.parse(lmStudioLink));
                  },
                  icon: const Icon(Icons.download_rounded, size: 16),
                  label: const Text('LM Studio'),
                ),
                const SizedBox(width: StitchTheme.spaceSM),
                Expanded(
                  child: Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: StitchTheme.spaceSM,
                      vertical: StitchTheme.spaceSM,
                    ),
                    decoration: BoxDecoration(
                      color: colors.surfaceContainerHighest,
                      borderRadius: BorderRadius.circular(StitchTheme.radiusSM),
                    ),
                    child: Row(
                      children: [
                        Icon(
                          Icons.terminal_rounded,
                          size: 14,
                          color: colors.onSurfaceVariant,
                        ),
                        const SizedBox(width: StitchTheme.spaceSM),
                        Expanded(
                          child: SelectableText(
                            ollamaCommand,
                            style: theme.textTheme.bodySmall?.copyWith(
                              fontFamily: 'monospace',
                            ),
                          ),
                        ),
                        IconButton(
                          icon: Icon(
                            Icons.copy_rounded,
                            size: 14,
                            color: colors.onSurfaceVariant,
                          ),
                          onPressed: () {
                            Clipboard.setData(
                              ClipboardData(text: ollamaCommand),
                            );
                            ScaffoldMessenger.of(context).showSnackBar(
                              const SnackBar(
                                content: Text('Copied to clipboard'),
                              ),
                            );
                          },
                          tooltip: 'Copy',
                          visualDensity: VisualDensity.compact,
                          padding: EdgeInsets.zero,
                          constraints: const BoxConstraints(
                            minWidth: 28,
                            minHeight: 28,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
