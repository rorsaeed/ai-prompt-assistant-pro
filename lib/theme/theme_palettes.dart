import 'package:flutter/material.dart';
import 'app_theme_data.dart';

/// Registry of all available theme palettes.
/// Each entry maps a theme ID to its [AppThemeData] definition.
final Map<String, AppThemeData> themePalettes = {
  'stitch': stitchPalette,
  'ocean': oceanPalette,
  'forest': forestPalette,
  'sunset': sunsetPalette,
  'rose': rosePalette,
  'slate': slatePalette,
  'corporate': corporatePalette,
  'high_contrast_light': highContrastLightPalette,
  'amoled': amoledPalette,
};

/// Ordered list of theme IDs for the UI picker.
final List<String> themeOrder = [
  'stitch',
  'ocean',
  'forest',
  'sunset',
  'rose',
  'slate',
  'corporate',
  'high_contrast_light',
  'amoled',
];

// ═══════════════════════════════════════════════════
// 1. STITCH PURPLE (original)
// ═══════════════════════════════════════════════════
const stitchPalette = AppThemeData(
  id: 'stitch',
  displayName: 'Stitch Purple',
  seedColor: Color(0xFF7C3AED),
  primaryColor: Color(0xFF7C3AED),
  deepColor: Color(0xFF6D28D9),
  lightColor: Color(0xFFA78BFA),
  paleColor: Color(0xFFEDE9FE),
  // Light
  lightSurface: Color(0xFFFAFAFC),
  lightSurfaceContainer: Color(0xFFF4F2F7),
  lightSidebarBg: Color(0xFFF8F6FC),
  lightBorder: Color(0xFFE2E0E8),
  lightTextPrimary: Color(0xFF1C1B1F),
  lightTextSecondary: Color(0xFF605D66),
  lightTextHint: Color(0xFF938F99),
  // Dark
  darkSurface: Color(0xFF1C1B2E),
  darkSurfaceContainer: Color(0xFF252438),
  darkSidebarBg: Color(0xFF201F33),
  darkBorder: Color(0xFF3D3B50),
  darkTextPrimary: Color(0xFFE6E1E5),
  darkTextSecondary: Color(0xFFCAC4D0),
  darkTextHint: Color(0xFF938F99),
  // Semantic
  successColor: Color(0xFF22C55E),
);

// ═══════════════════════════════════════════════════
// 2. OCEAN BLUE
// ═══════════════════════════════════════════════════
const oceanPalette = AppThemeData(
  id: 'ocean',
  displayName: 'Ocean Blue',
  seedColor: Color(0xFF2563EB),
  primaryColor: Color(0xFF2563EB),
  deepColor: Color(0xFF1D4ED8),
  lightColor: Color(0xFF60A5FA),
  paleColor: Color(0xFFDBEAFE),
  // Light
  lightSurface: Color(0xFFFAFBFD),
  lightSurfaceContainer: Color(0xFFF0F4FA),
  lightSidebarBg: Color(0xFFF5F8FC),
  lightBorder: Color(0xFFD6E0F0),
  lightTextPrimary: Color(0xFF0F172A),
  lightTextSecondary: Color(0xFF475569),
  lightTextHint: Color(0xFF94A3B8),
  // Dark
  darkSurface: Color(0xFF0F172A),
  darkSurfaceContainer: Color(0xFF1E293B),
  darkSidebarBg: Color(0xFF162033),
  darkBorder: Color(0xFF334155),
  darkTextPrimary: Color(0xFFE2E8F0),
  darkTextSecondary: Color(0xFF94A3B8),
  darkTextHint: Color(0xFF64748B),
  // Semantic
  successColor: Color(0xFF22C55E),
);

// ═══════════════════════════════════════════════════
// 3. FOREST GREEN
// ═══════════════════════════════════════════════════
const forestPalette = AppThemeData(
  id: 'forest',
  displayName: 'Forest Green',
  seedColor: Color(0xFF059669),
  primaryColor: Color(0xFF059669),
  deepColor: Color(0xFF047857),
  lightColor: Color(0xFF34D399),
  paleColor: Color(0xFFD1FAE5),
  // Light
  lightSurface: Color(0xFFFAFDFB),
  lightSurfaceContainer: Color(0xFFEFF8F3),
  lightSidebarBg: Color(0xFFF4FBF7),
  lightBorder: Color(0xFFD1E7DC),
  lightTextPrimary: Color(0xFF14241C),
  lightTextSecondary: Color(0xFF4B6358),
  lightTextHint: Color(0xFF8DA398),
  // Dark
  darkSurface: Color(0xFF0D1F17),
  darkSurfaceContainer: Color(0xFF162B21),
  darkSidebarBg: Color(0xFF12251C),
  darkBorder: Color(0xFF2D4A3C),
  darkTextPrimary: Color(0xFFE0F0E8),
  darkTextSecondary: Color(0xFFA3C4B5),
  darkTextHint: Color(0xFF6B8E7E),
  // Semantic
  successColor: Color(0xFF22C55E),
);

// ═══════════════════════════════════════════════════
// 4. SUNSET ORANGE
// ═══════════════════════════════════════════════════
const sunsetPalette = AppThemeData(
  id: 'sunset',
  displayName: 'Sunset Orange',
  seedColor: Color(0xFFEA580C),
  primaryColor: Color(0xFFEA580C),
  deepColor: Color(0xFFC2410C),
  lightColor: Color(0xFFFB923C),
  paleColor: Color(0xFFFFF7ED),
  // Light
  lightSurface: Color(0xFFFFFCFA),
  lightSurfaceContainer: Color(0xFFFFF3EA),
  lightSidebarBg: Color(0xFFFFF8F3),
  lightBorder: Color(0xFFEDE0D4),
  lightTextPrimary: Color(0xFF27180E),
  lightTextSecondary: Color(0xFF6B5744),
  lightTextHint: Color(0xFFA0917F),
  // Dark
  darkSurface: Color(0xFF1F150D),
  darkSurfaceContainer: Color(0xFF2D2018),
  darkSidebarBg: Color(0xFF261B12),
  darkBorder: Color(0xFF4D3C2C),
  darkTextPrimary: Color(0xFFF5E6D8),
  darkTextSecondary: Color(0xFFC9B5A0),
  darkTextHint: Color(0xFF8D7A66),
  // Semantic
  successColor: Color(0xFF22C55E),
);

// ═══════════════════════════════════════════════════
// 5. ROSE PINK
// ═══════════════════════════════════════════════════
const rosePalette = AppThemeData(
  id: 'rose',
  displayName: 'Rose Pink',
  seedColor: Color(0xFFE11D48),
  primaryColor: Color(0xFFE11D48),
  deepColor: Color(0xFFBE123C),
  lightColor: Color(0xFFFB7185),
  paleColor: Color(0xFFFFE4E6),
  // Light
  lightSurface: Color(0xFFFFFBFC),
  lightSurfaceContainer: Color(0xFFFFF0F2),
  lightSidebarBg: Color(0xFFFFF5F6),
  lightBorder: Color(0xFFF0D4D9),
  lightTextPrimary: Color(0xFF1F0A10),
  lightTextSecondary: Color(0xFF6B4450),
  lightTextHint: Color(0xFFA08890),
  // Dark
  darkSurface: Color(0xFF1F0D12),
  darkSurfaceContainer: Color(0xFF2D181E),
  darkSidebarBg: Color(0xFF261218),
  darkBorder: Color(0xFF4D2D36),
  darkTextPrimary: Color(0xFFF5E0E4),
  darkTextSecondary: Color(0xFFC9A8B0),
  darkTextHint: Color(0xFF8D6B75),
  // Semantic
  successColor: Color(0xFF22C55E),
);

// ═══════════════════════════════════════════════════
// 6. SLATE GRAY (neutral)
// ═══════════════════════════════════════════════════
const slatePalette = AppThemeData(
  id: 'slate',
  displayName: 'Slate Gray',
  seedColor: Color(0xFF475569),
  primaryColor: Color(0xFF475569),
  deepColor: Color(0xFF334155),
  lightColor: Color(0xFF94A3B8),
  paleColor: Color(0xFFF1F5F9),
  // Light
  lightSurface: Color(0xFFFAFAFB),
  lightSurfaceContainer: Color(0xFFF1F3F5),
  lightSidebarBg: Color(0xFFF5F6F8),
  lightBorder: Color(0xFFDDE1E8),
  lightTextPrimary: Color(0xFF0F172A),
  lightTextSecondary: Color(0xFF64748B),
  lightTextHint: Color(0xFF94A3B8),
  // Dark
  darkSurface: Color(0xFF0F1218),
  darkSurfaceContainer: Color(0xFF1A1E26),
  darkSidebarBg: Color(0xFF15191F),
  darkBorder: Color(0xFF333B48),
  darkTextPrimary: Color(0xFFE2E8F0),
  darkTextSecondary: Color(0xFF94A3B8),
  darkTextHint: Color(0xFF64748B),
  // Semantic
  successColor: Color(0xFF22C55E),
);

// ═══════════════════════════════════════════════════
// 7. CORPORATE BLUE (professional)
// ═══════════════════════════════════════════════════
const corporatePalette = AppThemeData(
  id: 'corporate',
  displayName: 'Corporate Blue',
  seedColor: Color(0xFF1E40AF),
  primaryColor: Color(0xFF1E40AF),
  deepColor: Color(0xFF1E3A8A),
  lightColor: Color(0xFF3B82F6),
  paleColor: Color(0xFFEFF6FF),
  // Light
  lightSurface: Color(0xFFFCFCFD),
  lightSurfaceContainer: Color(0xFFF3F5F9),
  lightSidebarBg: Color(0xFFF7F8FB),
  lightBorder: Color(0xFFD4DAE5),
  lightTextPrimary: Color(0xFF111827),
  lightTextSecondary: Color(0xFF4B5563),
  lightTextHint: Color(0xFF9CA3AF),
  // Dark
  darkSurface: Color(0xFF111827),
  darkSurfaceContainer: Color(0xFF1F2937),
  darkSidebarBg: Color(0xFF172033),
  darkBorder: Color(0xFF374151),
  darkTextPrimary: Color(0xFFF9FAFB),
  darkTextSecondary: Color(0xFFD1D5DB),
  darkTextHint: Color(0xFF6B7280),
  // Semantic
  successColor: Color(0xFF059669),
);

// ═══════════════════════════════════════════════════
// 8. HIGH CONTRAST LIGHT
// ═══════════════════════════════════════════════════
const highContrastLightPalette = AppThemeData(
  id: 'high_contrast_light',
  displayName: 'High Contrast',
  seedColor: Color(0xFF0000CC),
  primaryColor: Color(0xFF0000CC),
  deepColor: Color(0xFF000099),
  lightColor: Color(0xFF4444FF),
  paleColor: Color(0xFFE0E0FF),
  // Light — maximum contrast: pure white background, near-black text
  lightSurface: Color(0xFFFFFFFF),
  lightSurfaceContainer: Color(0xFFF0F0F0),
  lightSidebarBg: Color(0xFFF5F5F5),
  lightBorder: Color(0xFF888888),
  lightTextPrimary: Color(0xFF000000),
  lightTextSecondary: Color(0xFF222222),
  lightTextHint: Color(0xFF555555),
  // Dark — high contrast dark
  darkSurface: Color(0xFF000000),
  darkSurfaceContainer: Color(0xFF1A1A1A),
  darkSidebarBg: Color(0xFF0D0D0D),
  darkBorder: Color(0xFF666666),
  darkTextPrimary: Color(0xFFFFFFFF),
  darkTextSecondary: Color(0xFFDDDDDD),
  darkTextHint: Color(0xFFAAAAAA),
  // Semantic
  successColor: Color(0xFF008800),
);

// ═══════════════════════════════════════════════════
// 9. AMOLED BLACK
// ═══════════════════════════════════════════════════
const amoledPalette = AppThemeData(
  id: 'amoled',
  displayName: 'AMOLED Black',
  seedColor: Color(0xFF7C3AED),
  primaryColor: Color(0xFF7C3AED),
  deepColor: Color(0xFF6D28D9),
  lightColor: Color(0xFFA78BFA),
  paleColor: Color(0xFFEDE9FE),
  // Light — same as Stitch for light mode
  lightSurface: Color(0xFFFAFAFC),
  lightSurfaceContainer: Color(0xFFF4F2F7),
  lightSidebarBg: Color(0xFFF8F6FC),
  lightBorder: Color(0xFFE2E0E8),
  lightTextPrimary: Color(0xFF1C1B1F),
  lightTextSecondary: Color(0xFF605D66),
  lightTextHint: Color(0xFF938F99),
  // Dark — pure black for OLED panels
  darkSurface: Color(0xFF000000),
  darkSurfaceContainer: Color(0xFF0D0D14),
  darkSidebarBg: Color(0xFF000000),
  darkBorder: Color(0xFF2A2A3A),
  darkTextPrimary: Color(0xFFEEEEF6),
  darkTextSecondary: Color(0xFFB0B0C0),
  darkTextHint: Color(0xFF707080),
  // Semantic
  successColor: Color(0xFF22C55E),
);
