import 'package:flutter/material.dart';

/// Defines the color palette for a single theme.
/// Each theme provides its own light and dark palette colors,
/// while spacing, radii, shadows, and text styles remain shared.
class AppThemeData {
  final String id;
  final String displayName;
  final Color seedColor;

  // Brand / accent colors
  final Color primaryColor;
  final Color deepColor;
  final Color lightColor;
  final Color paleColor;

  // Light palette
  final Color lightSurface;
  final Color lightSurfaceContainer;
  final Color lightSidebarBg;
  final Color lightBorder;
  final Color lightTextPrimary;
  final Color lightTextSecondary;
  final Color lightTextHint;

  // Dark palette
  final Color darkSurface;
  final Color darkSurfaceContainer;
  final Color darkSidebarBg;
  final Color darkBorder;
  final Color darkTextPrimary;
  final Color darkTextSecondary;
  final Color darkTextHint;

  // Semantic colors
  final Color successColor;
  final Color errorLightColor;
  final Color errorDarkColor;

  const AppThemeData({
    required this.id,
    required this.displayName,
    required this.seedColor,
    required this.primaryColor,
    required this.deepColor,
    required this.lightColor,
    required this.paleColor,
    required this.lightSurface,
    required this.lightSurfaceContainer,
    required this.lightSidebarBg,
    required this.lightBorder,
    required this.lightTextPrimary,
    required this.lightTextSecondary,
    required this.lightTextHint,
    required this.darkSurface,
    required this.darkSurfaceContainer,
    required this.darkSidebarBg,
    required this.darkBorder,
    required this.darkTextPrimary,
    required this.darkTextSecondary,
    required this.darkTextHint,
    required this.successColor,
    this.errorLightColor = const Color(0xFFDC2626),
    this.errorDarkColor = const Color(0xFFF87171),
  });
}
