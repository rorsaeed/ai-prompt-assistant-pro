import 'package:flutter/material.dart';

/// Custom theme colors that extend Material's [ColorScheme].
/// Access via `Theme.of(context).extension<CustomColors>()!`.
class CustomColors extends ThemeExtension<CustomColors> {
  final Color sidebarBg;
  final Color sidebarBorder;
  final Color surfaceContainer;
  final Color successColor;

  const CustomColors({
    required this.sidebarBg,
    required this.sidebarBorder,
    required this.surfaceContainer,
    required this.successColor,
  });

  @override
  CustomColors copyWith({
    Color? sidebarBg,
    Color? sidebarBorder,
    Color? surfaceContainer,
    Color? successColor,
  }) {
    return CustomColors(
      sidebarBg: sidebarBg ?? this.sidebarBg,
      sidebarBorder: sidebarBorder ?? this.sidebarBorder,
      surfaceContainer: surfaceContainer ?? this.surfaceContainer,
      successColor: successColor ?? this.successColor,
    );
  }

  @override
  CustomColors lerp(covariant ThemeExtension<CustomColors>? other, double t) {
    if (other is! CustomColors) return this;
    return CustomColors(
      sidebarBg: Color.lerp(sidebarBg, other.sidebarBg, t)!,
      sidebarBorder: Color.lerp(sidebarBorder, other.sidebarBorder, t)!,
      surfaceContainer:
          Color.lerp(surfaceContainer, other.surfaceContainer, t)!,
      successColor: Color.lerp(successColor, other.successColor, t)!,
    );
  }
}
