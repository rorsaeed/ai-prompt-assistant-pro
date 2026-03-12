import 'package:flutter/material.dart';
import 'app_theme_data.dart';
import 'custom_colors.dart';
import 'theme_palettes.dart';

class StitchTheme {
  StitchTheme._();

  // ─── Spacing ───
  static const double spaceXS = 4.0;
  static const double spaceSM = 8.0;
  static const double spaceMD = 16.0;
  static const double spaceLG = 24.0;
  static const double spaceXL = 32.0;

  // ─── Radii ───
  static const double radiusSM = 8.0;
  static const double radiusMD = 12.0;
  static const double radiusLG = 16.0;
  static const double radiusXL = 24.0;

  // ─── Layout ───
  static const double sidebarWidth = 380.0;

  // ─── Elevation / Shadows ───
  static const List<BoxShadow> shadowSM = [
    BoxShadow(
      color: Color(0x0A000000),
      blurRadius: 4,
      offset: Offset(0, 1),
    ),
  ];

  static const List<BoxShadow> shadowMD = [
    BoxShadow(
      color: Color(0x14000000),
      blurRadius: 8,
      offset: Offset(0, 2),
    ),
  ];

  // ─── Text Styles (light defaults — use Theme.of(context) in widgets) ───
  static const TextStyle headingLarge = TextStyle(
    fontSize: 24,
    fontWeight: FontWeight.w700,
    letterSpacing: -0.5,
    height: 1.3,
  );

  static const TextStyle headingSmall = TextStyle(
    fontSize: 18,
    fontWeight: FontWeight.w600,
    letterSpacing: -0.2,
    height: 1.3,
  );

  static const TextStyle bodyLarge = TextStyle(
    fontSize: 16,
    fontWeight: FontWeight.w400,
    height: 1.5,
  );

  static const TextStyle bodyMedium = TextStyle(
    fontSize: 14,
    fontWeight: FontWeight.w400,
    height: 1.5,
  );

  static const TextStyle bodySmall = TextStyle(
    fontSize: 12,
    fontWeight: FontWeight.w400,
    height: 1.5,
  );

  static const TextStyle labelLarge = TextStyle(
    fontSize: 14,
    fontWeight: FontWeight.w600,
    letterSpacing: 0.1,
  );

  static const TextStyle labelMedium = TextStyle(
    fontSize: 12,
    fontWeight: FontWeight.w500,
    letterSpacing: 0.1,
  );

  // ─────────────────────────────────────────────
  // LEGACY GETTERS (default to Stitch palette)
  // ─────────────────────────────────────────────
  static ThemeData get lightTheme => buildLight(stitchPalette);
  static ThemeData get darkTheme => buildDark(stitchPalette);

  // ─────────────────────────────────────────────
  // PALETTE-AWARE BUILDERS
  // ─────────────────────────────────────────────
  static ThemeData buildLight(AppThemeData palette) {
    final colorScheme = ColorScheme.fromSeed(
      seedColor: palette.seedColor,
      brightness: Brightness.light,
      primary: palette.primaryColor,
      secondary: palette.deepColor,
      tertiary: palette.lightColor,
      surface: palette.lightSurface,
      error: palette.errorLightColor,
    );

    return _buildTheme(colorScheme, Brightness.light, palette);
  }

  static ThemeData buildDark(AppThemeData palette) {
    final colorScheme = ColorScheme.fromSeed(
      seedColor: palette.seedColor,
      brightness: Brightness.dark,
      primary: palette.lightColor,
      secondary: palette.primaryColor,
      tertiary: palette.paleColor,
      surface: palette.darkSurface,
      error: palette.errorDarkColor,
    );

    return _buildTheme(colorScheme, Brightness.dark, palette);
  }

  // ─────────────────────────────────────────────
  // SHARED BUILDER
  // ─────────────────────────────────────────────
  static ThemeData _buildTheme(
      ColorScheme scheme, Brightness brightness, AppThemeData palette) {
    final isDark = brightness == Brightness.dark;
    final surfaceContainer =
        isDark ? palette.darkSurfaceContainer : palette.lightSurfaceContainer;
    final borderColor = isDark ? palette.darkBorder : palette.lightBorder;
    final textPrimary =
        isDark ? palette.darkTextPrimary : palette.lightTextPrimary;
    final textSecondaryColor =
        isDark ? palette.darkTextSecondary : palette.lightTextSecondary;
    final sidebarBgColor =
        isDark ? palette.darkSidebarBg : palette.lightSidebarBg;
    final textHintColor = isDark ? palette.darkTextHint : palette.lightTextHint;

    return ThemeData(
      useMaterial3: true,
      brightness: brightness,
      colorScheme: scheme,
      scaffoldBackgroundColor: scheme.surface,
      fontFamily: null, // Uses default Roboto

      // ─── AppBar ───
      appBarTheme: AppBarTheme(
        elevation: 0,
        scrolledUnderElevation: 1,
        backgroundColor: isDark ? surfaceContainer : scheme.surface,
        foregroundColor: textPrimary,
        surfaceTintColor: Colors.transparent,
        titleTextStyle: headingSmall.copyWith(color: textPrimary),
        iconTheme: IconThemeData(color: textPrimary, size: 22),
        shape: Border(
          bottom: BorderSide(color: borderColor, width: 1),
        ),
      ),

      // ─── TabBar ───
      tabBarTheme: TabBarThemeData(
        labelColor: scheme.primary,
        unselectedLabelColor: textSecondaryColor,
        labelStyle: labelLarge,
        unselectedLabelStyle: labelMedium,
        indicatorSize: TabBarIndicatorSize.tab,
        indicator: BoxDecoration(
          borderRadius: BorderRadius.circular(radiusSM),
          color: scheme.primary.withValues(alpha: 0.12),
        ),
        dividerColor: Colors.transparent,
        overlayColor: WidgetStateProperty.resolveWith((states) {
          if (states.contains(WidgetState.hovered)) {
            return scheme.primary.withValues(alpha: 0.08);
          }
          return Colors.transparent;
        }),
      ),

      // ─── Drawer ───
      drawerTheme: DrawerThemeData(
        backgroundColor: sidebarBgColor,
        surfaceTintColor: Colors.transparent,
        elevation: 2,
        width: sidebarWidth,
        shape: const RoundedRectangleBorder(
          borderRadius: BorderRadius.only(
            topRight: Radius.circular(radiusLG),
            bottomRight: Radius.circular(radiusLG),
          ),
        ),
      ),

      // ─── Card ───
      cardTheme: CardThemeData(
        elevation: 0,
        color: isDark ? surfaceContainer : Colors.white,
        surfaceTintColor: Colors.transparent,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(radiusMD),
          side: BorderSide(color: borderColor, width: 1),
        ),
        margin: const EdgeInsets.only(bottom: spaceMD),
      ),

      // ─── Elevated Button ───
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: scheme.primary,
          foregroundColor: scheme.onPrimary,
          elevation: 0,
          padding: const EdgeInsets.symmetric(
            horizontal: spaceLG,
            vertical: spaceSM + spaceXS,
          ),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(radiusSM),
          ),
          textStyle: labelLarge,
        ),
      ),

      // ─── Outlined Button ───
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: scheme.primary,
          side: BorderSide(color: borderColor),
          padding: const EdgeInsets.symmetric(
            horizontal: spaceLG,
            vertical: spaceSM + spaceXS,
          ),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(radiusSM),
          ),
          textStyle: labelLarge,
        ),
      ),

      // ─── Text Button ───
      textButtonTheme: TextButtonThemeData(
        style: TextButton.styleFrom(
          foregroundColor: scheme.primary,
          textStyle: labelLarge,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(radiusSM),
          ),
        ),
      ),

      // ─── Icon Button ───
      iconButtonTheme: IconButtonThemeData(
        style: IconButton.styleFrom(
          foregroundColor: textSecondaryColor,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(radiusSM),
          ),
        ),
      ),

      // ─── Input Decoration ───
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: surfaceContainer,
        contentPadding: const EdgeInsets.symmetric(
          horizontal: spaceMD,
          vertical: spaceSM + spaceXS,
        ),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(radiusSM),
          borderSide: BorderSide(color: borderColor),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(radiusSM),
          borderSide: BorderSide(color: borderColor),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(radiusSM),
          borderSide: BorderSide(color: scheme.primary, width: 2),
        ),
        errorBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(radiusSM),
          borderSide: BorderSide(color: scheme.error),
        ),
        labelStyle: bodyMedium.copyWith(color: textSecondaryColor),
        hintStyle: bodyMedium.copyWith(color: textHintColor),
        floatingLabelStyle: labelMedium.copyWith(color: scheme.primary),
      ),

      // ─── Dropdown ───
      dropdownMenuTheme: DropdownMenuThemeData(
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: surfaceContainer,
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(radiusSM),
            borderSide: BorderSide(color: borderColor),
          ),
        ),
      ),

      // ─── Chip / FilterChip ───
      chipTheme: ChipThemeData(
        backgroundColor: surfaceContainer,
        selectedColor: scheme.primary.withValues(alpha: 0.15),
        side: BorderSide(color: borderColor),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(radiusSM),
        ),
        labelStyle: bodySmall.copyWith(color: textPrimary),
        secondaryLabelStyle: bodySmall.copyWith(color: scheme.primary),
        padding: const EdgeInsets.symmetric(
          horizontal: spaceSM,
          vertical: spaceXS,
        ),
        showCheckmark: true,
        checkmarkColor: scheme.primary,
      ),

      // ─── Divider ───
      dividerTheme: DividerThemeData(
        color: borderColor,
        thickness: 1,
        space: spaceXL,
      ),

      // ─── ExpansionTile ───
      expansionTileTheme: ExpansionTileThemeData(
        backgroundColor: surfaceContainer,
        collapsedBackgroundColor: Colors.transparent,
        tilePadding: const EdgeInsets.symmetric(
          horizontal: spaceMD,
          vertical: spaceXS,
        ),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(radiusSM),
          side: BorderSide(color: borderColor),
        ),
        collapsedShape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(radiusSM),
          side: BorderSide(color: borderColor),
        ),
        iconColor: textSecondaryColor,
        collapsedIconColor: textSecondaryColor,
      ),

      // ─── Radio / Checkbox ───
      radioTheme: RadioThemeData(
        fillColor: WidgetStateProperty.resolveWith((states) {
          if (states.contains(WidgetState.selected)) return scheme.primary;
          return textSecondaryColor;
        }),
      ),
      checkboxTheme: CheckboxThemeData(
        fillColor: WidgetStateProperty.resolveWith((states) {
          if (states.contains(WidgetState.selected)) return scheme.primary;
          return Colors.transparent;
        }),
        checkColor: WidgetStateProperty.all(scheme.onPrimary),
        side: BorderSide(color: textSecondaryColor),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(4),
        ),
      ),

      // ─── ListTile ───
      listTileTheme: ListTileThemeData(
        contentPadding: const EdgeInsets.symmetric(horizontal: spaceMD),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(radiusSM),
        ),
        dense: true,
      ),

      // ─── Dialog ───
      dialogTheme: DialogThemeData(
        backgroundColor: isDark ? surfaceContainer : Colors.white,
        surfaceTintColor: Colors.transparent,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(radiusLG),
        ),
        elevation: 4,
      ),

      // ─── Snackbar ───
      snackBarTheme: SnackBarThemeData(
        backgroundColor: textPrimary,
        contentTextStyle: bodyMedium.copyWith(
          color: isDark ? palette.darkSurface : Colors.white,
        ),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(radiusSM),
        ),
        behavior: SnackBarBehavior.floating,
      ),

      // ─── Progress Indicator ───
      progressIndicatorTheme: ProgressIndicatorThemeData(
        color: scheme.primary,
        linearTrackColor: scheme.primary.withValues(alpha: 0.15),
        circularTrackColor: scheme.primary.withValues(alpha: 0.15),
      ),

      // ─── Floating Action Button ───
      floatingActionButtonTheme: FloatingActionButtonThemeData(
        backgroundColor: scheme.primary,
        foregroundColor: scheme.onPrimary,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(radiusMD),
        ),
      ),

      // ─── Scrollbar ───
      scrollbarTheme: ScrollbarThemeData(
        thumbColor: WidgetStateProperty.all(
          textSecondaryColor.withValues(alpha: 0.3),
        ),
        radius: const Radius.circular(radiusSM),
        thickness: WidgetStateProperty.all(6),
      ),

      // ─── Text Theme ───
      textTheme: TextTheme(
        headlineLarge: headingLarge.copyWith(color: textPrimary),
        headlineMedium: headingSmall.copyWith(color: textPrimary),
        headlineSmall: headingSmall.copyWith(
          color: textPrimary,
          fontSize: 16,
        ),
        titleLarge: headingSmall.copyWith(color: textPrimary),
        titleMedium: labelLarge.copyWith(color: textPrimary),
        titleSmall: labelMedium.copyWith(color: textPrimary),
        bodyLarge: bodyLarge.copyWith(color: textPrimary),
        bodyMedium: bodyMedium.copyWith(color: textPrimary),
        bodySmall: bodySmall.copyWith(color: textSecondaryColor),
        labelLarge: labelLarge.copyWith(color: textPrimary),
        labelMedium: labelMedium.copyWith(color: textSecondaryColor),
        labelSmall: bodySmall.copyWith(
          color: textSecondaryColor,
          fontSize: 11,
        ),
      ),

      // ─── Theme Extensions ───
      extensions: [
        CustomColors(
          sidebarBg: sidebarBgColor,
          sidebarBorder: borderColor,
          surfaceContainer: surfaceContainer,
          successColor: palette.successColor,
        ),
      ],
    );
  }

  // ─── Helper: pick color based on current theme brightness ───
  static Color adaptive(
    BuildContext context, {
    required Color light,
    required Color dark,
  }) {
    return Theme.of(context).brightness == Brightness.dark ? dark : light;
  }

  // ─── Convenience: access CustomColors from context ───
  static CustomColors customColors(BuildContext context) =>
      Theme.of(context).extension<CustomColors>()!;

  static Color sidebarBg(BuildContext context) =>
      customColors(context).sidebarBg;

  static Color sidebarBorderColor(BuildContext context) =>
      customColors(context).sidebarBorder;

  static Color surfaceColor(BuildContext context) =>
      Theme.of(context).colorScheme.surface;

  static Color surfaceContainerColor(BuildContext context) =>
      customColors(context).surfaceContainer;
}
