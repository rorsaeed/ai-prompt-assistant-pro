import 'dart:math';
import 'package:flutter/material.dart';

/// A premium animated splash screen shown during app initialization.
///
/// Displays the app icon with scale/fade/glow animations, floating sparkle
/// particles, and a smooth transition to the main content once [onInitialize]
/// completes (minimum 3 seconds shown).
class SplashScreen extends StatefulWidget {
  /// Async callback that performs heavy initialization (DB, providers, etc.).
  /// Returns the fully-initialized main app widget.
  final Future<Widget> Function() onInitialize;

  /// Called when the splash is done and the initialized app widget is ready.
  final void Function(Widget app) onComplete;

  const SplashScreen({
    super.key,
    required this.onInitialize,
    required this.onComplete,
  });

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen>
    with TickerProviderStateMixin {
  // ─── Animation controllers ───
  late final AnimationController _iconController;
  late final AnimationController _glowController;
  late final AnimationController _titleController;
  late final AnimationController _particleController;
  late final AnimationController _fadeOutController;

  // ─── Animations ───
  late final Animation<double> _iconScale;
  late final Animation<double> _iconOpacity;
  late final Animation<double> _glowOpacity;
  late final Animation<double> _titleOpacity;
  late final Animation<Offset> _titleSlide;
  late final Animation<double> _fadeOut;

  // ─── Particle system ───
  final List<_Particle> _particles = [];
  final _random = Random();

  // ─── State ───
  Widget? _initializedApp;
  bool _initDone = false;
  bool _minTimePassed = false;

  @override
  void initState() {
    super.initState();
    _setupAnimations();
    _generateParticles();
    _startInitialization();
  }

  void _setupAnimations() {
    // Icon: scale + fade in
    _iconController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1200),
    );
    _iconScale = Tween<double>(begin: 0.5, end: 1.0).animate(
      CurvedAnimation(
        parent: _iconController,
        curve: Curves.elasticOut,
      ),
    );
    _iconOpacity = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(
        parent: _iconController,
        curve: const Interval(0.0, 0.5, curve: Curves.easeOut),
      ),
    );

    // Glow: pulsing ring around icon
    _glowController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 2000),
    )..repeat(reverse: true);
    _glowOpacity = Tween<double>(begin: 0.3, end: 0.8).animate(
      CurvedAnimation(parent: _glowController, curve: Curves.easeInOut),
    );

    // Title: fade + slide up
    _titleController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 800),
    );
    _titleOpacity = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _titleController, curve: Curves.easeOut),
    );
    _titleSlide = Tween<Offset>(
      begin: const Offset(0, 0.3),
      end: Offset.zero,
    ).animate(
      CurvedAnimation(parent: _titleController, curve: Curves.easeOut),
    );

    // Particle system
    _particleController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 10),
    )..repeat();

    // Fade out transition
    _fadeOutController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 600),
    );
    _fadeOut = Tween<double>(begin: 1.0, end: 0.0).animate(
      CurvedAnimation(parent: _fadeOutController, curve: Curves.easeInOut),
    );

    // Sequence the entrance animations
    _iconController.forward();
    Future.delayed(const Duration(milliseconds: 600), () {
      if (mounted) _titleController.forward();
    });
  }

  void _generateParticles() {
    for (int i = 0; i < 30; i++) {
      _particles.add(_Particle(
        x: _random.nextDouble(),
        y: _random.nextDouble(),
        size: _random.nextDouble() * 3 + 1,
        speed: _random.nextDouble() * 0.3 + 0.1,
        opacity: _random.nextDouble() * 0.6 + 0.2,
        phase: _random.nextDouble() * 2 * pi,
      ));
    }
  }

  Future<void> _startInitialization() async {
    // Start both the min-timer and the actual initialization
    final minDuration = Future.delayed(const Duration(milliseconds: 3000), () {
      _minTimePassed = true;
    });

    try {
      _initializedApp = await widget.onInitialize();
      _initDone = true;
    } catch (e) {
      // If initialization fails, still proceed — user sees the main app error
      _initDone = true;
    }

    await minDuration;

    if (mounted) {
      _transitionToApp();
    }
  }

  void _transitionToApp() {
    if (!_initDone || !_minTimePassed) return;
    if (_initializedApp == null) return;

    _fadeOutController.forward().then((_) {
      if (!mounted) return;
      widget.onComplete(_initializedApp!);
    });
  }

  @override
  void dispose() {
    _iconController.dispose();
    _glowController.dispose();
    _titleController.dispose();
    _particleController.dispose();
    _fadeOutController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _fadeOut,
      builder: (context, child) => Opacity(
        opacity: _fadeOut.value,
        child: child,
      ),
      child: Scaffold(
        body: Container(
          decoration: const BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [
                Color(0xFF0D0B2E), // deep navy
                Color(0xFF1A1145), // dark purple
                Color(0xFF2D1B69), // medium purple
                Color(0xFF1A1145), // dark purple
                Color(0xFF0D0B2E), // deep navy
              ],
              stops: [0.0, 0.25, 0.5, 0.75, 1.0],
            ),
          ),
          child: Stack(
            children: [
              // Floating sparkle particles
              AnimatedBuilder(
                animation: _particleController,
                builder: (context, _) => CustomPaint(
                  size: MediaQuery.of(context).size,
                  painter: _ParticlePainter(
                    particles: _particles,
                    progress: _particleController.value,
                  ),
                ),
              ),

              // Main content
              Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    // Icon with glow
                    AnimatedBuilder(
                      animation: Listenable.merge([
                        _iconController,
                        _glowController,
                      ]),
                      builder: (context, child) {
                        return Opacity(
                          opacity: _iconOpacity.value,
                          child: Transform.scale(
                            scale: _iconScale.value,
                            child: Stack(
                              alignment: Alignment.center,
                              children: [
                                // Glow ring
                                Container(
                                  width: 180,
                                  height: 180,
                                  decoration: BoxDecoration(
                                    shape: BoxShape.circle,
                                    boxShadow: [
                                      BoxShadow(
                                        color: const Color(0xFF7C3AED)
                                            .withValues(
                                                alpha:
                                                    _glowOpacity.value * 0.5),
                                        blurRadius: 40,
                                        spreadRadius: 10,
                                      ),
                                      BoxShadow(
                                        color: const Color(0xFF06B6D4)
                                            .withValues(
                                                alpha:
                                                    _glowOpacity.value * 0.3),
                                        blurRadius: 60,
                                        spreadRadius: 20,
                                      ),
                                    ],
                                  ),
                                ),
                                // Icon image
                                Container(
                                  width: 140,
                                  height: 140,
                                  decoration: BoxDecoration(
                                    borderRadius: BorderRadius.circular(28),
                                    boxShadow: [
                                      BoxShadow(
                                        color:
                                            Colors.black.withValues(alpha: 0.3),
                                        blurRadius: 20,
                                        offset: const Offset(0, 8),
                                      ),
                                    ],
                                  ),
                                  child: ClipRRect(
                                    borderRadius: BorderRadius.circular(28),
                                    child: Image.asset(
                                      'assets/icon/app_icon.png',
                                      fit: BoxFit.cover,
                                    ),
                                  ),
                                ),
                              ],
                            ),
                          ),
                        );
                      },
                    ),

                    const SizedBox(height: 48),

                    // Title text
                    SlideTransition(
                      position: _titleSlide,
                      child: FadeTransition(
                        opacity: _titleOpacity,
                        child: Column(
                          children: [
                            ShaderMask(
                              shaderCallback: (bounds) => const LinearGradient(
                                colors: [
                                  Color(0xFFE0E7FF), // light indigo
                                  Color(0xFFC4B5FD), // light purple
                                  Color(0xFF67E8F9), // light cyan
                                ],
                              ).createShader(bounds),
                              child: const Text(
                                'AI Prompt Assistant',
                                style: TextStyle(
                                  fontSize: 32,
                                  fontWeight: FontWeight.w700,
                                  color: Colors.white,
                                  letterSpacing: -0.5,
                                ),
                              ),
                            ),
                            const SizedBox(height: 12),
                            Text(
                              'Multi-provider AI image & video studio',
                              style: TextStyle(
                                fontSize: 14,
                                fontWeight: FontWeight.w400,
                                color: Colors.white.withValues(alpha: 0.5),
                                letterSpacing: 0.5,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
              ),

              // Bottom: loading indicator + version
              Positioned(
                left: 0,
                right: 0,
                bottom: 60,
                child: FadeTransition(
                  opacity: _titleOpacity,
                  child: Column(
                    children: [
                      SizedBox(
                        width: 200,
                        child: ClipRRect(
                          borderRadius: BorderRadius.circular(2),
                          child: LinearProgressIndicator(
                            backgroundColor:
                                Colors.white.withValues(alpha: 0.1),
                            valueColor: AlwaysStoppedAnimation<Color>(
                              const Color(0xFF7C3AED).withValues(alpha: 0.7),
                            ),
                            minHeight: 3,
                          ),
                        ),
                      ),
                      const SizedBox(height: 20),
                      Text(
                        'v1.0.0',
                        style: TextStyle(
                          fontSize: 12,
                          color: Colors.white.withValues(alpha: 0.3),
                          letterSpacing: 1,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ─── Particle Data ───
class _Particle {
  double x;
  double y;
  final double size;
  final double speed;
  final double opacity;
  final double phase;

  _Particle({
    required this.x,
    required this.y,
    required this.size,
    required this.speed,
    required this.opacity,
    required this.phase,
  });
}

// ─── Particle Painter ───
class _ParticlePainter extends CustomPainter {
  final List<_Particle> particles;
  final double progress;

  _ParticlePainter({required this.particles, required this.progress});

  @override
  void paint(Canvas canvas, Size size) {
    for (final p in particles) {
      final time = (progress + p.phase) % 1.0;
      final x = (p.x + sin(time * 2 * pi + p.phase) * 0.05) * size.width;
      final y = ((p.y - time * p.speed) % 1.0) * size.height;

      // Twinkle effect
      final twinkle = (sin(time * 4 * pi + p.phase) + 1) / 2;
      final alpha = p.opacity * twinkle;

      final paint = Paint()
        ..color = Color.lerp(
          const Color(0xFF7C3AED),
          const Color(0xFF67E8F9),
          (sin(p.phase) + 1) / 2,
        )!
            .withValues(alpha: alpha)
        ..maskFilter = MaskFilter.blur(BlurStyle.normal, p.size);

      canvas.drawCircle(Offset(x, y), p.size, paint);
    }
  }

  @override
  bool shouldRepaint(covariant _ParticlePainter oldDelegate) => true;
}
