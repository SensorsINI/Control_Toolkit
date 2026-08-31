#ifndef RPGD_PLATFORM_H
#define RPGD_PLATFORM_H

/*
 * Host (Linux/macOS/Windows): OpenMP or pthread workers, heap solver storage.
 * Bare metal: selected by -DRPGD_BAREMETAL / -DRPGD_BARE_METAL, or automatically
 * for ARM EABI builds which do not identify as a Unix/Apple/Windows host
 * (Vitis standalone / arm-none-eabi).
 *
 * Dual-core AMP/FreeRTOS workers can later #undef RPGD_FORCE_SINGLE_THREAD and
 * provide a parallel rollout loop. Sampling RNG must stay serial.
 */
#if defined(RPGD_BAREMETAL) || defined(RPGD_BARE_METAL)
#  define RPGD_PLATFORM_BAREMETAL 1
#elif defined(__arm__) && !defined(__unix__) && !defined(__APPLE__) && !defined(_WIN32)
#  define RPGD_PLATFORM_BAREMETAL 1
#endif

#ifndef RPGD_MAX_NUM_ROLLOUTS
#define RPGD_MAX_NUM_ROLLOUTS 16
#endif
#ifndef RPGD_MAX_HORIZON
#define RPGD_MAX_HORIZON 35
#endif
#ifndef RPGD_MAX_INTERMEDIATE_STEPS
#define RPGD_MAX_INTERMEDIATE_STEPS 10
#endif
#ifndef RPGD_MAX_OUTER_ITS
#define RPGD_MAX_OUTER_ITS 32
#endif

#define RPGD_MAX_TOTAL_STEPS (RPGD_MAX_HORIZON * RPGD_MAX_INTERMEDIATE_STEPS)
#define RPGD_MAX_STATE_BUF ((RPGD_MAX_TOTAL_STEPS + 1) * 6)
#define RPGD_MAX_Q_BUF (RPGD_MAX_NUM_ROLLOUTS * RPGD_MAX_HORIZON)

#if defined(__GNUC__)
#  define RPGD_ALIGN64 __attribute__((aligned(64)))
#else
#  define RPGD_ALIGN64
#endif

#if defined(RPGD_DUAL_CORE) && defined(RPGD_PLATFORM_BAREMETAL) && !defined(RPGD_WORKER_ONLY)
#  define RPGD_SHARED __attribute__((section(".amp_shared"), aligned(64)))
#else
#  define RPGD_SHARED RPGD_ALIGN64
#endif

#ifdef RPGD_PLATFORM_BAREMETAL
#  ifndef RPGD_FORCE_SINGLE_THREAD
#    define RPGD_FORCE_SINGLE_THREAD 1
#  endif
#endif

#if defined(__GNUC__)
#  define RPGD_HOT __attribute__((hot))
#  define RPGD_INLINE static inline __attribute__((always_inline))
#  if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#    define RPGD_THREAD_LOCAL _Thread_local
#  else
#    define RPGD_THREAD_LOCAL __thread
#  endif
#else
#  define RPGD_HOT
#  define RPGD_INLINE static inline
#  define RPGD_THREAD_LOCAL _Thread_local
#endif

#endif
