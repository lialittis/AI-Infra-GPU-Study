// LD_PRELOAD shim: opens of /proc/cpuinfo are redirected to $CPUINFO_OVERRIDE. Everything else
// passes through untouched. See README.md for why this exists.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char* redirect(const char* path) {
  const char* override = getenv("CPUINFO_OVERRIDE");
  return (override && path && strcmp(path, "/proc/cpuinfo") == 0) ? override : path;
}

#define REAL(name) ((__typeof__(&name))dlsym(RTLD_NEXT, #name))

// open/openat take a mode only with O_CREAT/O_TMPFILE; forwarding it unconditionally is harmless.
#define DEFINE_OPEN(name)                                \
  int name(const char* path, int flags, ...) {           \
    va_list ap;                                          \
    va_start(ap, flags);                                 \
    mode_t mode = va_arg(ap, mode_t);                    \
    va_end(ap);                                          \
    return REAL(name)(redirect(path), flags, mode);      \
  }
#define DEFINE_OPENAT(name)                              \
  int name(int dirfd, const char* path, int flags, ...) { \
    va_list ap;                                          \
    va_start(ap, flags);                                 \
    mode_t mode = va_arg(ap, mode_t);                    \
    va_end(ap);                                          \
    return REAL(name)(dirfd, redirect(path), flags, mode); \
  }

DEFINE_OPEN(open)
DEFINE_OPEN(open64)
DEFINE_OPENAT(openat)
DEFINE_OPENAT(openat64)

// Fortified variants used by code built with _FORTIFY_SOURCE.
int __open_2(const char* path, int flags) { return REAL(__open_2)(redirect(path), flags); }
int __open64_2(const char* path, int flags) { return REAL(__open64_2)(redirect(path), flags); }

FILE* fopen(const char* path, const char* mode) { return REAL(fopen)(redirect(path), mode); }
FILE* fopen64(const char* path, const char* mode) { return REAL(fopen64)(redirect(path), mode); }
