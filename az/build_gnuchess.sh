#!/usr/bin/env bash
# Build GNU Chess 5.07 from its release tarball without running its configure script (reference opponent).
#   az/build_gnuchess.sh gnuchess-5.07.tar.gz [OUT_DIR]   → OUT_DIR/gnuchess-5.07/src/gnuchess
# Fixes for a modern toolchain: generated config.h/GCint.h, lexpgn.c via flex, hide POSIX getline (-D_XOPEN_SOURCE=600),
# -fcommon, and input.c's `static pthread_t input_thread` vs the extern in common.h.
set -euo pipefail
TAR=$1; OUT=${2:-third_party}
mkdir -p "$OUT" && tar xzf "$TAR" -C "$OUT"
cd "$OUT/gnuchess-5.07/src"
cat > config.h <<'H'
#define HAVE_ERRNO_H 1
#define HAVE_FCNTL_H 1
#define HAVE_GETTIMEOFDAY 1
#define HAVE_INTTYPES_H 1
#define HAVE_MALLOC 1
#define HAVE_MEMSET 1
#define HAVE_PTHREAD 1
#define HAVE_STDINT_H 1
#define HAVE_STDLIB_H 1
#define HAVE_STRING_H 1
#define HAVE_SYS_TIME_H 1
#define HAVE_UNISTD_H 1
#define STDC_HEADERS 1
#define TIME_WITH_SYS_TIME 1
#define PACKAGE "gnuchess"
#define VERSION "5.07"
H
printf '#include <stdint.h>\n#include <inttypes.h>\n' > GCint.h
flex -olexpgn.c lexpgn.l
sed -i 's/^static pthread_t input_thread;/pthread_t input_thread;/' input.c
gcc -O2 -w -fcommon -std=c99 -D_XOPEN_SOURCE=600 -DHAVE_CONFIG_H -I. -o gnuchess *.c -lm -lpthread
echo "built $(pwd)/gnuchess"
