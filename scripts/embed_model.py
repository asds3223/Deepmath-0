#!/usr/bin/env python3
# 모델(.dce)을 C 헤더로 변환해 실행 파일에 내장한다.  embed_model.py IN.dce OUT.h
import sys
d = open(sys.argv[1], 'rb').read()
with open(sys.argv[2], 'w') as f:
    f.write('// generated from %s\n#pragma once\n#include <cstddef>\nstatic const unsigned char kModelData[] = {\n' % sys.argv[1])
    for i in range(0, len(d), 20):
        f.write('  ' + ','.join(str(b) for b in d[i:i + 20]) + ',\n')
    f.write('};\nstatic const size_t kModelSize = %d;\n' % len(d))
