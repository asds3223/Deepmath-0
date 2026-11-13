#!/usr/bin/env bash
# Rapfi 2025 릴리스(태그 250615)와 같은 시점의 가중치(rapfi-networks 918b757)를 받아 고정한다.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party
if [ ! -d third_party/rapfi/.git ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --branch 250615 https://github.com/dhbloo/rapfi third_party/rapfi
fi
git -C third_party/rapfi checkout -q 250615
if [ ! -d third_party/rapfi/Networks/.git ]; then
  rm -rf third_party/rapfi/Networks
  GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/dhbloo/rapfi-networks third_party/rapfi/Networks
fi
git -C third_party/rapfi/Networks checkout -q 918b757
mkdir -p data/net
N=third_party/rapfi/Networks
cp $N/config-example/config.toml $N/classical/*.bin $N/mix9svq/* data/net/
echo "rapfi $(git -C third_party/rapfi rev-parse --short HEAD), networks $(git -C $N rev-parse --short HEAD) -> data/net"
