#!/usr/bin/env bash
# Copy the conda env off quarkfs onto local disk.
#
# Serial cp is hopeless here: every small-file open on /mnt/afs costs 40-90ms, and the
# env holds ~10^5 files. Parallelism hides the per-file latency, so fan out over the
# directory entries that give the best balance (site-packages children dominate).

set -uo pipefail

SRC=/mnt/afs/huangxinyue/miniconda3/envs/verl
DST=/tmp/envs/verl
SP=lib/python3.12/site-packages
JOBS=48

echo "[$(date +%T)] src=$SRC dst=$DST jobs=$JOBS"
mkdir -p "$DST/$SP"

# 1) site-packages children: ~700 units, the bulk of the file count.
echo "[$(date +%T)] stage 1: site-packages"
ls -A "$SRC/$SP" \
    | xargs -P "$JOBS" -I{} cp -a "$SRC/$SP/{}" "$DST/$SP/"
echo "[$(date +%T)] stage 1 done, files so far: $(find "$DST" | wc -l)"

# 2) the rest of lib/python3.12 (stdlib), minus site-packages.
echo "[$(date +%T)] stage 2: stdlib"
ls -A "$SRC/lib/python3.12" | grep -vx site-packages \
    | xargs -P "$JOBS" -I{} cp -a "$SRC/lib/python3.12/{}" "$DST/lib/python3.12/"
echo "[$(date +%T)] stage 2 done, files so far: $(find "$DST" | wc -l)"

# 3) everything else: bin, include, share, ssl, conda-meta, lib/*.so ...
echo "[$(date +%T)] stage 3: remainder"
for top in $(ls -A "$SRC"); do
    if [ "$top" = lib ]; then
        ls -A "$SRC/lib" | grep -vx python3.12 \
            | xargs -P "$JOBS" -I{} cp -a "$SRC/lib/{}" "$DST/lib/"
    else
        echo "$top"
    fi
done | xargs -P "$JOBS" -I{} cp -a "$SRC/{}" "$DST/"

echo "[$(date +%T)] ALL DONE. total files: $(find "$DST" | wc -l), size: $(du -sh "$DST" | cut -f1)"
