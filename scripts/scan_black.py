#!/usr/bin/env python3
"""Scan mp4 videos for black frames.

Usage: scan_black.py <glob or dir> [threshold_mean] [threshold_std]
A video is flagged BLACK if the per-frame gray mean over up to 9 sampled
frames has min < threshold_mean AND overall std < threshold_std.
Default thresholds: mean 12.0, std 8.0 (calibrated against healthy Wan2.2
720p outputs, whose frame means sit well above 30).
Prints one line per video: name | frames | min_mean | avg_mean | std | verdict
Exit code: 1 if any BLACK, 0 otherwise.
"""
import av
import glob
import os
import statistics
import sys

def scan(path, threshold_mean, threshold_std):
    c = av.open(path)
    frames = list(c.decode(video=0))
    if not frames:
        return None, 0.0, 0.0, 0.0, "NO_FRAMES"
    step = max(1, len(frames) // 9)
    vals = []
    for i in range(0, len(frames), step):
        vals.append(float(frames[i].to_ndarray(format="gray").mean()))
    mn, av_ = min(vals), sum(vals) / len(vals)
    sd = statistics.pstdev(vals)
    verdict = "BLACK" if (mn < threshold_mean and sd < threshold_std) else "ok"
    return len(frames), mn, av_, sd, verdict

def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "/*.mp4"
    tm = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
    ts = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0
    files = sorted(glob.glob(target)) if not os.path.isdir(target) \
        else sorted(glob.glob(os.path.join(target, "*.mp4")))
    bad = 0
    for p in files:
        try:
            n, mn, av_, sd, v = scan(p, tm, ts)
        except Exception as e:
            print(f"{os.path.basename(p)} | ERROR | {str(e)[:60]}")
            bad += 1
            continue
        flag = " <<< BLACK" if v == "BLACK" else ""
        if v != "ok":
            bad += 1
        print(f"{os.path.basename(p)} | {n}f | min={mn:.1f} avg={av_:.1f} std={sd:.1f} | {v}{flag}")
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
