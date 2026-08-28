# MXAttention Full-72 Runbook

This runbook is the short operational handoff for completing the current
shared-output generation run. See `mxattention_full72_handoff.md` for the
full state and validation notes.

## Start missing prompts on GPUs 1–6

```bash
cd /dev/shm/.tmp_yi/workspace/bench-omni-wan
nohup ./run_mxattention_full72_gpu1to6.sh \
  > output/mxattention_full_common10/full72/gpu1to6.run.log 2>&1 &
```

The runner scans the shared output directory first, skips existing videos,
and assigns only missing prompts across six TP1 servers on ports 8121–8126.
Use `./run_mxattention_full72_gpu1to6.sh --dry-run` to inspect assignments.

## Check progress

```bash
find output/mxattention_full_common10/videos -maxdepth 1 \
  -type f -name '*.mp4' -size +0c | wc -l
tail -f output/mxattention_full_common10/full72/gpu1to6.run.log
```

The runner stops only stale vLLM servers on its target GPUs and refuses to
run if an unrelated compute process or an active generation client is found.

## Finish

After 72 videos exist, run:

```bash
./verify_mxattention_full72.py \
  --video-dir output/mxattention_full_common10/videos \
  --prompt-file subject_consistency.txt \
  --log-dir output/mxattention_full_common10/full72/gpu1to6/logs \
  --legacy-log-dir output/mxattention_full_common10/full72/logs \
  --run-manifest output/mxattention_full_common10/full72/gpu1to6/manifest.json \
  --output-root output/mxattention_full_common10/full72
```

Run VBench on released GPU 1, then append the resulting summary with
`write_mxattention_full72_results.py`.
