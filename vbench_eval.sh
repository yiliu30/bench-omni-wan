vbench_dir=/home/yiliu7/workspace/VBench
video_dir=/home/yiliu7/workspace/vllm-omni/bench-wan/output/default_bf16
video_dir=/mnt/ctrl/disk2/yiliu7/wan-res/default_real_sage3_attn_only/
# video_dir=/home/yiliu7/workspace/vllm-omni/bench-wan/output/default_mxfp4_linear_only
test_basename=$(basename $video_dir)
test_name="${test_basename}_subject_consistency"

CUDA_VISIBLE_DEVICES=5,6 \
/home/yiliu7/workspace/venvs/vbench/bin/python $vbench_dir/evaluate.py \
    --videos_path $video_dir \
    --dimension subject_consistency \
    --mode=custom_input \
    --output_path ./$test_name.json 