#!/bin/bash
# Qwen LoRA 재학습 파이프라인
#
# Stage 1: Qwen LoRA 모드로 평가 → 실패 프레임을 Qwen 학습 포맷으로 수집
# Stage 2: 기존 학습 데이터 + 실패 프레임 merge
# Stage 3: Qwen LoRA 추가 학습 (failure-augmented fine-tuning)
# Stage 4: 새 LoRA로 평가 → 성능 비교

set -eo pipefail
cd /home/choi/LGHA

PYTHON=/home/choi/miniconda3/envs/lgha/bin/python
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/qwen_retrain_${TIMESTAMP}.log"

ORIG_DATA="/home/choi/libero_object/qwen_finetune.jsonl"
FAIL_DATA="logs/qwen_failure_frames.jsonl"
MERGED_DATA="logs/qwen_merged_${TIMESTAMP}.jsonl"
NEW_LORA="checkpoints/qwen_lora_retrained"
QWEN_LORA="checkpoints/qwen_lora/final"

echo "[$(date)] ── Stage 1: 실패 프레임 수집 시작 ──" | tee -a "$LOG"
$PYTHON -m hierarchical_vla.pipeline.evaluate \
    --checkpoint  checkpoints/hierarchical_phase/final.pth \
    --suite       libero_object \
    --num_episodes 20 \
    --device      cuda:0 \
    --qwen_device cuda:1 \
    --phase \
    --qwen_lora        "$QWEN_LORA" \
    --qwen_stride      5 \
    --qwen_failure_data "$FAIL_DATA" \
    --grasp_timeout    50 \
    --no_video \
    2>&1 | tee -a "$LOG"

N_FAIL=$(wc -l < "$FAIL_DATA")
N_ORIG=$(wc -l < "$ORIG_DATA")
echo "[$(date)] ── Stage 1 완료: 실패 프레임 ${N_FAIL}건 수집 ──" | tee -a "$LOG"

if [ "$N_FAIL" -lt 3 ]; then
    echo "[$(date)] 실패 프레임 부족 (${N_FAIL}건). 종료." | tee -a "$LOG"
    exit 1
fi

echo "[$(date)] ── Stage 2: 데이터 merge (기존 ${N_ORIG} + 실패 ${N_FAIL}) ──" | tee -a "$LOG"
cat "$ORIG_DATA" "$FAIL_DATA" > "$MERGED_DATA"
echo "Merged: $(wc -l < "$MERGED_DATA")건 → $MERGED_DATA" | tee -a "$LOG"

echo "[$(date)] ── Stage 3: Qwen LoRA 재학습 시작 (DDP 2-GPU) ──" | tee -a "$LOG"
$PYTHON -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port=29500 \
    -m hierarchical_vla.pipeline.finetune_qwen \
    --data_path  "$MERGED_DATA" \
    --output_dir "$NEW_LORA" \
    --num_epochs 1 \
    2>&1 | tee -a "$LOG"

echo "[$(date)] ── Stage 4: 새 LoRA로 평가 시작 ──" | tee -a "$LOG"
$PYTHON -m hierarchical_vla.pipeline.evaluate \
    --checkpoint  checkpoints/hierarchical_phase/final.pth \
    --suite       libero_object \
    --num_episodes 20 \
    --device      cuda:0 \
    --qwen_device cuda:1 \
    --phase \
    --qwen_lora   "${NEW_LORA}/final" \
    --qwen_stride 5 \
    --grasp_timeout 50 \
    --no_video \
    2>&1 | tee -a "$LOG"

echo "[$(date)] ── 전체 파이프라인 완료 ──" | tee -a "$LOG"
grep -E "Overall|pick_up" "$LOG" | tail -15 | tee -a "$LOG"
