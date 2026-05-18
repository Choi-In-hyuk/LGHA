#!/bin/bash
# 전체 파이프라인 순차 실행 (cuda:0 단일 GPU — cuda:1은 타 사용자 점유 중)
#
# Step 1: Qwen LoRA 모드로 평가 → 실패 프레임 수집
# Step 2: 기존 학습 데이터 + 실패 프레임 merge
# Step 3: Qwen LoRA 재학습
# Step 4: 새 LoRA로 평가 → 성능 확인

set -eo pipefail
cd /home/choi/LGHA

PYTHON=/home/choi/miniconda3/envs/lgha/bin/python
LOG="logs/all_sequential_$(date +%Y%m%d_%H%M%S).log"

ORIG_DATA="/home/choi/libero_object/qwen_finetune.jsonl"
FAIL_DATA="logs/qwen_failure_frames.jsonl"
MERGED_DATA="logs/qwen_merged.jsonl"
NEW_LORA="checkpoints/qwen_lora_retrained"
# 이전 라운드에서 재학습된 LoRA가 있으면 사용, 없으면 초기 LoRA 사용
if [ -d "checkpoints/qwen_lora_retrained/final" ]; then
    QWEN_LORA="checkpoints/qwen_lora_retrained/final"
    echo "[$(date)] 이전 재학습 LoRA 사용: $QWEN_LORA" | tee -a "$LOG"
else
    QWEN_LORA="checkpoints/qwen_lora/final"
    echo "[$(date)] 초기 LoRA 사용: $QWEN_LORA" | tee -a "$LOG"
fi

echo "[$(date)] ════ Step 1: 실패 프레임 수집 (cuda:0) ════" | tee -a "$LOG"
$PYTHON -m hierarchical_vla.pipeline.evaluate \
    --checkpoint    checkpoints/hierarchical_phase/final.pth \
    --suite         libero_object \
    --num_episodes  50 \
    --device        cuda:0 \
    --qwen_device   cuda:0 \
    --phase \
    --qwen_lora     "$QWEN_LORA" \
    --qwen_stride   5 \
    --qwen_failure_data "$FAIL_DATA" \
    --grasp_timeout 50 \
    --no_video \
    2>&1 | tee -a "$LOG"

N_FAIL=$(wc -l < "$FAIL_DATA")
echo "[$(date)] Step 1 완료 — 실패 프레임 ${N_FAIL}건" | tee -a "$LOG"

if [ "$N_FAIL" -lt 3 ]; then
    echo "[$(date)] 실패 프레임 부족. 종료." | tee -a "$LOG"
    exit 1
fi

echo "[$(date)] ════ Step 2: 데이터 merge ════" | tee -a "$LOG"
N_ORIG=$(wc -l < "$ORIG_DATA")
cat "$ORIG_DATA" "$FAIL_DATA" > "$MERGED_DATA"
echo "[$(date)] merge 완료 — 기존 ${N_ORIG} + 실패 ${N_FAIL} = $(wc -l < "$MERGED_DATA")건" | tee -a "$LOG"

echo "[$(date)] ════ Step 3: Qwen LoRA 재학습 (cuda:0) ════" | tee -a "$LOG"
$PYTHON -m hierarchical_vla.pipeline.finetune_qwen \
    --data_path  "$MERGED_DATA" \
    --output_dir "$NEW_LORA" \
    --num_epochs 3 \
    --device     cuda:0 \
    2>&1 | tee -a "$LOG"

echo "[$(date)] ════ Step 4: 새 LoRA로 평가 (cuda:0) ════" | tee -a "$LOG"
$PYTHON -m hierarchical_vla.pipeline.evaluate \
    --checkpoint    checkpoints/hierarchical_phase/final.pth \
    --suite         libero_object \
    --num_episodes  50 \
    --device        cuda:0 \
    --qwen_device   cuda:0 \
    --phase \
    --qwen_lora     "${NEW_LORA}/final" \
    --qwen_stride   5 \
    --grasp_timeout 50 \
    --no_video \
    2>&1 | tee -a "$LOG"

echo "[$(date)] ════ 전체 완료 ════" | tee -a "$LOG"
grep -E "Overall|pick_up" "$LOG" | tail -12
