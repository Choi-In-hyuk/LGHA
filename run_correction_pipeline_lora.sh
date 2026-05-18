#!/bin/bash
# Stage 1: Qwen LoRA 모드로 grasp 실패 데이터 수집
#   → loc_token이 Qwen 예측값이라 gt와 차이 발생 → 진짜 localization error 학습 가능
# Stage 2: CorrectionModule 학습
# Stage 3: Correction retry 평가

set -eo pipefail
cd /home/choi/LGHA

PYTHON=/home/choi/miniconda3/envs/lgha/bin/python
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/correction_lora_pipeline_${TIMESTAMP}.log"
FAILURE_DATA="logs/grasp_failures_lora.jsonl"   # GT 데이터와 분리
CORRECTION_OUT="checkpoints/correction_lora"
QWEN_LORA="checkpoints/qwen_lora/final"

echo "[$(date)] ── Stage 1: Qwen LoRA 모드 데이터 수집 시작 ──" | tee -a "$LOG"
$PYTHON -m hierarchical_vla.pipeline.evaluate \
    --checkpoint  checkpoints/hierarchical_phase/final.pth \
    --suite       libero_object \
    --num_episodes 20 \
    --device      cuda:0 \
    --qwen_device cuda:1 \
    --phase \
    --qwen_lora   "$QWEN_LORA" \
    --qwen_stride 5 \
    --failure_data "$FAILURE_DATA" \
    --grasp_timeout 50 \
    --no_video \
    --debug \
    2>&1 | tee -a "$LOG"

N=$(wc -l < "$FAILURE_DATA")
echo "[$(date)] ── Stage 1 완료. 수집 레코드: ${N}건 ──" | tee -a "$LOG"

if [ "$N" -lt 5 ]; then
    echo "[$(date)] 실패 레코드가 너무 적습니다 (${N}건). 학습을 건너뜁니다." | tee -a "$LOG"
    exit 1
fi

echo "[$(date)] ── Stage 2: CorrectionModule 학습 시작 ──" | tee -a "$LOG"
$PYTHON -m hierarchical_vla.pipeline.correction_module \
    --data   "$FAILURE_DATA" \
    --out    "$CORRECTION_OUT" \
    --epochs 200 \
    --batch_size 32 \
    --lr     1e-3 \
    --device cuda:0 \
    2>&1 | tee -a "$LOG"

echo "[$(date)] ── Stage 3: Correction retry 평가 시작 ──" | tee -a "$LOG"
$PYTHON -m hierarchical_vla.pipeline.evaluate \
    --checkpoint  checkpoints/hierarchical_phase/final.pth \
    --suite       libero_object \
    --num_episodes 20 \
    --device      cuda:0 \
    --qwen_device cuda:1 \
    --phase \
    --qwen_lora   "$QWEN_LORA" \
    --qwen_stride 5 \
    --correction_module "${CORRECTION_OUT}/correction_module.pth" \
    --grasp_timeout 50 \
    --no_video \
    --debug \
    2>&1 | tee -a "$LOG"

echo "[$(date)] ── 전체 파이프라인 완료 ──" | tee -a "$LOG"
