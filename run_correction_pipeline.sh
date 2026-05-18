#!/bin/bash
# Stage 1: grasp 실패 데이터 수집
# Stage 2: CorrectionModule 학습
# 로그: logs/correction_pipeline_<timestamp>.log

set -eo pipefail
cd /home/choi/LGHA

PYTHON=/home/choi/miniconda3/envs/lgha/bin/python

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/correction_pipeline_${TIMESTAMP}.log"
FAILURE_DATA="logs/grasp_failures.jsonl"
CORRECTION_OUT="checkpoints/correction"

echo "[$(date)] ── Stage 1: 데이터 수집 시작 ──" | tee -a "$LOG"
$PYTHON -m hierarchical_vla.pipeline.evaluate \
    --checkpoint checkpoints/hierarchical_phase/final.pth \
    --suite      libero_object \
    --num_episodes 50 \
    --device     cuda:0 \
    --qwen_device cuda:1 \
    --phase \
    --failure_data "$FAILURE_DATA" \
    --grasp_timeout 50 \
    --no_video \
    --debug \
    2>&1 | tee -a "$LOG"

echo "[$(date)] ── Stage 1 완료. 수집 레코드 수: $(wc -l < "$FAILURE_DATA") ──" | tee -a "$LOG"

echo "[$(date)] ── Stage 2: CorrectionModule 학습 시작 ──" | tee -a "$LOG"
$PYTHON -m hierarchical_vla.pipeline.correction_module \
    --data   "$FAILURE_DATA" \
    --out    "$CORRECTION_OUT" \
    --epochs 100 \
    --batch_size 64 \
    --lr     1e-3 \
    --device cuda:0 \
    2>&1 | tee -a "$LOG"

echo "[$(date)] ── 전체 파이프라인 완료 ──" | tee -a "$LOG"
echo "모델 저장 위치: ${CORRECTION_OUT}/correction_module.pth" | tee -a "$LOG"
