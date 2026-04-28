# Hierarchical VLA: LLM-Orchestrated Sequential Manipulation

## 개요

LLM(GPT-4o)이 long-horizon task를 primitive 시퀀스로 분해하고,
MambaVLA가 각 primitive를 순차 실행하는 계층적 프레임워크.

---

## 프로젝트 구조

```
hierarchical_vla/
├── libero_bench/
│   ├── llm_orchestrator.py      # GPT-4o 오케스트레이터 + home pose
│   ├── libero_primitives.json   # scene별 primitive 목록
│   ├── eval_libero10_llm.py     # 평가 스크립트
│   ├── dataloader.py            # LIBERO hdf5 데이터 로더
│   └── train.py                 # MambaVLA 학습
├── language_embeddings/         # LIBERO task CLIP embeddings
├── checkpoints/
│   └── libero_10_subset/        # MambaVLA 체크포인트
└── conf/
    └── config.yaml
```

---

## 아키텍처

```
LIBERO-10 Task
      ↓
  GPT-4o  ← libero_primitives.json
      ↓
[prim_1, prim_2, ..., prim_N]
      ↓
MambaVLA(prim_1) → home pose → MambaVLA(prim_2) → ... → MambaVLA(prim_N)
```

---

## 성능

| 방법 | LIBERO-10 Success Rate |
|------|----------------------|
| VLA 단독 (OpenVLA-OFT) | 14.3% |
| LLM Orchestrator + MambaVLA | **90.0%** |

---

## 학습

```bash
python -m libero_bench.train \
    data_directory=LIBERO/libero/datasets/libero_10_subset \
    save_dir=./checkpoints/libero_10_subset \
    num_epochs=2000
```

## 평가

```bash
export $(cat hierarchical_vla/.env | xargs)
python -m hierarchical_vla.libero_bench.eval_libero10_llm \
    --checkpoint ./hierarchical_vla/checkpoints/libero_10_subset/final_model.pth \
    --openai_api_key $OPENAI_API_KEY \
    --num_rollouts 3
```

---

## 데이터셋

- **경로**: `LIBERO/libero/datasets/libero_10_subset/`
- LIBERO-10 수행에 필요한 LIBERO-90 primitive **15개** 선별
- 태스크당 50 demo, max demo length 400 steps
