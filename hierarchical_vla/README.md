# Hierarchical VLA: LLM-Orchestrated Sequential Manipulation

## 개요

LLM(GPT-4o)이 long-horizon task를 primitive 시퀀스로 분해하고,
MambaVLA가 각 primitive를 순차 실행하는 계층적 프레임워크.

**핵심 아이디어**: LLM은 "두뇌", VLA는 "몸". LLM이 고수준 계획을 담당하고 VLA는 각 primitive 실행에만 집중.

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

### MambaVLA 내부 구조

```
RGB images (num_cams × 3×128×128)
      ↓
ResNet ObsEncoder
      ↓
obs_embed (B, num_cams, 256)   +   CLIP lang_embed (B, 512)
      ↓
Mamba (SSM backbone, 5 layers)
      ↓
Flow Matching (action denoising, 4 steps)
      ↓
actions (B, action_seq_len, 7)
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

---

## 연구 방향 (Future Work)

### 현재 구조의 한계

현재 LLM은 task를 primitive 시퀀스로 분해하는 **텍스트 수준의 오케스트레이터** 역할만 한다.
LLM의 reasoning이 VLA의 내부 표현(latent space)에는 전혀 영향을 주지 않는다.

즉, LLM과 VLA 사이의 인터페이스가 `"pick up the red cup"` 같은 자연어 문자열에 불과하다.

### 목표: LLM reasoning → VLA latent space

LLM의 추론 결과를 VLA의 latent space에 직접 주입하여, VLA가 더 풍부한 맥락 정보를 활용하도록 한다.

```
LIBERO-10 Task
      ↓
  GPT-4o (reasoning)
      ↓
  latent vector z  ← LLM hidden state / embedding
      ↓
MambaVLA (z가 obs_embed에 영향)
      ↓
actions
```

### 영감: RD-VLA (Iterative Latent Refinement)

RD-VLA는 recurrent Transformer로 latent를 반복 정제하여 action을 개선한다.
우리의 목표는 이 latent 정제 과정을 **LLM reasoning으로 유도(guide)**하는 것:

- RD-VLA: 자기 자신의 latent를 반복 업데이트
- 목표 구조: LLM semantic understanding → latent 조건부 → VLA action 개선

### 핵심 Contribution 방향

1. **LLM-guided latent injection**: LLM이 생성한 scene 이해/계획 정보를 VLA의 obs_embed 또는 Mamba hidden state에 주입
2. **Cross-modal alignment**: LLM text embedding ↔ VLA visual embedding을 정렬하는 projection layer 학습
3. **Fine-tuning with LLM signal**: LLM reasoning을 조건(condition)으로 MambaVLA를 fine-tuning
