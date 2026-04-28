# Hierarchical VLA: LLM-Orchestrated Sequential Manipulation

## Overview

LIBERO-90으로 학습한 VLA 모델은 LIBERO-10 (long-horizon task) 평가에서 낮은 성능을 보인다.
주요 원인은 두 가지다:

1. **언어 이해 실패**: VLA가 복잡한 multi-step 지시를 단일 primitive로 분해하지 못함
2. **환경 변화 미적응**: sequential 실행 중 scene이 바뀌어 distribution shift 발생

이 프로젝트는 두 문제를 단계적으로 해결하는 계층적 프레임워크를 제안한다.

---

## 프로젝트 구조

```
hierarchical_vla/
│
├── libero_bench/                     # 공통 인프라
│   ├── dataloader.py                 # LIBERO hdf5 데이터 로더 (LiberoDataset)
│   ├── llm_orchestrator.py           # GPT-4o 오케스트레이터 + home pose + replan
│   ├── libero_primitives.json        # scene별 VLA primitive 목록 (15개)
│   ├── eval_libero10_llm.py          # Baseline (MambaVLA) 평가
│   ├── eval_libero10_llm_scene.py    # MambaVLAScene 평가
│   ├── train.py                      # MambaVLA 학습
│   └── train_scene.py                # MambaVLAScene 학습
│
├── models/                           # MambaVLAScene (Baseline 2)
│   ├── vjepa_encoder.py              # VJEPASceneEncoder
│   ├── mambavla_scene.py             # SceneAwareMambaVLAPolicy
│   ├── model_factory_scene.py        # create_mambavla_scene_model()
│   └── precompute_vjepa.py           # V-JEPA 캐시 사전 계산
│
├── idea2/                            # IDEA2 v2 (제안 방법)
│   ├── scene_encoder.py              # frozen ResNet-18 → z_scene
│   ├── context_encoder.py            # delta_history → context_vec
│   ├── memory.py                     # EpisodeMemory (JSON 기반)
│   ├── vqvae.py                      # VQVAEModel (delta → codebook)
│   ├── latent_vla.py                 # LatentVLAPolicy + IDEA2Model
│   ├── action_decoder.py             # codebook[k] + z_scene → actions
│   ├── model_factory.py              # create_idea2_model()
│   ├── precompute_resnet.py          # ResNet 캐시 사전 계산
│   ├── dataset.py                    # DeltaDataset, LatentActionDataset
│   ├── train_stage1.py               # VQ-VAE 학습
│   ├── train_stage2.py               # LatentVLAPolicy 학습
│   ├── train_stage3.py               # ActionDecoder 학습
│   └── eval_idea2.py                 # EpisodeMemory + replan 평가
│
├── world_model/                      # Future Work
│   └── ...
│
├── language_embeddings/              # LIBERO task CLIP embeddings
├── checkpoints/                      # 학습 체크포인트
│   ├── libero_10_subset/             # MambaVLA (Baseline)
│   └── idea2_v2/                     # IDEA2 v2
│       ├── stage1/                   # VQ-VAE
│       ├── stage2/                   # LatentVLAPolicy
│       └── stage3/                   # ActionDecoder + scaler
├── resnet_cache/                     # ResNet z_scene 캐시 (IDEA2 v2용)
│   └── libero_10_subset/
│       ├── scene_encoder.pt          # 일관성 보장용 가중치
│       └── *.pt                      # task별 z_scene 캐시
└── conf/
    └── config.yaml
```

---

## Baseline 1: LLM Orchestrator + MambaVLA

### 핵심 아이디어

LLM(GPT-4o)이 long-horizon task를 primitive 시퀀스로 분해하고, MambaVLA가 각 primitive를 순차 실행.
primitive 사이에 home pose 복귀를 삽입해 동작 간 안정성 확보.

### 아키텍처

```
LIBERO-10 Task
      ↓
  GPT-4o  ← libero_primitives.json
      ↓
[prim_1, prim_2, ..., prim_N]
      ↓
MambaVLA(prim_1) → home pose → MambaVLA(prim_2) → ... → MambaVLA(prim_N)
```

### 성능

| 방법 | LIBERO-10 Success Rate |
|------|----------------------|
| VLA 단독 (OpenVLA-OFT) | 14.3% |
| LLM Orchestrator + MambaVLA | **90.0%** |

### 학습

```bash
cd /home/choi/LGHA/hierarchical_vla
python -m libero_bench.train \
    data_directory=LIBERO/libero/datasets/libero_10_subset \
    save_dir=./checkpoints/libero_10_subset \
    num_epochs=2000 max_len_data=400 num_workers=0 wandb.enabled=false
```

### 평가

```bash
python -m libero_bench.eval_libero10_llm \
    --checkpoint ./checkpoints/libero_10_subset/final_model.pth \
    --openai_api_key $OPENAI_API_KEY
```

---

## Baseline 2: MambaVLAScene (V-JEPA Scene Conditioning)

현재 scene 상태를 V-JEPA 2로 encoding해 VLA에 추가 conditioning.
→ distribution shift를 줄이는 시도이나 V-JEPA 7B 모델이 inference 시 필요.

자세한 내용: [models/](models/)

---

## 제안 방법: IDEA2 v2 (Memory-Guided Context Injection)

자세한 내용: [IDEA2.md](IDEA2.md)

### 핵심 아이디어

이전 primitive들이 환경에 만든 변화(delta_z)를 **EpisodeMemory**에 기록하고,
다음 primitive 실행 시 **ContextEncoder**를 통해 context_vec으로 변환해 VLA에 주입.

```
primitive 1 실행 전: context_vec = 0
primitive 1 실행 후: memory = [{delta_z: ..., success: true}]

primitive 2 실행 전: context_vec = ContextEncoder(memory)
                   → ActionDecoder가 "이전에 이런 변화가 있었어"를 알고 동작
```

LLM은 실패 시 memory에서 피드백 텍스트를 받아 재계획(replan).

### 특징

- **V-JEPA 완전 제거**: inference 시 ResNet-18만 사용 (경량, 빠름)
- **z_scene 일관성**: precompute_resnet.py가 저장한 동일 가중치를 모든 단계에서 사용
- **context_vec**: 학습 파라미터 없음, 지수감쇠 가중합
- **LLM 양방향 소통**: VLA 실패 → delta 정보 → LLM replan

### 학습 순서

```bash
cd /home/choi/LGHA/hierarchical_vla

# 0. ResNet 캐시 (1회)
python -m hierarchical_vla.idea2.precompute_resnet \
    --data_dir LIBERO/libero/datasets/libero_10_subset \
    --cache_dir ./resnet_cache/libero_10_subset

# 1. Stage 1: VQ-VAE
python -m hierarchical_vla.idea2.train_stage1 \
    --data_dir LIBERO/libero/datasets/libero_10_subset \
    --cache_dir ./resnet_cache/libero_10_subset \
    --save_dir ./checkpoints/idea2_v2/stage1 --K 64 --epochs 200

# 2. Stage 2: Latent VLA Policy
python -m hierarchical_vla.idea2.train_stage2 \
    --data_dir LIBERO/libero/datasets/libero_10_subset \
    --cache_dir ./resnet_cache/libero_10_subset \
    --vqvae_path ./checkpoints/idea2_v2/stage1/vqvae.pt \
    --save_dir ./checkpoints/idea2_v2/stage2 --epochs 500

# 3. Stage 3: Action Decoder
python -m hierarchical_vla.idea2.train_stage3 \
    --data_dir LIBERO/libero/datasets/libero_10_subset \
    --cache_dir ./resnet_cache/libero_10_subset \
    --vqvae_path ./checkpoints/idea2_v2/stage1/vqvae.pt \
    --stage2_path ./checkpoints/idea2_v2/stage2/final_model.pth \
    --save_dir ./checkpoints/idea2_v2/stage3 --epochs 1000
```

### 평가

```bash
python -m hierarchical_vla.idea2.eval_idea2 \
    --checkpoint ./checkpoints/idea2_v2/stage3/final_model.pth \
    --vqvae_path ./checkpoints/idea2_v2/stage1/vqvae.pt \
    --cache_dir ./resnet_cache/libero_10_subset \
    --openai_api_key $OPENAI_API_KEY \
    --num_rollouts 3 --max_replans 2
```

---

## 비교 실험 계획

| 방법 | 계층구조 | Scene Conditioning | Memory/Context | LIBERO-10 SR |
|------|---------|-------------------|----------------|-------------|
| VLA 단독 (OpenVLA-OFT) | ❌ | ❌ | ❌ | 14.3% |
| MambaVLA + LLM (Baseline 1) | ✅ | ❌ | ❌ | ~90% |
| MambaVLAScene + LLM (Baseline 2) | ✅ | V-JEPA | ❌ | TBD |
| **IDEA2 v2 + LLM (제안)** | ✅ | ResNet | ✅ EpisodeMemory | **TBD** |

---

## 데이터셋

- **경로**: `LIBERO/libero/datasets/libero_10_subset/`
- LIBERO-10 태스크 수행에 필요한 LIBERO-90 primitive **15개** 선별 (전체 90개 중)
- 태스크당 50 demo, max demo length = 400 steps

선별된 primitive scenes:
```
KITCHEN_SCENE3, KITCHEN_SCENE4, KITCHEN_SCENE8
LIVING_ROOM_SCENE1, LIVING_ROOM_SCENE2, LIVING_ROOM_SCENE5, LIVING_ROOM_SCENE6
STUDY_SCENE1
```
