# IDEA2 v2: Memory-Guided Context Injection for Adaptive VLA

## 동기

LLM + VLA 프레임워크에서 VLA가 이전 primitive 결과로 달라진 환경을 인식하지 못하는 문제.

```
primitive 1 완료: 파란 접시가 테이블에서 치워짐
primitive 2 시작: VLA는 "파란 접시가 없다"는 사실을 모른 채 학습 시 분포와 다른 scene에서 동작
→ 성공률 저하
```

LLM은 이전 단계 결과로 환경 변화를 알고 있다.
하지만 그 정보를 VLA에게 텍스트로 줄 수 없다.
**이전 primitive들이 scene에 만든 변화(delta_z)를 latent로 누적해 VLA에 주입**하는 것이 핵심 아이디어.

---

## 전체 아키텍처

```
[Task 시작]
  memory = []   ← 비어있음

  LLM: task → ["pick_red_plate", "place_on_table", ...]

  ┌─ Primitive Loop ───────────────────────────────────────────────┐
  │                                                                 │
  │  1. memory → ContextEncoder → context_vec (256-dim)           │
  │     (처음엔 zero, 이전 단계 delta들의 지수감쇠 합)              │
  │                                                                 │
  │  2. SceneEncoder(image) → z_scene (256-dim)                   │
  │     (frozen ResNet-18, 학습/inference 완전 동일)               │
  │                                                                 │
  │  3. CLIP(primitive_text) → z_goal (512-dim)                   │
  │                                                                 │
  │  4. LatentVLAPolicy(obs_tokens, z_goal) → codebook_k          │
  │     (Mamba backbone + classification head)                      │
  │                                                                 │
  │  5. ActionDecoder(codebook[k], z_scene + context_vec)         │
  │     → action_seq (10, 7)                                       │
  │                                                                 │
  │  6. 로봇 실행 (max 250 steps)                                   │
  │                                                                 │
  │  7. z_before, z_after 측정                                     │
  │     delta = z_after - z_before                                  │
  │     success = reward OR ||delta - codebook[k]|| < threshold    │
  │                                                                 │
  │  8. memory에 기록:                                              │
  │     {primitive, delta_z, success, codebook_k}                  │
  │                                                                 │
  │  9. 실패 시: memory.get_feedback_for_llm() → LLM 재계획       │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘
```

---

## 모듈 구성

```
idea2/
├── scene_encoder.py       SceneEncoder: frozen ResNet-18 → z_scene (256)
│                          학습/inference 어디서나 동일 가중치 → 공간 일관성
│
├── context_encoder.py     ContextEncoder: delta_history → context_vec (256)
│                          지수감쇠 가중합: context = Σ decay^i * delta_{-i}
│                          학습 파라미터 없음
│
├── memory.py              EpisodeMemory: primitive 결과 JSON 저장
│                          - write(primitive, delta_z, success, k)
│                          - get_delta_history() → ContextEncoder 입력
│                          - get_feedback_for_llm() → LLM replan 텍스트
│
├── vqvae.py               VQVAEModel: delta_z → codebook index (Stage 1)
│                          ResNet delta 기반 (V-JEPA 불필요)
│
├── latent_vla.py          LatentVLAPolicy: obs_tokens + lang → codebook_k
│                          IDEA2Model: 전체 파이프라인 통합
│
├── action_decoder.py      ActionDecoder: codebook[k] + z_input → action_seq
│                          z_input = z_scene + context_vec
│
├── model_factory.py       create_idea2_model(), create_context_encoder()
│
├── precompute_resnet.py   SceneEncoder로 모든 frame z_scene 사전 계산
│                          scene_encoder.pt 저장 → 이후 모든 단계에서 로드
│
├── dataset.py             DeltaDataset (Stage 1), LatentActionDataset (Stage 2/3)
│                          context_vec simulation: 50% zero, 50% delta 누적
│
├── train_stage1.py        VQ-VAE 학습 (ResNet delta 기반)
├── train_stage2.py        LatentVLAPolicy 학습 (CrossEntropy)
├── train_stage3.py        ActionDecoder 학습 (MSE, z_scene live 계산)
└── eval_idea2.py          EpisodeMemory + replan 루프 포함 평가
```

---

## Stage별 학습 전략

### Stage 1: VQ-VAE (ResNet delta)

```
SceneEncoder(frame_t+1) - SceneEncoder(frame_t) → delta (256)
delta → VQ-VAE encoder → quantize → codebook[k]
Loss: reconstruction + commitment
```

- 기존 V-JEPA 기반 cache 대신 ResNet cache 사용
- inference와 완전 동일한 공간 → z_scene 일관성 보장

### Stage 2: LatentVLAPolicy

```
입력: obs_tokens (ResNet 2-cam), lang_emb (CLIP)
출력: codebook index k (CrossEntropy)

Token sequence (Mamba): [obs_0, obs_1, ..., lang] ← lang 마지막
→ Mamba causal: lang이 모든 obs 정보를 보고 예측

Context simulation:
  50% → context_vec = zeros (첫 primitive 상황)
  50% → context_vec = 이전 step delta 지수감쇠 합
```

### Stage 3: ActionDecoder

```
입력: codebook[k] (256), z_scene (256, scene_encoder로 live 계산), context_vec (256)
z_input = z_scene + context_vec
출력: action_seq (10, 7) MSE loss

핵심: z_scene을 학습 시에도 scene_encoder로 live 계산
→ 학습/inference 완전 동일한 z_scene 공간
```

---

## 기존 IDEA2 (V-JEPA 기반) 와의 차이

| | IDEA2 v1 (구) | IDEA2 v2 (현재) |
|---|---|---|
| z_scene 인코더 | V-JEPA (7B ViT-g) | ResNet-18 (frozen, 경량) |
| z_scene 일관성 | ❌ Stage마다 다른 proj 초기화 | ✅ precompute_resnet.pt 동일 로드 |
| context | ❌ 없음 | ✅ EpisodeMemory → delta 누적 |
| LLM 피드백 | ❌ 없음 | ✅ 실패 시 delta 정보 전달 replan |
| 성공 판단 | reward만 | reward + latent distance |
| inference 속도 | V-JEPA 7B 필요 | ResNet만 |

---

## 학습 순서

```bash
# 0. ResNet cache 사전 계산 (1회)
python -m hierarchical_vla.idea2.precompute_resnet \
    --data_dir LIBERO/libero/datasets/libero_10_subset \
    --cache_dir ./resnet_cache/libero_10_subset \
    --demos_per_task 50
# → resnet_cache/libero_10_subset/scene_encoder.pt 저장 (일관성 핵심)

# 1. Stage 1: VQ-VAE
python -m hierarchical_vla.idea2.train_stage1 \
    --data_dir LIBERO/libero/datasets/libero_10_subset \
    --cache_dir ./resnet_cache/libero_10_subset \
    --save_dir ./checkpoints/idea2_v2/stage1 \
    --K 64 --epochs 200

# 2. Stage 2: Latent VLA Policy
python -m hierarchical_vla.idea2.train_stage2 \
    --data_dir LIBERO/libero/datasets/libero_10_subset \
    --cache_dir ./resnet_cache/libero_10_subset \
    --vqvae_path ./checkpoints/idea2_v2/stage1/vqvae.pt \
    --save_dir ./checkpoints/idea2_v2/stage2 \
    --epochs 500

# 3. Stage 3: Action Decoder
python -m hierarchical_vla.idea2.train_stage3 \
    --data_dir LIBERO/libero/datasets/libero_10_subset \
    --cache_dir ./resnet_cache/libero_10_subset \
    --vqvae_path ./checkpoints/idea2_v2/stage1/vqvae.pt \
    --stage2_path ./checkpoints/idea2_v2/stage2/final_model.pth \
    --save_dir ./checkpoints/idea2_v2/stage3 \
    --epochs 1000
```

---

## 평가

```bash
python -m hierarchical_vla.idea2.eval_idea2 \
    --checkpoint ./checkpoints/idea2_v2/stage3/final_model.pth \
    --vqvae_path ./checkpoints/idea2_v2/stage1/vqvae.pt \
    --cache_dir ./resnet_cache/libero_10_subset \
    --openai_api_key $OPENAI_API_KEY \
    --num_rollouts 3 \
    --max_replans 2
```

---

## EpisodeMemory 파일 예시

```json
{
  "history": [
    {
      "primitive": "KITCHEN_SCENE3_pick_up_the_red_plate",
      "delta_z": [0.12, -0.03, ...],
      "success": true,
      "codebook_k": 23
    },
    {
      "primitive": "KITCHEN_SCENE3_place_on_the_counter",
      "delta_z": [0.01, 0.08, ...],
      "success": false,
      "codebook_k": 7
    }
  ]
}
```

실패 시 LLM에게 전달되는 피드백:
```
Step 1 [KITCHEN_SCENE3_pick_up_the_red_plate]: SUCCESS (scene_change=0.432, codebook_k=23)
Step 2 [KITCHEN_SCENE3_place_on_the_counter]: FAILED (scene_change=0.089, codebook_k=7)
```
