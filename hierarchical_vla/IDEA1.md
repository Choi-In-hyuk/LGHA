# IDEA 1: Shared Latent Working Memory for LLM-VLA Collaboration

## 동기 (Motivation)

현재 계층적 VLA 프레임워크의 근본적 병목:

```
GPT-4o → "pick up the cup" (text) → CLIP encode → z_lang → VLA policy
```

LLM 내부에는 태스크에 대한 풍부한 맥락 표현이 있지만, 텍스트 직렬화 과정에서 대부분 손실됨.
또한 VLA는 이전 primitive 실행 결과(scene 변화)를 다음 primitive 선택에 반영하지 못함.

**참고 논문**: LatentMAS (arxiv 2511.20639) — 여러 LLM 에이전트가 텍스트 대신 latent space를 직접 공유해 협업. 4× 빠른 추론, 14.6% 정확도 향상.

---

## 핵심 아이디어

LLM task planner와 VLA policy가 **공유 latent working memory**를 통해 소통.

```
                    ┌──────────────────────────────┐
                    │   Shared Latent Working Memory│
                    │   z_memory = f(z_task, z_scene)│
                    └──────────┬───────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         ↓                     ↓                     ↓
   LLM hidden state       V-JEPA z_scene         VLA policy
   (z_task)               (업데이트됨)            (z_memory 조건부)
```

각 primitive 실행 후 z_scene이 갱신 → z_memory 업데이트 → 다음 primitive 실행에 반영.

---

## 현재 구조 vs 제안 구조

| 항목 | 현재 (계층적 VLA) | 제안 (Shared Latent Memory) |
|---|---|---|
| Task planner | GPT-4o (text output) | 오픈소스 LLM (hidden state 접근) |
| Scene 표현 | V-JEPA z_scene (static, 학습 중만) | z_scene 매 primitive마다 업데이트 |
| LLM→VLA 전달 | CLIP-encoded text string | z_memory (latent 직접 전달) |
| Primitive 간 상태 공유 | 없음 (home pose return만) | z_memory에 누적 |
| 정보 손실 | 텍스트 직렬화로 손실 큼 | 없음 |

---

## 구현 계획

### Stage 1: LLM 교체
- GPT-4o → LLaMA-3 / Qwen2.5 등 오픈소스 LLM
- 이유: hidden state (last layer) 접근 필요
- z_task = LLM의 task description에 대한 마지막 hidden state (dim: ~4096)

### Stage 2: Shared Memory 모듈
```python
class SharedLatentMemory(nn.Module):
    def __init__(self, task_dim, scene_dim, memory_dim):
        self.task_proj  = nn.Linear(task_dim, memory_dim)
        self.scene_proj = nn.Linear(scene_dim, memory_dim)
        self.fusion     = nn.TransformerEncoder(...)  # 2 tokens: z_task, z_scene

    def update(self, z_task, z_scene):
        # primitive 실행 전 호출
        z_memory = self.fusion(z_task, z_scene)
        return z_memory  # (1, memory_dim)
```

### Stage 3: VLA 수정
- MambaVLA의 lang token을 z_lang (CLIP) 대신 z_memory로 교체
- 또는 z_memory를 추가 token으로 삽입: [sigma, z_memory, obs, action]

### Stage 4: 학습
- z_memory 모듈 + VLA policy 공동 학습
- LLM은 frozen (LoRA 선택적 적용)
- V-JEPA는 frozen (pre-computed cache 재사용 가능)

---

## 기대 효과

1. **LLM-VLA 간극 해소**: 텍스트 없이 latent에서 직접 소통
2. **sequential execution 향상**: 이전 primitive 결과가 z_scene 업데이트로 자동 반영
3. **Distribution shift 대응**: 매 primitive마다 현재 scene state 반영
4. **연구 contribution**: VLA에 LatentMAS 개념 최초 적용 (로보틱스 도메인)

---

## 선행 연구 관계

```
현재 연구 (MambaVLAScene)
  └→ Scene token으로 VLA에 scene 정보 추가
       └→ IDEA 1 (Shared Latent Memory)
            └→ LLM latent + scene latent 통합, 공유 메모리로 LLM-VLA 간극 해소
```

---

## 현실적 고려사항

- **GPT-4o 교체 비용**: 오픈소스 LLM이 task decomposition 품질 낮을 수 있음
  - 완화책: task decomposition은 GPT-4o 유지, z_task는 fine-tuned 소형 LLM에서 생성
- **latent space alignment**: LLM hidden dim (~4096) vs scene dim (256) 불일치
  - 완화책: projection layer로 memory_dim=512로 통일
- **학습 데이터 부족**: 공유 메모리 학습에 multi-step demonstration 필요
  - 완화책: LIBERO-10 데이터 + synthetic multi-step trajectories

---

## TODO (착수 시)

- [ ] 오픈소스 LLM 선정 (LLaMA-3-8B vs Qwen2.5-7B)
- [ ] SharedLatentMemory 모듈 구현
- [ ] MambaVLAScene을 z_memory 조건부로 수정
- [ ] LIBERO-10에서 실험: text comm vs latent memory 비교
- [ ] 논문 작성
