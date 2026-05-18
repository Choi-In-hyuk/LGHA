# LGHA: Language-Guided Hierarchical Action

Hierarchical Vision-Language-Action (VLA) system for robot manipulation on the LIBERO benchmark.  
High-level task planning (Qwen2-VL-7B) + low-level motor control (MambaVLA) with online failure-augmented LoRA retraining.

---

## Motivation

Standard VLA models couple language understanding and motor control in a single network, making them difficult to scale and adapt. LGHA decomposes the problem into two levels:

1. **High-level**: *Where* is the object, and *what phase* is the task in?
2. **Low-level**: *How* should the robot move to accomplish that phase?

This separation allows each module to be trained and improved independently. In particular, the high-level localizer (Qwen2-VL) can be refined online from failure cases *without* retraining the entire policy.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Environment                             │
│   obs: {agentview_image, eye_in_hand_image, robot_state}        │
└───────────────────┬─────────────────────────────────────────────┘
                    │
          ┌─────────▼──────────┐
          │   PhaseDetector    │  (rule-based, every step)
          │  gripper + eef     │
          │  → phase ∈ {0..4}  │
          └─────────┬──────────┘
                    │ phase_id
          ┌─────────▼──────────┐
          │  Qwen2-VL-7B       │  (every qwen_stride steps)
          │  + LoRA adapter    │
          │  image + instr     │
          │  → loc_token ∈     │
          │    {0..1023}       │
          └─────────┬──────────┘
                    │ (phase_id, loc_token)
          ┌─────────▼──────────┐
          │    MambaVLA        │  (every step)
          │  SSM-based policy  │
          │  → action ∈ R^7    │
          └─────────┬──────────┘
                    │
          ┌─────────▼──────────┐
          │   Robot Actuator   │
          │  7-DOF joint ctrl  │
          └────────────────────┘
```

---

## Components

### 1. PhaseDetector (rule-based)

Task phase는 gripper openness와 end-effector 위치로 규칙 기반 감지합니다.

| Phase | ID | Condition |
|-------|----|-----------|
| REACH | 0 | gripper open, eef far from object |
| GRASP | 1 | gripper open → closing, eef ≈ object |
| TRANSPORT | 2 | gripper closed, object lifted |
| PLACE | 3 | gripper closed, near target XY |
| RELEASE | 4 | gripper opening, object at target |

Gripper 상태는 `robot0_gripper_qpos` 합으로 판단:

$$\text{is\_open} = \sum_i |q_{\text{gripper},i}| \geq \theta_{\text{open}}, \quad \theta_{\text{open}} = 0.073$$

### 2. loc_token: 32×32 Grid Localization

이미지 상의 물체 위치를 단일 정수로 인코딩합니다:

$$\text{loc\_token} = r_{32} \times 32 + c_{32}, \quad \text{loc\_token} \in \{0, \ldots, 1023\}$$

여기서 $r_{32}, c_{32}$는 128×128 픽셀 좌표를 32×32 그리드로 양자화한 행/열입니다:

$$r_{32} = \left\lfloor \frac{p_{\text{row}}}{128} \times 31 \right\rfloor, \quad c_{32} = \left\lfloor \frac{p_{\text{col}}}{128} \times 31 \right\rfloor$$

3D 물체 위치 $\mathbf{x}_{\text{world}} \in \mathbb{R}^3$는 robosuite 카메라 투영행렬 $\mathbf{T} \in \mathbb{R}^{4 \times 4}$로 픽셀 좌표로 변환됩니다:

$$\mathbf{p} = \pi\bigl(\mathbf{T} \cdot \tilde{\mathbf{x}}_{\text{world}}\bigr) \in \mathbb{R}^2$$

**GT 모드**: 시뮬레이터 state에서 직접 계산 → 항상 정확  
**Qwen LoRA 모드**: Qwen2-VL이 이미지를 보고 `"object: NNN target: MMM"` 형식으로 예측

### 3. Qwen2-VL-7B Orchestrator (High-Level)

Qwen2-VL-7B-Instruct에 LoRA adapter를 붙여 loc_token 예측에 fine-tune합니다.

**입력**: agentview 이미지 + 자연어 명령  
**출력**: `"object: NNN target: MMM"` (object loc_token, target loc_token)

**추론 빈도**: 매 `qwen_stride = 5` 스텝마다 호출, 중간 스텝은 캐시된 값 사용

Phase에 따라 사용하는 loc_token 구분:

$$\text{loc\_token}_t = \begin{cases} \hat{l}_{\text{obj}} & \text{phase} \in \{\text{REACH, GRASP}\} \\ \hat{l}_{\text{target}} & \text{phase} \in \{\text{TRANSPORT, PLACE, RELEASE}\} \end{cases}$$

### 4. MambaVLA (Low-Level Policy)

SSM(State Space Model) 기반 diffusion policy. Flow matching으로 학습됩니다.

**입력 임베딩**:

$$\mathbf{e}_{\text{obs}} = f_{\text{img}}(I_{\text{agent}}, I_{\text{hand}}) \oplus \mathbf{W}_s \mathbf{s}_{\text{robot}}$$

$$\mathbf{e}_{\text{phase}} = \text{Emb}_{\text{phase}}(\text{phase\_id}) \in \mathbb{R}^{256}$$

$$\mathbf{e}_{\text{obj}} = g\bigl(\text{RoIAlign}(F_{\text{img}},\ r_{32}, c_{32})\bigr) \in \mathbb{R}^{256}$$

Mamba sequence에 phase 임베딩을 object 임베딩보다 앞에 배치하여, SSM이 task context를 먼저 처리한 후 위치 정보를 조건부로 받습니다:

$$\mathbf{h} = \text{Mamba}\bigl([\mathbf{e}_{\text{obs}};\ \mathbf{e}_{\text{phase}};\ \mathbf{e}_{\text{obj}}]\bigr)$$

**Flow Matching Loss**:

시간 $t \in [0,1]$에서 노이즈 액션 $\mathbf{a}_t = (1-t)\mathbf{\epsilon} + t\mathbf{a}^*$ 를 예측 velocity로 회귀:

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, \mathbf{a}^*, \mathbf{\epsilon}}\left[\left\|\mathbf{v}_\theta(\mathbf{a}_t, \mathbf{h}, t) - (\mathbf{a}^* - \mathbf{\epsilon})\right\|^2\right]$$

추론 시 ODE integration으로 액션 시퀀스 $\mathbf{a}^* \in \mathbb{R}^{10 \times 7}$ 생성.

---

## Failure-Augmented LoRA Retraining

Qwen LoRA가 loc_token을 잘못 예측할 경우, GRASP phase에서 `grasp_timeout` 스텝이 초과되면 실패로 판정하고 해당 프레임을 재학습 데이터로 수집합니다.

### 실패 감지 조건

$$\text{fail} \iff \text{phase} = \text{GRASP} \land (t - t_{\text{grasp\_start}}) \geq \tau_{\text{grasp}}, \quad \tau_{\text{grasp}} = 50$$

### 수집 데이터 포맷

실패 시점의 agentview 이미지에 GT loc_token을 정답으로 붙입니다:

```json
{
  "image": "<base64 PNG>",
  "instruction": "pick up the ketchup and place it in the basket",
  "response": "object: 412 target: 683",
  "source": "failure"
}
```

GT loc_token은 시뮬레이터 state에서 직접 계산하므로 항상 정확합니다.

### 재학습 파이프라인

```
Step 1. Eval (Qwen LoRA)  →  실패 프레임 수집
Step 2. Merge             →  D_orig (7680) ∪ D_fail
Step 3. LoRA Retrain      →  failure-augmented fine-tuning
Step 4. Eval (new LoRA)   →  성능 검증
```

**학습 Loss**: 동일한 cross-entropy, assistant 응답 토큰에만 적용

$$\mathcal{L}_{\text{LoRA}} = -\sum_{k} \log P_\theta\bigl(y_k \mid y_{<k},\ I,\ \text{instr}\bigr)$$

**LoRA Config**: $r=16$, $\alpha=32$, dropout $= 0.05$, target modules: `{q, k, v, o, gate, up, down}_proj`

---

## Results (LIBERO-Object, 20 ep/task)

| Task | Before Retrain | After Retrain | Δ |
|------|:--------------:|:-------------:|:-:|
| alphabet_soup | 95.0% | 90.0% | -5% |
| cream_cheese | 45.0% | 50.0% | +5% |
| salad_dressing | 90.0% | 95.0% | +5% |
| bbq_sauce | 60.0% | 50.0% | -10% |
| ketchup | 55.0% | 90.0% | **+35%** |
| tomato_sauce | 85.0% | 80.0% | -5% |
| butter | 65.0% | 75.0% | +10% |
| milk | 60.0% | 55.0% | -5% |
| chocolate_pudding | 100.0% | 95.0% | -5% |
| orange_juice | 100.0% | 100.0% | 0% |
| **Overall** | **75.5%** | **78.0%** | **+2.5%** |

1 epoch, 실패 프레임 111건 추가만으로 +2.5% 향상. ketchup은 +35%로 가장 큰 폭 개선.  
태스크별 편차가 있는 것은 에피소드 수(20개)가 적어 분산이 크기 때문이며, 더 많은 실패 데이터 수집 및 epoch 증가로 개선 가능.

---

## Setup

```bash
conda activate lgha

# 평가
python -m hierarchical_vla.pipeline.evaluate \
    --checkpoint checkpoints/hierarchical_phase/final.pth \
    --suite libero_object --num_episodes 20 \
    --device cuda:0 --qwen_device cuda:0 \
    --phase --qwen_lora checkpoints/qwen_lora/final \
    --qwen_stride 5 --grasp_timeout 50 --no_video

# 전체 파이프라인 (실패 수집 → merge → 재학습 → 재평가)
bash run_all_sequential.sh
```

---

## Future Work

- **반복 재학습 (Iterative DAgger-style)**: 재학습된 LoRA로 다시 실패 수집 → 반복
- **Epoch 및 데이터 스케일 확장**: 현재 1 epoch / 111건 → 더 많은 실패 케이스와 epoch으로 수렴 여부 확인
- **태스크별 분리 학습**: cream_cheese, bbq_sauce 등 낮은 성능 태스크에 실패 데이터 집중
- **DDP 학습**: cuda:1이 가용해지면 두 GPU로 데이터 병렬 학습으로 속도 개선
- **LIBERO-Spatial / LIBERO-Goal 확장**: 다른 suite로 일반화 검증
