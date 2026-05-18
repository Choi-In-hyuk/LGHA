"""
Qwen2-VL LoRA 파인튜닝 스크립트. 단일 GPU 및 DDP 모두 지원.

단일 GPU:
  python -m hierarchical_vla.pipeline.finetune_qwen \
      --data_path /home/choi/libero_object/qwen_finetune.jsonl \
      --output_dir checkpoints/qwen_lora \
      --num_epochs 3 --device cuda:1

DDP (2 GPU):
  python -m torch.distributed.run --nproc_per_node=2 --master_port=29500 \
      -m hierarchical_vla.pipeline.finetune_qwen \
      --data_path /home/choi/libero_object/qwen_finetune.jsonl \
      --output_dir checkpoints/qwen_lora \
      --num_epochs 3
"""

import argparse
import base64
import json
import logging
import os
from io import BytesIO

import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

_USER_TEMPLATE = (
    "Task: {instruction}\n"
    "Look at the image and output the grid positions (0-1023) of:\n"
    "1) the object to pick up\n"
    "2) the target location to place it\n"
    "Format: object: NNN target: MMM"
)


class LocDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.records = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        log.info(f"Loaded {len(self.records)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_bytes = base64.b64decode(rec["image"])
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
        return {
            "image":       image,
            "instruction": rec["instruction"],
            "response":    rec["response"],
        }


def finetune(
    data_path: str,
    output_dir: str,
    model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
    device: str = "cuda:1",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    save_steps: int = 200,
):
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from peft import LoraConfig, get_peft_model, TaskType

    # ── DDP 초기화 ────────────────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_ddp = local_rank >= 0

    if is_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        world_size = dist.get_world_size()
    else:
        world_size = 1

    is_main = (not is_ddp) or (local_rank == 0)

    if is_main:
        log.info(f"DDP={'ON world_size=' + str(world_size) if is_ddp else 'OFF'} | device={device}")

    # ── 1. 모델 로드 ───────────────────────────────────────────────────────────
    if is_main:
        log.info(f"Loading {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)

    if is_ddp:
        # device_map은 DDP와 호환 안 됨 → CPU 로드 후 GPU로 이동
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        ).to(device)
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )

    # ── 2. LoRA 적용 ──────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    if is_main:
        model.print_trainable_parameters()
    model.train()

    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    # ── 3. 데이터셋 ───────────────────────────────────────────────────────────
    dataset = LocDataset(data_path)

    if is_ddp:
        sampler = DistributedSampler(dataset, num_replicas=world_size,
                                     rank=local_rank, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                num_workers=2, collate_fn=lambda x: x)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                num_workers=2, collate_fn=lambda x: x)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
    )

    if is_main:
        os.makedirs(output_dir, exist_ok=True)

    global_step = 0

    # ── 4. 학습 루프 ──────────────────────────────────────────────────────────
    for epoch in range(1, num_epochs + 1):
        if is_ddp:
            sampler.set_epoch(epoch)
        epoch_loss = 0.0

        for batch in dataloader:
            messages_list = []
            for sample in batch:
                messages_list.append([
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": sample["image"]},
                            {"type": "text",  "text": _USER_TEMPLATE.format(
                                instruction=sample["instruction"]
                            )},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": sample["response"]}],
                    },
                ])

            from qwen_vl_utils import process_vision_info

            texts = [
                processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False
                )
                for msgs in messages_list
            ]
            all_images = []
            for msgs in messages_list:
                imgs, _ = process_vision_info(msgs)
                if imgs:
                    all_images.extend(imgs)

            inputs = processor(
                text=texts,
                images=all_images if all_images else None,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            labels = inputs["input_ids"].clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100

            optimizer.zero_grad()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if is_main and global_step % 50 == 0:
                log.info(f"  step={global_step} loss={loss.item():.4f}")

            if is_main and global_step % save_steps == 0:
                raw = model.module if is_ddp else model
                ckpt = os.path.join(output_dir, f"step_{global_step:05d}")
                raw.save_pretrained(ckpt)
                processor.save_pretrained(ckpt)
                log.info(f"  Saved → {ckpt}")

        if is_main:
            avg = epoch_loss / len(dataloader)
            log.info(f"Epoch {epoch}/{num_epochs} | avg_loss={avg:.4f}")

    # ── 5. 최종 저장 (rank 0만) ───────────────────────────────────────────────
    if is_main:
        raw = model.module if is_ddp else model
        final_path = os.path.join(output_dir, "final")
        raw.save_pretrained(final_path)
        processor.save_pretrained(final_path)
        log.info(f"Fine-tuning complete → {final_path}")

    if is_ddp:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",      required=True)
    parser.add_argument("--output_dir",     default="./checkpoints/qwen_lora")
    parser.add_argument("--model_id",       default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--device",         default="cuda:1",
                        help="단일 GPU 모드에서만 사용 (DDP 시 LOCAL_RANK로 자동 결정)")
    parser.add_argument("--num_epochs",     type=int,   default=3)
    parser.add_argument("--batch_size",     type=int,   default=4)
    parser.add_argument("--learning_rate",  type=float, default=2e-4)
    parser.add_argument("--lora_r",         type=int,   default=16)
    parser.add_argument("--save_steps",     type=int,   default=200)
    args = parser.parse_args()

    finetune(
        data_path=args.data_path,
        output_dir=args.output_dir,
        model_id=args.model_id,
        device=args.device,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        save_steps=args.save_steps,
    )


if __name__ == "__main__":
    main()
