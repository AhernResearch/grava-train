"""Local multimodal inference, with one serialized generation stream per device."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


class _TransformersWorker:
    """Single-device transformers inference worker."""

    def __init__(
        self,
        model: str,
        device: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._lock = asyncio.Lock()
        self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
        self.processor.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        template_path = Path(model) / "chat_template_no_think.jinja"
        self.chat_template = template_path.read_text() if template_path.exists() else None


class TransformersInferenceClient:
    """Multi-worker transformers inference client with async wrapper."""

    def __init__(
        self,
        model: str,
        device: str = "cuda:0",
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
        devices: Sequence[str] | None = None,
    ):
        self.devices = list(devices) if devices else [device]
        self.workers = [
            _TransformersWorker(
                model=model,
                device=worker_device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            for worker_device in self.devices
        ]
        self.n_clients = len(self.workers)

    @staticmethod
    def _build_prompt_and_images(
        worker: _TransformersWorker, sample: Dict, enable_thinking: bool,
    ) -> tuple[str, List[Image.Image]]:
        text = sample["messages"][0]["content"].replace("<image>", "").strip()
        images = []
        for path in sample["images"]:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        conversation = [{
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in images],
                {"type": "text", "text": text},
            ],
        }]
        prompt = worker.processor.apply_chat_template(
            conversation,
            chat_template=worker.chat_template if not enable_thinking else None,
            enable_thinking=enable_thinking,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt, images

    @staticmethod
    def _clean_response(response: str) -> str:
        response = response.lstrip()
        if response.startswith("<|endoftext|>"):
            response = response[len("<|endoftext|>"):].lstrip()
        im_end = "<|im_end|>"
        first_im_end = response.find(im_end)
        if first_im_end != -1:
            response = response[:first_im_end + len(im_end)]
        return response.strip()

    def _infer_batch_sync(
        self, worker: _TransformersWorker, samples: List[Dict], enable_thinking: bool,
    ) -> List[str]:
        prompts: List[str] = []
        batch_images: List[List[Image.Image]] = []
        for sample in samples:
            prompt, images = self._build_prompt_and_images(worker, sample, enable_thinking)
            prompts.append(prompt)
            batch_images.append(images)
        inputs = worker.processor(
            text=prompts,
            images=batch_images,
            return_tensors="pt",
            padding=True,
        ).to(worker.model.device)
        with torch.no_grad():
            outputs = worker.model.generate(
                **inputs,
                max_new_tokens=worker.max_new_tokens,
                temperature=max(worker.temperature, 1e-7),
                top_p=worker.top_p,
                do_sample=worker.temperature > 0,
                eos_token_id=worker.processor.tokenizer.convert_tokens_to_ids('<|im_end|>'),
            )
        input_width = inputs["input_ids"].shape[1]
        responses: List[str] = []
        for output in outputs:
            response_ids = output[input_width:]
            response = worker.processor.decode(response_ids, skip_special_tokens=False)
            responses.append(self._clean_response(response or ""))
        return responses

    async def infer_batch(
        self, samples: List[Dict], client_idx: int = 0, enable_thinking: bool = False,
    ) -> List[str]:
        worker = self.workers[client_idx % self.n_clients]
        async with worker._lock:
            return await asyncio.to_thread(self._infer_batch_sync, worker, samples, enable_thinking)

    async def infer(
        self, sample: Dict, client_idx: int = 0, enable_thinking: bool = False,
    ) -> str:
        results = await self.infer_batch([sample], client_idx=client_idx, enable_thinking=enable_thinking)
        return results[0]
