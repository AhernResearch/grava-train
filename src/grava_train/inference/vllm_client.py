"""Async multimodal chat inference over OpenAI-compatible HTTP endpoints."""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, List

import openai


class VLLMInferenceClient:
    """Send ordinary image files and user prompts to one or more endpoints."""

    def __init__(
        self,
        urls: List[str],
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        self.clients = [
            openai.AsyncOpenAI(base_url=url, api_key="EMPTY")
            for url in urls
        ]
        self.model = model
        self._gen_kwargs: Dict[str, Any] = {
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if top_p < 1.0:
            self._gen_kwargs["top_p"] = top_p

    @property
    def n_clients(self) -> int:
        return len(self.clients)

    @staticmethod
    def build_payload(sample: Dict) -> List[Dict]:
        """Build OpenAI chat messages (chat mode)."""
        content = sample["messages"][0]["content"]
        text = content.replace("<image>", "").strip()

        content_parts: List[Dict] = []
        for img_path in sample["images"]:
            raw = Path(img_path).read_bytes()
            b64 = base64.b64encode(raw).decode("utf-8")
            suffix = Path(img_path).suffix.lstrip(".") or "png"
            mime = f"image/{suffix}" if suffix != "jpg" else "image/jpeg"
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })
        content_parts.append({"type": "text", "text": text})
        return [{"role": "user", "content": content_parts}]

    async def infer(
        self,
        sample: Dict,
        client_idx: int = 0,
        enable_thinking: bool = False,
    ) -> str:
        """Run single-sample inference. Returns prediction text."""
        client = self.clients[client_idx % self.n_clients]

        response = await client.chat.completions.create(
            model=self.model,
            messages=self.build_payload(sample),
            extra_body={
                "skip_special_tokens": False,
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
            },
            **self._gen_kwargs,
        )
        return response.choices[0].message.content or ""
