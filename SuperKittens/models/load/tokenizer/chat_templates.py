from __future__ import annotations
from typing import Sequence


def deepseek_template(messages: Sequence[dict], add_generation_prompt: bool = True) -> str:
    role_tag = {
        "user": "<｜user｜>",
        "assistant": "<｜assistant｜>",
        "system": "<｜system｜>",
    }
    parts = []
    for m in messages:
        role = m.get("role", "user").lower()
        tag = role_tag.get(role, role_tag["user"])
        parts.append(f"{tag}<｜start_of_turn｜>{m.get('content','')}<｜end_of_turn｜>")
    if add_generation_prompt:
        parts.append(f"{role_tag['assistant']}<｜start_of_turn｜>")
    return "".join(parts)


def qwen_template(messages: Sequence[dict], add_generation_prompt: bool = True) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "user").lower()
        parts.append(f"<|im_start|>{role}\n{m.get('content','')}<|im_end|>\n")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def gemma_template(messages: Sequence[dict], add_generation_prompt: bool = True) -> str:
    role_map = {"user": "user", "assistant": "model", "model": "model", "system": "user"}
    parts = []
    for m in messages:
        role = role_map.get(m.get("role", "user").lower(), "user")
        parts.append(f"<start_of_turn>{role}\n{m.get('content','')}<end_of_turn>\n")
    if add_generation_prompt:
        parts.append("<start_of_turn>model\n")
    return "".join(parts)


def gemma4_template(messages: Sequence[dict], add_generation_prompt: bool = True) -> str:
    """Gemma 4 chat format. Distinct from Gemma 1-3: uses <|turn>role\\n ... <turn|>\\n
    markers (tokens 105 / 106) rather than <start_of_turn>/<end_of_turn>.
    Source: model_weights/gemma-4-E2B-it/chat_template.jinja.
    NOTE: <bos> is NOT prepended here -- callers must opt in via Tokenizer.chat(bos=True).
    """
    role_map = {"user": "user", "assistant": "model", "model": "model", "system": "system"}
    parts = []
    for m in messages:
        role = role_map.get(m.get("role", "user").lower(), "user")
        content = m.get("content", "")
        if isinstance(content, str):
            content = content.strip()
        parts.append(f"<|turn>{role}\n{content}<turn|>\n")
    if add_generation_prompt:
        parts.append("<|turn>model\n")
    return "".join(parts)


CHAT_TEMPLATES = {
    "deepseek": deepseek_template,
    "ds4": deepseek_template,
    "qwen": qwen_template,
    "qwen3": qwen_template,
    "chatml": qwen_template,
    "gemma": gemma_template,
    "gemma4": gemma4_template,
}
