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


def llama_template(messages: Sequence[dict], add_generation_prompt: bool = True) -> str:
    """Llama-3 header format (also used by Llama-3.1-Nemotron-Nano).
    <|begin_of_text|> is NOT prepended here -- callers opt in via chat(bos=True).
    Nemotron-Nano's non-reasoning default is a "detailed thinking off" system
    message; pass it explicitly in `messages` when that mode is wanted.
    """
    parts = []
    for m in messages:
        role = m.get("role", "user").lower()
        parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{m.get('content','')}<|eot_id|>")
    if add_generation_prompt:
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


def qwen_template(messages: Sequence[dict], add_generation_prompt: bool = True) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "user").lower()
        parts.append(f"<|im_start|>{role}\n{m.get('content','')}<|im_end|>\n")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def mistral_template(messages: Sequence[dict], add_generation_prompt: bool = True) -> str:
    """Mistral-Instruct v0.2/v0.3 format: [INST] user [/INST] assistant</s>.
    <s> is NOT prepended here -- callers opt in via chat(bos=True). A leading
    system message is folded into the first user turn (Mistral has no system role
    token). add_generation_prompt is implicit: a trailing [INST] block with no
    closing assistant turn already cues the model to answer.
    """
    parts = []
    system = ""
    pending_user = None
    for m in messages:
        role = m.get("role", "user").lower()
        content = m.get("content", "")
        if role == "system":
            system = (system + "\n\n" + content).strip() if system else content
        elif role == "user":
            text = f"{system}\n\n{content}".strip() if system else content
            system = ""
            pending_user = text
            parts.append(f"[INST] {text} [/INST]")
        else:  # assistant
            pending_user = None
            parts.append(f"{content}</s>")
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
    "llama": llama_template,
    "nemotron": llama_template,
    "mistral": mistral_template,
}
