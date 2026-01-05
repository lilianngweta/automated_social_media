from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any, Literal

import httpx
from dotenv import load_dotenv


Format = Literal["markdown", "json"]


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str


@dataclass(frozen=True)
class MastodonConfig:
    base_url: str
    access_token: str


@dataclass(frozen=True)
class ReplicateConfig:
    api_token: str
    model_version: str | None = None
    model: str | None = None  # owner/name


@dataclass(frozen=True)
class TelegramConfig:
    bot_token: str
    chat_id: str


def load_brand_docs(docs_dir: Path) -> str:
    files = sorted(glob(str(docs_dir / "*.md")))
    if not files:
        raise FileNotFoundError(f"No .md files found in: {docs_dir}")

    parts: list[str] = []
    for fp in files:
        path = Path(fp)
        text = path.read_text(encoding="utf-8")
        parts.append(f"\n\n=== SOURCE: {path.name} ===\n{text.strip()}\n")

    joined = "\n".join(parts).strip()
    # Keep prompt size sane if docs grow later.
    if len(joined) > 80_000:
        joined = joined[:80_000] + "\n\n[TRUNCATED: docs exceeded 80k characters]"
    return joined


def build_reply_messages(
    *,
    brand_context: str,
    original_status: dict[str, Any],
    context_thread: list[dict[str, Any]],
    count: int,
) -> list[dict[str, str]]:
    """Build messages for generating replies to an existing Mastodon post."""
    system = (
        "You are a senior social media community manager for Maua Real Estate Agency. "
        "You must ONLY use the provided brand documents as your factual source. "
        "Write professional, helpful, and engaging replies that align with the brand voice. "
        "Keep replies conversational, relevant to the original post, and concise (under 400 characters). "
        "Do NOT invent contact details, addresses, prices, or specific offers not in the brand docs. "
        "Avoid being overly promotional—focus on being genuinely helpful and building community."
    )

    # Extract original post details
    author = original_status.get("account", {}).get("username", "unknown")
    content = original_status.get("content", "").strip()
    # Remove HTML tags for cleaner context
    import re
    clean_content = re.sub(r"<[^>]+>", "", content)

    # Build thread context if available
    thread_context = ""
    if context_thread:
        thread_context = "\n\nConversation thread (most recent first):\n"
        for idx, msg in enumerate(context_thread[:3], 1):  # Limit to 3 for brevity
            msg_author = msg.get("account", {}).get("username", "unknown")
            msg_content = re.sub(r"<[^>]+>", "", msg.get("content", "")).strip()
            thread_context += f"{idx}. @{msg_author}: {msg_content[:200]}\n"

    user = (
        f"Brand documents (source-of-truth):\n{brand_context}\n\n"
        f"Original post by @{author}:\n{clean_content}\n"
        f"{thread_context}\n\n"
        f"Task: Generate {count} possible reply {'comments' if count > 1 else 'comment'} to this post. "
        "Each reply should:\n"
        "- Be conversational and directly address the topic of the original post\n"
        "- Show genuine interest or add value (answer questions, share insights, express support)\n"
        "- Subtly reflect Maua Real Estate's expertise if relevant (without being salesy)\n"
        "- Use 1–3 hashtags maximum, and only if they naturally fit\n"
        "- Include '@' mentions only if directly replying to someone specific in the thread\n"
        "- Be authentic and human—avoid corporate jargon\n\n"
        "Output format: Return STRICT JSON only (no markdown, no commentary).\n"
        "Return a JSON array of objects with keys: reply_text, tone (e.g., 'helpful', 'supportive', 'curious')."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_messages(*, brand_context: str, platform: str, count: int) -> list[dict[str, str]]:
    system = (
        "You are a senior social media copywriter for Maua Real Estate Agency. "
        "You must ONLY use the provided brand documents as your factual source. "
        "If something is not in the documents (addresses, phone numbers, pricing, locations, offers), "
        "do not invent it—keep it generic or omit it. "
        "Write in a warm, knowledgeable, confident, professional voice."
    )

    platform_rules = ""
    if platform.strip().lower() == "mastodon":
        platform_rules = (
            "Platform-specific constraints (Mastodon):\n"
            "- Keep each caption concise and scannable; prefer 1–2 short paragraphs.\n"
            "- Aim for <= 450 characters for the caption text (excluding hashtags) to stay comfortably within common limits.\n"
            "- Use 3–6 hashtags (Mastodon performs better with fewer, more relevant tags).\n\n"
        )

    user = (
        f"Brand documents (source-of-truth):\n{brand_context}\n\n"
        f"Task: Generate {count} social media posts for platform: {platform}.\n"
        "Requirements:\n"
        "- Each post should be ready to publish and aligned with the brand voice and differentiators.\n"
        "- Vary the angles across posts (buying, selling, renting, investing, property management, commercial, tech-forward process, community focus).\n"
        "- Include a clear but non-pushy call-to-action (CTA) in each post (e.g., ‘DM us’, ‘Book a consult’).\n"
        "- Add 5–12 relevant hashtags per post (no location-specific hashtags unless explicitly in the docs), unless the platform rules below say otherwise.\n"
        "- Avoid claims that require proof (e.g., ‘#1’, ‘best’, ‘guaranteed returns’).\n\n"
        f"{platform_rules}"
        "Output format: Return STRICT JSON only (no markdown, no commentary).\n"
        "Return a JSON array of objects with keys: platform, hook, caption, hashtags (array of strings), cta.\n"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def chat_completions(*, cfg: LLMConfig, messages: list[dict[str, str]], temperature: float = 0.7) -> str:
    url = cfg.base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
        "temperature": temperature,
    }

    with httpx.Client(timeout=60) as client:
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    # OpenAI-compatible shape
    return data["choices"][0]["message"]["content"]


def telegram_send_message(*, cfg: TelegramConfig, text: str) -> dict[str, Any]:
    url = f"https://api.telegram.org/bot{cfg.bot_token}/sendMessage"
    payload = {
        "chat_id": cfg.chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }
    with httpx.Client(timeout=30) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()


def telegram_get_updates(*, cfg: TelegramConfig, offset: int | None, timeout_s: int = 25) -> list[dict[str, Any]]:
    url = f"https://api.telegram.org/bot{cfg.bot_token}/getUpdates"
    params: dict[str, Any] = {"timeout": timeout_s}
    if offset is not None:
        params["offset"] = offset
    with httpx.Client(timeout=timeout_s + 5) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
    if not data.get("ok"):
        return []
    return data.get("result", [])


def _telegram_update_chat_id(update: dict[str, Any]) -> str | None:
    msg = update.get("message") or update.get("edited_message")
    if not msg:
        return None
    chat = msg.get("chat") or {}
    cid = chat.get("id")
    return str(cid) if cid is not None else None


def _telegram_update_text(update: dict[str, Any]) -> str | None:
    msg = update.get("message") or update.get("edited_message")
    if not msg:
        return None
    text = msg.get("text")
    return str(text) if text is not None else None


def telegram_get_latest_update_id(cfg: TelegramConfig) -> int | None:
    updates = telegram_get_updates(cfg=cfg, offset=None, timeout_s=1)
    if not updates:
        return None
    return max(int(u.get("update_id")) for u in updates if u.get("update_id") is not None)


def telegram_wait_for_approval(
    *,
    cfg: TelegramConfig,
    approval_code: str,
    expected_chat_id: str,
    start_offset: int | None,
    timeout_s: int,
    poll_timeout_s: int = 25,
) -> tuple[bool, int | None]:
    """Wait for a Telegram message 'approve <code>' or 'reject <code>'.

    Returns: (approved, next_offset)
    """
    deadline = time.time() + timeout_s
    offset = start_offset

    while time.time() < deadline:
        updates = telegram_get_updates(cfg=cfg, offset=offset, timeout_s=poll_timeout_s)
        for u in updates:
            uid = u.get("update_id")
            if uid is not None:
                offset = int(uid) + 1

            chat_id = _telegram_update_chat_id(u)
            if chat_id is None or chat_id != expected_chat_id:
                continue

            text = _telegram_update_text(u)
            if not text:
                continue

            lowered = text.strip().lower()
            if lowered == f"approve {approval_code}" or lowered == f"yes {approval_code}":
                return True, offset
            if lowered == f"reject {approval_code}" or lowered == f"no {approval_code}":
                return False, offset

        # keep polling
    return False, offset


def normalize_hashtags(hashtags: Any) -> list[str]:
    if hashtags is None:
        return []
    if isinstance(hashtags, str):
        # allow "#a #b" or "a b"
        parts = [p.strip() for p in hashtags.replace("\n", " ").split(" ") if p.strip()]
        hashtags_list = parts
    elif isinstance(hashtags, list):
        hashtags_list = [str(h).strip() for h in hashtags if str(h).strip()]
    else:
        hashtags_list = [str(hashtags).strip()]

    normalized: list[str] = []
    for tag in hashtags_list:
        if not tag:
            continue
        normalized.append(tag if tag.startswith("#") else f"#{tag}")
    # de-dupe while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for t in normalized:
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def build_mastodon_status(post: dict[str, Any], *, max_chars: int = 500) -> str:
    hook = (post.get("hook") or "").strip()
    caption = (post.get("caption") or "").strip()
    cta = (post.get("cta") or "").strip()
    hashtags = normalize_hashtags(post.get("hashtags"))
    hashtag_line = " ".join(hashtags).strip()

    blocks: list[str] = []
    if hook:
        blocks.append(hook)
    if caption:
        blocks.append(caption)
    if cta:
        blocks.append(cta)
    if hashtag_line:
        blocks.append(hashtag_line)

    status = "\n\n".join(blocks).strip()

    if len(status) <= max_chars:
        return status

    # If too long, preserve hashtags and CTA as much as possible.
    suffix_parts: list[str] = []
    if cta:
        suffix_parts.append(cta)
    if hashtag_line:
        suffix_parts.append(hashtag_line)
    suffix = "\n\n" + "\n\n".join(suffix_parts) if suffix_parts else ""

    # Reserve room for suffix + ellipsis.
    reserve = len(suffix)
    available = max_chars - reserve
    if available <= 10:
        # Worst case: drop suffix first
        status = status[: max_chars - 1].rstrip() + "…"
        return status

    main_parts: list[str] = []
    if hook:
        main_parts.append(hook)
    if caption:
        main_parts.append(caption)
    main = "\n\n".join(main_parts).strip()
    if len(main) > available:
        main = main[: available - 1].rstrip() + "…"
    return (main + suffix).strip()


def mastodon_get_status(
    *,
    cfg: MastodonConfig,
    status_id: str,
) -> dict[str, Any]:
    """Fetch a single status by ID."""
    url = cfg.base_url.rstrip("/") + f"/api/v1/statuses/{status_id}"
    headers = {"Authorization": f"Bearer {cfg.access_token}"}

    with httpx.Client(timeout=60) as client:
        resp = client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.json()


def mastodon_get_status_context(
    *,
    cfg: MastodonConfig,
    status_id: str,
) -> dict[str, Any]:
    """Fetch the conversation context (ancestors/descendants) of a status."""
    url = cfg.base_url.rstrip("/") + f"/api/v1/statuses/{status_id}/context"
    headers = {"Authorization": f"Bearer {cfg.access_token}"}

    with httpx.Client(timeout=60) as client:
        resp = client.get(url, headers=headers)
        resp.raise_for_status()
        return resp.json()


def mastodon_upload_media(
    *,
    cfg: MastodonConfig,
    file_path: Path,
    description: str | None = None,
) -> dict[str, Any]:
    """Upload a media file to Mastodon and return the media ID."""
    url = cfg.base_url.rstrip("/") + "/api/v2/media"
    headers = {"Authorization": f"Bearer {cfg.access_token}"}

    with open(file_path, "rb") as f:
        files = {"file": (file_path.name, f, "image/png")}
        data = {}
        if description:
            data["description"] = description

        with httpx.Client(timeout=120) as client:
            resp = client.post(url, headers=headers, files=files, data=data)
            resp.raise_for_status()
            return resp.json()


def mastodon_post_status(
    *,
    cfg: MastodonConfig,
    status: str,
    visibility: str = "public",
    language: str | None = None,
    in_reply_to_id: str | None = None,
    media_ids: list[str] | None = None,
) -> dict[str, Any]:
    url = cfg.base_url.rstrip("/") + "/api/v1/statuses"
    headers = {
        "Authorization": f"Bearer {cfg.access_token}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "status": status,
        "visibility": visibility,
    }
    if language:
        payload["language"] = language
    if in_reply_to_id:
        payload["in_reply_to_id"] = in_reply_to_id
    if media_ids:
        payload["media_ids"] = media_ids

    with httpx.Client(timeout=60) as client:
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()


def build_image_prompt(post: dict[str, Any]) -> str:
    """Build a safe, brand-aligned image prompt without inventing facts."""
    hook = (post.get("hook") or "").strip()
    caption = (post.get("caption") or "").strip()
    # Keep prompt concise; Flux typically does well with clear art direction.
    core = ". ".join([p for p in [hook, caption] if p])
    if len(core) > 600:
        core = core[:600].rsplit(" ", 1)[0] + "…"

    return (
        "Modern, premium real-estate social media visual for a brand named 'Maua Real Estate Agency'. "
        "Clean composition, professional marketing style, warm and trustworthy tone. "
        "No phone numbers, no addresses, no logos, no watermarks, no text overlays. "
        f"Concept inspiration: {core}"
    ).strip()


def replicate_create_prediction(
    *,
    cfg: ReplicateConfig,
    prompt: str,
    aspect_ratio: str,
    output_format: str,
    extra_input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Token {cfg.api_token}",
        "Content-Type": "application/json",
    }

    # Common inputs used by many Flux variants.
    model_input: dict[str, Any] = {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "output_format": output_format,
    }
    if extra_input:
        model_input.update(extra_input)

    if cfg.model_version:
        url = "https://api.replicate.com/v1/predictions"
        payload: dict[str, Any] = {"version": cfg.model_version, "input": model_input}
    elif cfg.model:
        owner, name = cfg.model.split("/", 1)
        url = f"https://api.replicate.com/v1/models/{owner}/{name}/predictions"
        payload = {"input": model_input}
    else:
        raise ValueError("Replicate config requires model_version or model")

    with httpx.Client(timeout=60) as client:
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()


def replicate_wait_for_output(
    *,
    cfg: ReplicateConfig,
    prediction_id: str,
    timeout_s: int = 300,
    poll_s: float = 1.0,
) -> dict[str, Any]:
    headers = {"Authorization": f"Token {cfg.api_token}"}
    url = f"https://api.replicate.com/v1/predictions/{prediction_id}"
    deadline = time.time() + timeout_s

    with httpx.Client(timeout=60) as client:
        while True:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status")

            if status in {"succeeded", "failed", "canceled"}:
                return data

            if time.time() >= deadline:
                raise TimeoutError(f"Replicate prediction timed out after {timeout_s}s: {prediction_id}")

            time.sleep(poll_s)


def download_url_to_file(*, url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        resp = client.get(url)
        resp.raise_for_status()
        out_path.write_bytes(resp.content)


def ensure_suffix(path: Path, suffix: str) -> Path:
    if path.suffix.lower() == suffix.lower():
        return path
    return path.with_suffix(suffix)


def render_markdown(posts: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for idx, post in enumerate(posts, start=1):
        hashtags = post.get("hashtags", [])
        if isinstance(hashtags, list):
            hashtag_line = " ".join(str(h) for h in hashtags)
        else:
            hashtag_line = str(hashtags)

        lines.append(f"## Post {idx} ({post.get('platform','')})")
        if post.get("hook"):
            lines.append(f"**Hook:** {post['hook']}")
        if post.get("caption"):
            lines.append(post["caption"])
        if post.get("image_path"):
            lines.append("")
            lines.append(f"![]({post['image_path']})")
        if hashtag_line.strip():
            lines.append("")
            lines.append(hashtag_line)
        if post.get("cta"):
            lines.append("")
            lines.append(f"**CTA:** {post['cta']}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def parse_posts(raw: str) -> list[dict[str, Any]]:
    raw = raw.strip()
    # Some models wrap JSON in ```json ... ```; try to recover.
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json\n", "", 1).strip()

    return json.loads(raw)


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Generate social media posts from docs/*.md")
    parser.add_argument("--docs-dir", default="docs", help="Directory containing brand .md docs")
    parser.add_argument("--platform", default=os.getenv("DEFAULT_PLATFORM", "instagram"))
    parser.add_argument("--count", type=int, default=int(os.getenv("DEFAULT_COUNT", "10")))
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--out", default="generated_posts.md", help="Output file path")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--temperature", type=float, default=0.7)

    # Images (Replicate)
    parser.add_argument(
        "--images",
        action="store_true",
        help="Generate one image per post using a Replicate-hosted Flux model",
    )
    parser.add_argument("--image-dir", default="generated_images", help="Directory to write generated images")
    parser.add_argument(
        "--replicate-model-version",
        default=os.getenv("REPLICATE_MODEL_VERSION", ""),
        help="Replicate model version hash (recommended)",
    )
    parser.add_argument(
        "--replicate-model",
        default=os.getenv("REPLICATE_MODEL", ""),
        help="Replicate model in the form owner/name (alternative to version)",
    )
    parser.add_argument(
        "--replicate-input-json",
        default="",
        help="Extra Replicate model input as JSON string (merged into input)",
    )
    parser.add_argument(
        "--image-aspect-ratio",
        default="1:1",
        help="Aspect ratio sent to the model (common values: 1:1, 4:5, 16:9)",
    )
    parser.add_argument(
        "--image-format",
        default="png",
        choices=["png", "jpg", "jpeg", "webp"],
        help="Output format hint for the model; file saved with this extension",
    )
    parser.add_argument(
        "--replicate-timeout-s",
        type=int,
        default=300,
        help="Timeout per image prediction in seconds",
    )

    # Publishing
    parser.add_argument(
        "--publish",
        action="store_true",
        help="If set, publish generated posts to Mastodon (requires MASTODON_* env vars)",
    )
    parser.add_argument("--mastodon-base-url", default=os.getenv("MASTODON_BASE_URL", ""))
    parser.add_argument("--mastodon-access-token", default=os.getenv("MASTODON_ACCESS_TOKEN", ""))
    parser.add_argument(
        "--visibility",
        default=os.getenv("MASTODON_VISIBILITY", "public"),
        choices=["public", "unlisted", "private", "direct"],
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=500,
        help="Max characters per Mastodon status (default 500)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --publish, do not actually post; just print what would be posted",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional BCP-47 language tag for Mastodon (e.g., en, en-GB)",
    )

    # Telegram approval gate
    parser.add_argument(
        "--require-telegram-approval",
        action="store_true",
        help="If set with --publish, sends each draft to Telegram and waits for approval before posting",
    )
    parser.add_argument(
        "--telegram-bot-token",
        default=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        help="Telegram bot token (or set TELEGRAM_BOT_TOKEN)",
    )
    parser.add_argument(
        "--telegram-chat-id",
        default=os.getenv("TELEGRAM_CHAT_ID", ""),
        help="Telegram chat id to send approvals to (or set TELEGRAM_CHAT_ID)",
    )
    parser.add_argument(
        "--approval-timeout-s",
        type=int,
        default=int(os.getenv("TELEGRAM_APPROVAL_TIMEOUT_S", "900")),
        help="How long to wait for Telegram approval per post (seconds)",
    )

    # Reply/comment mode
    parser.add_argument(
        "--reply-to",
        default="",
        help="Mastodon status ID to reply to (generates contextual replies instead of new posts)",
    )
    parser.add_argument(
        "--reply-count",
        type=int,
        default=1,
        help="Number of reply variations to generate (default: 1)",
    )

    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing OPENAI_API_KEY. Create a .env file (see .env.example) or export OPENAI_API_KEY."
        )

    cfg = LLMConfig(api_key=api_key, base_url=args.base_url, model=args.model)

    telegram_cfg: TelegramConfig | None = None
    telegram_offset: int | None = None
    if args.require_telegram_approval:
        if not args.telegram_bot_token or not args.telegram_chat_id:
            raise SystemExit(
                "Missing Telegram config. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (or pass --telegram-bot-token/--telegram-chat-id)."
            )
        if not str(args.telegram_chat_id).lstrip("-").isdigit():
            raise SystemExit(
                "TELEGRAM_CHAT_ID must be a numeric chat id (e.g., 123456789). See README for how to fetch it from getUpdates()."
            )
        telegram_cfg = TelegramConfig(bot_token=args.telegram_bot_token, chat_id=args.telegram_chat_id)
        latest = telegram_get_latest_update_id(telegram_cfg)
        telegram_offset = (latest + 1) if latest is not None else None

    docs_dir = Path(args.docs_dir)
    brand_context = load_brand_docs(docs_dir)

    # Reply mode
    if args.reply_to:
        if args.platform.strip().lower() != "mastodon":
            raise SystemExit("--reply-to is only supported when --platform mastodon")
        if not args.mastodon_base_url or not args.mastodon_access_token:
            raise SystemExit(
                "Missing Mastodon config for reply mode. Set MASTODON_BASE_URL and MASTODON_ACCESS_TOKEN."
            )

        mcfg = MastodonConfig(base_url=args.mastodon_base_url, access_token=args.mastodon_access_token)

        # Fetch the original status
        original_status = mastodon_get_status(cfg=mcfg, status_id=args.reply_to)
        print(f"Replying to status {args.reply_to} by @{original_status.get('account', {}).get('username', 'unknown')}")

        # Fetch conversation context
        context_data = mastodon_get_status_context(cfg=mcfg, status_id=args.reply_to)
        # Ancestors are older, descendants are newer replies
        thread = context_data.get("ancestors", []) + [original_status] + context_data.get("descendants", [])

        reply_count = args.reply_count if args.reply_count > 0 else 1
        messages = build_reply_messages(
            brand_context=brand_context,
            original_status=original_status,
            context_thread=context_data.get("descendants", []),
            count=reply_count,
        )

        raw = chat_completions(cfg=cfg, messages=messages, temperature=args.temperature)
        replies = parse_posts(raw)  # Expect JSON array

        # Generate images for replies if requested
        if args.images:
            replicate_token = os.getenv("REPLICATE_API_TOKEN")
            if not replicate_token:
                raise SystemExit(
                    "Missing REPLICATE_API_TOKEN. Set it in .env (see .env.example) or export it."
                )

            model_version = args.replicate_model_version.strip() or None
            model = args.replicate_model.strip() or None
            if not model_version and not model:
                raise SystemExit(
                    "Missing Replicate model. Set REPLICATE_MODEL_VERSION (recommended) or REPLICATE_MODEL."
                )

            extra_input: dict[str, Any] | None = None
            if args.replicate_input_json.strip():
                try:
                    parsed = json.loads(args.replicate_input_json)
                except json.JSONDecodeError as e:
                    raise SystemExit(f"Invalid --replicate-input-json: {e}")
                if not isinstance(parsed, dict):
                    raise SystemExit("--replicate-input-json must be a JSON object")
                extra_input = parsed

            rcfg = ReplicateConfig(api_token=replicate_token, model_version=model_version, model=model)
            image_dir = Path(args.image_dir)

            for idx, reply_data in enumerate(replies, start=1):
                # Build simpler prompt for reply images
                prompt = (
                    "Modern, premium real-estate social media visual for Maua Real Estate Agency. "
                    "Clean composition, professional marketing style, warm and trustworthy tone. "
                    "No text overlays, no watermarks."
                )
                
                prediction = replicate_create_prediction(
                    cfg=rcfg,
                    prompt=prompt,
                    aspect_ratio=args.image_aspect_ratio,
                    output_format=args.image_format,
                    extra_input=extra_input,
                )
                prediction_id = prediction.get("id")
                if not prediction_id:
                    raise RuntimeError("Replicate response missing prediction id")

                print(f"Generating image for reply {idx}...")
                final = replicate_wait_for_output(
                    cfg=rcfg,
                    prediction_id=prediction_id,
                    timeout_s=args.replicate_timeout_s,
                )
                if final.get("status") != "succeeded":
                    raise RuntimeError(f"Replicate prediction failed: {final.get('error') or final}")

                output = final.get("output")
                if isinstance(output, list) and output:
                    url = str(output[0])
                else:
                    url = str(output)

                ext = ".jpg" if args.image_format == "jpeg" else f".{args.image_format}"
                out_path = ensure_suffix(image_dir / f"reply_{idx:02d}", ext)
                download_url_to_file(url=url, out_path=out_path)
                reply_data["image_path"] = str(out_path)
                print(f"Image saved: {out_path}")

        # Post replies
        for idx, reply_data in enumerate(replies, start=1):
            reply_text = reply_data.get("reply_text", "").strip()
            if not reply_text:
                print(f"Skipping empty reply {idx}")
                continue

            # Telegram approval gate
            if args.publish and args.require_telegram_approval and not args.dry_run:
                code = uuid.uuid4().hex[:8]
                image_hint = f"\nImage: {reply_data.get('image_path')}" if reply_data.get("image_path") else ""
                preview = (
                    f"Mastodon reply approval required\n"
                    f"Reply target: {args.reply_to}\n"
                    f"Code: {code}\n\n"
                    f"Draft reply:\n{reply_text}\n"
                    f"{image_hint}\n\n"
                    f"Send: approve {code}  OR  reject {code}"
                )
                # Telegram message limit is ~4096 chars; keep it safe.
                if len(preview) > 3500:
                    preview = preview[:3500] + "\n…(truncated)\n" + f"approve {code} / reject {code}"
                assert telegram_cfg is not None
                telegram_send_message(cfg=telegram_cfg, text=preview)
                approved, telegram_offset = telegram_wait_for_approval(
                    cfg=telegram_cfg,
                    approval_code=code,
                    expected_chat_id=str(args.telegram_chat_id),
                    start_offset=telegram_offset,
                    timeout_s=args.approval_timeout_s,
                )
                if not approved:
                    print(f"Reply {idx} not approved; skipping.")
                    continue

            # Upload media if image exists
            media_ids: list[str] = []
            if reply_data.get("image_path"):
                image_path = Path(reply_data["image_path"])
                if image_path.exists():
                    if not args.dry_run:
                        print(f"Uploading image: {image_path}")
                        media_resp = mastodon_upload_media(
                            cfg=mcfg,
                            file_path=image_path,
                            description=reply_text[:200],
                        )
                        media_ids.append(media_resp["id"])
                        print(f"Image uploaded: {media_resp['id']}")
                    else:
                        print(f"Would upload image: {image_path}")

            if args.dry_run:
                print(f"\n--- DRY RUN: would post reply {idx} ---")
                print(reply_text)
                print(f"Tone: {reply_data.get('tone', 'N/A')}")
                if media_ids or reply_data.get("image_path"):
                    print(f"With image: {reply_data.get('image_path')}")
                continue

            resp = mastodon_post_status(
                cfg=mcfg,
                status=reply_text,
                visibility=args.visibility,
                language=args.language,
                in_reply_to_id=args.reply_to,
                media_ids=media_ids if media_ids else None,
            )
            print(f"Posted reply {idx}/{len(replies)}: {resp.get('url') or resp.get('id')}")

        print(f"\nReplied to status {args.reply_to} with {len(replies)} comment(s).")
        return

    # Standard post generation mode
    messages = build_messages(brand_context=brand_context, platform=args.platform, count=args.count)

    raw = chat_completions(cfg=cfg, messages=messages, temperature=args.temperature)

    posts = parse_posts(raw)

    if args.images:
        replicate_token = os.getenv("REPLICATE_API_TOKEN")
        if not replicate_token:
            raise SystemExit(
                "Missing REPLICATE_API_TOKEN. Set it in .env (see .env.example) or export it."
            )

        model_version = args.replicate_model_version.strip() or None
        model = args.replicate_model.strip() or None
        if not model_version and not model:
            raise SystemExit(
                "Missing Replicate model. Set REPLICATE_MODEL_VERSION (recommended) or REPLICATE_MODEL."
            )

        extra_input: dict[str, Any] | None = None
        if args.replicate_input_json.strip():
            try:
                parsed = json.loads(args.replicate_input_json)
            except json.JSONDecodeError as e:
                raise SystemExit(f"Invalid --replicate-input-json: {e}")
            if not isinstance(parsed, dict):
                raise SystemExit("--replicate-input-json must be a JSON object")
            extra_input = parsed

        rcfg = ReplicateConfig(api_token=replicate_token, model_version=model_version, model=model)
        image_dir = Path(args.image_dir)

        for idx, post in enumerate(posts, start=1):
            prompt = build_image_prompt(post)
            prediction = replicate_create_prediction(
                cfg=rcfg,
                prompt=prompt,
                aspect_ratio=args.image_aspect_ratio,
                output_format=args.image_format,
                extra_input=extra_input,
            )
            prediction_id = prediction.get("id")
            if not prediction_id:
                raise RuntimeError("Replicate response missing prediction id")

            final = replicate_wait_for_output(
                cfg=rcfg,
                prediction_id=prediction_id,
                timeout_s=args.replicate_timeout_s,
            )
            if final.get("status") != "succeeded":
                raise RuntimeError(f"Replicate prediction failed: {final.get('error') or final}")

            output = final.get("output")
            # Many models return a list of URLs; handle both list and single string.
            if isinstance(output, list) and output:
                url = str(output[0])
            else:
                url = str(output)

            ext = ".jpg" if args.image_format == "jpeg" else f".{args.image_format}"
            out_path = ensure_suffix(image_dir / f"post_{idx:02d}", ext)
            download_url_to_file(url=url, out_path=out_path)
            post["image_path"] = str(out_path)
    out_path = Path(args.out)

    if args.format == "json":
        out_path.write_text(json.dumps(posts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        out_path.write_text(render_markdown(posts), encoding="utf-8")

    print(f"Wrote {len(posts)} posts to: {out_path}")

    if args.publish:
        if args.platform.strip().lower() != "mastodon":
            raise SystemExit("--publish is only supported when --platform mastodon")
        if not args.mastodon_base_url or not args.mastodon_access_token:
            raise SystemExit(
                "Missing Mastodon config. Set MASTODON_BASE_URL and MASTODON_ACCESS_TOKEN (see .env.example)."
            )

        mcfg = MastodonConfig(base_url=args.mastodon_base_url, access_token=args.mastodon_access_token)
        posted = 0
        for post in posts:
            status = build_mastodon_status(post, max_chars=args.max_chars)

            # Telegram approval gate
            if args.require_telegram_approval and not args.dry_run:
                code = uuid.uuid4().hex[:8]
                image_hint = f"\nImage: {post.get('image_path')}" if post.get("image_path") else ""
                preview = (
                    f"Mastodon post approval required\n"
                    f"Code: {code}\n\n"
                    f"Draft:\n{status}\n"
                    f"{image_hint}\n\n"
                    f"Send: approve {code}  OR  reject {code}"
                )
                if len(preview) > 3500:
                    preview = preview[:3500] + "\n…(truncated)\n" + f"approve {code} / reject {code}"
                assert telegram_cfg is not None
                telegram_send_message(cfg=telegram_cfg, text=preview)
                approved, telegram_offset = telegram_wait_for_approval(
                    cfg=telegram_cfg,
                    approval_code=code,
                    expected_chat_id=str(args.telegram_chat_id),
                    start_offset=telegram_offset,
                    timeout_s=args.approval_timeout_s,
                )
                if not approved:
                    print("Post not approved; skipping.")
                    continue
            
            # Upload media if image exists
            media_ids: list[str] = []
            if post.get("image_path"):
                image_path = Path(post["image_path"])
                if image_path.exists():
                    if not args.dry_run:
                        print(f"Uploading image: {image_path}")
                        media_resp = mastodon_upload_media(
                            cfg=mcfg,
                            file_path=image_path,
                            description=post.get("hook") or post.get("caption", "")[:200],
                        )
                        media_ids.append(media_resp["id"])
                        print(f"Image uploaded: {media_resp['id']}")
                    else:
                        print(f"Would upload image: {image_path}")
            
            if args.dry_run:
                print("\n--- DRY RUN: would post ---\n")
                print(status)
                if media_ids or post.get("image_path"):
                    print(f"With image: {post.get('image_path')}")
                continue

            resp = mastodon_post_status(
                cfg=mcfg,
                status=status,
                visibility=args.visibility,
                language=args.language,
                media_ids=media_ids if media_ids else None,
            )
            posted += 1
            print(f"Posted {posted}/{len(posts)}: {resp.get('url') or resp.get('id')}")


if __name__ == "__main__":
    main()
