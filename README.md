# Social Media Post Generator

Generates social media posts using the Maua Real Estate Agency brand documents in `docs/`.

## Setup

1. Create a virtualenv (optional but recommended):
   - `python -m venv .venv`
   - `source .venv/bin/activate`
2. Install requirements:
   - `pip install -r requirements.txt`
3. Configure env:
   - Copy `.env.example` to `.env` using command `cp .env.example .env` and set `OPENAI_API_KEY` and all the other required variables in `.env`.
     

## Generate posts and publish to Mastodon

1. Create an access token on your Mastodon instance:
   - Preferences → Development → New application → copy the access token
2. Set env vars in `.env`:
   - `MASTODON_BASE_URL` (e.g. `https://mastodon.social`)
   - `MASTODON_ACCESS_TOKEN`
3. Generate + publish:
   - `python generate_social_posts.py --platform mastodon --count 1 --publish --visibility public --out posts.md`


## Reply to Mastodon posts

Reply to other users' posts with contextual, brand-aligned comments:

1. Find a post to reply to (get the status ID from the URL, e.g., `https://mastodon.social/@lilianngweta/115841460058086337`)
2. Generate and post a reply:
   - `python generate_social_posts.py --reply-to 115841460058086337 --visibility public`
3. Generate multiple reply variations (dry-run):
   - `python generate_social_posts.py --reply-to 115841460058086337 --reply-count 1 --dry-run`

The reply generator:
- Fetches the original post and conversation context
- Uses the LLM to craft relevant, helpful replies aligned with Maua Real Estate's brand voice
- Posts as threaded replies maintaining proper conversation structure

## Generate images (Replicate Flux fine-tune)

This project can optionally generate an image per post using a fine-tuned Flux model hosted on Replicate.

1. Create a Replicate API token:
   - https://replicate.com/account/api-tokens
2. Set env vars in `.env` (see `.env.example`):
   - `REPLICATE_API_TOKEN`
   - Either `REPLICATE_MODEL_VERSION` (recommended) OR `REPLICATE_MODEL` (owner/name)
3. Generate posts + images:
   - `python generate_social_posts.py --platform mastodon --count 1 --images --image-dir generated_images --out posts.md`

### Post to Mastodon with images

Generate posts, create images, and publish to Mastodon with attached media:
```bash
python generate_social_posts.py --platform mastodon --count 1 \
  --images --image-aspect-ratio 1:1 --image-format png \
  --publish --visibility public
```

Reply to a post with an attached image:
```bash
python generate_social_posts.py --reply-to 115841460058086337 \
  --images --visibility public
```


## Telegram approval (recommended)

If you want a human-in-the-loop gate before anything posts to Mastodon, enable Telegram approval.

1. Create a Telegram bot via @BotFather and copy the bot token.
2. Get your chat id:
3. Set in `.env`:
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
4. Run with approval gate:
   - `python generate_social_posts.py --platform mastodon --count 1 --images --publish --require-telegram-approval`

You’ll receive a Telegram message for each draft. Reply with `approve <code>` or `reject <code>`.


## Notes

- The script reads all `docs/*.md` and provides them to the LLM as the source-of-truth.
- If you use a non-OpenAI provider, set `OPENAI_BASE_URL` to an OpenAI-compatible endpoint.
- For Replicate Flux model, the script sends common inputs (`prompt`, `aspect_ratio`, `output_format`) and you can pass additional inputs via `--replicate-input-json`.
- Images are uploaded to Mastodon before posting (using `/api/v2/media` endpoint)

  This project was done at [Sundai Club](https://research.sundai.club)
