import os
import asyncio
import json
import aiohttp
import discord
import io
from discord import app_commands
from discord.ext import commands

# ---- Configuration ----
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")  # put in DO App env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # put in DO App env
ALLOWED_CHANNEL_NAME = "sora"  # only respond in #sora

# Sensible defaults per Sora 2 Cookbook:
# model: "sora-2" or "sora-2-pro"
# size:  "1280x720" or "720x1280" (and on Pro also 1024x1792, 1792x1024)
# seconds: "4", "8", or "12"
DEFAULT_SIZE = "1280x720"
DEFAULT_SECONDS_STD = "8"
DEFAULT_SECONDS_PRO = "12"

INTENTS = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=INTENTS)

# --- helpers ---

async def create_video_job(session: aiohttp.ClientSession, prompt: str, pro: bool,
                           size: str, seconds: str) -> str:
    """
    Create a Sora 2 generation job via OpenAI Videos API.
    Returns the job/video id to poll.
    """
    model = "sora-2-pro" if pro else "sora-2"

    # Endpoint per OpenAI docs (Videos API)
    url = "https://api.openai.com/v1/videos"

    payload = {
        "model": model,
        "prompt": prompt,
        "size": size,        # e.g., "1280x720"
        "seconds": seconds,  # "4" | "8" | "12"
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    async with session.post(url, headers=headers, data=json.dumps(payload)) as resp:
        if resp.status >= 400:
            text = await resp.text()
            raise RuntimeError(f"OpenAI create video error {resp.status}: {text}")
        data = await resp.json()

    # Typical shape: {"id": "...", "status": "queued", ...}
    job_id = data.get("id")
    if not job_id:
        raise RuntimeError(f"OpenAI did not return an id: {data}")
    return job_id


async def poll_video_until_ready(session: aiohttp.ClientSession, job_id: str,
                                 timeout_sec: int = 1200, poll_every: float = 5.0):
    """
    Poll the video job until it's ready or fails.
    Returns a dict with whatever final payload OpenAI returns.
    """
    url = f"https://api.openai.com/v1/videos/{job_id}"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    stop = asyncio.get_event_loop().time() + timeout_sec
    last_status = None

    while True:
        async with session.get(url, headers=headers) as resp:
            if resp.status >= 400:
                text = await resp.text()
                raise RuntimeError(f"OpenAI poll error {resp.status}: {text}")
            data = await resp.json()

        status = data.get("status")
        if status != last_status:
            last_status = status
            # You could add logging here

        if status in ("succeeded", "completed", "ready"):
            return data
        if status in ("failed", "error", "cancelled"):
            raise RuntimeError(f"Video job failed: {data}")

        if asyncio.get_event_loop().time() > stop:
            raise TimeoutError("Timed out waiting for video generation.")

        await asyncio.sleep(poll_every)


def extract_video_url(final_payload: dict) -> str | None:
    """
    Different API versions sometimes nest the output differently.
    Try a few likely places to find a playable URL.
    """
    # Common patterns:
    # 1) {"assets":{"video": "https://...mp4"}}
    assets = final_payload.get("assets") or {}
    if isinstance(assets, dict):
        vid = assets.get("video")
        if isinstance(vid, str):
            return vid

    # 2) {"url": "https://...mp4"}
    if isinstance(final_payload.get("url"), str):
        return final_payload["url"]

    # 3) {"generations": [{"url": "..."}]}
    gens = final_payload.get("generations")
    if isinstance(gens, list) and gens:
        first = gens[0]
        if isinstance(first, dict) and isinstance(first.get("url"), str):
            return first["url"]

    return None


async def generate_and_send(interaction: discord.Interaction, prompt: str, pro: bool):
    if interaction.channel is None or interaction.channel.name != ALLOWED_CHANNEL_NAME:
        await interaction.response.send_message(
            f"Please use this command in #{ALLOWED_CHANNEL_NAME}.", ephemeral=True
        )
        return

    await interaction.response.defer(thinking=True)

    size = DEFAULT_SIZE
    seconds = DEFAULT_SECONDS_PRO if pro else DEFAULT_SECONDS_STD

    async with aiohttp.ClientSession() as session:
        try:
            job_id = await create_video_job(session, prompt, pro, size, seconds)
        except Exception as e:
            # Fail fast if we couldn’t even create a job
            await interaction.followup.send(f"⚠️ Failed to start generation: `{e}`")
            return

    # Close out the interaction quickly (no long waits tied to webhook token)
    title = "Sora 2 Pro" if pro else "Sora 2"
    await interaction.followup.send(f"🎬 **{title}** job started for: `{prompt}`\n🆔 Job: `{job_id}`\nI'll post the video here when it's ready.")

    # Now poll in the background and post with channel.send()
    async def _track_and_post(channel: discord.abc.Messageable, title: str, prompt: str, job_id: str):
        async with aiohttp.ClientSession() as s:
            try:
                final = await poll_video_until_ready(s, job_id, timeout_sec=900, poll_every=3.0)

                # Try to extract a public URL first (fast path)
                video_url = extract_video_url(final)

                if video_url:
                    await channel.send(f"✅ **{title}** result for `{prompt}`\n{video_url}")
                    return

                # No URL provided → fetch the binary and upload to Discord
                data, ext = await fetch_video_bytes(s, job_id)

                # Discord default upload limit is ~25MB; wrap in BytesIO
                fp = io.BytesIO(data)
                fp.seek(0)
                filename = f"{job_id}{ext}"

                try:
                    await channel.send(
                        content=f"✅ **{title}** result for `{prompt}` (uploaded)",
                        file=discord.File(fp, filename=filename),
                    )
                except discord.HTTPException as e:
                    # If too large or other upload error, fall back to a message with size hint
                    await channel.send(
                        f"✅ Generated, but I couldn’t upload the file (likely too large). "
                        f"File size was ~{len(data) / 1_000_000:.1f} MB. "
                        f"Consider increasing Discord upload limits, or we can auto-upload to a storage bucket and link here."
                    )

            except TimeoutError:
                await channel.send(f"⏳ Still generating `{prompt}` (job `{job_id}`)… I’ll keep checking.")
            except Exception as e:
                await channel.send(f"⚠️ Error while generating `{prompt}`: `{e}`")

    # Fire and forget
    asyncio.create_task(_track_and_post(interaction.channel, title, prompt, job_id))


async def fetch_video_bytes(session: aiohttp.ClientSession, job_id: str) -> tuple[bytes, str]:
    """
    Try several likely content endpoints for a completed video job and return (data, ext).
    Raises RuntimeError if none work.
    """
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    # Try a few common patterns (first one that returns 200 wins)
    candidates = [
        f"https://api.openai.com/v1/videos/{job_id}/content",
        f"https://api.openai.com/v1/videos/{job_id}/content/video",
        # If your org has a generations-style split, uncomment this once you have a generation_id:
        # f"https://api.openai.com/v1/video/generations/{generation_id}/content/video",
    ]

    last_error = None
    for url in candidates:
        async with session.get(url, headers=headers) as resp:
            if resp.status == 200:
                data = await resp.read()
                # Try to infer extension; default to .mp4
                ctype = resp.headers.get("Content-Type", "").lower()
                ext = ".mp4" if "mp4" in ctype or not ctype else (".webm" if "webm" in ctype else ".mp4")
                return data, ext
            else:
                last_error = f"{resp.status} {await resp.text()}"
    raise RuntimeError(f"Unable to fetch video content. Last error: {last_error}")


# --- slash commands ---

@bot.tree.command(name="sora2", description="Generate a Sora 2 video from a prompt.")
@app_commands.describe(prompt="Describe the video you want.")
async def sora2(interaction: discord.Interaction, prompt: str):
    await generate_and_send(interaction, prompt=prompt, pro=False)

@bot.tree.command(name="sora2-pro", description="Generate a Sora 2 Pro video from a prompt.")
@app_commands.describe(prompt="Describe the video you want.")
async def sora2_pro(interaction: discord.Interaction, prompt: str):
    await generate_and_send(interaction, prompt=prompt, pro=True)

@bot.event
async def on_ready():
    try:
        await bot.tree.sync()
        print(f"Synced commands. Logged in as {bot.user} (ID: {bot.user.id})")
    except Exception as e:
        print(f"Slash command sync failed: {e}")

if __name__ == "__main__":
    if not DISCORD_TOKEN or not OPENAI_API_KEY:
        raise SystemExit("Please set DISCORD_TOKEN and OPENAI_API_KEY env vars.")
    bot.run(DISCORD_TOKEN)
