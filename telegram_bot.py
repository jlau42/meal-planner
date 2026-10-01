"""Kenji as a Telegram bot.

Run:  uv run telegram_bot.py

Environment:
  ANTHROPIC_API_KEY         Claude API key (same as the CLI)
  TELEGRAM_BOT_TOKEN        token from @BotFather (required)
  KENJI_ADMIN_TELEGRAM_IDS  optional: comma-separated Telegram user ids allowed to claim
                            existing profiles that no one has linked yet (e.g. "jon")

Each Telegram user links to one profile with /link. Kenji answers every message in a
private chat; in groups it answers only when @mentioned or replied to. Conversations
are kept per user per chat, in memory; /newchat starts over. Plans Kenji saves are sent
as .md and .pdf documents.
"""

import asyncio
import html
import logging
import os
import re
import secrets
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anthropic
from telegram import BotCommand, Message, Update
from telegram.constants import ChatAction, ChatType, MessageLimit, ParseMode
from telegram.error import BadRequest
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

import config  # noqa: F401  loads .env before anything reads env vars
import tools
from agent import AGENT_NAME, run_turn
from md_to_pdf import convert as md_to_pdf

log = logging.getLogger("kenji")

CHUNK_LIMIT = MessageLimit.MAX_TEXT_LENGTH - 500  # headroom for HTML tags added by formatting
IDLE_RESET_SECONDS = 6 * 3600       # a conversation idle this long starts fresh
INVITE_TTL_SECONDS = 15 * 60
MAX_ATTACHMENT_BYTES = 100_000
TEXT_SUFFIXES = {".txt", ".md", ".json", ".csv"}
TYPING_REFRESH_SECONDS = 4          # Telegram's typing indicator lasts ~5 seconds


def _id_set(var: str) -> set[int]:
    return {int(x) for x in os.environ.get(var, "").replace(" ", "").split(",") if x}


# ---------------------------------------------------------------------------
# Pure helpers (no Telegram objects) - easy to test
# ---------------------------------------------------------------------------

def chunk_text(text: str, limit: int = CHUNK_LIMIT) -> List[str]:
    """Split text into Telegram-sized messages, preferring paragraph, then line, then word breaks."""
    chunks = []
    text = text.strip()
    while len(text) > limit:
        cut = text.rfind("\n\n", 0, limit)
        if cut < limit // 2:
            cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = text.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        chunks.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        chunks.append(text)
    return chunks


def markdown_to_telegram_html(text: str) -> str:
    """Convert the Markdown Claude writes into Telegram's small HTML subset.

    Handles code spans, bold, italics, links and headings; everything else is escaped."""
    text = html.escape(text, quote=False)
    text = re.sub(r"```(?:\w+)?\n?(.*?)```", lambda m: f"<pre>{m.group(1)}</pre>", text, flags=re.S)
    text = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\[([^\]\n]+)\]\((https?://[^)\s]+)\)", r'<a href="\2">\1</a>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)([^*\n]+?)\*(?![\w*])", r"<i>\1</i>", text)
    text = re.sub(r"(?<!\w)_(?!\s)([^_\n]+?)_(?!\w)", r"<i>\1</i>", text)
    text = re.sub(r"^#{1,6}\s+(.+)$", r"<b>\1</b>", text, flags=re.M)
    return text


@dataclass
class Conversation:
    profile: str
    messages: list = field(default_factory=list)
    last_active: float = field(default_factory=time.monotonic)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class Conversations:
    """Chat histories keyed by (telegram user id, chat id)."""

    def __init__(self, idle_reset: float = IDLE_RESET_SECONDS):
        self._convs: Dict[Tuple[int, int], Conversation] = {}
        self._idle_reset = idle_reset

    def get(self, user_id: int, chat_id: int, profile: str) -> Conversation:
        key = (user_id, chat_id)
        conv = self._convs.get(key)
        stale = conv is None or conv.profile != profile or (
            time.monotonic() - conv.last_active > self._idle_reset and not conv.lock.locked())
        if stale:
            conv = self._convs[key] = Conversation(profile)
        conv.last_active = time.monotonic()
        return conv

    def reset(self, user_id: int, chat_id: Optional[int] = None) -> None:
        for key in [k for k in self._convs if k[0] == user_id and chat_id in (None, k[1])]:
            del self._convs[key]


class Invites:
    """One-time codes that let another Telegram user join a profile. In memory only."""

    def __init__(self, ttl: float = INVITE_TTL_SECONDS):
        self._codes: Dict[str, Tuple[str, float]] = {}
        self._ttl = ttl

    def create(self, profile: str) -> str:
        code = secrets.token_hex(4)
        self._codes[code] = (profile, time.monotonic() + self._ttl)
        return code

    def redeem(self, profile: str, code: Optional[str]) -> bool:
        key = (code or "").strip().lower()
        entry = self._codes.get(key)
        if entry is None or entry[0] != profile or entry[1] < time.monotonic():
            return False
        del self._codes[key]
        return True


def link_profile(name: str, telegram_id: int, code: Optional[str], admins: set[int], invites: Invites) -> str:
    """Link a Telegram user to a profile, enforcing who may join which profile.

    New profile names are created and linked freely. Joining a profile someone has
    already linked needs an invite code from them; claiming an existing profile no one
    has linked (e.g. one made in the CLI) is limited to admins. Raises PermissionError."""
    profile = tools.slugify(name)
    if not profile:
        raise PermissionError("Profile names need at least one letter or digit.")
    if profile in tools.list_users():
        linked = tools.linked_telegram_ids(profile)
        if str(telegram_id) in linked:
            return f"You're already linked to {profile}."
        if linked and not invites.redeem(profile, code):
            raise PermissionError(f"{profile} belongs to someone else. Ask them to send /invite "
                                  f"and give you the code, then send: /link {profile} <code>")
        if not linked and telegram_id not in admins:
            raise PermissionError(f"{profile} exists but isn't linked to Telegram yet; only a bot "
                                  "admin can claim it. Pick a new name to create your own profile.")
        created = False
    else:
        created = True
    tools.set_active_user(profile, remember=False)  # creates the profile file if new
    tools.link_telegram(profile, telegram_id)
    return (f"Created profile {profile} and linked it to you." if created
            else f"Linked you to profile {profile}.")


def snapshot_plans(profile: str) -> Dict[Path, float]:
    plans_dir = tools.PLANS_DIR / profile
    return {p: p.stat().st_mtime for p in plans_dir.glob("*.md")} if plans_dir.exists() else {}


def changed_plans(profile: str, before: Dict[Path, float]) -> List[Path]:
    plans_dir = tools.PLANS_DIR / profile
    if not plans_dir.exists():
        return []
    return [p for p in sorted(plans_dir.glob("*.md")) if before.get(p) != p.stat().st_mtime]


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------

HELP = f"""{AGENT_NAME} plans meals for your household. Chat with me here, or @mention me in a group.

/link <profile> - link your Telegram account to a profile (a new name creates one)
/link <profile> <code> - join someone's profile with their invite code
/invite - get a one-time code so someone else can join your profile
/profile - show your household
/favorites - list your favorite recipes
/newchat - start a fresh conversation
/unlink - disconnect your account

Paste recipes straight into the chat, or send them as .txt files. Saved plans come back as files.
Your household's allergies and preferences show up in my replies, so private chat is the most private place to plan."""

COMMANDS = [
    BotCommand("link", "Link your account to a profile"),
    BotCommand("invite", "Get a code for someone to join your profile"),
    BotCommand("profile", "Show your household"),
    BotCommand("favorites", "List your favorite recipes"),
    BotCommand("newchat", "Start a fresh conversation"),
    BotCommand("unlink", "Disconnect your account"),
    BotCommand("help", "How to use Kenji"),
]


class KenjiBot:
    def __init__(self) -> None:
        self.conversations = Conversations()
        self.invites = Invites()
        self.admins = _id_set("KENJI_ADMIN_TELEGRAM_IDS")

    # -- sending ----------------------------------------------------------

    @staticmethod
    async def send(message: Message, text: str) -> None:
        """Send Markdown-ish text as formatted HTML, falling back to plain text if
        Telegram rejects the markup."""
        for part in chunk_text(text):
            try:
                await message.get_bot().send_message(message.chat_id, markdown_to_telegram_html(part),
                                                     parse_mode=ParseMode.HTML,
                                                     disable_web_page_preview=True)
            except BadRequest:
                await message.get_bot().send_message(message.chat_id, part, disable_web_page_preview=True)

    # -- commands ---------------------------------------------------------

    @staticmethod
    async def _private_only(update: Update) -> bool:
        if update.effective_chat.type != ChatType.PRIVATE:
            await update.message.reply_text("For privacy, send that to me in a private chat.")
            return False
        return True

    async def _require_profile(self, update: Update) -> Optional[str]:
        profile = tools.user_for_telegram(update.effective_user.id)
        if profile is None:
            await update.message.reply_text("You're not linked yet: send /link <name> to create a profile.")
            return None
        tools.set_active_user(profile, remember=False)  # this handler's context only
        return profile

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(HELP)

    async def link(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._private_only(update):
            return
        if not context.args:
            await update.message.reply_text("Send /link <profile name>, or /link <profile> <code> to join one.")
            return
        code = context.args[1] if len(context.args) > 1 else None
        try:
            msg = link_profile(context.args[0], update.effective_user.id, code, self.admins, self.invites)
            self.conversations.reset(update.effective_user.id)
        except PermissionError as e:
            msg = str(e)
        await update.message.reply_text(msg)

    async def unlink(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        profile = tools.unlink_telegram(update.effective_user.id)
        self.conversations.reset(update.effective_user.id)
        await update.message.reply_text(
            f"Unlinked from {profile}. The profile itself is kept." if profile else "You weren't linked.")

    async def invite(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._private_only(update) or (profile := await self._require_profile(update)) is None:
            return
        code = self.invites.create(profile)
        await update.message.reply_text(
            f"Give this to the person joining {profile} - they send it to me in a private chat:\n\n"
            f"/link {profile} {code}\n\nIt works once and expires in {INVITE_TTL_SECONDS // 60} minutes. "
            "They'll see your household's allergies and preferences.")

    async def profile(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._private_only(update) or await self._require_profile(update) is None:
            return
        await update.message.reply_text(tools.describe_profile())

    async def favorites(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._private_only(update) or await self._require_profile(update) is None:
            return
        names = [r["name"] for r in tools.favorite_recipes()]
        await update.message.reply_text(
            "\n".join(f"- {n}" for n in names) or "No favorites yet. Ask me to favorite a recipe.")

    async def newchat(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        self.conversations.reset(update.effective_user.id, update.effective_chat.id)
        await update.message.reply_text("Fresh start - what are we planning?")

    # -- chat -------------------------------------------------------------

    @staticmethod
    def _addressed_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[str]:
        """The message text meant for Kenji, or None if a group message isn't for us."""
        message = update.message
        text = message.text or message.caption or ""
        if update.effective_chat.type == ChatType.PRIVATE:
            return text
        mention = f"@{context.bot.username}"
        replied_to_bot = (message.reply_to_message is not None and message.reply_to_message.from_user
                          and message.reply_to_message.from_user.id == context.bot.id)
        if mention.lower() in text.lower():
            return re.sub(re.escape(mention), "", text, flags=re.I).strip()
        return text if replied_to_bot else None

    async def _read_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
        doc = update.message.document
        if doc is None:
            return ""
        if Path(doc.file_name or "").suffix.lower() not in TEXT_SUFFIXES and not (
                doc.mime_type or "").startswith("text/"):
            await update.message.reply_text("I can only read text files (.txt, .md) - paste other recipes as text.")
            return ""
        if (doc.file_size or 0) > MAX_ATTACHMENT_BYTES:
            await update.message.reply_text(f"That file is over {MAX_ATTACHMENT_BYTES // 1000} KB.")
            return ""
        data = await (await context.bot.get_file(doc.file_id)).download_as_bytearray()
        return f"\n\nAttached file {doc.file_name}:\n\n{bytes(data).decode('utf-8', errors='replace')}"

    async def on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message is None or update.effective_user is None or update.effective_user.is_bot:
            return
        text = self._addressed_text(update, context)
        if text is None:
            return
        text = (text + await self._read_document(update, context)).strip()
        if not text:
            await update.message.reply_text(HELP)
            return
        profile = tools.user_for_telegram(update.effective_user.id)
        if profile is None:
            await update.message.reply_text("Hi! First link your Telegram account to a profile: "
                                            "send /link <name> to create one. /help for more.")
            return
        await self._run_agent(update, context, profile, text)

    async def _keep_typing(self, context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> None:
        while True:
            try:
                await context.bot.send_chat_action(chat_id, ChatAction.TYPING)
            except Exception:
                log.debug("typing indicator failed", exc_info=True)
            await asyncio.sleep(TYPING_REFRESH_SECONDS)

    async def _run_agent(self, update: Update, context: ContextTypes.DEFAULT_TYPE, profile: str, text: str) -> None:
        message = update.message
        conv = self.conversations.get(update.effective_user.id, update.effective_chat.id, profile)
        if conv.lock.locked():
            await message.reply_text("Still working on your last message - I'll get to this one next.")
        async with conv.lock:
            loop = asyncio.get_running_loop()

            def post(reply: str) -> None:
                # Called from the worker thread; wait so replies stay in order.
                asyncio.run_coroutine_threadsafe(self.send(message, reply), loop).result()

            def work() -> None:
                tools.set_active_user(profile, remember=False)  # this thread's context only
                run_turn(conv.messages, on_text=post, on_notice=post)

            before = snapshot_plans(profile)
            checkpoint = len(conv.messages)
            conv.messages.append({"role": "user", "content": text})
            typing = asyncio.create_task(self._keep_typing(context, update.effective_chat.id))
            error = None
            try:
                await asyncio.to_thread(work)
            except anthropic.RateLimitError:
                error = "I'm being rate limited - try again in a minute."
            except anthropic.APIConnectionError:
                error = "I couldn't reach the Claude API - try again shortly."
            except anthropic.APIStatusError as e:
                error = f"Claude API error {e.status_code}. Try again, or /newchat if it keeps happening."
                log.warning("API error for %s: %s", profile, e)
            except Exception:
                log.exception("Agent turn failed for %s", profile)
                error = "Something went wrong on my end. Try again, or /newchat to start fresh."
            finally:
                typing.cancel()
            if error:
                del conv.messages[checkpoint:]  # keep the history valid for the next attempt
                await message.reply_text(error)
                return
            await self._send_plans(context, update.effective_chat.id, profile, before)

    async def _send_plans(self, context: ContextTypes.DEFAULT_TYPE, chat_id: int, profile: str,
                          before: Dict[Path, float]) -> None:
        for md in changed_plans(profile, before):
            paths = [md]
            try:
                paths.append(await asyncio.to_thread(md_to_pdf, md))
            except Exception:
                log.exception("PDF conversion failed for %s", md)
            for path in paths:
                with open(path, "rb") as f:
                    await context.bot.send_document(chat_id, f, filename=path.name)

    async def on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        log.error("Telegram handler error", exc_info=context.error)


def build_application(token: str) -> Application:
    bot = KenjiBot()

    async def post_init(app: Application) -> None:
        await app.bot.set_my_commands(COMMANDS)

    # concurrent_updates: one person's long planning turn mustn't block everyone else.
    app = Application.builder().token(token).concurrent_updates(True).post_init(post_init).build()
    app.add_handler(CommandHandler(["start", "help"], bot.help))
    app.add_handler(CommandHandler("link", bot.link))
    app.add_handler(CommandHandler("unlink", bot.unlink))
    app.add_handler(CommandHandler("invite", bot.invite))
    app.add_handler(CommandHandler("profile", bot.profile))
    app.add_handler(CommandHandler("favorites", bot.favorites))
    app.add_handler(CommandHandler("newchat", bot.newchat))
    app.add_handler(MessageHandler((filters.TEXT | filters.Document.ALL) & ~filters.COMMAND, bot.on_message))
    app.add_error_handler(bot.on_error)
    return app


def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        sys.exit("Set TELEGRAM_BOT_TOKEN to the token @BotFather gave you.")
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # it logs every polling request
    build_application(token).run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
