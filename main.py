"""Interactive meal-planning agent.

Run:  uv run main.py
"""

import json
import sys
from pathlib import Path

import anthropic

from tools import ALL_TOOLS, search_recipes

MODEL = "claude-opus-5"
AGENT_NAME = "Kenji"

SYSTEM_PROMPT = f"""You are {AGENT_NAME}, a practical meal-planning assistant. You help the user \
plan meals for a day, a week, or any span they ask for, and hand them a saved plan with a \
shopping list. You're an AI assistant, not a real chef; don't claim to be any real person.

How to work:
- Start by reading the user's profile with get_profile. If key facts are missing for the \
request (household size, dietary restrictions/allergies, calorie or protein goals, how much \
time they have to cook), ask briefly, then save the answers with update_profile so you don't \
ask again next time. Don't interrogate: if the user wants to skip questions, use sensible defaults.
- Allergies are hard constraints: always pass them to search_recipes as exclude_ingredients \
and never schedule a recipe containing them. Dislikes should be avoided when possible.
- The user's preferred recipe source is Serious Eats (seriouseats.com). You cannot browse the \
web; the user brings Serious Eats recipes to you by pasting them into the chat.
- When the user pastes a recipe, save it with add_recipe: use its ingredient quantities, \
serving count and nutrition if listed (otherwise estimate), set source_url if they gave one, \
and write the steps as a brief summary in your own words (the full method stays on the linked \
page). Flag any allergy conflicts before saving.
- When planning, favor saved Serious Eats recipes (search_recipes lists them first and shows \
source_url). Fill a slot with another saved recipe only when no Serious Eats recipe fits the \
constraints (diet, time, household), and say which slots those are. When Serious Eats coverage \
is thin, suggest a few specific dishes the user could look up on seriouseats.com and paste \
back, so the library grows toward their preference. Respect the weeknight time limit; \
Serious Eats recipes can be long.
- Favorites: when the user says they love a recipe or asks to favorite/unfavorite one, call \
set_favorite (search first if you need the id). When planning, work favorites in regularly \
(they're listed first in search_recipes), while keeping variety across the week. After a plan, \
if the user praises a meal, offer to favorite it.
- Plan like a real home cook: reuse ingredients across the week, schedule leftovers of \
batch recipes (make-ahead tag) on later days to cut waste and cooking time, keep weeknight \
dinners within the user's time limit, and include variety.
- Call save_meal_plan to produce the plan. It returns computed per-person daily nutrition; \
compare it with the user's targets and, if days are meaningfully off (more than ~10%), \
adjust (swap meals, add a snack) and save again. Trust its numbers over your own estimates.
- Finish with a short summary: the saved file path, the days at a glance, how the plan \
tracks against targets, and any prep-ahead tips. Keep chat replies concise."""

HELP = """Commands:
  /paste        paste a multi-line recipe; finish with a line containing only /end
  /file <path>  send a text file's contents (e.g. a recipe you saved)
  /favorites    list your favorite recipes
  quit          exit
To add or remove a favorite, just ask, e.g. "favorite the coconut curry"."""

client = anthropic.Anthropic()


def print_assistant(message) -> None:
    for block in message.content:
        if block.type == "text" and block.text.strip():
            print(f"\n{AGENT_NAME}: {block.text}")
        elif block.type == "tool_use":
            print(f"  [calling {block.name}]")


def run_turn(messages: list) -> None:
    """Run the agent until it stops calling tools, appending everything to `messages`."""
    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        tools=ALL_TOOLS,
        messages=messages,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        # If a safety classifier declines a (benign) request, the API retries it
        # on Anthropic's recommended fallback model instead of returning a refusal.
        fallbacks="default",
        betas=["server-side-fallback-2026-07-01"],
        # Cache the stable prefix (tools + system + history) across turns.
        cache_control={"type": "ephemeral"},
    )
    for message in runner:
        # The runner keeps its own history; mirror it so the chat continues next turn.
        messages.append({"role": "assistant", "content": message.content})
        if message.stop_reason == "refusal":
            print(f"\n{AGENT_NAME}: (The request was declined. Try rephrasing.)")
            return
        if message.stop_reason == "max_tokens":
            print(f"\n{AGENT_NAME}: (Response hit the length limit; ask me to continue.)")
        print_assistant(message)
        tool_response = runner.generate_tool_call_response()
        if tool_response is not None:
            messages.append(tool_response)
    print()


def read_user_input() -> str | None:
    """Read one chat message, expanding /paste and /file. Returns None to skip the turn."""
    line = input("You: ").strip()
    if line == "/help":
        print(HELP + "\n")
        return None
    if line == "/favorites":
        result = search_recipes.call({"favorites_only": True})
        if result.startswith("["):  # JSON list; otherwise it's the "no matches" message
            for r in json.loads(result):
                print(f"  * {r['name']} ({r['id']})")
        else:
            print("  No favorites yet. Ask me to favorite a recipe.")
        print()
        return None
    if line == "/paste":
        print("Paste the recipe (source URL on the first line helps), then /end on its own line:")
        lines = []
        while (next_line := input()).strip() != "/end":
            lines.append(next_line)
        text = "\n".join(lines).strip()
        return f"Please save this recipe to my library:\n\n{text}" if text else None
    if line.startswith("/file "):
        path = Path(line[6:].strip().strip('"'))
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as e:
            print(f"Couldn't read {path}: {e}\n")
            return None
        return f"Contents of {path.name}:\n\n{text}"
    return line or None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to a legacy code page
    print(f"Hi, I'm {AGENT_NAME}, your meal planner. Describe what you need (e.g. 'plan my dinners for next week').")
    print("Serious Eats recipes are preferred: use /paste to add one. Type /help for commands, 'quit' to exit.\n")
    messages: list = []
    while True:
        try:
            user_input = read_user_input()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input is None:
            continue
        if user_input.lower() in {"quit", "exit"}:
            break
        checkpoint = len(messages)
        messages.append({"role": "user", "content": user_input})
        try:
            run_turn(messages)
            continue
        except anthropic.AuthenticationError:
            print("Authentication failed - set ANTHROPIC_API_KEY (or run `ant auth login`).")
            break
        except anthropic.RateLimitError:
            print("Rate limited - wait a moment and try again.")
        except anthropic.APIStatusError as e:
            print(f"API error {e.status_code}: {e.message}")
        except anthropic.APIConnectionError:
            print("Couldn't reach the API - check your connection.")
        # Drop the failed turn so the history stays valid for the next attempt.
        del messages[checkpoint:]
        print()


if __name__ == "__main__":
    main()
