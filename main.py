"""Kenji, the meal-planning agent: command-line front end.

Run:  uv run main.py [profile]
"""

import json
import sys
from pathlib import Path

import anthropic

import tools
from agent import AGENT_NAME, run_turn

HELP = """Commands:
  /paste           paste a multi-line recipe; finish with a line containing only /end
  /file <path>     send a text file's contents (e.g. a recipe you saved)
  /favorites       list this profile's favorite recipes
  /profile         show the active profile and its household
  /profile <name>  switch to another profile (creates it if new; starts a fresh chat)
  /profiles        list all profiles
  quit             exit
Household members, allergies and targets: just tell me, e.g. "add Ava, age 6, allergic to peanuts".
To add or remove a favorite, just ask, e.g. "favorite the coconut curry".
To drop a recipe you don't like: "remove the pasta primavera, the kids hated it"."""


def describe_profile() -> str:
    profile = json.loads(tools.get_profile.call({}))
    lines = [f"Profile: {profile['user']} (household of {profile['household_size']})"]
    for m in profile["members"]:
        details = [f"allergies: {', '.join(m['allergies'])}" if m["allergies"] else "",
                   f"diet: {', '.join(m['diet'])}" if m["diet"] else "",
                   f"{m['daily_calorie_target']} kcal/day" if m["daily_calorie_target"] else ""]
        lines.append(f"  - {m['name']}" + (f" ({'; '.join(d for d in details if d)})" if any(details) else ""))
    if not profile["members"]:
        lines.append("  No household members yet - tell me who's eating.")
    return "\n".join(lines)


def switch_profile(name: str) -> None:
    user, created = tools.set_active_user(name)
    print(("Created new profile " if created else "Switched to profile ") + f"'{user}'.")
    print(describe_profile() + "\n")


def choose_profile() -> None:
    if len(sys.argv) > 1:
        switch_profile(sys.argv[1])
        return
    users, default = tools.list_users(), tools.last_user()
    if users:
        print("Profiles: " + ", ".join(users))
    prompt = f"Which profile? [{default}] " if default else "Profile name (new profiles are created): "
    name = input(prompt).strip() or default
    while not name:
        name = input("Profile name: ").strip()
    switch_profile(name)


def read_user_input() -> str | None:
    """Read one chat message, handling commands. Returns None when there's nothing to send."""
    line = input("You: ").strip()
    if line == "/help":
        print(HELP + "\n")
        return None
    if line == "/profiles":
        print("Profiles: " + ", ".join(tools.list_users()) + f" (active: {tools.active_user()})\n")
        return None
    if line == "/profile":
        print(describe_profile() + "\n")
        return None
    if line.startswith("/profile "):
        return "/switch " + line[9:].strip()
    if line == "/favorites":
        result = tools.search_recipes.call({"favorites_only": True})
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
    print(f"Hi, I'm {AGENT_NAME}, your meal planner.")
    try:
        choose_profile()
    except (EOFError, KeyboardInterrupt):
        return
    print("Describe what you need (e.g. 'plan my dinners for next week').")
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
        if user_input.startswith("/switch "):
            try:
                switch_profile(user_input[8:])
            except ValueError as e:
                print(f"{e}\n")
                continue
            messages = []  # the conversation belonged to the previous profile
            continue
        checkpoint = len(messages)
        messages.append({"role": "user", "content": user_input})
        try:
            run_turn(messages,
                     on_text=lambda text: print(f"\n{AGENT_NAME}: {text}"),
                     on_tool=lambda name: print(f"  [calling {name}]"),
                     on_notice=lambda text: print(f"\n{AGENT_NAME}: {text}"))
            print()
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
