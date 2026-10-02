# Meal Planner

Kenji is a meal-planning agent built on the Claude API. Tell it what you need
("plan dinners for next week") and it builds a plan from your recipe library that's
safe for everyone eating, then saves it as Markdown with per-person nutrition, a cook
list, and a shopping list grouped by store section.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and an Anthropic API key (workspace-scoped).

Secrets live in a `.env` file in this folder, which is git-ignored. Copy the template
and fill it in:

```powershell
Copy-Item .env.example .env   # then edit .env: ANTHROPIC_API_KEY=sk-ant-...
uv run main.py                # asks which profile (defaults to the last one used)
uv run main.py jon            # or name it directly
```

Values already set as real environment variables take precedence over `.env`. Never
commit `.env`; `.env.example` is the committed template listing every setting.

## Profiles and households

Each **profile** is one user of Kenji, with their own favorites, removed recipes and
saved plans. The recipe library is shared by everyone.

Each profile has a **household** of members, and every member has their own
allergies, diet, dislikes and calorie/protein targets. Just tell Kenji, e.g. "add Ava,
age 6, allergic to peanuts, 1400 calories a day".

- Searches automatically hide recipes containing an allergen of anyone eating the meal.
  `allergy_exceptions` whitelist safe look-alikes (coconut milk for a dairy allergy).
- Saving a plan fails if anyone is scheduled to eat one of their allergens. A member can
  get their own dish for a meal while everyone else shares the main one.
- Plans report each member's nutrition against their own targets.

To try Kenji with example data, copy the sample in as a profile:
`Copy-Item data\sample_profile.json data\profiles\sample.json`, then `uv run main.py sample`.

## Telegram bot

Kenji also runs as a Telegram bot (`telegram_bot.py`), sharing the same profiles,
recipes and plans as the CLI.

**One-time setup**

1. In Telegram, message **@BotFather**, send `/newbot`, and pick a name and username.
   Copy the token it gives you.
2. Find your own Telegram user id (message **@userinfobot**). Admins are the only users
   who can issue invite and share codes; they can also create profiles without an invite
   and claim existing ones such as `jon`.
3. Put both in `.env` (`TELEGRAM_BOT_TOKEN=...`, `KENJI_ADMIN_TELEGRAM_IDS=<your id>`)
   and run it:

```powershell
uv run telegram_bot.py
```

The bot polls Telegram, so it needs no public URL or open ports; keep it running on
any always-on machine.

**Using it**

| Command | What it does |
| --- | --- |
| `/invite` | *Admin only.* Get a code (24 h) so someone can create **their own** profile |
| `/share` | *Admin only.* Get a code (15 min) so someone can join **your** household |
| `/link <name> <invite code>` | Create your own profile |
| `/link <profile> <share code>` | Join an existing household |
| `/profile`, `/favorites` | Show your household / favorites |
| `/newchat` | Start a fresh conversation |
| `/unlink`, `/help` | Disconnect your account / show help |

Kenji is **invite-only**, and only admins (`KENJI_ADMIN_TELEGRAM_IDS`) can issue codes, so
nobody can use the bot, or your API key, without an admin's say-so. `/invite` and
`/share` only appear in admins' command menus. Each code works once; unused
codes survive bot restarts (`data/profiles/.invites.json`).

Then just chat. Paste recipes directly or send them as `.txt` files; saved plans come
back as `.md` and `.pdf` documents. In a private chat Kenji answers everything; in a
group it answers only when @mentioned or replied to. Replies mention household
allergies, so `/link`, `/invite`, `/share`, `/profile` and `/favorites` only work in
private chat.

Profiles that exist but aren't linked yet (like ones made in the CLI) can only be
claimed by an id in `KENJI_ADMIN_TELEGRAM_IDS`. Chat history is kept in memory per user per chat and resets
after 6 idle hours or a bot restart; profiles, favorites and plans are saved on disk.

## Chat commands

| Command | What it does |
| --- | --- |
| `/paste` | Paste a multi-line recipe (put the source URL on the first line), end with `/end` |
| `/file <path>` | Send a text file's contents, e.g. a saved recipe |
| `/favorites` | List this profile's favorite recipes |
| `/profile` | Show the active profile and its household |
| `/profile <name>` | Switch profiles (creates new ones); starts a fresh chat |
| `/profiles` | List all profiles |
| `/help` | Show commands |
| `quit` | Exit |

To favorite a recipe, just ask: "favorite the coconut curry". To drop one you don't
like: "remove the pasta primavera, the kids hated it". Removal is per profile: the
recipe is hidden for you, won't be re-added, and can be restored on request.

## How it works

- `agent.py`: the agent loop (Anthropic SDK tool runner, `claude-opus-5`) and system
  prompt, shared by every front end. Output goes through callbacks.
- `main.py`: command-line front end. `telegram_bot.py`: Telegram front end.
- `tools.py`: tools the agent calls: profile and household members, recipe
  search/get/add, favorites, removal, and `save_meal_plan`, which does allergy checks,
  nutrition and the shopping list in code rather than trusting model arithmetic. The
  active profile is context-local, so a multi-user front end can serve several people
  at once.
- `data/recipes.json`: shared recipe library. Favorites, then Serious Eats recipes
  (by `source_url`), are ranked first.
- `data/profiles/<name>.json`: one file per profile. The folder is git-ignored because
  profiles hold personal details (names, allergies, linked Telegram ids); back it up
  separately. `data/sample_profile.json` is a committed example of the format.
- `plans/<name>/`: that profile's saved plans. Convert one to PDF with
  `uv run md_to_pdf.py plans/<name>/<file>.md`. Git-ignored like profiles, since plans
  name household members; `data/sample_plan.md` is a committed example.

Serious Eats blocks Anthropic's web crawler, so the agent can't browse it: add
recipes from there with `/paste`.
