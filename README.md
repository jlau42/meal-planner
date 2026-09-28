# Meal Planner

A command-line meal-planning agent built on the Claude API. Tell it what you need
("plan dinners for next week, family of 4, dairy-free") and it builds a plan from
your recipe library, then saves it as Markdown with per-person nutrition, a cook
list, and a shopping list grouped by store section.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and an Anthropic API key (workspace-scoped).

```powershell
[Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY", "sk-ant-...", "User")  # then open a new terminal
uv run main.py
```

## Chat commands

| Command | What it does |
| --- | --- |
| `/paste` | Paste a multi-line recipe (put the source URL on the first line), end with `/end` |
| `/file <path>` | Send a text file's contents, e.g. a saved recipe |
| `/favorites` | List favorite recipes |
| `/help` | Show commands |
| `quit` | Exit |

To favorite a recipe, just ask: "favorite the coconut curry".

## How it works

- `main.py`: chat loop using the Anthropic SDK tool runner (`claude-opus-5`).
- `tools.py`: tools the agent calls: profile read/update, recipe search/get/add,
  favorites, and `save_meal_plan`, which computes nutrition and the shopping list in
  code rather than trusting model arithmetic.
- `data/recipes.json`: recipe library. Serious Eats recipes (by `source_url`) and
  favorites are ranked first.
- `data/profile.json`: household size, allergies, targets, pantry staples, favorites.
- `plans/`: saved plans. Convert one to PDF with `uv run md_to_pdf.py plans/<file>.md`.

Serious Eats blocks Anthropic's web crawler, so the agent can't browse it: add
recipes from there with `/paste`.
