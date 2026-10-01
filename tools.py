"""Tools the meal-planning agent can call.

Claude decides *what* to eat; these functions own the facts and the math
(recipe data, nutrition totals, shopping-list aggregation) so the numbers in a
saved plan come from code, not from the model's arithmetic.
"""

import json
import math
import re
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import List, Optional, TypedDict

from anthropic import beta_tool

ROOT = Path(__file__).parent
RECIPES_PATH = ROOT / "data" / "recipes.json"
PROFILE_PATH = ROOT / "data" / "profile.json"
REMOVED_PATH = ROOT / "data" / "removed_recipes.json"
PLANS_DIR = ROOT / "plans"
PREFERRED_SOURCE = "seriouseats.com"

DEFAULT_PROFILE = {
    "household_size": 3,
    "diet": [],
    "allergies": [],
    "dislikes": [],
    "daily_calorie_target": None,
    "daily_protein_target_g": None,
    "max_weeknight_minutes": None,
    "pantry_staples": [],
    "notes": "",
    "favorite_recipe_ids": [],
}


def _load_recipes() -> list:
    return json.loads(RECIPES_PATH.read_text(encoding="utf-8"))


def _save_recipes(recipes: list) -> None:
    RECIPES_PATH.write_text(json.dumps(recipes, indent=2), encoding="utf-8")


def _load_profile() -> dict:
    if PROFILE_PATH.exists():
        return {**DEFAULT_PROFILE, **json.loads(PROFILE_PATH.read_text(encoding="utf-8"))}
    return dict(DEFAULT_PROFILE)


def _summary(r: dict, favorites: set) -> dict:
    return {
        "favorite": r["id"] in favorites,
        "id": r["id"],
        "name": r["name"],
        "meal_types": r["meal_types"],
        "tags": r["tags"],
        "prep_minutes": r["prep_minutes"],
        "servings_per_batch": r["servings"],
        "nutrition_per_serving": r["nutrition_per_serving"],
        "ingredients": [i["item"] for i in r["ingredients"]],
        "source_url": r.get("source_url"),
    }


def _is_preferred(r: dict) -> bool:
    return PREFERRED_SOURCE in (r.get("source_url") or "")


@beta_tool
def get_profile() -> str:
    """Get the user's saved food profile: household size, diet, allergies,
    dislikes, calorie/protein targets, cooking-time limits, pantry staples and notes.
    Call this at the start of every planning session."""
    return json.dumps(_load_profile(), indent=2)


@beta_tool
def update_profile(
    household_size: Optional[int] = None,
    diet: Optional[List[str]] = None,
    allergies: Optional[List[str]] = None,
    dislikes: Optional[List[str]] = None,
    daily_calorie_target: Optional[int] = None,
    daily_protein_target_g: Optional[int] = None,
    max_weeknight_minutes: Optional[int] = None,
    pantry_staples: Optional[List[str]] = None,
    notes: Optional[str] = None,
) -> str:
    """Save changes to the user's food profile. Only the fields you pass are
    changed; list fields replace the existing list, so include existing entries you want to keep.

    Args:
        household_size: Number of people eating each meal.
        diet: Diet labels matching recipe tags, e.g. ["vegetarian"], ["pescatarian", "gluten-free"].
        allergies: Ingredients that must never appear, e.g. ["peanut", "shrimp"].
        dislikes: Ingredients to avoid when possible.
        daily_calorie_target: Target calories per person per day.
        daily_protein_target_g: Target protein grams per person per day.
        max_weeknight_minutes: Longest prep time acceptable on Mon-Thu dinners.
        pantry_staples: Items the user always has, left off shopping lists (e.g. "olive oil", "cumin").
        notes: Free-form preferences (cuisines, budget, batch-cooking habits).
    """
    profile = _load_profile()
    updates = {k: v for k, v in locals().items() if k != "profile" and v is not None}
    profile.update(updates)
    PROFILE_PATH.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return "Profile updated:\n" + json.dumps(profile, indent=2)


@beta_tool
def search_recipes(
    meal_type: Optional[str] = None,
    required_tags: Optional[List[str]] = None,
    exclude_ingredients: Optional[List[str]] = None,
    max_prep_minutes: Optional[int] = None,
    text: Optional[str] = None,
    favorites_only: bool = False,
) -> str:
    """Search the saved recipe library. All filters are optional and combined with AND.
    Returns summaries including favorite status, nutrition per serving, ingredient names
    and source_url. The user's favorites are listed first, then Serious Eats recipes.

    Args:
        meal_type: One of "breakfast", "lunch", "dinner", "snack".
        required_tags: Recipe must have all of these tags, e.g. ["vegan", "quick"].
            Known tags include vegetarian, vegan, pescatarian, gluten-free, dairy-free,
            low-carb, high-protein, high-fiber, quick, make-ahead, budget.
        exclude_ingredients: Drop recipes containing any ingredient whose name includes one of these
            substrings (case-insensitive). Use for allergies and dislikes.
        max_prep_minutes: Maximum prep time.
        text: Substring to match against the recipe name or ingredients.
        favorites_only: Only return the user's favorite recipes.
    """
    favorites = set(_load_profile()["favorite_recipe_ids"])
    results = []
    for r in _load_recipes():
        if favorites_only and r["id"] not in favorites:
            continue
        ingredients = " | ".join(i["item"].lower() for i in r["ingredients"])
        if meal_type and meal_type.lower() not in r["meal_types"]:
            continue
        if required_tags and not all(t.lower() in r["tags"] for t in required_tags):
            continue
        if exclude_ingredients and any(x.lower() in ingredients for x in exclude_ingredients):
            continue
        if max_prep_minutes is not None and r["prep_minutes"] > max_prep_minutes:
            continue
        if text and text.lower() not in (r["name"].lower() + " " + ingredients):
            continue
        results.append(r)
    if not results:
        return ("No saved recipes matched. Loosen the filters, or suggest a Serious Eats dish "
                "the user could paste in.")
    results.sort(key=lambda r: (r["id"] not in favorites, not _is_preferred(r)))
    return json.dumps([_summary(r, favorites) for r in results], indent=2)


@beta_tool
def set_favorite(recipe_id: str, favorite: bool = True) -> str:
    """Mark a recipe as one of the user's favorites, or remove it from favorites.
    Favorites are listed first in search_recipes and should be scheduled often.

    Args:
        recipe_id: The recipe id from search_recipes.
        favorite: True to add to favorites, False to remove.
    """
    names = {r["id"]: r["name"] for r in _load_recipes()}
    if recipe_id not in names:
        raise ValueError(f"Unknown recipe id {recipe_id!r}. Use search_recipes to find the id.")
    profile = _load_profile()
    ids = [i for i in profile["favorite_recipe_ids"] if i != recipe_id]
    if favorite:
        ids.append(recipe_id)
    profile["favorite_recipe_ids"] = ids
    PROFILE_PATH.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    current = ", ".join(names.get(i, i) for i in ids) or "none"
    action = "Added to" if favorite else "Removed from"
    return f"{action} favorites: {names[recipe_id]}. Current favorites: {current}."


def _load_removed() -> list:
    if REMOVED_PATH.exists():
        return json.loads(REMOVED_PATH.read_text(encoding="utf-8"))
    return []


def _find_removed(recipe_id: str, source_url: Optional[str] = None) -> Optional[dict]:
    for entry in _load_removed():
        r = entry["recipe"]
        if r["id"] == recipe_id or (source_url and r.get("source_url") == source_url):
            return entry
    return None


@beta_tool
def remove_recipe(recipe_id: str, reason: Optional[str] = None) -> str:
    """Remove a recipe the user doesn't like from the library. It also comes off their
    favorites, and add_recipe will refuse to re-add it. The recipe is kept in
    data/removed_recipes.json so restore_recipe can bring it back.

    Args:
        recipe_id: The recipe id from search_recipes.
        reason: Why the user didn't like it, in their words (e.g. "too spicy for the kids").
            Use it to steer future suggestions away from similar recipes.
    """
    recipes = _load_recipes()
    recipe = next((r for r in recipes if r["id"] == recipe_id), None)
    if recipe is None:
        raise ValueError(f"Unknown recipe id {recipe_id!r}. Use search_recipes to find the id.")

    removed = _load_removed()
    removed.append({"recipe": recipe, "reason": reason, "removed_on": date.today().isoformat()})
    REMOVED_PATH.write_text(json.dumps(removed, indent=2), encoding="utf-8")
    _save_recipes([r for r in recipes if r["id"] != recipe_id])

    profile = _load_profile()
    if recipe_id in profile["favorite_recipe_ids"]:
        profile["favorite_recipe_ids"] = [i for i in profile["favorite_recipe_ids"] if i != recipe_id]
        PROFILE_PATH.write_text(json.dumps(profile, indent=2), encoding="utf-8")
        return f"Removed {recipe['name']} from the library and from favorites."
    return f"Removed {recipe['name']} from the library."


@beta_tool
def list_removed_recipes() -> str:
    """List recipes the user has removed, with their reasons. Check this to avoid
    suggesting dishes like ones the user disliked."""
    removed = _load_removed()
    if not removed:
        return "No removed recipes."
    return json.dumps([{"id": e["recipe"]["id"], "name": e["recipe"]["name"], "reason": e["reason"],
                        "removed_on": e["removed_on"]} for e in removed], indent=2)


@beta_tool
def restore_recipe(recipe_id: str) -> str:
    """Put a previously removed recipe back in the library. Only when the user asks for it.

    Args:
        recipe_id: The id from list_removed_recipes.
    """
    removed = _load_removed()
    entry = next((e for e in removed if e["recipe"]["id"] == recipe_id), None)
    if entry is None:
        raise ValueError(f"No removed recipe with id {recipe_id!r}. Use list_removed_recipes.")
    recipes = _load_recipes()
    if not any(r["id"] == recipe_id for r in recipes):
        recipes.append(entry["recipe"])
        _save_recipes(recipes)
    REMOVED_PATH.write_text(json.dumps([e for e in removed if e is not entry], indent=2), encoding="utf-8")
    return f"Restored {entry['recipe']['name']} to the library."


@beta_tool
def get_recipe(recipe_id: str) -> str:
    """Get the full recipe (quantities and steps) for one recipe id.

    Args:
        recipe_id: The recipe id from search_recipes.
    """
    for r in _load_recipes():
        if r["id"] == recipe_id:
            return json.dumps(r, indent=2)
    raise ValueError(f"Unknown recipe id {recipe_id!r}")


class Ingredient(TypedDict):
    item: str
    qty: float
    unit: str
    section: str


class Nutrition(TypedDict):
    calories: int
    protein_g: int
    carbs_g: int
    fat_g: int


@beta_tool
def add_recipe(
    name: str,
    meal_types: List[str],
    tags: List[str],
    prep_minutes: int,
    servings: int,
    nutrition_per_serving: Nutrition,
    ingredients: List[Ingredient],
    steps: List[str],
    source_url: Optional[str] = None,
) -> str:
    """Save a recipe to the library, usually one the user pasted from Serious Eats. Use the source's
    ingredient quantities; if it lists no nutrition, give realistic, conservative estimates.

    Args:
        name: Recipe name.
        meal_types: Any of "breakfast", "lunch", "dinner", "snack".
        tags: Lowercase tags, e.g. ["vegetarian", "quick"]. Tag every diet the recipe satisfies.
        prep_minutes: Total active + cook time.
        servings: Servings one batch makes.
        nutrition_per_serving: calories, protein_g, carbs_g, fat_g per serving.
        ingredients: For one batch. section is the store aisle: produce, meat, seafood,
            dairy, bakery, deli, pantry, frozen.
        steps: Brief cooking steps in your own words (a quick-reference summary, not the
            source's text; the full method stays at source_url).
        source_url: Page the recipe came from, e.g. a https://www.seriouseats.com/... URL.
    """
    recipes = _load_recipes()
    recipe_id = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if any(r["id"] == recipe_id for r in recipes):
        raise ValueError(f"A recipe with id {recipe_id!r} already exists")
    if (prior := _find_removed(recipe_id, source_url)) is not None:
        raise ValueError(
            f"The user removed {prior['recipe']['name']!r} on {prior['removed_on']} "
            f"(reason: {prior['reason'] or 'not given'}). Don't re-add it; if the user explicitly "
            f"wants it back, use restore_recipe."
        )
    recipes.append({
        "id": recipe_id,
        "name": name,
        "meal_types": [m.lower() for m in meal_types],
        "tags": [t.lower() for t in tags],
        "prep_minutes": prep_minutes,
        "servings": servings,
        "nutrition_per_serving": dict(nutrition_per_serving),
        "ingredients": [dict(i) for i in ingredients],
        "steps": steps,
        **({"source_url": source_url} if source_url else {}),
    })
    _save_recipes(recipes)
    return f"Added recipe with id {recipe_id!r}."


class PlannedMeal(TypedDict):
    meal_type: str
    recipe_id: str
    portions: int


class PlanDay(TypedDict):
    day: str
    meals: List[PlannedMeal]


def _fmt_qty(q: float) -> str:
    return str(int(q)) if float(q).is_integer() else f"{q:.2f}".rstrip("0")


@beta_tool
def save_meal_plan(title: str, days: List[PlanDay], prep_notes: Optional[str] = None) -> str:
    """Validate and save a meal plan. Computes per-person daily nutrition and a
    consolidated shopping list (batches are rounded up, so leftovers get reused
    across days), writes a Markdown file, and returns the totals so you can check
    them against the user's targets and revise if needed.

    Args:
        title: Plan title, e.g. "Week of Oct 5".
        days: One entry per day. Each meal names a recipe_id and portions (people eating it,
            usually the household size). Repeating a recipe on later days uses leftovers.
        prep_notes: Optional batch-prep / leftover instructions to include in the file.
    """
    recipes = {r["id"]: r for r in _load_recipes()}
    profile = _load_profile()
    household = max(1, profile["household_size"])

    unknown = {m["recipe_id"] for d in days for m in d["meals"]} - recipes.keys()
    if unknown:
        raise ValueError(f"Unknown recipe ids: {sorted(unknown)}. Use search_recipes or add_recipe first.")

    portions_by_recipe = defaultdict(int)
    daily = []
    for d in days:
        totals = defaultdict(float)
        for m in d["meals"]:
            r = recipes[m["recipe_id"]]
            portions_by_recipe[r["id"]] += m["portions"]
            # Per-person nutrition: each eater gets one serving.
            for k, v in r["nutrition_per_serving"].items():
                totals[k] += v
        daily.append((d["day"], {k: round(v) for k, v in totals.items()}))

    batches = {rid: math.ceil(p / recipes[rid]["servings"]) for rid, p in portions_by_recipe.items()}
    staples = [s.lower() for s in profile["pantry_staples"]]
    shopping = defaultdict(float)  # (section, item, unit) -> qty
    for rid, n in batches.items():
        for ing in recipes[rid]["ingredients"]:
            if any(s in ing["item"].lower() for s in staples):
                continue
            shopping[(ing["section"], ing["item"].lower(), ing["unit"])] += ing["qty"] * n

    lines = [f"# {title}", "", f"Household size: {household}", ""]
    for d, (day_name, totals) in zip(days, daily):
        lines.append(f"## {day_name}")
        for m in d["meals"]:
            r = recipes[m["recipe_id"]]
            name = f"[{r['name']}]({r['source_url']})" if r.get("source_url") else r["name"]
            lines.append(f"- **{m['meal_type'].title()}**: {name} ({r['prep_minutes']} min) x{m['portions']}")
        lines.append(
            f"- _Per person: {totals.get('calories', 0)} kcal, {totals.get('protein_g', 0)}g protein, "
            f"{totals.get('carbs_g', 0)}g carbs, {totals.get('fat_g', 0)}g fat_"
        )
        lines.append("")
    if prep_notes:
        lines += ["## Prep notes", "", prep_notes, ""]
    lines += ["## Cook list", ""]
    for rid, n in batches.items():
        r = recipes[rid]
        lines.append(f"- {r['name']}: {n} batch(es) = {n * r['servings']} servings, {portions_by_recipe[rid]} planned")
    lines += ["", "## Shopping list", ""]
    for section in sorted({s for s, _, _ in shopping}):
        lines.append(f"### {section.title()}")
        for (s, item, unit), qty in sorted(shopping.items()):
            if s == section:
                lines.append(f"- [ ] {item}: {_fmt_qty(qty)} {unit}")
        lines.append("")
    if staples:
        lines.append(f"_Pantry staples not listed: {', '.join(profile['pantry_staples'])}_")

    PLANS_DIR.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "plan"
    path = PLANS_DIR / f"{date.today().isoformat()}-{slug}.md"
    path.write_text("\n".join(lines), encoding="utf-8")

    return json.dumps({
        "saved_to": str(path),
        "targets": {
            "daily_calorie_target": profile["daily_calorie_target"],
            "daily_protein_target_g": profile["daily_protein_target_g"],
        },
        "per_person_daily_nutrition": dict(daily),
        "batches_to_cook": batches,
        "shopping_list_item_count": len(shopping),
    }, indent=2)


ALL_TOOLS = [get_profile, update_profile, search_recipes, get_recipe, add_recipe, set_favorite,
             remove_recipe, list_removed_recipes, restore_recipe, save_meal_plan]
