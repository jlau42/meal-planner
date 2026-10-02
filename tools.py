"""Tools the meal-planning agent can call.

Claude decides *what* to eat; these functions own the facts and the math
(recipe data, allergy checks, nutrition totals, shopping-list aggregation) so the
numbers and safety checks in a saved plan come from code, not the model.

Data layout:
  data/recipes.json          shared recipe library
  data/profiles/<user>.json  one profile per user: household members (each with their
                             own allergies, diet, targets), favorites, removed recipes
  plans/<user>/              that user's saved plans

The active user is context-local (a ContextVar), so a multi-user front end such as a
Telegram bot can serve several people at once: set_active_user() in each request's
context, then run the agent there (asyncio.to_thread copies the context along).
"""

import functools
import json
import math
import re
import threading
from collections import defaultdict
from contextvars import ContextVar
from datetime import date
from pathlib import Path
from typing import List, NotRequired, Optional, TypedDict

from anthropic import beta_tool

ROOT = Path(__file__).parent
RECIPES_PATH = ROOT / "data" / "recipes.json"
PROFILES_DIR = ROOT / "data" / "profiles"
LAST_USER_PATH = PROFILES_DIR / ".last_user"
PLANS_DIR = ROOT / "plans"
PREFERRED_SOURCE = "seriouseats.com"

DEFAULT_PROFILE = {
    "household_size": 1,
    "members": [],
    "max_weeknight_minutes": None,
    "pantry_staples": [],
    "notes": "",
    "favorite_recipe_ids": [],
    "removed_recipes": [],
    "telegram_user_ids": [],
}

DEFAULT_MEMBER = {
    "name": "",
    "allergies": [],
    "allergy_exceptions": [],
    "diet": [],
    "dislikes": [],
    "daily_calorie_target": None,
    "daily_protein_target_g": None,
    "notes": "",
}

_active_user: ContextVar[Optional[str]] = ContextVar("active_user", default=None)
# Serializes read-modify-write of profiles and the recipe library across threads.
_write_lock = threading.RLock()


def _locked(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _write_lock:
            return fn(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# Users (called by main.py, not by the agent)
# ---------------------------------------------------------------------------

def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def list_users() -> List[str]:
    return sorted(p.stem for p in PROFILES_DIR.glob("*.json") if not p.name.startswith("."))  # skip .invites.json etc.


def last_user() -> Optional[str]:
    if LAST_USER_PATH.exists():
        name = LAST_USER_PATH.read_text(encoding="utf-8").strip()
        if name in list_users():
            return name
    return None


@_locked
def set_active_user(name: str, remember: bool = True) -> tuple[str, bool]:
    """Make `name` the active user in the current context, creating an empty profile if
    needed. remember=True records it as the CLI's default. Returns (user id, created)."""
    user = slugify(name)
    if not user:
        raise ValueError("Profile names need at least one letter or digit.")
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    created = not _profile_path(user).exists()
    if created:
        _profile_path(user).write_text(json.dumps(DEFAULT_PROFILE, indent=2), encoding="utf-8")
    _active_user.set(user)
    if remember:
        LAST_USER_PATH.write_text(user, encoding="utf-8")
    return user, created


def active_user() -> str:
    user = _active_user.get()
    if user is None:
        raise RuntimeError("No active profile. Call set_active_user() first.")
    return user


def _read_profile_file(user: str) -> dict:
    profile = {**DEFAULT_PROFILE, **json.loads(_profile_path(user).read_text(encoding="utf-8"))}
    profile.pop("discord_user_ids", None)  # legacy field from the dropped Discord front end
    return profile


def linked_telegram_ids(user: str) -> List[str]:
    """Telegram user ids linked to a profile."""
    return list(_read_profile_file(slugify(user))["telegram_user_ids"])


def user_for_telegram(telegram_user_id: int | str) -> Optional[str]:
    """The profile linked to a Telegram user id, or None if they haven't linked one."""
    tid = str(telegram_user_id)
    return next((u for u in list_users() if tid in _read_profile_file(u)["telegram_user_ids"]), None)


@_locked
def link_telegram(user: str, telegram_user_id: int | str) -> None:
    """Link a Telegram user to a profile (moving the link if it pointed elsewhere).
    Several Telegram users may share one profile, e.g. two adults in a household."""
    tid = str(telegram_user_id)
    if slugify(user) not in list_users():
        raise ValueError(f"No profile named {user!r}.")
    for u in list_users():
        profile = _read_profile_file(u)
        ids = [i for i in profile["telegram_user_ids"] if i != tid]
        if u == slugify(user):
            ids.append(tid)
        if ids != profile["telegram_user_ids"]:
            profile["telegram_user_ids"] = ids
            _profile_path(u).write_text(json.dumps(profile, indent=2), encoding="utf-8")


@_locked
def unlink_telegram(telegram_user_id: int | str) -> Optional[str]:
    """Remove a Telegram user's link. Returns the profile it was linked to, if any."""
    tid = str(telegram_user_id)
    user = user_for_telegram(tid)
    if user:
        profile = _read_profile_file(user)
        profile["telegram_user_ids"] = [i for i in profile["telegram_user_ids"] if i != tid]
        _profile_path(user).write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return user


def describe_profile() -> str:
    """Short human-readable summary of the active profile and its household."""
    profile = _load_profile()
    lines = [f"Profile: {active_user()} (household of {_household_size(profile)})"]
    for m in profile["members"]:
        details = [f"allergies: {', '.join(m['allergies'])}" if m["allergies"] else "",
                   f"diet: {', '.join(m['diet'])}" if m["diet"] else "",
                   f"{m['daily_calorie_target']} kcal/day" if m["daily_calorie_target"] else ""]
        details = [d for d in details if d]
        lines.append(f"  - {m['name']}" + (f" ({'; '.join(details)})" if details else ""))
    if not profile["members"]:
        lines.append("  No household members yet - tell me who's eating.")
    return "\n".join(lines)


def favorite_recipes() -> List[dict]:
    """The active profile's favorite recipes, in the order they were favorited."""
    favorites = _load_profile()["favorite_recipe_ids"]
    by_id = {r["id"]: r for r in _load_recipes()}
    return [by_id[i] for i in favorites if i in by_id]


def _profile_path(user: Optional[str] = None) -> Path:
    return PROFILES_DIR / f"{user or active_user()}.json"


def _load_profile() -> dict:
    profile = _read_profile_file(active_user())
    profile["members"] = [{**DEFAULT_MEMBER, **m} for m in profile["members"]]
    return profile


def _save_profile(profile: dict) -> None:
    _profile_path().write_text(json.dumps(profile, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_recipes() -> list:
    return json.loads(RECIPES_PATH.read_text(encoding="utf-8"))


def _save_recipes(recipes: list) -> None:
    RECIPES_PATH.write_text(json.dumps(recipes, indent=2), encoding="utf-8")


def _is_preferred(r: dict) -> bool:
    return PREFERRED_SOURCE in (r.get("source_url") or "")


def _household_size(profile: dict) -> int:
    return max(1, profile["household_size"], len(profile["members"]))


def _find_member(profile: dict, name: str) -> Optional[dict]:
    return next((m for m in profile["members"] if m["name"].lower() == name.strip().lower()), None)


def _resolve_members(profile: dict, names: Optional[List[str]]) -> List[dict]:
    """Members for `names`, or every member when names is None. Unknown names raise."""
    if names is None:
        return profile["members"]
    members, unknown = [], []
    for n in names:
        m = _find_member(profile, n)
        (members.append(m) if m else unknown.append(n))
    if unknown:
        known = ", ".join(m["name"] for m in profile["members"]) or "none yet"
        raise ValueError(f"Unknown household member(s): {unknown}. Members: {known}.")
    return members


# A "dairy" allergy also matches these, since cheeses and dairy products are often named
# without the words milk/cheese/cream (e.g. "feta", "ricotta", "ghee").
DAIRY_TERMS = (
    "dairy", "milk", "butter", "cheese", "cream", "yogurt", "ghee", "whey", "casein", "lactose",
    "kefir", "half-and-half", "parmesan", "pecorino", "mozzarella", "cheddar", "feta", "ricotta",
    "paneer", "mascarpone", "gruyere", "provolone", "halloumi", "burrata", "queso", "brie",
    "gouda", "monterey jack", "colby", "swiss", "ranch", "alfredo", "tzatziki", "custard",
)
# Non-dairy ingredients whose names contain a dairy term. Ignored only when checking
# dairy terms, so "cashew butter" still matches a cashew allergy.
DAIRY_LOOKALIKES = (
    "peanut butter", "almond butter", "cashew butter", "sunflower butter", "nut butter",
    "cocoa butter", "apple butter", "butternut", "butter lettuce", "butter beans",
    "coconut milk", "coconut cream", "cream of coconut", "almond milk", "oat milk", "soy milk",
    "rice milk", "cashew milk", "cream of tartar", "swiss chard",
)


def is_dairy(item: str) -> bool:
    """Whether an ingredient name looks like dairy, ignoring known non-dairy look-alikes."""
    item = item.lower()
    for lookalike in DAIRY_LOOKALIKES:
        item = item.replace(lookalike, " ")
    return any(term in item for term in DAIRY_TERMS)


def _allergy_conflicts(recipe: dict, member: dict) -> List[str]:
    """Ingredients in `recipe` that match one of the member's allergens.

    Matching is by case-insensitive substring, so "milk" also catches "buttermilk". A
    "dairy" allergy also matches DAIRY_TERMS; dairy terms ignore DAIRY_LOOKALIKES such as
    peanut butter. allergy_exceptions whitelists specific safe ingredients ("coconut milk")."""
    allergens = [a.lower() for a in member["allergies"]]
    exceptions = [e.lower() for e in member["allergy_exceptions"]]
    dairy_allergy = any(a == "dairy" or is_dairy(a) for a in allergens)
    other_allergens = [a for a in allergens if not is_dairy(a) and a != "dairy"]
    hits = []
    for ing in recipe["ingredients"]:
        item = ing["item"].lower()
        if any(e in item for e in exceptions):
            continue
        if (dairy_allergy and is_dairy(item)) or any(a in item for a in other_allergens):
            hits.append(ing["item"])
    return hits


def _soft_issues(recipe: dict, member: dict) -> List[str]:
    """Non-safety mismatches: diet tags the recipe lacks, disliked ingredients."""
    issues = [f"not tagged {d}" for d in member["diet"] if d.lower() not in recipe["tags"]]
    ingredients = " | ".join(i["item"].lower() for i in recipe["ingredients"])
    issues += [f"contains disliked {d}" for d in member["dislikes"] if d.lower() in ingredients]
    return issues


def _summary(r: dict, favorites: set, members: List[dict]) -> dict:
    summary = {
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
    issues = {m["name"]: i for m in members if (i := _soft_issues(r, m))}
    if issues:
        summary["member_issues"] = issues
    return summary


# ---------------------------------------------------------------------------
# Profile and household members
# ---------------------------------------------------------------------------

@beta_tool
def get_profile() -> str:
    """Get the active user's profile: household size, each household member's allergies,
    allergy exceptions, diet, dislikes and calorie/protein targets, plus cooking-time
    limit, pantry staples, notes, favorites and removed recipes.
    Call this at the start of every planning session."""
    profile = _load_profile()
    profile.pop("telegram_user_ids", None)  # account plumbing, not food preferences
    return json.dumps({"user": active_user(), **profile,
                       "household_size": _household_size(profile)}, indent=2)


@beta_tool
@_locked
def update_profile(
    household_size: Optional[int] = None,
    max_weeknight_minutes: Optional[int] = None,
    pantry_staples: Optional[List[str]] = None,
    notes: Optional[str] = None,
) -> str:
    """Save household-wide settings for the active user. Per-person details (allergies,
    diet, targets) belong to household members: use set_member for those. Only the fields
    you pass are changed; list fields replace the existing list.

    Args:
        household_size: Total people eating, including members without a saved profile
            (e.g. a guest). Never less than the number of saved members.
        max_weeknight_minutes: Longest prep time acceptable on Mon-Thu dinners.
        pantry_staples: Items always on hand, left off shopping lists (e.g. "olive oil").
        notes: Free-form household preferences (cuisines, budget, batch-cooking habits).
    """
    profile = _load_profile()
    for key, value in (("household_size", household_size), ("max_weeknight_minutes", max_weeknight_minutes),
                       ("pantry_staples", pantry_staples), ("notes", notes)):
        if value is not None:
            profile[key] = value
    _save_profile(profile)
    return "Profile updated:\n" + get_profile.call({})


@beta_tool
@_locked
def set_member(
    name: str,
    allergies: Optional[List[str]] = None,
    allergy_exceptions: Optional[List[str]] = None,
    diet: Optional[List[str]] = None,
    dislikes: Optional[List[str]] = None,
    daily_calorie_target: Optional[int] = None,
    daily_protein_target_g: Optional[int] = None,
    notes: Optional[str] = None,
) -> str:
    """Add a household member or update one (matched by name, case-insensitive). Only
    the fields you pass are changed; list fields replace the existing list.

    Args:
        name: The person's name, e.g. "Jon" or "Ava".
        allergies: Hard exclusions, matched as substrings of ingredient names. Be broad:
            for a dairy allergy use ["dairy", "milk", "butter", "cheese", "cream", "yogurt", ...].
        allergy_exceptions: Ingredients that match an allergy term but are safe, e.g.
            ["coconut milk", "coconut cream", "peanut-free butter"].
        diet: Diet labels matching recipe tags, e.g. ["vegetarian"], ["gluten-free"].
        dislikes: Ingredients this person avoids when possible.
        daily_calorie_target: This person's daily calorie target.
        daily_protein_target_g: This person's daily protein target in grams.
        notes: Free-form notes, e.g. "age 6, mild spice only".
    """
    profile = _load_profile()
    member = _find_member(profile, name)
    created = member is None
    if created:
        member = {**DEFAULT_MEMBER, "name": name.strip()}
        profile["members"].append(member)
    for key, value in (("allergies", allergies), ("allergy_exceptions", allergy_exceptions),
                       ("diet", diet), ("dislikes", dislikes),
                       ("daily_calorie_target", daily_calorie_target),
                       ("daily_protein_target_g", daily_protein_target_g), ("notes", notes)):
        if value is not None:
            member[key] = value
    _save_profile(profile)
    return f"{'Added' if created else 'Updated'} member {member['name']}:\n" + json.dumps(member, indent=2)


@beta_tool
@_locked
def remove_member(name: str) -> str:
    """Remove a household member from the active user's profile.

    Args:
        name: The member's name.
    """
    profile = _load_profile()
    member = _find_member(profile, name)
    if member is None:
        _resolve_members(profile, [name])  # raises with the list of known members
    profile["members"] = [m for m in profile["members"] if m is not member]
    _save_profile(profile)
    return f"Removed member {member['name']}. Household size is now {_household_size(profile)}."


# ---------------------------------------------------------------------------
# Recipes
# ---------------------------------------------------------------------------

@beta_tool
def search_recipes(
    meal_type: Optional[str] = None,
    required_tags: Optional[List[str]] = None,
    exclude_ingredients: Optional[List[str]] = None,
    max_prep_minutes: Optional[int] = None,
    text: Optional[str] = None,
    favorites_only: bool = False,
    safe_for: Optional[List[str]] = None,
) -> str:
    """Search the shared recipe library. All filters are optional and combined with AND.

    Recipes containing an allergen of any member in safe_for (default: every household
    member) are excluded automatically. Recipes the active user removed are hidden.
    Results list favorites first, then Serious Eats recipes, and include member_issues:
    diet tags a recipe lacks or disliked ingredients, per member (not safety problems).

    Args:
        meal_type: One of "breakfast", "lunch", "dinner", "snack".
        required_tags: Recipe must have all of these tags, e.g. ["vegan", "quick"].
            Known tags include vegetarian, vegan, pescatarian, gluten-free, dairy-free,
            low-carb, high-protein, high-fiber, quick, make-ahead, budget.
        exclude_ingredients: Also drop recipes containing any of these substrings.
        max_prep_minutes: Maximum prep time.
        text: Substring to match against the recipe name or ingredients.
        favorites_only: Only return the active user's favorite recipes.
        safe_for: Member names who will eat this meal, when it's not the whole household
            (e.g. ["Ava", "Leo"] for a kids' lunch).
    """
    profile = _load_profile()
    favorites = set(profile["favorite_recipe_ids"])
    removed = {e["id"] for e in profile["removed_recipes"]}
    eaters = _resolve_members(profile, safe_for)
    results, hidden_unsafe = [], 0
    for r in _load_recipes():
        if r["id"] in removed or (favorites_only and r["id"] not in favorites):
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
        if any(_allergy_conflicts(r, m) for m in eaters):
            hidden_unsafe += 1
            continue
        results.append(r)
    note = (f" ({hidden_unsafe} more hidden for allergies of: "
            f"{', '.join(m['name'] for m in eaters)})") if hidden_unsafe else ""
    if not results:
        return ("No saved recipes matched" + note + ". Loosen the filters, or suggest a "
                "Serious Eats dish the user could paste in.")
    results.sort(key=lambda r: (r["id"] not in favorites, not _is_preferred(r)))
    body = json.dumps([_summary(r, favorites, eaters) for r in results], indent=2)
    return body + (f"\n\nNote:{note}" if note else "")


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


@beta_tool
@_locked
def set_favorite(recipe_id: str, favorite: bool = True) -> str:
    """Mark a recipe as one of the active user's favorites, or remove it from favorites.
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
    _save_profile(profile)
    current = ", ".join(names.get(i, i) for i in ids) or "none"
    action = "Added to" if favorite else "Removed from"
    return f"{action} favorites: {names[recipe_id]}. Current favorites: {current}."


@beta_tool
@_locked
def remove_recipe(recipe_id: str, reason: Optional[str] = None) -> str:
    """Remove a recipe the active user doesn't like. It's hidden from their searches and
    plans and taken off their favorites; the shared library and other users are unaffected.
    add_recipe refuses to re-add it, and restore_recipe brings it back.

    Args:
        recipe_id: The recipe id from search_recipes.
        reason: Why they didn't like it, in their words (e.g. "too spicy for the kids").
            Use it to steer future suggestions away from similar recipes.
    """
    recipe = next((r for r in _load_recipes() if r["id"] == recipe_id), None)
    if recipe is None:
        raise ValueError(f"Unknown recipe id {recipe_id!r}. Use search_recipes to find the id.")
    profile = _load_profile()
    if any(e["id"] == recipe_id for e in profile["removed_recipes"]):
        return f"{recipe['name']} was already removed."
    profile["removed_recipes"].append({
        "id": recipe_id, "name": recipe["name"], "source_url": recipe.get("source_url"),
        "reason": reason, "removed_on": date.today().isoformat(),
    })
    was_favorite = recipe_id in profile["favorite_recipe_ids"]
    profile["favorite_recipe_ids"] = [i for i in profile["favorite_recipe_ids"] if i != recipe_id]
    _save_profile(profile)
    return f"Removed {recipe['name']}" + (" (and from favorites)." if was_favorite else ".")


@beta_tool
def list_removed_recipes() -> str:
    """List recipes the active user has removed, with their reasons. Check this to avoid
    suggesting dishes like ones they disliked."""
    removed = _load_profile()["removed_recipes"]
    if not removed:
        return "No removed recipes."
    return json.dumps([{k: e[k] for k in ("id", "name", "reason", "removed_on")} for e in removed], indent=2)


@beta_tool
@_locked
def restore_recipe(recipe_id: str) -> str:
    """Bring back a recipe the active user removed. Only when they ask for it.

    Args:
        recipe_id: The id from list_removed_recipes.
    """
    profile = _load_profile()
    entry = next((e for e in profile["removed_recipes"] if e["id"] == recipe_id), None)
    if entry is None:
        raise ValueError(f"No removed recipe with id {recipe_id!r}. Use list_removed_recipes.")
    profile["removed_recipes"] = [e for e in profile["removed_recipes"] if e is not entry]
    _save_profile(profile)
    return f"Restored {entry['name']}."


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
@_locked
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
    """Save a recipe to the shared library, usually one the user pasted from Serious Eats.
    Use the source's ingredient quantities; if it lists no nutrition, give realistic,
    conservative estimates.

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
    recipe_id = slugify(name)
    recipes = _load_recipes()

    def same(r: dict) -> bool:
        return (r["id"] == recipe_id or slugify(r["name"]) == recipe_id
                or bool(source_url and r.get("source_url") == source_url))

    prior = next((e for e in _load_profile()["removed_recipes"] if same(e)), None)
    if prior is not None:
        raise ValueError(
            f"This user removed {prior['name']!r} on {prior['removed_on']} "
            f"(reason: {prior['reason'] or 'not given'}). Don't re-add it; if they explicitly "
            f"want it back, use restore_recipe."
        )
    if (existing := next((r for r in recipes if same(r)), None)) is not None:
        raise ValueError(f"Already in the library as {existing['name']!r} (id {existing['id']!r}).")
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


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------

class PlannedMeal(TypedDict):
    meal_type: str
    recipe_id: str
    eaters: NotRequired[List[str]]
    portions: NotRequired[int]


class PlanDay(TypedDict):
    day: str
    meals: List[PlannedMeal]


def _fmt_qty(q: float) -> str:
    return str(int(q)) if float(q).is_integer() else f"{q:.2f}".rstrip("0")


def _fmt_target(value: float, target: Optional[int], unit: str) -> str:
    if not target:
        return f"{round(value)}{unit}"
    pct = round(100 * (value - target) / target)
    return f"{round(value)}{unit} (target {target}, {pct:+d}%)"


@beta_tool
def save_meal_plan(title: str, days: List[PlanDay], prep_notes: Optional[str] = None) -> str:
    """Validate and save a meal plan for the active user's household.

    Refuses to save if any member is scheduled to eat one of their allergens. Otherwise
    computes each member's daily nutrition against their own targets and a consolidated
    shopping list (batches are rounded up, so leftovers get reused across days), writes a
    Markdown file, and returns the totals so you can check them and revise if needed.

    Args:
        title: Plan title, e.g. "Week of Oct 5".
        days: One entry per day. Each meal names a recipe_id and, optionally, eaters: the
            member names eating it, when it's not the whole household (use this for
            per-person swaps, e.g. a separate dairy-free dish for one member). portions
            defaults to the number of eaters (the whole household size when eaters is
            omitted); set it higher only to cook extra on purpose.
        prep_notes: Optional batch-prep / leftover instructions to include in the file.
    """
    recipes = {r["id"]: r for r in _load_recipes()}
    profile = _load_profile()
    household = _household_size(profile)
    members = profile["members"]

    unknown = {m["recipe_id"] for d in days for m in d["meals"]} - recipes.keys()
    if unknown:
        raise ValueError(f"Unknown recipe ids: {sorted(unknown)}. Use search_recipes or add_recipe first.")

    # Safety first: every named eater must be allergen-free for every meal they eat.
    problems = []
    for d in days:
        for m in d["meals"]:
            r = recipes[m["recipe_id"]]
            for member in _resolve_members(profile, m.get("eaters")):
                if hits := _allergy_conflicts(r, member):
                    problems.append(f"{d['day']} {m['meal_type']}: {r['name']} contains "
                                    f"{', '.join(hits)} - {member['name']} is allergic")
    if problems:
        raise ValueError("Plan NOT saved - allergy conflicts:\n" + "\n".join(problems) +
                         "\nSwap these meals, or give the allergic member a separate safe dish "
                         "and list everyone else as eaters of the original.")

    portions_by_recipe = defaultdict(int)
    # day -> person -> nutrient totals; "Each person" stands in when no members are saved.
    nutrition = []
    for d in days:
        per_person = defaultdict(lambda: defaultdict(float))
        for m in d["meals"]:
            r = recipes[m["recipe_id"]]
            eaters = _resolve_members(profile, m.get("eaters"))
            default_portions = len(eaters) if "eaters" in m else household
            portions_by_recipe[r["id"]] += max(m.get("portions") or default_portions, 1)
            for person in ([e["name"] for e in eaters] if members else ["Each person"]):
                for k, v in r["nutrition_per_serving"].items():
                    per_person[person][k] += v
        nutrition.append((d["day"], per_person))

    batches = {rid: math.ceil(p / recipes[rid]["servings"]) for rid, p in portions_by_recipe.items()}
    staples = [s.lower() for s in profile["pantry_staples"]]
    shopping = defaultdict(float)  # (section, item, unit) -> qty
    for rid, n in batches.items():
        for ing in recipes[rid]["ingredients"]:
            if any(s in ing["item"].lower() for s in staples):
                continue
            shopping[(ing["section"], ing["item"].lower(), ing["unit"])] += ing["qty"] * n

    targets = {m["name"]: m for m in members}
    member_names = ", ".join(m["name"] for m in members)
    lines = [f"# {title}", "", f"Household size: {household}" + (f" ({member_names})" if members else ""), ""]
    for d, (day_name, per_person) in zip(days, nutrition):
        lines.append(f"## {day_name}")
        for m in d["meals"]:
            r = recipes[m["recipe_id"]]
            name = f"[{r['name']}]({r['source_url']})" if r.get("source_url") else r["name"]
            who = f" - for {', '.join(m['eaters'])}" if m.get("eaters") else ""
            lines.append(f"- **{m['meal_type'].title()}**: {name} ({r['prep_minutes']} min){who}")
        for person, t in per_person.items():
            target = targets.get(person, DEFAULT_MEMBER)
            lines.append(
                f"- _{person}: {_fmt_target(t['calories'], target['daily_calorie_target'], ' kcal')}, "
                f"{_fmt_target(t['protein_g'], target['daily_protein_target_g'], 'g protein')}, "
                f"{round(t['carbs_g'])}g carbs, {round(t['fat_g'])}g fat_"
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

    plans_dir = PLANS_DIR / active_user()
    plans_dir.mkdir(parents=True, exist_ok=True)
    path = plans_dir / f"{date.today().isoformat()}-{slugify(title) or 'plan'}.md"
    path.write_text("\n".join(lines), encoding="utf-8")

    return json.dumps({
        "saved_to": str(path),
        "daily_nutrition_by_person": {
            day: {
                person: {
                    "calories": round(t["calories"]), "protein_g": round(t["protein_g"]),
                    "calorie_target": targets.get(person, DEFAULT_MEMBER)["daily_calorie_target"],
                    "protein_target_g": targets.get(person, DEFAULT_MEMBER)["daily_protein_target_g"],
                }
                for person, t in per_person.items()
            }
            for day, per_person in nutrition
        },
        "batches_to_cook": batches,
        "shopping_list_item_count": len(shopping),
    }, indent=2)


ALL_TOOLS = [get_profile, update_profile, set_member, remove_member, search_recipes, get_recipe,
             add_recipe, set_favorite, remove_recipe, list_removed_recipes, restore_recipe,
             save_meal_plan]
