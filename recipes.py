import pandas as pd
import random
import sys
import ast
from colorama import init, Fore, Style

init(autoreset=True)

recipes_df = pd.read_csv("recipes.csv")
nutrition_df = pd.read_csv("nutrition_facts_with_clusters.csv")
nutrition_df.rename(columns=lambda x: "Ingredient" if x.strip().lower() == "ingredient" else x, inplace=True)

NUTRIENT_WEIGHTS = {
    "Protein": 3,
    "Fiber, total dietary": 2,
    "Vitamin A, RAE": 1,
    "Calcium, Ca": 1,
    "Iron, Fe": 1
}

def generate_smart_menu():
    def top_ingredients_by_nutrient(nutrient, top_n=20):
        df = nutrition_df[["Ingredient", nutrient]].dropna()
        return df.sort_values(by=nutrient, ascending=False).head(top_n)["Ingredient"].tolist()

    breakfast = random.sample(top_ingredients_by_nutrient("Fiber, total dietary"), 1) + \
                random.sample(top_ingredients_by_nutrient("Calcium, Ca"), 1) + \
                random.sample(top_ingredients_by_nutrient("Vitamin A, RAE"), 1)

    lunch = random.sample(top_ingredients_by_nutrient("Protein"), 1) + \
            random.sample(top_ingredients_by_nutrient("Iron, Fe"), 1) + \
            random.sample(top_ingredients_by_nutrient("Fiber, total dietary"), 1)

    dinner = random.sample(top_ingredients_by_nutrient("Protein"), 1) + \
             random.sample(top_ingredients_by_nutrient("Vitamin A, RAE"), 1) + \
             random.sample(top_ingredients_by_nutrient("Calcium, Ca"), 1)

    return {
        "Завтрак": breakfast,
        "Обед": lunch,
        "Ужин": dinner
    }

# 🔧 FIX: без деления на 100, работаем с %DV напрямую
def rate_ingredient(ingredient):
    row = nutrition_df[nutrition_df['Ingredient'].str.lower() == ingredient.lower()]
    if row.empty:
        return 0.0
    score = 0.0
    for nutrient, weight in NUTRIENT_WEIGHTS.items():
        val = row[nutrient].values[0] if nutrient in row else 0
        score += val * weight  # <--- основное исправление
    return round(score, 2)

def find_similar_recipes(ingredients, top_n=3):
    def count_matches(row):
        return sum(any(ing in item for item in row) for ing in ingredients)

    filtered = recipes_df.copy()
    filtered["ingredients"] = filtered["ingredients"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    filtered["match_score"] = filtered["ingredients"].apply(count_matches)
    top_matches = filtered[filtered["match_score"] > 0]
    top_matches = top_matches.sort_values(by="match_score", ascending=False)
    return top_matches.head(top_n)

def list_all_ingredients():
    ingredients = nutrition_df['Ingredient'].dropna().unique()
    print(Fore.CYAN + "📋 Доступные ингредиенты:")
    for ingr in sorted(ingredients):
        print("-", ingr)

if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) == 1 and args[0] == "--menu":
        menu = generate_smart_menu()
        for meal, items in menu.items():
            print(Fore.BLUE + f"\n🍽️ {meal}:")
            for ingr in items:
                score = rate_ingredient(ingr)
                print(Fore.GREEN + f"- {ingr}" + Fore.YELLOW + f" (рейтинг пользы: {score})")

            matches = find_similar_recipes(items)
            if matches.empty:
                print(Fore.RED + "  🥲 Нет подходящих рецептов\n")
            else:
                for i, (_, row) in enumerate(matches.iterrows(), 1):
                    print(Fore.MAGENTA + f"  {i}. {row['title']} " + Fore.YELLOW + f"(оценка: {row['rating']})")
                    print("     " + Fore.GREEN + "Ингредиенты:", ", ".join(row['ingredients']))
                    print("     " + Fore.CYAN + "URL:", row['url'])

    elif len(args) >= 1 and args[0] == "--rate":
        input_str = ' '.join(args[1:])
        ingredients = [i.strip().lower() for i in input_str.split(",") if i.strip()]

        if not ingredients:
            print(Fore.RED + "⚠️ Нет ингредиентов для анализа")
            sys.exit()

        total_score = 0.0
        for ingr in ingredients:
            score = rate_ingredient(ingr)
            print(Fore.GREEN + f"- {ingr}:" + Fore.YELLOW + f" {score}")
            total_score += score

        avg = total_score / len(ingredients)

        # ✅ FIX: изменена шкала
        if avg > 150:
            rating = "great"
        elif avg > 75:
            rating = "so-so"
        else:
            rating = "bad"

        print(Fore.CYAN + f"\n⭐ Общий рейтинг: {rating} ({round(avg, 2)})")

    elif len(args) == 1 and args[0] == "--list":
        list_all_ingredients()

    else:
        print(Fore.CYAN + """\n📘 Использование:
  python3 recipes.py --menu
  python3 recipes.py --rate "ингредиент1, ингредиент2"
  python3 recipes.py --list
""")
