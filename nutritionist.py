import pandas as pd
import numpy as np
import joblib
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 📦 Загрузка данных и моделей
nutrition_df = pd.read_csv("nutrition_facts_with_clusters.csv")
scaler = joblib.load("scaler.joblib")
kmeans = joblib.load("kmeans.joblib")

# 🧪 Показать кластер по ингредиенту
def show_cluster(ingredient):
    ingredient_col = next((col for col in nutrition_df.columns if col.lower() == "ingredient"), None)
    cluster_col = next((col for col in nutrition_df.columns if col.lower() == "cluster"), None)

    if not ingredient_col or not cluster_col:
        print("🧲 Не найдены нужные колонки ('Ingredient' и 'Cluster')")
        return

    row = nutrition_df[nutrition_df[ingredient_col].str.lower() == ingredient.lower()]
    if row.empty:
        print(f"🧲 Нет данных по ингредиенту: {ingredient}")
        return

    cluster = row[cluster_col].values[0]
    print(f"🧬 Ингредиент: {ingredient.lower()}")
    print(f"Кластер: {cluster}")

    top_nutrients = row.drop([ingredient_col, cluster_col], axis=1).T.sort_values(by=row.index[0], ascending=False).head(5)
    print("🔝 Топ-5 нутриентов:")
    for name, val in top_nutrients.itertuples():
        print(f"- {name}: {val:.1f}% от дневной нормы")

# 📋 Показать все ингредиенты по кластерам
def list_ingredients_by_cluster():
    ingredient_col = next((col for col in nutrition_df.columns if col.lower() == "ingredient"), None)
    cluster_col = next((col for col in nutrition_df.columns if col.lower() == "cluster"), None)

    if not ingredient_col or not cluster_col:
        print("🧲 Не найдены нужные колонки ('Ingredient' и 'Cluster')")
        return

    print("📊 Ингредиенты по кластерам:")
    for cluster in sorted(nutrition_df[cluster_col].unique()):
        items = nutrition_df[nutrition_df[cluster_col] == cluster][ingredient_col].tolist()
        print(f"\n🔹 Кластер {cluster} ({len(items)} ингредиентов):")
        for item in items:
            print("-", item)

# 📊 Сравнение N ингредиентов по нутриентам
def compare_multiple_ingredients(ingredients):
    ingredient_col = next((col for col in nutrition_df.columns if col.lower() == "ingredient"), None)
    cluster_col = next((col for col in nutrition_df.columns if col.lower() == "cluster"), None)

    data = []
    names = []

    for ingr in ingredients:
        row = nutrition_df[nutrition_df[ingredient_col].str.lower() == ingr.lower()]
        if row.empty:
            print(f"🧲 Нет данных по ингредиенту: {ingr}")
            continue
        names.append(ingr.lower())
        data.append(row.drop([ingredient_col, cluster_col], axis=1).values.flatten())

    if len(data) < 2:
        print("🧲 Не хватает данных для сравнения")
        return

    df = pd.DataFrame(data, index=names, columns=nutrition_df.columns.drop([ingredient_col, cluster_col]))
    print(f"🔬 Сравнение ингредиентов: {', '.join(names)}")

    for nut in df.columns:
        values = ", ".join([f"{ingr}: {df.loc[ingr, nut]:.1f}%" for ingr in df.index if df.loc[ingr, nut] > 0])
        if values:
            print(f"📌 {nut} → {values}")

# 🔝 Топ-N ингредиентов по нутриенту
def show_top_by_nutrient(nutrient_name, top_n=10):
    nutrient_col = next((col for col in nutrition_df.columns if nutrient_name.lower() in col.lower()), None)
    ingredient_col = next((col for col in nutrition_df.columns if col.lower() == "ingredient"), None)

    if not nutrient_col or not ingredient_col:
        print("🧲 Не найдены нужные колонки ('Ingredient' или указанного нутриента)")
        return

    top = nutrition_df[[ingredient_col, nutrient_col]].sort_values(by=nutrient_col, ascending=False).head(top_n)
    print(f"🏆 Топ-{top_n} по нутриенту: {nutrient_col}")
    for i, row in top.iterrows():
        print(f"- {row[ingredient_col]}: {row[nutrient_col]:.1f}% от дневной нормы")

# 🔮 Предсказание кластера по ингредиенту
def predict_cluster_by_ingredient(ingredient_name):
    ingredient_col = next((col for col in nutrition_df.columns if col.lower() == "ingredient"), None)
    cluster_col = next((col for col in nutrition_df.columns if col.lower() == "cluster"), None)

    row = nutrition_df[nutrition_df[ingredient_col].str.lower() == ingredient_name.lower()]
    if row.empty:
        print(f"🧲 Нет данных по ингредиенту: {ingredient_name}")
        return

    features = row.drop([ingredient_col, cluster_col], axis=1)
    try:
        features_scaled = scaler.transform(features)
        cluster = kmeans.predict(features_scaled)[0]
        print(f"🔮 Предсказанный кластер для {ingredient_name.lower()}: {cluster}")
    except Exception as e:
        print(f"❌ Ошибка предсказания: {e}")

# 🧈 Справка
HELP_TEXT = """
👩‍🔬 Nutritionist CLI:

Команды:
- python3 nutritionist.py pasta                           → кластер и топ-5 нутриентов
- python3 nutritionist.py milk,honey,jam                  → сравнение нескольких ингредиентов
- python3 nutritionist.py --list                          → ингредиенты по кластерам
- python3 nutritionist.py --top protein 5                 → топ-5 по нутриенту

"""

if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) == 1 and ',' in args[0]:
        args = [arg.strip() for arg in args[0].split(',')]

    if len(args) == 1:
        if args[0] == "--list":
            list_ingredients_by_cluster()
        elif args[0] in ["--help", "-h"]:
            print(HELP_TEXT)
        elif args[0].lower().endswith(".csv"):
            print("📂 Работа с CSV пока не реализована")
        else:
            show_cluster(args[0])

    elif len(args) >= 2:
        if args[0] == "--top":
            try:
                show_top_by_nutrient(args[1], int(args[2]) if len(args) > 2 else 10)
            except:
                print("⚠️ Неверный формат команды для --top. Пример: --top protein 5")
        elif args[0] == "--predict":
            predict_cluster_by_ingredient(args[1])
        elif args[0] == "--predict-rating":
            try:
                from recipes import classify_ingredients
                prediction = classify_ingredients(args[1])
                print(f"🍽 Прогноз качества блюда: {prediction}")
            except Exception as e:
                print(f"❌ Ошибка предсказания рейтинга: {e}")
        else:
            compare_multiple_ingredients(args)

    else:
        print("⚠️ Использование: python3 nutritionist.py [ingredient] | --list | [ingr1, ingr2, ...] | --top [nutrient] | --predict [ingredient] | --predict-rating [ingr1,ingr2,...] | --help")
