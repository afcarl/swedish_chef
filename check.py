from tqdm import tqdm
import lzma
import myio.myio as myio
import sys

index = int(sys.argv[1])
table = myio.load_pickle("tmp/recipe_table")
recipes = table.get_recipes()
recipe = recipes[index]
ingredients = recipe.get_ingredients_list()
text = recipe.get_text()
print("Ingredients: " + str(ingredients))
print("Recipe text: " + str(text))

#bitvec = []
#for rec in table:
#    if rec.get_text() != "" and len(rec.get_ingredients_list()) != 0:
#        bitvec.append(1)
#    else:
#        bitvec.append(0)
#coverage = sum(bitvec) / float(len(bitvec))
#coverage *= 100.0
#print("Coverage (% of recipes with ingredients and texts): " + str(coverage) + "%")
#
#compresseds = []
#all_ingredients = table.get_all_ingredients()
#for ing in tqdm(all_ingredients):
#    fv = table.ingredient_to_feature_vector(ing)
#    fv_str = ""
#    for i in fv:
#        fv_str += str(i)
#    compressed = lzma.compress(fv_str.encode("utf-8"))
#    compresseds.append(compressed)
#
#lens = [len(c) for c in compresseds]
#longest = max(lens)
#print("Longest: " + str(longest))











