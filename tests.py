from ice.recipe import recipe
from typing import List
import search_tree

async def complete_test():
  prompt = "A man goes to the doctor, says he is very depressed. The doctor says,"
  answer = await recipe.agent().complete(prompt = prompt)
  return await recipe.agent().predict(context = prompt)

async def test_generate_tree() -> List[str]:
  prompt = "A man goes to the doctor, says he is very depressed. The doctor says,"
  tree = search_tree.generateTree(prompt, '.')
  results : List[str] = []
  for _ in range(2):
    _, _, result, tree = await tree.rolloutAndUpdate(search_tree.mcrule(0.9), search_tree.icePredictAgent)
    results.append(result)
  return results

recipe.main(test_generate_tree)
