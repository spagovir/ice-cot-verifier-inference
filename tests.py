from ice.recipe import recipe
from ice.apis.openai import openai_complete
from typing import List
import search_tree
prompt = "A man goes to the doctor, says he is very depressed. The doctor says,"
async def complete_test():
  answer = await recipe.agent().complete(prompt = prompt)
  return await recipe.agent().predict(context = prompt)

async def test_generate_tree() -> List[str]:
  tree = search_tree.generateTree(prompt, '.')
  results : List[str] = []
  for _ in range(2):
    _, _, result, tree = await tree.rolloutAndUpdate(search_tree.mcrule(0.9), search_tree.icePredictAgent)
    results.append(result)
  return results

async def test_openai_logprobs(): 
  return await openai_complete(prompt = prompt, logprobs = 5)

recipe.main(test_openai_logprobs)
