from ice.recipe import recipe
from ice.agent import Agent
from ice.trace import TracedABC
from ice.apis.openai import openai_complete
from typing import List
import search_tree
prompt = "A man goes to the doctor, says he is very depressed. The doctor says,"

'''
Scores generated text based on whether it's a Pagliacci joke 
but still returns the text.
'''
def pagliacci(generated : str) -> search_tree.SearchTreeNode[str]:
  return search_tree.ScoreNode(0 if "Pagliacci" in generated else search_tree.LOG_0, search_tree.Leaf(generated))

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

async def test_complete_tree() -> List[str]:
  tree = search_tree.CompleteNode(prompt, "", pagliacci)
  results : List[str] = []
  for _ in range(100):
    _, _, result, tree = await tree.rolloutAndUpdate(search_tree.mcrule(0.9), search_tree.WrappedICEAgent())
    results.append(result)
  return results

class TestAgent(TracedABC):
  def __init__(self, prefix : str) -> None:
    self.prefix = prefix
  async def test_trace(self, instr : str) -> str:
    return self.prefix + instr
  
async def test_test_agent():
  test_agent = TestAgent("Hello: ")
  return await test_agent.test_trace('World!')

recipe.main(test_complete_tree)
