from ice.recipe import recipe
from ice.agent import Agent
from ice.trace import TracedABC
from ice.apis.openai import openai_complete
from typing import List, Tuple
import programs
import search_tree
prompt = """
In his latest speech, Boris Johnson quoted the opening line of the Odyssey: 

"""

'''
Scores generated text based on whether it's a Pagliacci joke 
but still returns the text.
'''
def pagliacci(generated : str) -> search_tree.SearchTreeNode[str]:
  return search_tree.ScoreNode(0 if "Pagliacci" in generated else search_tree.LOG_0, search_tree.Leaf(generated))

async def complete_test():
  answer = await recipe.agent().complete(prompt = prompt)
  return await recipe.agent().predict(context = prompt)

'''
async def test_generate_tree() -> List[Tuple[str,float]]:
  tree = search_tree.generateTree(prompt, '.')
  results : List[Tuple[str,float]] = []
  for _ in range(2):
    score, _, result, tree = await tree.rolloutAndUpdate(search_tree.mcrule(), search_tree.icePredictAgent)
    results.append((result,score))
  return results
'''

async def test_openai_logprobs(): 
  return await openai_complete(prompt = prompt, logprobs = 5)

async def test_complete_tree() -> List[str]:
  tree = search_tree.CompleteNode(prompt, "", lambda text : search_tree.Leaf(text))
  results : List[str] = []
  for _ in range(40):
    _, _, result, tree = await tree.rolloutAndUpdate(search_tree.mcrule, search_tree.WrappedICEAgent())
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


async def test_qa() -> list[str]:
  tree = programs.qa_program("This man wrote that 'All of Gaul was divided into three parts.' Who was this man?", lambda x : search_tree.Leaf(x))
  results : List[str] = []
  for _ in range(40):
    _, _, result, tree = await tree.rolloutAndUpdate(search_tree.mcrule, search_tree.WrappedICEAgent())
    results.append(result)
  return results
  
async def test_qa_verifier_attempt() -> list[str]:
  tree = programs.cot_verifier_attempt("Robert Nozick makes 70k a year pre-tax. The first 10k dollars of income are exempt from tax, the marginal tax rate is 20%% between $10k and $50k, and 35%% on every dollar of income after $50k. What is Robert Nozick's post-tax income?")
  results : List[str] = []
  for _ in range(40):
    _, _, result, tree = await tree.rolloutAndUpdate(search_tree.mcrule, search_tree.WrappedICEAgent())
    results.append(result)
  return results

recipe.main(test_qa_verifier_attempt)
