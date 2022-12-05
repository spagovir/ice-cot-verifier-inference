# %%
from __future__ import annotations

from ice.recipe import recipe
import numpy as np
import numpy.random as rand
import numpy.typing as npt
import math
from abc import ABC, abstractmethod

from typing import Generic, TypeVar, Tuple, Awaitable, Callable, List

# %%
RT = TypeVar('RT')
IT = TypeVar('IT')
OT = TypeVar('OT')
UpdateRule = Callable[[npt.NDArray[np.float32], int, float, float], npt.NDArray[np.float32]]
Agent = Callable[[IT], Awaitable[Tuple[npt.NDArray[np.float32], List[OT]]]]

class SearchTreeNode(ABC, Generic[IT, OT, RT]):
    @abstractmethod
    async def rolloutAndUpdate(self, rule : UpdateRule, agent: Agent[IT, OT]) -> Tuple[float, float, RT, SearchTreeNode[IT,OT,RT]]:
        pass

class Leaf(SearchTreeNode[IT, OT, RT]):
    def __init__(self, result : RT) -> None:
        self.result = result
    async def rolloutAndUpdate(self, rule : UpdateRule, agent: Agent[IT,OT]) -> Tuple[float, float, RT, SearchTreeNode[IT,OT,RT]]:
        return (0.0, 0.0, self.result, self)

class ScoreNode(SearchTreeNode[IT, OT, RT]):
    def __init__(self, score : float, child : SearchTreeNode[IT, OT, RT]):
        self.score = score
        self.child = child
    async def rolloutAndUpdate(self, rule : UpdateRule, agent : Agent[IT, OT]) -> Tuple[float, float, RT, SearchTreeNode[IT,OT,RT]]:
        score, logprob, result, update = await self.child.rolloutAndUpdate(rule, agent)
        return (score + self.score, logprob, result, ScoreNode(self.score, update))

class ChoiceNode(SearchTreeNode[IT, OT, RT]):
    def __init__(self, scores : npt.NDArray[np.float32], logprobs : npt.NDArray[np.float32], children : List[SearchTreeNode[IT,OT,RT]]):
        self.scores = scores
        self.logprobs = logprobs
        self.children = children
    async def rolloutAndUpdate(self, rule : UpdateRule, agent : Agent[IT, OT]) -> Tuple[float, float, RT, SearchTreeNode[IT,OT,RT]]:
        idx = rand.choice(len(self.logprobs), None, p = np.exp(self.logprobs)/np.sum(np.exp(self.logprobs)))
        score, logprob, result, update = await self.children[idx].rolloutAndUpdate(rule, agent)
        score += self.scores[idx]
        logprob += self.logprobs[idx]
        nlogprobs = rule(self.logprobs, idx, score, logprob)
        nchildren = self.children.copy()
        nchildren[idx] = update
        return (score, logprob, result, ChoiceNode(self.scores, nlogprobs, nchildren))

Program = Callable[[OT], SearchTreeNode[IT,OT,RT]]
class DelayedNode(SearchTreeNode[IT, OT, RT]):
    def __init__(self, prompt : IT, program : Program[OT, IT, RT]):
        self.prompt = prompt
        self.program = program
    async def rolloutAndUpdate(self, rule: UpdateRule, agent : Agent[IT, OT]) -> Tuple[float, float, RT, SearchTreeNode[IT,OT,RT]]:
        logits, outputs = await agent(self.prompt)
        children = [self.program(output) for output in outputs]
        return await ChoiceNode(logits, logits, children).rolloutAndUpdate(rule, agent)

"""
# Markov Chain Rule

Update rule designed so that the logprobs should converge to scores such that sampling from the search tree leads follows the correct posterior distribution.

Let z be the choice of child, y be the choice of subsequent children, and z the event we're conditioning/scoring on. 
Let q_z, q_y|z, q_x|yz denote the known values for p(z), p(y|z), and p(x|y,z). 
(ie, q_z is carried by the score fields of this choice node, q_y|z the product of the score fields of downstream choice nodes,
and q_x is the product of the score fields of downstream score nodes)
Let p_z denote the probabilities associated with each child. 
We desire that p_z converges to p(x|z)p(z) = sum_y p(x|y,z)p(y|z)p(z) = q_z(sum_y q_y|z q_x|yz)
so that when normalized the p_z represent p(z|x).
Denote this desired value as q_z|x

Let 1_z denote the vector where the z'th entry is 1 and the others 0. 
We note that sampling y,z using p_z and p_y|z, 
$E[1_z q_z q_y|z q_x|yz / (p_z p_y|z)] 
= sum_z p_z 1_z q_z / p_z E[q_y|z q_x|yz / p_y|z]
= sum_z 1_z q_z sum_y p_y|z q_y|z q_x|yz /p_y|z 
= sum_z 1_z q_z|x
= (q_0|x, q_1|x, ... , q_n|x)$
which is the desired value we want the probs to converge to. 

We thus calculate our tree by sampling the above value and averaging 
it into our probability weights. 
"""
def mcrule(beta : float) -> UpdateRule:
    def mcruleb(logprobs : npt.NDArray[np.float32], idx : int, score : float, logprob : float) -> npt.NDArray[np.float32]:
        nonlocal beta
        nweights = np.zeros_like(logprobs)
        nweights[idx] = math.exp(score - logprob)
        return np.log(beta * np.exp(logprobs) + (1-beta) * nweights)
    return mcruleb

EOT = '<|endoftext|>'

def cleanUnicode(string : str) -> str:
    if string.startswith('bytes:'):
        return string[6:].encode('utf-8').decode('unicode_escape')
    return string
def checkRepeatWhitespace(string : str) -> bool:
    return (len(string) > 3) and string[3:].isspace()
def generateStepR(prompt : str, stop : str | None, generated : str, prev_token : str) -> SearchTreeNode[str,str,str]:
    generated = generated + cleanUnicode(prev_token)
    if prev_token == EOT or (stop is not None and prev_token == stop) or checkRepeatWhitespace(generated):
        return Leaf(generated)
    return DelayedNode(prompt + generated, generate(prompt, stop, generated))
def generate(prompt : str, stop : str | None, generated : str = "") -> Program[str, str, str]:
    return (lambda prev_token: generateStepR(prompt, stop, generated, prev_token))
def generateTree(prompt :str, stop : str | None) -> SearchTreeNode[str, str, str]:
    return DelayedNode(prompt, generate(prompt, stop, ""))

async def icePredictAgent(prompt : str) -> Tuple[npt.NDArray[np.float32], List[str]]:
    probs = await recipe.agent().predict(context = prompt)
    outputs = list(probs)
    logprobs = np.log(np.fromiter((probs[output] for output in outputs), np.float32, len(probs)))
    return (logprobs, outputs)



    