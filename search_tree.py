# %%
from __future__ import annotations
import numpy as np
import numpy.random as rand
import numpy.typing as npt
import math
from abc import ABC, abstractmethod
from collections import deque

from typing import Generic, TypeVar, Tuple, Callable, List, Dict

# %%
RT = TypeVar('RT')
IT = TypeVar('IT')
OT = TypeVar('OT')
UpdateRule = Callable[[npt.NDArray[np.float32], int, float, float], npt.NDArray[np.float32]]

"""
To represent a wrapper around ice.Agent and the _openai complete method call 
"""
class Agent(ABC):
    @abstractmethod
    async def predict(self, prompt : str) -> Tuple[npt.NDArray[np.float32], List[str]]:
        pass
    @abstractmethod
    async def complete(self, prompt : str, stop : List[str]) -> Tuple[str, List[Tuple[npt.NDArray[np.float32], List[str], int]]]:
        pass
    @abstractmethod
    async def classify(self, prompt : str, choices : Tuple[str, ...]) -> Dict[str,float]:
        pass

class SearchTreeNode(ABC, Generic[RT]):
    """
    seq represents a sequence of choices used to force rollouts to use an already explored path through the search tree.
    predict/complete nodes currently ignore seq.
    seq can be a partial path. 
    """
    @abstractmethod
    async def rolloutAndUpdate(self, rule : UpdateRule, agent: Agent, seq : deque[int] = deque([])) -> Tuple[float, float, RT, SearchTreeNode[RT]]:
        pass

class Leaf(SearchTreeNode[RT]):
    def __init__(self, result : RT) -> None:
        self.result = result
    async def rolloutAndUpdate(self, rule : UpdateRule, agent: Agent, seq : deque[int] = deque([])) -> Tuple[float, float, RT, SearchTreeNode[RT]]:
        return (0.0, 0.0, self.result, self)

class ScoreNode(SearchTreeNode[RT]):
    def __init__(self, score : float, child : SearchTreeNode[RT]):
        self.score = score
        self.child = child
    async def rolloutAndUpdate(self, rule : UpdateRule, agent : Agent, seq : deque[int] = deque([])) -> Tuple[float, float, RT, SearchTreeNode[RT]]:
        score, logprob, result, update = await self.child.rolloutAndUpdate(rule, agent, seq)
        return (score + self.score, logprob, result, ScoreNode(self.score, update))

class ChoiceNode(SearchTreeNode[RT]):
    def __init__(self, scores : npt.NDArray[np.float32], logprobs : npt.NDArray[np.float32], children : List[SearchTreeNode[RT]]):
        self.scores = scores
        self.logprobs = logprobs
        self.children = children
    async def rolloutAndUpdate(self, rule : UpdateRule, agent : Agent, seq : deque[int] = deque([]))-> Tuple[float, float, RT, SearchTreeNode[RT]]:
        if seq:
            idx = seq.popleft()
        else: 
            idx : int = rand.choice(len(self.logprobs), None, p = np.exp(self.logprobs)/np.sum(np.exp(self.logprobs)))
        score, logprob, result, update = await self.children[idx].rolloutAndUpdate(rule, agent, seq)
        score += self.scores[idx]
        logprob += self.logprobs[idx]
        nlogprobs = rule(self.logprobs, idx, score, logprob)
        nchildren = self.children.copy()
        nchildren[idx] = update
        return (score, logprob, result, ChoiceNode(self.scores, nlogprobs, nchildren))

Program = Callable[[IT], SearchTreeNode[RT]]
class PredictNode(SearchTreeNode[RT]):
    def __init__(self, prompt : str, program : Program[str, RT]):
        self.prompt = prompt
        self.program = program
    async def rolloutAndUpdate(self, rule: UpdateRule, agent : Agent, seq : deque[int] = deque([])) -> Tuple[float, float, RT, SearchTreeNode[RT]]:
        logits, outputs = await formatPredictAgent(self.prompt, agent)
        children = [self.program(output) for output in outputs]
        return await ChoiceNode(logits, logits, children).rolloutAndUpdate(rule, agent)

class CompleteNode(SearchTreeNode[RT]):
    def __init__(self, prompt : str, prefix : str,  program : Program[str, RT], stop : List[str] = []):
        self.prompt = prompt
        self.program = program
        self.stop = stop
        self.prefix = prefix
    async def rolloutAndUpdate(self, rule: UpdateRule, agent: Agent, seq : deque[int] = deque([])) -> Tuple[float, float, RT, SearchTreeNode[RT]]:
        result, steps = await agent.complete(self.prompt + self.prefix, self.stop)
        prefix = self.prefix
        nodes : List[SearchTreeNode[RT]]= []
        prev : Tuple[ChoiceNode[RT], int] | None = None
        seq = deque([])
        for step in steps:
            logprobs, child_strs, child = step
            seq.append(child)
            children : List[SearchTreeNode[RT]] = [CompleteNode(self.prompt, prefix + child_str, self.program, self.stop) for child_str in child_strs]
            this_node = ChoiceNode[RT](logprobs, logprobs, children)
            if prev is not None: 
                prev_node, prev_child_idx = prev
                prev_node.children[prev_child_idx] = this_node
            prev = (this_node, child)
            nodes.append(this_node)
            prefix = prefix + child_strs[child]
        if prev is not None:
            prev_node, prev_child_idx = prev
            prev_node.children[prev_child_idx] = self.program(self.prefix + result)
            return await nodes[0].rolloutAndUpdate(rule, agent, seq)
        else:
            return await self.program(self.prefix + result).rolloutAndUpdate(rule, agent, seq)
        


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
def generateStepR(prompt : str, stop : str | None, generated : str, prev_token : str) -> SearchTreeNode[str]:
    generated = generated + cleanUnicode(prev_token)
    if prev_token == EOT or (stop is not None and prev_token == stop) or checkRepeatWhitespace(generated):
        return Leaf(generated)
    return PredictNode(prompt + generated, generate(prompt, stop, generated))
def generate(prompt : str, stop : str | None, generated : str = "") -> Program[str, str]:
    return (lambda prev_token: generateStepR(prompt, stop, generated, prev_token))
def generateTree(prompt :str, stop : str | None) -> SearchTreeNode[str]:
    return PredictNode(prompt, generate(prompt, stop, ""))

async def formatPredictAgent(prompt : str, agent : Agent) -> Tuple[npt.NDArray[np.float32], List[str]]:
    probs = await agent.predict(context = prompt)
    outputs = list(probs)
    logprobs = np.log(np.fromiter((probs[output] for output in outputs), np.float32, len(probs)))
    return (logprobs, outputs)



    