from __future__ import annotations
from search_tree import SearchTreeNode, CompleteNode, ClassifyNode, Leaf, Program, ScoreNode, LOG_0
# from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeVar, Callable
import math

path = Path(__file__).parent
qa_prefix : str = (path / "qa_examples.txt").open('r').read()
verify_prefix : str = (path / "verifier_examples.txt").open('r').read()

IT = TypeVar("IT")
OT = TypeVar("OT") 
RT = TypeVar("RT")

def condition_on_yes(retNode : SearchTreeNode[RT]) -> Program[dict[str,float],RT]:
    def program(cats : dict[str,float]) -> SearchTreeNode[RT]:
        nonlocal retNode
        return ScoreNode(math.log(cats[' yes']) if cats[' yes'] != 0 else LOG_0, retNode)
    return program

def make_qa_prompt(question : str) -> str: 
    return qa_prefix + "Q: " + question + "\nA: Let's think step by step. "

def qa_program(question : str, continuation : Callable[[str], SearchTreeNode[RT]]) -> SearchTreeNode[RT]:
    return CompleteNode(make_qa_prompt(question), "", lambda x : continuation(x))

# We also have verifier do chain-of-thought reasoning. 
def verifier_program(question : str, continuation : Callable[[str], SearchTreeNode[RT]]) -> Program[str, RT]:
    def make_verification_prompt(question : str, answer : str) -> str:
        return verify_prefix \
            + f"""Consider the following exchange:

            > Teacher: {question}
            > Student: {answer}

            Q: Is the student's answer correct? Answer choices: yes, no.

            A: Let's examine each step of their reasoning. 
            """
    def verify_answer(answer : str) -> SearchTreeNode[RT]:
        prompt = make_verification_prompt(question, answer)
        def process_verifier_response(response : str) -> SearchTreeNode[RT]:
            nonlocal prompt
            nonlocal answer
            if "he answer is " in response:
                idx = response.rindex('he answer is ')
                return ScoreNode(0 if 'yes' in response[idx:] else LOG_0, continuation(answer))
            else:
                nonlocal prompt
                return ClassifyNode(prompt + response + ' The answer is', (' yes', ' no'), condition_on_yes(continuation(answer)))
        return CompleteNode(prompt, "", process_verifier_response) 
    return verify_answer
def extract(prompt : str) -> Program[str, str]:
    def extract_answer(answer : str) -> SearchTreeNode[str]:
        nonlocal prompt
        if 'he answer is ' in answer:
            return Leaf(answer[answer.rindex('he answer is ') + 13:].strip())
        else:
            return CompleteNode(prompt + answer + ' The answer is ', '', lambda s : Leaf(s.strip()))
    return extract_answer
def force_answer(prompt : str, continuation : Program[str, str]) -> Program[str, str]:
    def force(answer : str) -> SearchTreeNode[str]:
        nonlocal prompt
        if 'he answer is ' in answer:
            return continuation(answer)
        else:
            return CompleteNode(prompt, answer + ' The answer is ', continuation)
    return force

def cot_verifier_attempt(question : str) -> SearchTreeNode[str]:
    return qa_program(question
    , force_answer(make_qa_prompt(question), verifier_program(question, extract(make_qa_prompt(question)))))

"""
class ProgramRMonad(ABC, Generic[IT]):
    @abstractmethod
    def rBind(self, program : search_tree.Program[IT,RT]) -> search_tree.SearchTreeNode[RT]:
        pass
    def bind(self, mf : Callable[[IT], ProgramRMonad[RT]]) -> ProgramRMonad[RT]:

    def rBindEmpty(self, ret : search_tree.SearchTreeNode[RT]) -> search_tree.SearchTreeNode[RT]:
        return self.rBind(lambda x : ret)
    def fmap(self, f : Callable[[IT],RT]) -> search_tree.SearchTreeNode[RT]:
        return self.rBind(lambda x : search_tree.Leaf(f(x)))
class QA_Generate(ProgramRMonad[str]):
    def __init__(self, question : str): 
        self.question = question
    def __init__ 
"""