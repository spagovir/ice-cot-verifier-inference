import programs
import search_tree
from math import exp
from ice.recipe import recipe

default_question = "Robert Nozick makes 70k a year pre-tax. The first 10k dollars of income are exempt from tax, the marginal tax rate is 20%% between $10k and $50k, and 35%% on every dollar of income after $50k. What is Robert Nozick's post-tax income?"

# We try to marginalize over different chains of thoughts behind an answer, 
# trying to choose the answer with the highest _posterior probability_ 
# P(answer | chain-of-thought is correct) = Sum_{chains of thought} P(chain of thought, answer) P_verifier(correct | chain of thought, answer)

# We sample different chains of thoughts and accumulate the probabilities for each answer;
# we estimate the posterior probability by weighting each result by the posterior probability / the chance its sampled.
# We note that initially this is equal to the p(correct) provided by the verifier
# but we our sampling strategy should converge to sampling each answer with probability proportional 
# to its actual posterior probability. 
async def self_consistency_qa(question : str = default_question, attempts : int = 20) -> str:
    tree = search_tree.TracedSearchTreeWrapper(programs.cot_verifier_attempt(question))
    answers : dict[str,float]= {}
    for _ in range(attempts):
        score, logprob, answer, tree = await tree.rolloutAndUpdate(search_tree.mcrule, search_tree.WrappedICEAgent())
        if answer not in answers:
            answers[answer] = 0
        answers[answer] += exp(score - logprob)
    return max(answers, key = answers.get) # type: ignore

if __name__ == '__main__':
    recipe.main(self_consistency_qa)