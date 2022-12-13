import programs
import search_tree
from math import exp
from ice.recipe import recipe

# Questions. 

amc8_2022_p7 = "When the World Wide Web first became popular in the $1990$s, download speeds reached a maximum of about 56 kilobits per second. Approximately how many minutes would the download of a 4.2-megabyte song have taken at that speed? (Note that there are 8000 kilobits in a megabyte.) "

# This ones also impossible. I think it does better when there's lots of steps but the first step is really easy. 
"How many keystrokes are needed to type the numbers from 1 to 500? Answer Choices: (a) 1156 (b) 1392 (c) 1480 (d) 1562 (e) 178"

# It just can't do the one below: 
"John and Adam review drivers license applications. Together, they can review 1000 applications per hour. If John can review 500 more applications per hour than Adam, how many hours would it take Adam to review the number of applications that John can review in 2 hours?"

# It finds this one pretty easy (usually gets it right after aggregating 10 answers)
beth_bakes = "Beth bakes 4x 2 dozen batches of cookies in a week. If these cookies are shared amongst 16 people equally, how many cookies does each person consume?"

# It takes a lot of tries at all to get the right answer, 
# I need to give it about 30-40 chain-of-thought attempts to aggregate for 
# the end result to be reliably correct. 
nozicks_taxes = "Robert Nozick makes 70k a year pre-tax. The first 10k dollars of income are exempt from tax, the marginal tax rate is 20%% between $10k and $50k, and 35%% on every dollar of income after $50k. What is Robert Nozick's post-tax income?"

default_question = nozicks_taxes

# We try to marginalize over different chains of thoughts behind an answer, 
# trying to choose the answer with the highest _posterior probability_ 
# P(answer | chain-of-thought is correct) = Sum_{chains of thought} P(chain of thought, answer) P_verifier(correct | chain of thought, answer)

# We sample different chains of thoughts and accumulate the probabilities for each answer;
# we estimate the posterior probability by weighting each result by the posterior probability / the chance its sampled.
# We note that initially this is equal to the p(correct) provided by the verifier
# but we our sampling strategy should converge to sampling each answer with probability proportional 
# to its actual posterior probability. 
async def self_consistency_qa(question : str = default_question, attempts : int = 40) -> str:
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