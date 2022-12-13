# Chain-of-thought self-consistency with verifiers. 

Chain-of-thought self-consistency (https://arxiv.org/abs/2203.11171) is a technique where we use a large language model to sample multiple chains-of-thought in response to a question, and return the answer supported by the most chains of thought.
We can integrate self-consistency with verifiers by weighting the contribution of each chain-of-thought by the probability the verifier outputs that the chain-of-thought is correct.

However, if the question is rather difficult,
we often sample many chains-of-thought that contribute very little to the final decision for the answer because they are rejected by the verifier. 
Each sample would be more informative if we could directly sample from the distribution of chains-of-thought conditional on the verifier outputting correct. 

We can create a sampling method that converges to this distribution by 
1. asking GPT-3 to return the top logprobs used to sample each token it generates. 
2. caching generation paths in a tree, where each generated node keeps the probabilities of choosing each of its children. 
3. start each sample by randomly going down the tree until we reached a possible next token choice we haven't tried yet, and letting GPT-3 complete the generation from there. 
4. updating the weights each generated node keeps for the next token choice toward the probability of the next token conditional on the verifier passing as estimated from the result of the sample. 

This project implements that method and uses it to implement chain-of-thought self-consistency with verifiers. 

## Modules
- search_tree.py implements the search tree used to sample generations. 
It supports 'programs' made by chaining multiple LLM requests together by having
completion request nodes take continuations that determine what to do with the result of the completion requests; the tree stores the return values of such programs on its leaves. 
- programs.py implements programs/search trees for chain of thought prompting/answer extraction and verifiers. 
Programs are designed to take continuations so they can be chained with other programs that use their outputs. 
- self_consistency.py uses the chain-of-thought & verifier programs in programs to implement self consistency with verifiers. It can be run as main, in which case it will try to answer the question specified by the default_question constant. 

## Installing
This project depends on `ought-ice`, which can be installed by following the instructions here: https://github.com/oughtinc/ice. 

## What's it good for?
Right now it seems that it shows the biggest improvement vs vanilla GPT-3 at doing ~6th grade level math word problems.

## Possible Next Steps
- Integrate with amplification/subquestions?
- Summarize the reasoning path marginalized over to get the answer? (+ possibly verifier criticism?)
- Add option to request human input when verifier is uncertain?
- Return a confidence score (posterior probability of answer w/ uncertainty adjustment based on total probability mass explored and size of region that passes verifier) + runner up answers?
- I'm still don't think unicode handling is totally correct. 
- Better integration with rest of ICE framework.
- Allow running multiple exploration attempts in parallel and then doing a batch update. At the difficulties it's most useful for, it's pretty excruciatingly slow. 
- Sometimes answers overflow and start trying to imagine a next question. Figure out a better way to prevent that. 
- Sometimes the verifier hallucinates reasons the solution is wrong. Try to figure out why that happens/how to prevent?