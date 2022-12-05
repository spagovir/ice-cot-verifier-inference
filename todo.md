- change unicode handling to reject all unicode (ie, set likelihood to 0), 
unless I can figure out how the OpenAI API handles unicode.
- add handling for using openai complete and w/ logprobs enabled instead of individual predicts in order to not mess up caching on their end and cut down on network traffic/latency issues.
  - possibly pull request ice api to enable this, though this would mess with human mode. possibly subclass? 
