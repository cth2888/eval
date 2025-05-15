from rllm.rewards.code_reward import rllm_reward_fn_code 
from rllm.rewards.math_reward import rllm_reward_fn_math
from rllm.rewards.math_verify_reward import reward_fn_math_verify
from typing import Union, List
import json 

def reward_fn(data_source: str, llm_solution: str, ground_truth: Union[str, List[str]], extra_info={}, **kwargs):
    '''
        if data_source in ["apps", "taco", "code_contests", "codeforces", "livecodebench", "kodcode", "leetcode", "primeintellect", "humanevalplus"]:
        try:
            ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError:
            return False 
        return rllm_reward_fn_code(data_source, llm_solution, ground_truth, **kwargs)
    #elif data_source in ['deepscaler']:
        #return rllm_reward_fn_math(data_source, llm_solution, ground_truth, extra_info, **kwargs)
    '''


    return reward_fn_math_verify(data_source, llm_solution, ground_truth, extra_info, **kwargs)