# Monte Carlo Tree Search Based Fast-Adaptation (BatchWise Prompt Optimization)
# Reflective Adaptation | Use Groq for fast iteration | CUDA device could also be fast (?)
from data import CodeCompletionGenerator, load_config
from human_eval.data import write_jsonl, read_problems
from tqdm import tqdm as tqdm
from src.fast_adapt import *
from src.prompt import *
import numpy as np

config_name = "phi_3_instruct"
code_gen = CodeCompletionGenerator.from_config(config_name, device="mps")

batches = np.array_split(list(range(162)), 60)
max_search = 30
pb = tqdm(total=len(batches), desc="MCTS Search Batches")
for i, indices in enumerate(batches):
    pb.desc = "MCTS Search Batches: %d/%d" % (i + 1, len(batches))
    # Initialize Sub-Batch Indices
    init_prompt = "Do not fuck it up"

    # Initialize the MCTS object
    mcts = MCTS(init_prompt, 0, indices, code_gen, max_search=max_search)

    # Run the search loop
    while not mcts.is_search_complete():
        node_to_expand = mcts.select_node()
        mcts.expand_node(node_to_expand)
        pb.update(1)
    pb.n = max_search * (i + 1)
    # Retrieve the best prompt after the search is complete
    best_prompt = mcts.get_best_prompt()
    print("Best Prompt on Batch %d: %s" % (i, best_prompt))
    mcts.save_best_node()