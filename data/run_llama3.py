from sample_hf import CodeCompletionGenerator
from human_eval.data import write_jsonl, read_problems
from tqdm import tqdm

# model_name = "microsoft/Phi-3-mini-128k-instruct"
model_name="meta-llama/Meta-Llama-3-8B"
is_instruct = False
file_name = "eval_llama3_pretrain"

code_gen = CodeCompletionGenerator(model_name=model_name, device="mps", is_instruct=is_instruct)

problems = read_problems()
# problems = {k: problems[k] for k in list(problems.keys())[:1]} # SubSample Problems out of the 163 cases

num_samples_per_task = 2
total_tasks = len(problems)
pb = tqdm(total=total_tasks * num_samples_per_task, desc="Generating completions")
samples = []
for task_id in problems:
    for _ in range(num_samples_per_task):
        completion = code_gen.generate_one_completion(problems[task_id]["prompt"])
        samples.append(dict(task_id=task_id, completion=completion))
        pb.update(1)
pb.close()
write_jsonl(f"{file_name}.jsonl", samples)