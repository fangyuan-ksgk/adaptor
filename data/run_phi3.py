from sample_hf import CodeCompletionGenerator
from human_eval.data import write_jsonl, read_problems
from tqdm import tqdm

model_name = "microsoft/Phi-3-mini-128k-instruct"
# model_name="meta-llama/Meta-Llama-3-8B"
is_instruct = True
file_name = "eval_phi_3_instruct"

# Run on cloud servce, assume access to cuda device here
code_gen = CodeCompletionGenerator(model_name=model_name, device="cuda", is_instruct=is_instruct)

problems = read_problems()

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