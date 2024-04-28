from operator import is_
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from human_eval.data import write_jsonl, read_problems, stream_jsonl, HUMAN_EVAL
from human_eval.evaluation import evaluate_functional_correctness
import os, json
from tqdm import tqdm
from mlx_lm import load, generate

os.environ["TOKENIZERS_PARALLELISM"] = "true"
log_dir = "data/log/"
config_dir = "data/config/"

# Set the device to GPU (CUDA) if available
# device = "cuda" if torch.cuda.is_available() else "cpu"
pretrained_prompt_template = """[INSTURCTION] {system_prompt} [\INSTRUCTION] \n{prompt}"""

def load_config(config_name):
    config_path = f"{config_dir}{config_name}.json"
    config = json.load(open(config_path, 'r'))
    return config

def save_config(config_name, config):
    config_path = f"{config_dir}{config_name}.json"
    json.dump(config, open(config_path, 'w'), indent=4)


def generate_one_completion_pretrained(prompt: str, system_prompt: str,  model, tokenizer, device):
    """
    For pre-trained model HumanEval relies on direct continuation
    """
    tokenizer.pad_token = tokenizer.eos_token
    format_prompt = pretrained_prompt_template.format(system_prompt=system_prompt, prompt=prompt)
    inputs = tokenizer(format_prompt, return_tensors="pt", truncation=True, max_length=4096)

    # Generate || The configuration is designed for model to be more 'creative' therefore giving more freedom to generate and find the 'correct' answer
    generate_ids = model.generate(
        inputs.input_ids.to(device),
        max_new_tokens=384,
        do_sample=True,
        top_p=0.75,
        top_k=40,
        temperature=0.1,
    )
    completion = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    completion = completion.replace(prompt, "").split("\n\n\n")[0]

    return completion


def split_and_trim_code(text: str):
    # Find the index where "def " starts
    def_index = text.find('def ')
    if def_index == -1:
        return ""  # return an empty string if "def " is not found
    # Extract the code starting from "def "
    code_after_def = text[def_index:]
    # Find the end of the "def " line
    def_line_end_index = code_after_def.find('\n')
    if def_line_end_index != -1:
        # Skip the entire line where "def " is found
        code_after_def = code_after_def[def_line_end_index+1:]
    # Find the index of "```" that comes after "def "
    end_code_index = code_after_def.find('```')
    # If "```" is found, return the code until just before "```"
    if end_code_index != -1:
        code = code_after_def[:end_code_index]
    else:
        # If no "```" is found, return the code until the end of the string
        code = code_after_def
    # Removing leading and trailing whitespaces and newlines
    return "    " + code.strip()

def split_and_trim_code_mlx(text: str):
    text = split_and_trim_code(text)
    r = text.split("<|end|>")[0]
    lines = r.split('\n')
    # recursively pop out lines after the return statement
    while lines and not (lines[-1].strip()).startswith('return'):
        lines.pop()
    r = '\n'.join(lines)
    return r

def trim_response(response):
    if "if __name__" in response:
        return response.split("if __name__")[0]
    response = '    ' + response.lstrip()
    return response


def get_mlx_completion_pretrained(prompt, system_prompt, model, tokenizer, verbose=True):
    format_prompt = pretrained_prompt_template.format(system_prompt=system_prompt, prompt=prompt)
    response = generate(model, tokenizer, prompt=format_prompt, max_tokens=512, temp=0.2, top_p=0.95, verbose=verbose)
    return trim_response(response)

def get_mlx_completion_instruct(prompt, system_prompt, model, tokenizer, verbose=True):
    messages = [
        {
            "role": "system",
            "content": f"You are a friendly chatbot. [INSTRUCTION] {system_prompt} [\INSTRUCTION] WRITE THE FULL COMPLETE FUNCTION (EG WITH def ....) END CODE WITH '```'. NOTE YOU ABSOLUTELY MUST END THE CODE WITH END CODE WITH '```' OR ELSE THE CODE WILL NOT BE INTERPRETTED!!!!",
        },
        {"role": "user", "content": prompt},
    ]
    format_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    response = generate(model, tokenizer, prompt=format_prompt, max_tokens=512, temp=0.2, top_p=0.95, verbose=verbose)
    response = split_and_trim_code_mlx(response)
    # response = response.split("<|im_end|>")[0]
    return response


def generate_one_completion_instruct(prompt, system_prompt, model, tokenizer, device):
    # Format the prompt using the tokenizer's chat template
    messages = [
        {
            "role": "system",
            "content": f"You are a friendly chatbot. [INSTRUCTION] {system_prompt} [\INSTRUCTION] WRITE THE FULL COMPLETE FUNCTION (EG WITH def ....) END CODE WITH '```'. NOTE YOU ABSOLUTELY MUST END THE CODE WITH END CODE WITH '```' OR ELSE THE CODE WILL NOT BE INTERPRETTED!!!!",

        },
        {"role": "user", "content": prompt},
    ]

    # Use Pipeline for that
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    generation_args = {
        "max_new_tokens": 512,
        "return_full_text": False,
        "temperature": 0.2,
        "do_sample": True,
        "top_p": 0.95,
    }

    output = pipe(messages, **generation_args)
    output_text = output[0]['generated_text'] # Extract output text

    return split_and_trim_code(output_text)


# def generate_completion_mlx(model, tokenizer, )


# I still suspect llama3-8b is not fairly evaluated here ... 
# How come they do not release the code they've used to evaluate it on HumanEval?

class CodeCompletionGenerator:
    def __init__(self, model_name, device="mps", is_instruct=False, system_prompt="", file_name = None):
        self.device = device
        self.model_name = model_name
        self.is_instruct = is_instruct
        self.system_prompt = system_prompt
        self.file_name = file_name
        if self.device == "mps":
            self.model, self.tokenizer = load(model_name)
        else: # Cuda device
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

    @classmethod
    def from_config(cls, config_name, device="mps"):
        config = load_config(config_name)
        return cls(**config, device=device)

    def generate_one_completion(self, prompt, system_prompt=""):
        if system_prompt != "":
            self.system_prompt = system_prompt

        if self.device == "mps":
            if self.is_instruct:
                return get_mlx_completion_instruct(prompt, self.system_prompt, self.model, self.tokenizer)
            else:
                return get_mlx_completion_pretrained(prompt, self.system_prompt, self.model, self.tokenizer)
        else: # Cuda device
            if self.is_instruct:
                return generate_one_completion_instruct(prompt, self.system_prompt, self.model, self.tokenizer, self.device)
            else:
                return generate_one_completion_pretrained(prompt, self.system_prompt, self.model, self.tokenizer, self.device)
            
    def run_human_eval_test(self, indices=None, global_system_prompt="", id=""):
        if indices is None:
            file_name = self.file_name
        else: # with specific indices, name of the stored file is automatically updated here
            file_name = f"{self.file_name}-{'-'.join(map(str, indices))}"

        problems = read_problems()
        if indices is not None:
            problems = {list(problems.keys())[i]: problems[list(problems.keys())[i]] for i in indices}
        # Code Completion
        num_samples_per_task = 1
        total_tasks = len(problems)
        pb = tqdm(total=total_tasks * num_samples_per_task, desc="Generating code completions")
        samples = []
        for task_id in problems:
            for _ in range(num_samples_per_task):
                completion = self.generate_one_completion(problems[task_id]["prompt"], global_system_prompt)
                samples.append(dict(task_id=task_id, completion=completion))
                pb.update(1)
        pb.close()
        write_jsonl(f"{log_dir}{file_name}_{id}.jsonl", samples)

        # Error Message & Information Parsing
        check_performance(file_name, indices, id)

        if indices is None:
            # Standard HumanEval Benchmarking
            results = benchmark_performance(file_name)
            print("----- HumanEval Benchmarking Results [All Cases] -----")
            for k, v in results.items():
                print(f"{k}: {v}")

    def get_info(self, indices=None, id=""):
        info_dicts = get_performance_info(self.file_name, indices, id)
        infos = []
        for info in info_dicts:
            infos.append(info)
        return infos

            
    
def check_code(task_id, problem, completion, timeout=10.0):
    """ 
    Subprocess-base code test
    """
    # Construct the check program and run it. This is literally the python file which is supposed to be executed
    check_program = (
        problem["prompt"] + completion + "\n" +
        problem["test"] + "\n" +
        f"check({problem['entry_point']})"
    ) 
    import subprocess
    try:
        out = subprocess.run(["python", "-c", check_program], capture_output=True, text=True, timeout=timeout, check=False)
        success = out.returncode == 0
        message = out.stderr
    except subprocess.TimeoutExpired:
        success = False
        message = "Timeout expired after {} seconds".format(timeout) + " Possibly infinite loop."
    return check_program, success, message


def check_performance_stream(sample_file, indices=None):
    """ 
    Streaming code checking
    """
    problems = read_problems()
    if indices is not None:
        problems = {list(problems.keys())[i]: problems[list(problems.keys())[i]] for i in indices}

    for sample in stream_jsonl(sample_file): # This is a generator
        task_id = sample['task_id']
        problem = problems[task_id]
        check_program, success, message = check_code(task_id, problem, sample['completion'])
        sample['prompt'] = problem['prompt']
        sample['entry_point'] = problem['entry_point']
        sample['canonical_solution'] = problem['canonical_solution']
        sample['program'] = check_program
        sample['success'] = success
        sample['message'] = message
        yield sample

def check_performance(filename, indices=None, id=""):
    """ 
    Pass / Error Message, useful for reflective debugging
    """
    sample_file = log_dir + filename + f"_{id}.jsonl"  
    out_file = sample_file.replace(f"_{id}.jsonl", f"_{id}_info.jsonl")
    print(f"Writing error information to {out_file}...")
    n_samples = sum(1 for _ in stream_jsonl(sample_file))
    write_jsonl(out_file, tqdm(check_performance_stream(sample_file, indices), total=n_samples))

def get_performance_info(filename, indices=None, id=""):
    if indices == None:
        sample_file = log_dir + filename + f"_{id}.jsonl"
    else:
        sample_file = log_dir + filename + "-" + "-".join(map(str, indices)) + f"_{id}_info.jsonl"
    return stream_jsonl(sample_file)

def benchmark_performance(filename, k=[1]):
    """ 
    Official HumanEval Benchmarking :: Only success / fail but no further information
    """
    sample_file = log_dir + filename + ".jsonl"
    n_workers: int = 1
    timeout: float = 3.0
    problem_file: str = HUMAN_EVAL
    results = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file)
    return results



