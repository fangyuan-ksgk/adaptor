from operator import is_
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from human_eval.data import write_jsonl, read_problems, stream_jsonl
import os
from tqdm import tqdm
from mlx_lm import load, generate
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Set the device to GPU (CUDA) if available
# device = "cuda" if torch.cuda.is_available() else "cpu"


def generate_one_completion_pretrained(prompt: str, model, tokenizer, device):
    """
    For pre-trained model HumanEval relies on direct continuation
    """
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)

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


def trim_response(response):
    if "if __name__" in response:
        return response.split("if __name__")[0]
    response = '    ' + response.lstrip()
    return response

def get_mlx_completion_pretrained(prompt, model, tokenizer, verbose=True):
    response = generate(model, tokenizer, prompt=prompt, max_tokens=512, temp=0.2, top_p=0.95, verbose=verbose)
    return trim_response(response)

def get_mlx_completion_instruct(prompt, model, tokenizer, verbose=True):
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot. WRITE THE FULL COMPLETE FUNCTION (EG WITH def ....) END CODE WITH '```'. NOTE YOU ABSOLUTELY MUST END THE CODE WITH END CODE WITH '```' OR ELSE THE CODE WILL NOT BE INTERPRETTED!!!!",

        },
        {"role": "user", "content": prompt},
    ]
    format_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    response = generate(model, tokenizer, prompt=format_prompt, max_tokens=512, temp=0.2, top_p=0.95, verbose=verbose)
    response = response.split("<|im_end|>")[0]
    return response


def generate_one_completion_instruct(prompt, model, tokenizer, device):
    # Format the prompt using the tokenizer's chat template
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot. WRITE THE FULL COMPLETE FUNCTION (EG WITH def ....) END CODE WITH '```'. NOTE YOU ABSOLUTELY MUST END THE CODE WITH END CODE WITH '```' OR ELSE THE CODE WILL NOT BE INTERPRETTED!!!!",

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
    def __init__(self, model_name, device="mps", is_instruct=False):
        self.device = device
        self.model_name = model_name
        self.is_instruct = is_instruct
        if self.device == "mps":
            self.model, self.tokenizer = load(model_name)
        else: # Cuda device
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

    def generate_one_completion(self, prompt):
        if self.device == "mps":
            if self.is_instruct:
                return get_mlx_completion_instruct(prompt, self.model, self.tokenizer)
            else:
                return get_mlx_completion_pretrained(prompt, self.model, self.tokenizer)
        else: # Cuda device
            if self.is_instruct:
                return generate_one_completion_instruct(prompt, self.model, self.tokenizer, self.device)
            else:
                return generate_one_completion_pretrained(prompt, self.model, self.tokenizer, self.device)
            
    
def check_code(task_id, problem, completion, timeout=10.0):
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


def check_performance_stream(sample_file):
    problems = read_problems()
    for sample in stream_jsonl(sample_file): # This is a generator
        task_id = sample['task_id']
        problem = problems[task_id]
        check_program, success, message = check_code(task_id, problem, sample['completion'])
        sample['program'] = check_program
        sample['success'] = success
        sample['message'] = message
        yield sample

def check_performance(filename):
    sample_file = filename + ".jsonl"  
    out_file = sample_file + "_results.jsonl"
    print(f"Writing results to {out_file}...")
    n_samples = sum(1 for _ in stream_jsonl(sample_file))
    write_jsonl(out_file, tqdm(check_performance_stream(sample_file), total=n_samples))
