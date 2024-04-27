from operator import is_
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from human_eval.data import write_jsonl, read_problems

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
    return code.strip()


def generate_one_completion_instruct(prompt, model, tokenizer, device):
    # Format the prompt using the tokenizer's chat template
    global i 
    i+=1 
    print("Generating {}".format(i))
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot. WRITE THE FULL COMPLETE FUNCTION (EG WITH def ....) END CODE WITH '```'. NOTE YOU ABSOLUTELY MUST END THE CODE WITH END CODE WITH '```' OR ELSE THE CODE WILL NOT BE INTERPRETTED!!!!",

        },
        {"role": "user", "content": prompt},
    ]
    # Format the prompt using the tokenizer's chat template
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Generate a response using the model
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.2, top_p=0.95)
    
    # Extract the generated text
    text = outputs[0]["generated_text"]

    print("###"*50)
    # print(split_and_trim_code(text.split("<|assistant|>")[1]))
    # return text
    return split_and_trim_code(text.split("<|assistant|>")[1])


# I still suspect llama3-8b is not fairly evaluated here ... 
# How come they do not release the code they've used to evaluate it on HumanEval?




class CodeCompletionGenerator:
    def __init__(self, model_name, device="mps", is_instruct=False):
        self.device = device
        self.model_name = model_name
        self.is_instruct = is_instruct
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

    def generate_one_completion(self, prompt):
        if self.is_instruct:
            return generate_one_completion_instruct(prompt, self.model, self.tokenizer, self.device)
        else:
            return generate_one_completion_pretrained(prompt, self.model, self.tokenizer, self.device)
