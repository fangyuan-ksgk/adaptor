from groq import Groq 
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer, util
import torch
from .prompt import *
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
GROQ_API_KEY1 = os.environ["GROQ_API_KEY1"]
groq_client = Groq(api_key=GROQ_API_KEY)
groq_client1 = Groq(api_key=GROQ_API_KEY1)

###################
# Tool Definition #
###################
# Similar to how Human Acquire Scientific Discovery
# Gain Knowledge
# Retrieve Knowledge
# Propose Insight (Prompt)

prompt_rewrite_tool = {
    "type": "function",
    "function": {
        "name": "prompt_rewrite_tool",
        "description": "Reflect on the previous solution, analyze the error and provide a prompt to help the student avoids similar errors and reach the correct solution.",
        "parameters": {
            "type": "object",
            "properties": {
                "error_analysis": {
                    "type": "string",
                    "description": "Specific analysis on why the error message occurs, which line is causing it and why."
                },
                "prompt": {
                    "type": "string",
                    "description": "The suggestion to help student providing the correct solution."
                }
            },
            "required": ["error_analysis", "prompt"],
        }
    }
}

# I feel like this sort of self-RAG pipeline would be very beneficial -- it's very similar to building a self-knowledge-base
retrieve_knowledge_tool = {
    "type": "function",
    "function": {
        "name": "retrieve_knowledge_tool",
        "description": "Get the previous knowledge & experience & error from the previous attemps in solving the problem, based on your query. For instance, if you are unsure whehter some command is working, you could use this tool to check if the student has encountered an error with this command before.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search for the previous error."
                }
            },
            "required": ["query"]
        }
    }
}

add_knowledge_tool = {
    "type": "function",
    "function": {
        "name": "add_knowledge_tool",
        "description": "Add knowledge & experience & error from your attemps in teaching the student better solve the problem.",
        "parameters": {
            "type": "object",
            "properties": {
                "error": {
                    "type": "string",
                    "description": "The error encountered in the previous solution. Ensure the description is coherant and contains useful information. For instance, [CODE] leads to [ERROR] because of [REASON]"
                },
                "success": {
                    "type": "string",
                    "description": "The success of the previous solution. Ensure the description is coherant and contains useful information. For instance, [CODE] leads to [GOAL] because of [REASON]",
                },
                "knowledge": {
                    "type": "string",
                    "description": "The knowledge to add to the knowledge base of the LLM. Ensure the description is coherant and contains useful information. For instance, to achieve [GOAL], use [CODE]."
                }
            },
            "required": ["knowledge"]
        }
    }
}

def add_knowledge(knowledge):
    knowledge_path = "data/knowledge/knowledge.txt"
    os.makedirs(os.path.dirname(knowledge_path), exist_ok=True)
    with open(knowledge_path, "a", encoding="utf-8") as file:
        file.write(knowledge + "\n")

model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_knowledge(query, top_k=5):
    knowledge_path = "data/knowledge/knowledge.txt"
    if not os.path.exists(knowledge_path):
        return "No knowledge base found."
    
    with open(knowledge_path, "r", encoding="utf-8") as file:
        knowledge_content = file.readlines()
    
    if not knowledge_content:
        return "Knowledge base is empty."
    
    knowledge_embeddings = model.encode(knowledge_content)
    query_embedding = model.encode([query])
    cos_scores = util.cos_sim(query_embedding, knowledge_embeddings)[0]
    
    # Adjust top_k if there are fewer entries than requested
    actual_top_k = min(top_k, len(knowledge_content))
    top_results = torch.topk(cos_scores, k=actual_top_k)[1].tolist()
    
    return [knowledge_content[idx].strip() for idx in top_results]

def call_function(function_name, function_args):
    rewrite_prompt = None
    retrieve_infos = []
    if function_name == "prompt_rewrite_tool":
        rewrite_prompt = function_args['prompt']
        # return this rewritten prompt
    if function_name == "retrieve_knowledge_tool":
        retrieve_infos = retrieve_knowledge(function_args['query'], top_k=3)
    if function_name == "add_knowledge_tool":
        add_knowledge(function_args['knowledge'])
    return rewrite_prompt, retrieve_infos

def process_call(response):
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    prompt = None
    infos = None
    if tool_calls:
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            print("Call Function: ", function_name, function_args)
            rewrite_prompt, retrieve_infos = call_function(function_name, function_args)
            if rewrite_prompt is not None:
                prompt = rewrite_prompt
            if retrieve_infos:
                infos = retrieve_infos

    return prompt, infos


# Knowledge Base is accessible as an extra feature
SYSTEM_PROMPT_REWRITER = """You are a helpful assistant, helping to guide student to write better code. Given his previous error, you should provide a suggestion to help the student avoid similar errors and reach the correct solution. You could also check the previous knowledge base to see how you could help the student better. It's important to store your knowledge on how to help the student better."""

############
# Pipeline #
############
# 1. Initialize Agent with System Prompt [Done for both instruction & pretrained model]
# 2. Run HumanEval Test --> [Included into the CodeGen class run_human_eval_test functional with custome indices]
# 3. Get error message --> [Included into the CodeGen class run_human_eval_test functional with custome indices]
# 4. Update knowledge & prompt & Rewrite the prompt for better performance --> [Case Info Prepared | ]
# 5. Loop
# 6. FineTune & Pretrain for Long-Term memory update
# 7. Port in fine-tuned adaptor model for fast & slow adaptation loop 

# Template for Error Message Collection & Program Debugging 
def form_case_info(info):
    if info['success']:
        return correct_info_template_concise.format(**info)
    else:
        return error_info_template_concise.format(**info)
    
def form_case_infos(infos):
    return [form_case_info(info) for info in infos]
    
def form_rewrite_message_with_rag(case_info, retrieve_infos):
    rewrite_message = rewrite_message_template.format(case_info=case_info)
    numbered_retrieve_infos = "\n".join(f"{idx+1}. {info}" for idx, info in enumerate(retrieve_infos))
    knowledge_message = knowledge_message_template.format(numbered_retrieve_infos=numbered_retrieve_infos)
    return rewrite_with_rag_template.format(rewrite_message=rewrite_message, knowledge_message=knowledge_message)

def form_batch_rewrite_message_with_rag(case_infos, retrieve_infos = []):
    rewrite_message = batch_rewrite_message_template(case_infos=case_infos)
    if not retrieve_infos:
        return rewrite_message
    else:
        numbered_retrieve_infos = "\n".join(f"{idx+1}. {info}" for idx, info in enumerate(retrieve_infos))
        knowledge_message = knowledge_message_template.format(numbered_retrieve_infos=numbered_retrieve_infos)
        return rewrite_with_rag_template.format(rewrite_message=rewrite_message, knowledge_message=knowledge_message)
    

def chat_complete_groq(model_name, messages, tools, tool_choice="auto", max_tokens=4096):
    """ 
    Use alternative Groq API key for backup
    """
    try:
        response = groq_client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=4096,
        )
    except:
        response = groq_client1.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=4096,
        )
    return response

        
def call_rewriter(case_infos):
    
    rewrite_message = form_batch_rewrite_message_with_rag(case_infos=case_infos)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_REWRITER},
        {"role": "user", "content": rewrite_message}
    ]

    MODEL_NAME = "llama3-70b-8192"

    # Direct Rewrite + Knowledge Update + Retrieve Knowledge
    response = chat_complete_groq(
        model_name=MODEL_NAME,
        messages=messages,
        tools=[prompt_rewrite_tool, add_knowledge_tool, retrieve_knowledge_tool],
        tool_choice="auto",
        max_tokens=4096,
    )

    # Parse tool call: 1. get rewrite prompt 2. update knowledge base 3. obtain retrieved knowledge
    rewrite_prompt, retrieve_infos = process_call(response)

    # Add retrieved knowledge into rewrite messages
    rewrite_message = form_batch_rewrite_message_with_rag(case_infos=case_infos, retrieve_infos=retrieve_infos)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_REWRITER},
        {"role": "user", "content": rewrite_message}
    ]

    response = chat_complete_groq(
        model_name=MODEL_NAME,
        messages=messages,
        tools=[prompt_rewrite_tool],
        tool_choice="auto",
        max_tokens=4096,
    )

    rewrite_prompt_with_rag, retrieve_infos = process_call(response)

    prompts = [rewrite_prompt, rewrite_prompt_with_rag]
    return [p for p in prompts if p is not None]


#############################################
# Search Algorithm: Monte Carlo Tree Search #
#############################################

# Key is to incorporate the information into Node, then MCTS is just one way to play with it
class Node:
    def __init__(self, infos, prompt, score, id, parent=None):
        self.infos = infos
        self.prompt = prompt
        self.score = score
        self.id = id
        self.visits = 1
        self.parent = parent
        self.children = []

    @classmethod
    def from_prompt(cls, indices, prompt, id, code_gen, parent=None):
        code_gen.run_human_eval_test(indices=indices, global_system_prompt=prompt, id=id)
        infos = code_gen.get_info(indices, id)
        score = np.array([info['success'] for info in infos]).mean()
        return cls(infos=infos, prompt=prompt, score=score, id=id, parent=parent)

    def update_score(self, additional_score):
        self.score += additional_score
        self.visits += 1
        if self.parent:
            self.parent.update_score(additional_score / 5) # Trace back to parent node -- add score along the trajectory

    def spawn_child_nodes(self, indices, code_gen):
        # Evolve Up to 2 Child Nodes from Parent Node
        case_infos = form_case_infos(self.infos)
        prompts = call_rewriter(case_infos)
        for prompt in prompts:
            self.children.append(Node.from_prompt(indices, prompt, self.id + 1, code_gen, parent=self))
        return self.children
    
    def save_node(self, indices, text=""):
        info_dict = {"indices": indices, "prompt": self.prompt, "id": self.id}
        file_name = "data/log/best_node-"+ text + "-".join(map(str, indices)) + ".json"
        with open(file_name, "w") as f:
            json.dump(info_dict, f)


class MCTS:
    def __init__(self, prompt, id, indices, code_gen, max_search=50):
        self.indices = indices
        self.code_gen = code_gen
        self.root = Node.from_prompt(indices, prompt, id, code_gen)
        self.nodes = [self.root]
        self.max_search = max_search

    def select_node(self):
        # Select the node with the highest UCB1 score
        best_score = -1
        best_node = None
        for node in self.nodes:
            if node.visits == 0:
                return node
            score = node.score / node.visits + np.sqrt(2 * np.log(sum(n.visits for n in self.nodes)) / node.visits)
            if score > best_score:
                best_score = score
                best_node = node
        return best_node

    def expand_node(self, node):
        new_nodes = node.spawn_child_nodes(self.indices, self.code_gen)
        self.nodes.extend(new_nodes) # Add new nodes to tree

    def is_search_complete(self):
        # Define a condition to stop the search, e.g., a maximum number of nodes
        return len(self.nodes) > self.max_search
    
    @property
    def best_node(self):
        return max(self.nodes, key=lambda n: n.score / n.visits if n.visits > 0 else 0)

    def get_best_prompt(self):
        # Return the prompt of the node with the highest average score
        return self.best_node.prompt

    def get_prompt_score(self, prompt):
        # Find the node with the given prompt and return its score
        for node in self.nodes:
            if node.prompt == prompt:
                return node.score / node.visits if node.visits > 0 else 0
        return 0
    
    def save_best_node(self): # Save the best node
        self.best_node.save_node(self.indices)
    

    


















