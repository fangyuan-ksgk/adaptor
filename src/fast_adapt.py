from groq import Groq 
import os
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
groq_client = Groq(api_key=GROQ_API_KEY)

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
        "description": "Reflect on the previous solution, analyze the error and provide a prompt to help the LLM avoids similar errors and reach the correct solution.",
        "parameters": {
            "type": "object",
            "properties": {
                "error_analysis": {
                    "type": "string",
                    "description": "The error analysis of the previous solution."
                },
                "prompt": {
                    "type": "string",
                    "description": "The prompt to help the LLM avoid similar errors and reach the correct solution."
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
        "description": "Get the previous knowledge & experience & error from the previous attemps in solving the problem, based on your query. For instance, if you are unsure whehter some command is working, you could use this tool to check if the LLM has encountered an error with this command before.",
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
        "description": "Add knowledge & experience & error from your attemps in solving the problem. This could be used to update the knowledge base of the LLM, and help the LLM to avoid similar errors in the future.",
        "parameters": {
            "type": "object",
            "properties": {
                "error": {
                    "type": "string",
                    "description": "The error encountered in the previous solution. Ensure the description is coherant and contains useful information. For instance, [CODE] leads to [ERROR] because of [REASON]"
                },
                "knowledge": {
                    "type": "string",
                    "description": "The knowledge to add to the knowledge base of the LLM. Ensure the description is coherant and contains useful information. For instance, to achieve [GOAL], use [CODE]."
                }
            },
            "required": ["error", "knowledge"]
        }
    }
}

############
# Pipeline #
############
# 1. Initialize Agent with System Prompt [Done for both instruction & pretrained model]
# 2. Run HumanEval Test --> [Included into the CodeGen class run_human_eval_test functional with custome indices]
# 3. Get error message --> [Included into the CodeGen class run_human_eval_test functional with custome indices]
# 4. Update knowledge & prompt & Rewrite the prompt for better performance --> 
# 5. Loop
# 6. FineTune & Pretrain for Long-Term memory update
# 7. Port in fine-tuned adaptor model for fast & slow adaptation loop 













