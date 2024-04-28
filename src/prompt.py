error_info_template_concise = """You wrote the '{entry_point}' function in the following python program: 
<program>
{program}
</program>

Execution of your program resulted in the following error message: 
<error>
{message}
</error>
"""

correct_info_template_concise = """You've correctly implemented the '{entry_point}' function:
<code>
{prompt}{completion}
</code>

Excecution of your function is correct and provides desired output, great job!
"""

error_info_template_full = """You wrote the '{entry_point}' function in the following python program: 
<program>
{program}
</program>

Execution of your program resulted in the following error message: 
<error>
{message}
</error>

Entry point to the function is 
<entry_point>
{entry_point}
</entry_point>

Your errored completion is:
<completion>
{completion}
</completion>

The canonical completion is:
<canonical>
{canonical}
</canonical>
"""

knowledge_message_template = """<retrieve_knowledge>
{numbered_retrieve_infos}
</retrieve_knowledge>"""

rewrite_message_template = """Previous Case Information: {case_info}"""
rewrite_message_with_rag_template = """Previous Case Information: {case_info}
{knowledge_message}"""

rewrite_with_rag_template = """{rewrite_message}
{knowledge_message}"""

def batch_rewrite_message_template(case_infos):
    rewrite_template = ""
    for i, case_info in enumerate(case_infos):
        rewrite_template += f"""Previous Case ({i}) Information: {case_info} \n"""
    return rewrite_template

def batch_rewrite_message_with_rag_template(case_infos, knowledge_message):
    return batch_rewrite_message_template(case_infos) + f"""{knowledge_message}"""


