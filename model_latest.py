import chromadb

import autogen
from autogen import UserProxyAgent, config_list_from_json
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from system_msg import cs_manager_system_message, cs_critique_system_message, custom_rag_prompt, \
    customer_support_chat_manager_system_prompt
from system_msg import pm_verify_system_message, pm_verify_critique_system_message
from system_msg import OUTPUT_FORMAT_TASK_VERIFICATION
from system_msg import pm_task_delegation_system_message, OUTPUT_FORMAT_TASK_DELEGATION
from system_msg import pm_task_delegation_critic_system_message
from system_msg import manager_system_message
from system_msg import dev_agent_system_message, pm_agent_system_message, \
    team_gc_manager_system_message
from autogen import AssistantAgent
from system_msg import pm_agent_system_message, OUTPUT_FORMAT_CATALOG_FIELDS

# Load LLM inference endpoints from an env variable or a file
# See https://microsoft.github.io/autogen/docs/FAQ#set-your-api-endpoints
# and OAI_CONFIG_LIST
autogen.ChatCompletion.start_logging()

# Change this to your path
# config_list = config_list_from_json(
#     env_or_file="/Users/ojasvsingh/personal_projects/multi-agent/hackathon2023/autogen/OAI_CONFIG_LIST")


config_list = config_list_from_json(
    env_or_file="OAI_CONFIG_LIST")
default_llm_config = {"config_list": config_list, "seed": 12, "request_timeout": 600}

default_user_msg = """
How can I create a flatfile source?
"""

docs_path = 'collect.txt'


def get_cs_help(user_msg):
    user_proxy = UserProxyAgent("user_proxy",
                                code_execution_config={"work_dir": "coding"},
                                is_termination_msg=lambda x: print_and_return(x),
                                human_input_mode="TERMINATE",
                                max_consecutive_auto_reply=2,
                                system_message="Human agent. Interact with customer support manager to either solve "
                                               "the query or come up with a step by step plan to solve the query. "
                                               "Reply TERMINATE if the task has been solved at full satisfaction. "
                                               "Otherwise, reply CONTINUE, or the reason why the task is not solved "
                                               "yet.")

    customer_support_critique = RetrieveAssistantAgent(
        name="customer_support_critique",
        system_message=cs_critique_system_message,
        llm_config=default_llm_config
    )

    customer_support_manager = RetrieveAssistantAgent(
        name="customer_support_manager",
        system_message=cs_manager_system_message,
        llm_config=default_llm_config
    )

    customer_support_rag = RetrieveUserProxyAgent(
        name="customer_support_rag",
        max_consecutive_auto_reply=50,
        human_input_mode="NEVER",
        retrieve_config={
            "task": "qa",
            "docs_path": "collect.pdf",
            "collection_name": "zeoai",
            "chunk_token_size": 2000,
            "model": config_list[0]["model"],
            "client": chromadb.PersistentClient(path="./chromadb_folder"),
            "embedding_model": "all-mpnet-base-v2",
            "customized_prompt": custom_rag_prompt,
            "get_or_create": True
        },
    )

    cs_group_chat = autogen.GroupChat(
        agents=[user_proxy, customer_support_manager, customer_support_critique, customer_support_rag],
        max_round=20,
        messages=[]
    )

    cs_group_chat_manager = autogen.GroupChatManager(
        groupchat=cs_group_chat,
        llm_config=default_llm_config,
        name="cs_group_chat_manager",
        system_message=customer_support_chat_manager_system_prompt.format(user_msg)
    )

    user_proxy.initiate_chat(cs_group_chat_manager, message=user_msg)
                             # search_string=user_msg, clear_history=False)
    user_proxy.stop_reply_at_receive(customer_support_manager)
    user_proxy.send(
        message="Give me the result you came up with. Only return the step by step plan. Do not return anything else",
        request_reply=True,
        recipient=customer_support_manager
    )
    return user_proxy.last_message(customer_support_manager)["content"]


def pm_verify_instructions_from_manager(user_msg, manager_instructions, critical_prompt):
    print(f"user_msg: {user_msg}, manager_instructions: {manager_instructions}, critical_prompt: {critical_prompt}")

    user_proxy = UserProxyAgent("user_proxy",
                                code_execution_config={"work_dir": "coding"},
                                is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
                                human_input_mode="TERMINATE",
                                max_consecutive_auto_reply=2,
                                system_message="Human agent. Reply TERMINATE if the task has been solved at full satisfaction. Otherwise, reply CONTINUE, or the reason why the task is not solved yet.")

    pm_expert_critique = RetrieveAssistantAgent(
        name="pm_expert_critique",
        system_message=pm_verify_critique_system_message.format(CRITICAL_PROMPT=critical_prompt),
        llm_config={"config_list": config_list, "temperature": 0.5}
    )

    pm = AssistantAgent(
        name="pm",
        system_message=pm_verify_system_message,
        llm_config=default_llm_config
    )

    pm_rag = RetrieveUserProxyAgent(
        name="pm_rag",
        max_consecutive_auto_reply=50,
        human_input_mode="NEVER",
        retrieve_config={
            "task": "qa",
            "docs_path": "collect.pdf",
            "collection_name": "zeoai",
            "chunk_token_size": 2000,
            "model": config_list[0]["model"],
            "client": chromadb.PersistentClient(path="./chromadb_folder"),
            "embedding_model": "all-mpnet-base-v2",
            "customized_prompt": custom_rag_prompt
        },
    )

    # pm_rag.initiate_chat(pm, problem=manager_instructions, search_string=manager_instructions)

    pm_group_chat = autogen.GroupChat(
        agents=[user_proxy, pm, pm_expert_critique, pm_rag],
        max_round=20,
        messages=[]
    )

    pm_group_chat_manager = autogen.GroupChatManager(
        groupchat=pm_group_chat,
        llm_config=default_llm_config,
        name="pm_group_chat_manager",
        system_message="make sure feedback given by the expert is incorporated in the instructions. ",  # TODO: Improve
    )

    user_proxy.initiate_chat(pm_group_chat_manager, message=user_msg, search_string=manager_instructions)
    user_proxy.stop_reply_at_receive(pm)
    msg = f"""
    Give the output of your previous conversation in the following json format."
                "Keys: correct_instructions: . Output format: {OUTPUT_FORMAT_TASK_VERIFICATION}
"""
    user_proxy.send(
        message=msg,
        request_reply=True,
        recipient=pm_group_chat_manager
    )
    return user_proxy.last_message(pm_group_chat_manager)["content"]


global_context = {}


# Manager instructions are instructions given by PM
def pm_task_division_after_verification(manager_instructions):
    global global_context
    user_proxy = UserProxyAgent("user_proxy",
                                is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
                                human_input_mode="TERMINATE",
                                max_consecutive_auto_reply=20,
                                system_message="Human agent. Reply TERMINATE if the task has been solved at full satisfaction. Otherwise, reply CONTINUE, and the reason why the task is not solved yet.")

    pm_task_delegator = AssistantAgent(
        name="pm_task_delegator",
        system_message=pm_task_delegation_system_message,
        llm_config={"config_list": config_list, "temperature": 0.5}
    )

    pm_task_delegator_critic = RetrieveAssistantAgent(
        name="pm_task_delegator_critic",
        system_message=pm_task_delegation_critic_system_message.format(USER_QUERY=manager_instructions),
        llm_config={"config_list": config_list, "temperature": 0}
    )

    pm_task_delegator_rag = RetrieveUserProxyAgent(
        name="pm_task_delegator_rag",
        max_consecutive_auto_reply=50,
        human_input_mode="NEVER",
        retrieve_config={
            "docs_path": "collect.pdf",
            "collection_name": "zeoai",
            "chunk_token_size": 2000,
            "model": config_list[0]["model"],
            "client": chromadb.PersistentClient(path="./chromadb_folder"),
            "embedding_model": "all-mpnet-base-v2",
            "get_or_create": True,
        }
        #     todo: add custom rag prompt
    )

    pm_delegate_group_chat = autogen.GroupChat(
        agents=[user_proxy, pm_task_delegator, pm_task_delegator_critic, pm_task_delegator_rag],
        max_round=20,
        messages=[]
    )

    pm_delegate_group_chat_manager = autogen.GroupChatManager(
        groupchat=pm_delegate_group_chat,
        llm_config=default_llm_config,
        name="pm_delegate_group_chat_manager"
    )

    user_proxy.initiate_chat(pm_delegate_group_chat_manager, message=manager_instructions,
                             search_string=manager_instructions)

    user_proxy.stop_reply_at_receive(pm_task_delegator)
    user_proxy.send(
        message=f"Give the output of your previous conversation in the following json format. {OUTPUT_FORMAT_TASK_DELEGATION}",
        request_reply=True,
        recipient=pm_task_delegator
    )
    return user_proxy.last_message(pm_task_delegator)["content"]


def verify_delegate_instructions_from_manager(instruction, user_msg):
    user_proxy_new = UserProxyAgent("user_proxy",
                                    is_termination_msg=lambda x: print_and_return(x),
                                    max_consecutive_auto_reply=4,
                                    human_input_mode="TERMINATE",
                                    system_message="Human agent. Interact with manager to verify if instructions recieved are correct or not. Reply TERMINATE if the task has been solved at full satisfaction. Otherwise, reply CONTINUE, or the reason why the task is not solved yet.")

    manager_llm_config = {"config_list": config_list,
                          "functions": [
                              {
                                  "name": "pm_verify_instructions_from_manager",
                                  "description": "Verify if the given set of instructions are complete, correct and sufficient to solve the task.",
                                  "parameters": {
                                      "type": "object",
                                      "properties": {
                                          "user_msg": {
                                              "type": "string",
                                              "description": "The query asked by user",
                                          },
                                          "manager_instructions": {
                                              "type": "string",
                                              "description": "Instruction given by user proxy to solve the task. These instructions need to be verified",
                                          },
                                          "critical_prompt": {
                                              "type": "string",
                                              "description": "Prompt to be used for critical feedback.",
                                          },
                                      },
                                      "required": ["user_msg", "manager_instructions", "critical_prompt"],
                                  },
                              },
                              {
                                  "name": "pm_task_division_after_verification",
                                  "description": "This categorizes the every single task from a list of tasks by manager and delegate every task to appropriate agent for completion.",
                                  "parameters": {
                                      "type": "object",
                                      "properties": {
                                          "manager_instructions": {
                                              "type": "string",
                                              "description": "Instruction given by manager to solve the task.",
                                          }
                                      },
                                      "required": ["manager_instructions"],
                                  },
                              }
                          ],
                          "seed": 12,
                          "request_timeout": 1000}

    manager = autogen.AssistantAgent(
        max_consecutive_auto_reply=10,
        name="manager",
        system_message=manager_system_message.format(USER_INSTRUCTIONS=instruction, USER_QUERY=user_msg,
                                                     CRITICAL_PROMPT="Verify if each step mentioned by the manager is correct, complete and sufficient to solve the user query. Evaluate from the pov that a developer agent and PM agent would be perfomring the task. The instructions should be catering to them so they can produce the desired output. Only call pm_task_division_after_verification if instructions are successfully verified.",
                                                     OUTPUT_FORMAT_TASK_DELEGATION=OUTPUT_FORMAT_TASK_DELEGATION),
        llm_config=manager_llm_config)

    user_proxy_new.register_function(
        function_map={"pm_verify_instructions_from_manager": pm_verify_instructions_from_manager,
                      "pm_task_division_after_verification": pm_task_division_after_verification}
    )
    user_proxy_new.initiate_chat(manager, message=instruction, clear_history=False)
    user_proxy_new.stop_reply_at_receive(manager)
    user_proxy_new.send(
        message=f"Give the result of your result generated in your last conversation in the json format. To remind you, the format is {OUTPUT_FORMAT_TASK_DELEGATION}",
        request_reply=True,
        recipient=manager
    )
    print(user_proxy_new.last_message(manager)["content"])


def generate_catalog_fields():
    catalog_attr = get_list_catalog_attr_pm_brainstorm()
    # TODO: call the function to create catalog fields
