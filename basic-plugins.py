import asyncio
from typing import AsyncGenerator
from jinja2 import Environment, FileSystemLoader
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.function_choice_behavior import (
    FunctionChoiceBehavior,
)
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import (
    OpenAIChatCompletion,
)
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.connectors.ai.prompt_execution_settings import (
    PromptExecutionSettings,
)
from semantic_kernel.prompt_template.prompt_template_config import PromptTemplateConfig
from semantic_kernel.kernel import Kernel

# Import the modified plugin class
from plugins.shadow_insights_plugin import ShadowInsightsPlugin
from utils.log_chat_history import log_chat_history, extract_history

# Load Jinja2 environment
PROMPT_DIR = "prompts"  # Directory where your XML templates are stored
env = Environment(loader=FileSystemLoader(PROMPT_DIR), autoescape=True)

# This sample allows for a streaming response verus a non-streaming response
streaming = True

# 7) Define the chat history
chat = ChatHistory()

# Define the agent name and instructions
AGENT_NAME = "ShadowAgent"
agent_prompt = env.get_template("agent_prompt.xml")
AGENT_INSTRUCTIONS = agent_prompt.render()

def manage_file(filename, data):
    """
    Creates a file if it doesn't exist or appends data to the file if it already exists.

    :param filename: The name of the file to create or append to.
    :param data: The data to write to the file.
    """
    try:
        with open(filename, "w") as file:  # Open the file in write mode
            file.write(data + "\n")
        print(f"Data added to {filename}")
    except Exception as e:
        print(f"An error occurred: {e}")

#@log_chat_history(chat, parser_function=extract_history)
async def invoke_agent_by_line(
    agent: ChatCompletionAgent, query: str, chat: ChatHistory, streaming: bool) -> AsyncGenerator[str, None]:
    
    """Invoke the agent with the user input."""
    chat.add_user_message(query)
    print(chat.serialize())
    print(f"# {AuthorRole.USER}: \n'{query}'\n")
    if streaming:
        async for content in agent.invoke_stream(chat):
             # content.content is a chunk (could have multiple lines).
            # We split by newline (or use splitlines(True) to keep the newline).
            lines = content.content.splitlines(True)
            for line in lines:
                # Yield each line individually
                yield line
        chat.add_message(content)
    else:
        # Non-streaming approach
        async for content in agent.invoke(chat):
            # Instead of printing, yield to main
            yield f"# {content.role} - {content.name or '*'}: '{content.content}'\n"
        chat.add_message(content)


async def main():
    # 1) Create the instance of the Kernel
    kernel = Kernel()

    # Setup the Agent
    service_id = "shadow_agent"
    kernel.add_service(
        OpenAIChatCompletion(ai_model_id="gpt-4o", service_id=service_id)
    )

    # 2) Retrieve the PromptExecutionSettings
    settings = kernel.get_prompt_execution_settings_from_service_id(
        service_id=service_id
    )
    # Configure the function choice behavior to auto-invoke kernel functions
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    # 3) Create a PromptTemplateConfig (configure as needed)
    my_prompt_template_config = PromptTemplateConfig(
        # You can set any custom configuration here
        # E.g., temperature=0.7, max_tokens=2000, etc.
    )

    # 4) Instantiate ShadowInsightsPlugin with the config
    shadow_plugin = ShadowInsightsPlugin(
        prompt_template_config=my_prompt_template_config
    )

    # 5) Register plugin with the Kernel
    kernel.add_plugin(shadow_plugin, plugin_name="shadowRetrievalPlugin")

    # 6) Create the agent
    agent = ChatCompletionAgent(
        service_id="shadow_agent",
        kernel=kernel,
        name=AGENT_NAME,
        instructions=AGENT_INSTRUCTIONS,
        execution_settings=settings,
    )

    # 8) Loop for user input
    while True:
        query = input("\nAsk GPT: ")
        if query.lower() == "exit":
            exit(0)

        # Decide if streaming or not (assume you have a `streaming` flag)
        if streaming:
            #print("\n[Streaming line-by-line...]")
            async for line in invoke_agent_by_line(agent, query, chat, streaming=True):
                print(line, end="", flush=True)
        else:
            #print("\n[Non-streaming response...]")
            async for chunk in invoke_agent_by_line(agent, query, chat, streaming=False):
                print(chunk, end="", flush=True)

        #print(chat.serialize())

if __name__ == "__main__":
    asyncio.run(main())
