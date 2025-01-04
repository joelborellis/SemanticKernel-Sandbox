import asyncio
from typing import Annotated

from tools.searchshadow import SearchShadow
from tools.searchcustomer import SearchCustomer

from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.kernel import Kernel
from jinja2 import Environment, FileSystemLoader

import time

###################################################################
# The following sample demonstrates how to create a simple,       #
# non-group agent that utilizes plugins defined as part of        #
# the Kernel.                                                     #
###################################################################

search_client = SearchShadow()
search_customer_client = SearchCustomer()

# Load Jinja2 environment
PROMPT_DIR = "prompts"  # Directory where your XML templates are stored
env = Environment(loader=FileSystemLoader(PROMPT_DIR), autoescape=True)

# This sample allows for a streaming response verus a non-streaming response
streaming = True

# Define the agent name and instructions
AGENT_NAME = "Shadow"
# Render select_file_prompt template
create_content_prompt = env.get_template("shadow_prompt.xml")
AGENT_INSTRUCTIONS = create_content_prompt.render()

def manage_file(filename, data):
    """
    Creates a file if it doesn't exist or appends data to the file if it already exists.

    :param filename: The name of the file to create or append to.
    :param data: The data to write to the file.
    """
    try:
        with open(filename, 'w') as file:  # Open the file in append mode
            file.write(data + '\n')  # Append the new data
        print(f"Data added to {filename}")
    except Exception as e:
        print(f"An error occurred: {e}")


# Define the ShadowPlugin
class ShadowPlugin:
    """A sample Shadow Plugin used for the concept sample."""

    @kernel_function(name="get_sales_docs", description="Given a user query search the shadow sales strategy index.")
    def get_sales_docs(self, query: Annotated[str, "The query from the user."]
    ) -> Annotated[str, "Returns documents from the shadow sales strategy index."]:
        print(f"user_query:  {query}")
        docs = search_client.search_hybrid(query)
        return docs
    
    @kernel_function(name="get_customer_docs", description="Given a user query determine if a company name was mentioned.  Use the company name and the query information to search the shadow customer index.")
    def get_customer_docs(self, query: Annotated[str, "The query and the customer name from the user."]
    ) -> Annotated[str, "Returns documents from the shadow customer index."]:
        print(f"user_customer_query:  {query}")
        docs = search_customer_client.search_hybrid(query)
        return docs

# A helper method to invoke the agent with the user input
async def invoke_agent(agent: ChatCompletionAgent, query: str, chat: ChatHistory) -> None:
    """Invoke the agent with the user input."""
    chat.add_user_message(query)

    print(f"# {AuthorRole.USER}: \n'{query}'\n")

    if streaming:
        contents = []
        agent_name = ""
        async for content in agent.invoke_stream(chat):
            agent_name = content.name
            contents.append(content)
        message_content = "".join([content.content for content in contents])
        # Simulate typing by adding characters one by one
        if message_content:
            print(f"# {content.role} - {agent_name or '*'}:")
            for char in message_content:
                print(char, end="", flush=True)
                # Adjust sleep time to control the "typing" speed
                await asyncio.sleep(0.01)
        chat.add_assistant_message(message_content)
    else:
        async for content in agent.invoke(chat):
            print(f"# {content.role} - {content.name or '*'}: '{content.content}'")
        chat.add_message(content)


async def main():
    # Create the instance of the Kernel
    kernel = Kernel()

    service_id = "shadow_agent"
    kernel.add_service(OpenAIChatCompletion(ai_model_id="gpt-4o", service_id=service_id))

    settings = kernel.get_prompt_execution_settings_from_service_id(service_id=service_id)
    # Configure the function choice behavior to auto invoke kernel functions
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    kernel.add_plugin(ShadowPlugin(), plugin_name="shadow")

    # Create the agent
    agent = ChatCompletionAgent(
        service_id="shadow_agent", kernel=kernel, name=AGENT_NAME, instructions=AGENT_INSTRUCTIONS, execution_settings=settings
    )

    # Define the chat history
    chat = ChatHistory()
    
    while True:
        # Get user query
        query = input(f"\nAsk GPT: ")
        if query.lower() == "exit":
            exit(0)

        # Respond invoke the Shadow agent with the Plugins
        #print(chat)
        await invoke_agent(agent, query, chat)

        manage_file('chat_history.json', chat.serialize())

if __name__ == "__main__":
    asyncio.run(main())