import asyncio
from semantic_kernel.kernel import Kernel
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.prompt_template.prompt_template_config import PromptTemplateConfig
from semantic_kernel.connectors.ai import PromptExecutionSettings
from plugins.conversation_summary_plugin import ConversationSummaryPlugin
from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
)

async def main():
    # Initialize the kernel
    kernel = Kernel()
    
    service_id = "chat"
    
    kernel.add_service(OpenAIChatCompletion(service_id=service_id, ai_model_id="gpt-4o"))

    # Create a prompt template configuration
    #prompt_template_config = PromptTemplateConfig()
    
    # The following execution settings are used for the ConversationSummaryPlugin
    execution_settings = PromptExecutionSettings(
        service_id=service_id, max_tokens=ConversationSummaryPlugin._max_tokens, temperature=0.1, top_p=0.5
    )
    prompt_template_config = PromptTemplateConfig(
        template=ConversationSummaryPlugin._summarize_conversation_prompt_template,
        description="Given a section of a conversation transcript, summarize the part of the conversation.",
        execution_settings=execution_settings,
    )
    
    # Initialize the ConversationSummaryPlugin
    conversation_summary_plugin = ConversationSummaryPlugin(prompt_template_config)

    # Add the plugin to the kernel
    kernel.add_plugin(conversation_summary_plugin, 'summarizer')

    # Define a long conversation transcript
    conversation_transcript = """
    Alice: Hi Bob, how are you?
    Bob: I'm good, Alice. How about you?
    Alice: I'm doing well, thanks. Have you finished the project?
    Bob: Yes, I completed it yesterday. I'll send you the report.
    Alice: Great! Let's discuss it in the meeting tomorrow.
    Bob: Sure, see you then.
    """

    # Create kernel arguments
    arguments = KernelArguments()

    # Summarize the conversation
    summarized_arguments = await conversation_summary_plugin.summarize_conversation(
        input=conversation_transcript,
        kernel=kernel,
        arguments=arguments
    )

    # Print the summarized conversation
    print(summarized_arguments[conversation_summary_plugin.return_key])

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())