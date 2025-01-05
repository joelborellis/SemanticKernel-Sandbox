from typing import Annotated

from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.prompt_template.prompt_template_config import PromptTemplateConfig

from tools.searchshadow import SearchShadow
from tools.searchcustomer import SearchCustomer

# Create the search clients inside the plugin file
search_client = SearchShadow()
search_customer_client = SearchCustomer()

class ShadowInsightsPlugin:
    """Plugin class that accepts a PromptTemplateConfig for advanced configuration."""

    # The max tokens to process in a single semantic function call.
    _max_tokens = 1024

    def __init__(self, prompt_template_config: PromptTemplateConfig):
        """
        :param prompt_template_config: A PromptTemplateConfig object used for advanced template configuration.
        """
        self.prompt_template_config = prompt_template_config

    @kernel_function(
        name="get_sales_docs",
        description="Given a user query, search the shadow sales strategy index."
    )
    def get_sales_docs(self, query: Annotated[str, "The query from the user."]
    ) -> Annotated[str, "Returns documents from the shadow sales strategy index."]:
        print(f"user_query:  {query}")
        docs = search_client.search_hybrid(query)
        # Optionally, you can make use of self.prompt_template_config if needed
        # e.g., config_params = self.prompt_template_config.parameters
        return docs

    @kernel_function(
        name="get_customer_docs",
        description="Given a user query determine if a company name was mentioned. Use the company name and the query information to search the shadow customer index."
    )
    def get_customer_docs(self, query: Annotated[str, "The query and the customer name from the user."]
    ) -> Annotated[str, "Returns documents from the shadow customer index."]:
        print(f"user_customer_query:  {query}")
        docs = search_customer_client.search_hybrid(query)
        # Optionally, you can make use of self.prompt_template_config if needed
        return docs