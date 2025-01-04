import asyncio
from typing import Annotated
from semantic_kernel.kernel import Kernel
from conversation_summary import test_azure_summarize_conversation_using_plugin

###################################################################
# The following sample demonstrates how to create a simple,       #
# non-group agent that utilizes plugins defined as part of        #
# the Kernel.                                                     #
###################################################################


async def main():
    # Create the instance of the Kernel
    kernel = Kernel()

    await test_azure_summarize_conversation_using_plugin(kernel)


if __name__ == "__main__":
    asyncio.run(main())