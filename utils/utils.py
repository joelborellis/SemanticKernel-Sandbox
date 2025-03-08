# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
import os
import platform
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)


async def retry(
    func: Callable[..., Awaitable[Any]],
    retries: int = 20,
    reset: Callable[..., None] | None = None,
    name: str | None = None,
):
    """Retry the function if it raises an exception.

    Args:
        func (function): The function to retry.
        retries (int): Number of retries.
        reset (function): Function to reset the state of any variables used in the function

    """
    logger.info(f"Running {retries} retries with func: {name or func.__module__}")
    for i in range(retries):
        logger.info(f"   Try {i + 1} for {name or func.__module__}")
        try:
            if reset:
                reset()
            return await func()
        except Exception as e:
            logger.warning(f"   On try {i + 1} got this error: {e}")
            if i == retries - 1:  # Last retry
                raise
            # Binary exponential backoff
            backoff = 2**i
            logger.info(f"   Sleeping for {backoff} seconds before retrying")
            await asyncio.sleep(backoff)
    return None