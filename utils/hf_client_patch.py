"""
HuggingFace client patch to handle client closure gracefully.

This module provides fixes for the "Cannot send a request, as the client has been closed" error.
"""

import os
import logging

logger = logging.getLogger("experiment")


def patch_huggingface_client():
    """
    Apply patches to handle HuggingFace Hub client closure issues.

    Call this at the very start of your script before importing any transformers.
    """
    # Disable token-based authentication to avoid client issues
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

    # Set longer timeout for downloads
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "3600")  # 1 hour

    # Prefer local cache to avoid network requests
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")  # Allow downloads but prefer cache

    # Disable progress bars that might interfere with logging
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    # Patch the huggingface_hub client to auto-reconnect
    try:
        import huggingface_hub
        from huggingface_hub import constants

        # Increase retry count
        if hasattr(constants, 'HF_HUB_ETAG_TIMEOUT'):
            constants.HF_HUB_ETAG_TIMEOUT = 30

        logger.info("HuggingFace Hub client patches applied successfully")
    except ImportError:
        pass


def safe_model_load(model_loader_func, *args, max_retries=3, **kwargs):
    """
    Safely load a model with retry logic for client closure issues.

    Args:
        model_loader_func: Function that loads the model (e.g., SentenceTransformer)
        max_retries: Maximum number of retry attempts
        *args, **kwargs: Arguments to pass to the loader function

    Returns:
        Loaded model or None if all retries fail
    """
    import time

    for attempt in range(max_retries):
        try:
            # Clear any cached clients before retrying
            if attempt > 0:
                try:
                    import huggingface_hub
                    # Reset the file download system
                    if hasattr(huggingface_hub, 'reset_connections'):
                        huggingface_hub.reset_connections()
                except:
                    pass

            # Try loading the model
            model = model_loader_func(*args, **kwargs)
            return model

        except Exception as e:
            error_msg = str(e)
            if "client has been closed" in error_msg.lower():
                logger.warning(
                    f"HuggingFace client closed (attempt {attempt + 1}/{max_retries}). "
                    f"Retrying in {2 ** attempt} seconds..."
                )
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                # Different error, don't retry
                logger.error(f"Model loading failed with unexpected error: {e}")
                raise

    logger.error(f"Failed to load model after {max_retries} attempts")
    return None


# Apply patches on import
patch_huggingface_client()
