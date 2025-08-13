"""
Central configuration for all system limits, timeouts, and OpenAI API constraints.
"""

import os

# File and data limits
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
MAX_NUMERIC_COLUMNS_TO_PROFILE = 10  # Limit for data profiling
MAX_OBJECT_COLUMNS_TO_PROFILE = 10  # Limit for data profiling

# Execution limits
EXECUTION_TIMEOUT = 300  # 30 seconds for script execution
MEMORY_LIMIT = 2 * 1024 * 1024 * 1024  # 2GB in bytes
TOTAL_REQUEST_TIMEOUT = 300  # 5 minutes total timeout

# Iteration limits
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))  # Configurable, default 3
MAX_RETRIES = 3  # Max retries for API calls

# OpenAI token limits
ARCHITECT_INITIAL_MAX_TOKENS = 20000  # For initial requirements generation
ARCHITECT_VALIDATION_MAX_TOKENS = 20000  # For validation responses
CODER_MAX_TOKENS = 20000  # For code generation and revision

# Model constraints (from tokens_reference.md)
GPT5_MAX_INPUT_TOKENS = 272000
GPT5_MAX_OUTPUT_TOKENS = 128000
GPT5_TOTAL_CONTEXT_WINDOW = 400000

# Pydantic model limits
MAX_ACCEPTANCE_CRITERIA = 5  # Maximum number of acceptance criteria items