import os
import logging
import asyncio
from pathlib import Path
from typing import Optional
import openai
import yaml
import time
from limits import (
    MAX_RETRIES,
    CODER_MAX_TOKENS
)

logger = logging.getLogger(__name__)

# ==================== CODER AGENT PROMPTS ====================

# System prompt template for initial code generation
CODER_INITIAL_SYSTEM_PROMPT = """You are a Python data analysis code generator. Generate complete, executable Python scripts that:
1. Read data from a file path provided via --data command line argument
2. Output valid Markdown to stdout with links to saved PNG images
3. Use only: pandas, numpy, matplotlib, seaborn, scipy, sklearn, plotly
4. Save plot images as PNG files in the current working directory
5. Include proper error handling

The data file is a {file_ext} file."""

# User prompt template for initial code generation
CODER_INITIAL_USER_PROMPT = """Generate a complete Python script that implements the following requirements:

{requirements}

Acceptance Criteria (must be met):
{acceptance_criteria}

The script MUST:
1. Use argparse to accept a --data argument for the input file path
2. Read the {file_extension} file from the provided path
3. Perform all requested analyses and generate visualizations
4. Output a complete, valid Markdown document to stdout
5. Save all plots as PNG files in the current working directory with descriptive names
6. Include links to the saved images in the Markdown output using relative paths
7. Include proper error handling for common issues
8. Use only these packages: pandas, numpy, matplotlib, seaborn, scipy, sklearn, plotly

Structure the Markdown output with:
- A clear title and sections using Markdown headers (#, ##, ###)
- In the introduction of the report, summarize the requirements you have been given
- Tables formatted using Markdown table syntax
- Clear labels and descriptions for all results
- Links to images using Markdown syntax: ![Description](filename.png)
- IMPORTANT: For each plot/visualization, include a one-sentence takeaway text that explains what the plot intends to communicate (the evaluator cannot see plots, only text)

Image Guidelines:
- Keep all generated images small and low resolution to minimize file size
- Use figure sizes no larger than (6, 4) inches
- Set DPI to 72 for web display (not print quality)
- Save as PNG with compression
- Use descriptive filenames (e.g., 'distribution_plot.png', 'correlation_matrix.png')
- Close matplotlib figures after saving to free memory

Example of saving and linking an image:
```python
plt.figure(figsize=(6, 4), dpi=72)
plt.plot(data)
plt.title('Data Trend')
plt.savefig('data_trend.png', dpi=72, bbox_inches='tight')
plt.close()
print("![Data trend over time](data_trend.png)")
```

Do not:
- Make any network requests
- Use packages not in the allowed list
- Create subdirectories (save all files in the current working directory)

Generate the complete, executable Python script."""

# System prompt template for code revision
CODER_REVISION_SYSTEM_PROMPT = """You are a Python data analysis code generator. Fix the provided code based on feedback.
Maintain the same structure but fix the specific issues mentioned.
The data file is a {file_ext} file."""

# User prompt template for code revision
CODER_REVISION_USER_PROMPT = """Fix the Python script below based on the provided feedback.

Previous Code:
{previous_code}

Previous Output (what your code produced):
{previous_output}

Original Requirements:
{requirements}

Acceptance Criteria (must be met):
{acceptance_criteria}

Previous Grade:
{grade}

Grade Justification:
{grade_justification}

Feedback/Issues to Fix:
{feedback}

Generate the complete FIXED Python script that addresses all the feedback while maintaining the original requirements and meeting the acceptance criteria. Make sure to:
1. Fix any errors or issues mentioned in the feedback
2. Maintain the same overall structure and approach
3. Keep all the working parts of the previous code
4. Ensure the script still meets all original requirements AND acceptance criteria
5. Output Markdown to stdout with links to saved PNG images
6. Keep images small and low resolution (figure size ≤ (6,4), DPI ≤ 72)
7. Save all plots as PNG files with descriptive names
8. For each plot/visualization, include a one-sentence takeaway text that explains what the plot intends to communicate (the evaluator cannot see plots, only text)

Generate the complete, executable Python script with all fixes applied."""

class CoderAgent:
    def __init__(self):
        self.client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = self._load_model_config()
        self.max_retries = MAX_RETRIES
        self.base_delay = 1
        
    def _load_model_config(self) -> str:
        config_path = Path(__file__).parent.parent / "config" / "models.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('coder_model', 'gpt-5')
        return 'gpt-5'
    
    
    async def generate_code(self, requirements: str, acceptance_criteria: list, data_path: Path) -> str:
        logger.info("\n" + "="*80)
        logger.info("CODER: Starting code generation")
        logger.info("="*80)
        logger.info(f"Data path: {data_path}")
        logger.info(f"\nRequirements to implement:")
        logger.info("-" * 40)
        logger.info(requirements)
        logger.info("-" * 40)
        
        file_ext = data_path.suffix.lower()
        
        # Use the prompts defined at the top of the file
        system_message = CODER_INITIAL_SYSTEM_PROMPT.format(file_ext=file_ext)
        user_message = CODER_INITIAL_USER_PROMPT.format(
            requirements=requirements,
            acceptance_criteria="\n".join(f"- {c}" for c in acceptance_criteria),
            file_extension=file_ext
        )

        
        for attempt in range(self.max_retries):
            try:
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]
                
                logger.info(f"\n{'='*60}")
                logger.info(f"💻 CODER -> OpenAI API Call (Attempt {attempt + 1}/{self.max_retries})")
                logger.info(f"{'='*60}")
                logger.info(f"Model: {self.model}")
                logger.info(f"Max Tokens: {CODER_MAX_TOKENS}")
                logger.info(f"\n--- SYSTEM MESSAGE ---")
                logger.info(system_message)
                logger.info(f"\n--- USER MESSAGE (first 1000 chars) ---")
                logger.info(user_message[:1000] + ("...\n[TRUNCATED]" if len(user_message) > 1000 else ""))
                
                start_time = time.time()
                
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=CODER_MAX_TOKENS
                )
                
                elapsed_time = time.time() - start_time
                
                code = response.choices[0].message.content
                
                logger.info(f"\n{'='*60}")
                logger.info(f"✅ OpenAI -> CODER Response")
                logger.info(f"{'='*60}")
                logger.info(f"Response Time: {elapsed_time:.2f} seconds")
                logger.info(f"Finish Reason: {response.choices[0].finish_reason}")
                if hasattr(response, 'usage'):
                    logger.info(f"Token Usage:")
                    logger.info(f"  - Prompt Tokens: {response.usage.prompt_tokens}")
                    logger.info(f"  - Completion Tokens: {response.usage.completion_tokens}")
                    logger.info(f"  - Total Tokens: {response.usage.total_tokens}")
                
                # Extract code if wrapped in markdown code blocks
                original_length = len(code)
                if '```python' in code:
                    code = code.split('```python')[1].split('```')[0]
                elif '```' in code:
                    code = code.split('```')[1].split('```')[0]
                
                logger.info(f"\n--- GENERATED CODE ---")
                logger.info(f"Code length: {len(code)} characters")
                logger.info(f"First 500 characters of generated code:")
                logger.info("-" * 40)
                logger.info(code[:500] + ("...\n[TRUNCATED]" if len(code) > 500 else ""))
                logger.info("-" * 40)
                logger.info(f"\nCODER: Successfully generated Python analysis script")
                return code.strip()
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                else:
                    raise Exception(f"Failed to generate code after {self.max_retries} attempts: {str(e)}")
    
    async def revise_code(self, previous_code: str, requirements: str, acceptance_criteria: list, feedback: str, data_path: Path, grade: str, grade_justification: str, previous_output: Optional[str] = None) -> str:
        logger.info("\n" + "="*80)
        logger.info("CODER: Starting code revision")
        logger.info("="*80)
        logger.info(f"Previous code length: {len(previous_code)} characters")
        logger.info(f"\nPrevious Grade: {grade}")
        logger.info(f"Grade Justification: {grade_justification}")
        logger.info(f"\nFeedback to address:")
        logger.info("-" * 40)
        logger.info(feedback)
        logger.info("-" * 40)
        
        file_ext = data_path.suffix.lower()
        
        # Use the prompts defined at the top of the file
        system_message = CODER_REVISION_SYSTEM_PROMPT.format(file_ext=file_ext)
        user_message = CODER_REVISION_USER_PROMPT.format(
            previous_code=previous_code,
            previous_output=previous_output if previous_output else "Not available",
            requirements=requirements,
            acceptance_criteria="\n".join(f"- {c}" for c in acceptance_criteria),
            grade=grade,
            grade_justification=grade_justification,
            feedback=feedback
        )

        
        for attempt in range(self.max_retries):
            try:
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]
                
                logger.info(f"\n{'='*60}")
                logger.info(f"🔄 CODER -> OpenAI Revision Call (Attempt {attempt + 1}/{self.max_retries})")
                logger.info(f"{'='*60}")
                logger.info(f"Model: {self.model}")
                logger.info(f"Max Tokens: {CODER_MAX_TOKENS}")
                logger.info(f"\n--- REVISION SYSTEM MESSAGE ---")
                logger.info(system_message)
                logger.info(f"\n--- REVISION USER MESSAGE (first 1500 chars) ---")
                logger.info(user_message[:1500] + ("...\n[TRUNCATED]" if len(user_message) > 1500 else ""))
                
                start_time = time.time()
                
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=CODER_MAX_TOKENS
                )
                
                elapsed_time = time.time() - start_time
                
                code = response.choices[0].message.content
                
                logger.info(f"\n{'='*60}")
                logger.info(f"✅ OpenAI -> CODER Revision Response")
                logger.info(f"{'='*60}")
                logger.info(f"Response Time: {elapsed_time:.2f} seconds")
                logger.info(f"Finish Reason: {response.choices[0].finish_reason}")
                if hasattr(response, 'usage'):
                    logger.info(f"Token Usage:")
                    logger.info(f"  - Prompt Tokens: {response.usage.prompt_tokens}")
                    logger.info(f"  - Completion Tokens: {response.usage.completion_tokens}")
                    logger.info(f"  - Total Tokens: {response.usage.total_tokens}")
                
                # Extract code if wrapped in markdown code blocks
                if '```python' in code:
                    code = code.split('```python')[1].split('```')[0]
                elif '```' in code:
                    code = code.split('```')[1].split('```')[0]
                
                logger.info(f"\n--- REVISED CODE ---")
                logger.info(f"Revised code length: {len(code)} characters")
                logger.info(f"First 500 characters of revised code:")
                logger.info("-" * 40)
                logger.info(code[:500] + ("...\n[TRUNCATED]" if len(code) > 500 else ""))
                logger.info("-" * 40)
                logger.info(f"\nCODER: Successfully revised Python script based on feedback")
                return code.strip()
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                else:
                    raise Exception(f"Failed to revise code after {self.max_retries} attempts: {str(e)}")