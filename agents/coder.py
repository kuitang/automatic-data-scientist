import os
import logging
import asyncio
from pathlib import Path
from typing import Optional
import openai
import yaml
import time

logger = logging.getLogger(__name__)

class CoderAgent:
    def __init__(self):
        self.client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = self._load_model_config()
        self.max_retries = 3
        self.base_delay = 1
        
    def _load_model_config(self) -> str:
        config_path = Path(__file__).parent.parent / "config" / "models.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('coder_model', 'gpt-5')
        return 'gpt-5'
    
    async def generate_code(self, requirements: str, data_path: Path) -> str:
        logger.info("\n" + "="*80)
        logger.info("CODER: Starting code generation")
        logger.info("="*80)
        logger.info(f"Data path: {data_path}")
        logger.info(f"\nRequirements to implement:")
        logger.info("-" * 40)
        logger.info(requirements)
        logger.info("-" * 40)
        
        prompt = self._load_prompt('coder_initial.txt')
        
        file_ext = data_path.suffix.lower()
        
        system_message = f"""You are a Python data analysis code generator. Generate complete, executable Python scripts that:
1. Read data from a file path provided via --data command line argument
2. Output valid HTML to stdout with embedded charts (SVG or base64 PNG)
3. Use only: pandas, numpy, matplotlib, seaborn, scipy, sklearn, plotly
4. Do not access network or write files (except temp files for plotting)
5. Include proper error handling

The data file is a {file_ext} file."""

        user_message = prompt.format(
            requirements=requirements,
            file_extension=file_ext
        )
        
        if not user_message:
            # Fallback if prompt file doesn't exist
            user_message = f"""Generate a Python script that:
{requirements}

The script should:
- Accept data file path via argparse as --data argument
- Read the {file_ext} file
- Perform the requested analysis
- Output a complete HTML document to stdout with all results and visualizations embedded
- Use base64 encoding or SVG for all charts
- Include proper styling and formatting"""

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
                logger.info(f"Temperature: 0.3")
                logger.info(f"Max Tokens: 4000")
                logger.info(f"\n--- SYSTEM MESSAGE ---")
                logger.info(system_message)
                logger.info(f"\n--- USER MESSAGE (first 1000 chars) ---")
                logger.info(user_message[:1000] + ("...\n[TRUNCATED]" if len(user_message) > 1000 else ""))
                
                start_time = time.time()
                
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=4000
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
    
    async def revise_code(self, previous_code: str, requirements: str, feedback: str, data_path: Path) -> str:
        logger.info("\n" + "="*80)
        logger.info("CODER: Starting code revision")
        logger.info("="*80)
        logger.info(f"Previous code length: {len(previous_code)} characters")
        logger.info(f"\nFeedback to address:")
        logger.info("-" * 40)
        logger.info(feedback)
        logger.info("-" * 40)
        
        prompt = self._load_prompt('coder_revision.txt')
        
        file_ext = data_path.suffix.lower()
        
        system_message = f"""You are a Python data analysis code generator. Fix the provided code based on feedback.
Maintain the same structure but fix the specific issues mentioned.
The data file is a {file_ext} file."""

        user_message = prompt.format(
            previous_code=previous_code,
            requirements=requirements,
            feedback=feedback
        )
        
        if not user_message:
            # Fallback if prompt file doesn't exist
            user_message = f"""Fix this Python script based on the feedback:

Previous Code:
{previous_code}

Original Requirements:
{requirements}

Feedback/Errors to Fix:
{feedback}

Generate the complete fixed script."""

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
                logger.info(f"Temperature: 0.3")
                logger.info(f"Max Tokens: 4000")
                logger.info(f"\n--- REVISION SYSTEM MESSAGE ---")
                logger.info(system_message)
                logger.info(f"\n--- REVISION USER MESSAGE (first 1500 chars) ---")
                logger.info(user_message[:1500] + ("...\n[TRUNCATED]" if len(user_message) > 1500 else ""))
                
                start_time = time.time()
                
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=4000
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
    
    def _load_prompt(self, filename: str) -> str:
        prompt_path = Path(__file__).parent.parent / "config" / "prompts" / filename
        if prompt_path.exists():
            return prompt_path.read_text()
        return ""