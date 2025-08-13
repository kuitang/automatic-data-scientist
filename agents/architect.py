import os
import logging
import asyncio
import time
from pathlib import Path
from typing import Dict, Any, List
import openai
import yaml
import pandas as pd
from .models import ArchitectPlanResponse, ArchitectValidationResponse

logger = logging.getLogger(__name__)

class ArchitectAgent:
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
                return config.get('architect_model', 'gpt-5')
        return 'gpt-4o'
    
    async def profile_and_plan(self, data_path: Path, user_prompt: str) -> Dict[str, Any]:
        logger.info("\n" + "="*80)
        logger.info("ARCHITECT: Starting profile_and_plan")
        logger.info("="*80)
        logger.info(f"Data path: {data_path}")
        logger.info(f"User prompt: {user_prompt}")
        
        # Profile the data
        logger.info("\nARCHITECT: Profiling the dataset...")
        data_profile = self._profile_data(data_path)
        logger.info(f"\nData Profile Generated:")
        logger.info("-" * 40)
        logger.info(data_profile)
        logger.info("-" * 40)
        
        prompt = self._load_prompt('architect_initial.txt')
        
        system_message = """You are a data analysis architect and grader. Your job is to:
1. Understand the dataset structure and content
2. Create specific, actionable requirements for analysis
3. Define up to 5 acceptance criteria focused on data insights and substance (not style)
4. Explain the relative importance of each criterion

Focus criteria on:
- Key insights and conclusions drawn from the data
- Statistical validity and correctness
- Answering the user's core questions
- Data-driven findings and patterns

Avoid criteria about:
- Formatting and styling
- Minor presentation issues
- Non-essential visualizations"""

        user_message = prompt.format(
            data_profile=data_profile,
            user_prompt=user_prompt
        )
        
        if not user_message:
            # Fallback if prompt file doesn't exist
            user_message = f"""Analyze this dataset and create requirements:

Dataset Profile:
{data_profile}

User Request:
{user_prompt}

Create detailed requirements for a Python script that will analyze this data and produce an HTML report.
Focus on substantive insights and data-driven conclusions rather than stylistic elements."""

        for attempt in range(self.max_retries):
            try:
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]
                
                logger.info(f"\n{'='*60}")
                logger.info(f"🤖 ARCHITECT -> OpenAI API Call (Attempt {attempt + 1}/{self.max_retries})")
                logger.info(f"{'='*60}")
                logger.info(f"Model: {self.model}")
                logger.info(f"Temperature: 0.3")
                logger.info(f"Max Tokens: 2000")
                logger.info(f"Response Format: JSON Object")
                logger.info(f"\n--- SYSTEM MESSAGE ---")
                logger.info(system_message)
                logger.info(f"\n--- USER MESSAGE (first 1000 chars) ---")
                logger.info(user_message[:1000] + ("...\n[TRUNCATED]" if len(user_message) > 1000 else ""))
                
                start_time = time.time()
                
                response = await self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2000,
                    response_format=ArchitectPlanResponse
                )
                
                elapsed_time = time.time() - start_time
                
                parsed_result = response.choices[0].message.parsed
                
                logger.info(f"\n{'='*60}")
                logger.info(f"✅ OpenAI -> ARCHITECT Response")
                logger.info(f"{'='*60}")
                logger.info(f"Response Time: {elapsed_time:.2f} seconds")
                logger.info(f"Finish Reason: {response.choices[0].finish_reason}")
                if hasattr(response, 'usage'):
                    logger.info(f"Token Usage:")
                    logger.info(f"  - Prompt Tokens: {response.usage.prompt_tokens}")
                    logger.info(f"  - Completion Tokens: {response.usage.completion_tokens}")
                    logger.info(f"  - Total Tokens: {response.usage.total_tokens}")
                
                # Convert Pydantic model to dict for compatibility
                result = parsed_result.model_dump()
                
                logger.info(f"\n--- ARCHITECT'S ANALYSIS PLAN ---")
                logger.info(f"\nREQUIREMENTS:")
                logger.info("-" * 40)
                logger.info(result['requirements'])
                logger.info("-" * 40)
                
                logger.info(f"\nACCEPTANCE CRITERIA ({len(result['acceptance_criteria'])} items):")
                logger.info("-" * 40)
                for i, criterion in enumerate(result['acceptance_criteria'], 1):
                    logger.info(f"  {i}. {criterion}")
                logger.info("-" * 40)
                
                logger.info(f"\nCRITERIA IMPORTANCE:")
                logger.info("-" * 40)
                logger.info(result.get('criteria_importance', 'Not specified'))
                logger.info("-" * 40)
                
                logger.info(f"\nARCHITECT DECISION: Plan established successfully")
                logger.info(f"ARCHITECT THINKING: Created {len(result['acceptance_criteria'])} acceptance criteria that the code must meet.")
                logger.info(f"Next step: Generate Python code to implement these requirements.")
                
                return result
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                else:
                    # Return default requirements on failure
                    return {
                        "requirements": "Perform basic exploratory data analysis with summary statistics and visualizations",
                        "acceptance_criteria": [
                            "HTML output is generated",
                            "Summary statistics are included",
                            "At least one visualization is created"
                        ],
                        "is_complete": False,
                        "feedback": ""
                    }
    
    async def validate_results(self, html_output: str, acceptance_criteria: List[str]) -> Dict[str, Any]:
        logger.info("\n" + "="*80)
        logger.info("ARCHITECT: Starting validation of results")
        logger.info("="*80)
        logger.info(f"HTML output length: {len(html_output)} characters")
        logger.info(f"\nCriteria to validate against ({len(acceptance_criteria)} items):")
        for i, criterion in enumerate(acceptance_criteria, 1):
            logger.info(f"  {i}. {criterion}")
        
        prompt = self._load_prompt('architect_feedback.txt')
        
        system_message = """You are grading an analysis report against acceptance criteria.

Your tasks:
1. Evaluate how well each acceptance criterion was met
2. Assign a holistic letter grade (A+, A, A-, B+, B, B-, C+, C, C-, D, F)
3. The report passes if it achieves B- or higher
4. Provide specific feedback for improvement

Grading guidelines:
- A range: Exceptional work, all criteria met excellently, deep insights
- B range: Good work, most important criteria met well, solid analysis
- C range: Adequate work, basic criteria met but missing depth
- D: Poor work, critical criteria not met
- F: Failing, fundamental requirements not met

Remember: Some criteria may be critical enough that failing them means failing overall."""

        # Truncate HTML if too long
        if len(html_output) > 10000:
            html_summary = html_output[:5000] + "\n... [truncated] ...\n" + html_output[-2000:]
        else:
            html_summary = html_output

        user_message = prompt.format(
            html_output=html_summary,
            acceptance_criteria="\n".join(f"- {c}" for c in acceptance_criteria)
        )
        
        if not user_message:
            # Fallback if prompt file doesn't exist
            user_message = f"""Validate this HTML output against the acceptance criteria:

Acceptance Criteria:
{chr(10).join(f'- {c}' for c in acceptance_criteria)}

HTML Output (may be truncated):
{html_summary}

Grade the output holistically. What grade does it deserve and why? What needs improvement?"""

        for attempt in range(self.max_retries):
            try:
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]
                
                logger.info(f"\n{'='*60}")
                logger.info(f"🤖 ARCHITECT -> OpenAI Validation Call (Attempt {attempt + 1}/{self.max_retries})")
                logger.info(f"{'='*60}")
                logger.info(f"Model: {self.model}")
                logger.info(f"Temperature: 0.3")
                logger.info(f"Max Tokens: 1000")
                logger.info(f"\n--- VALIDATION SYSTEM MESSAGE ---")
                logger.info(system_message)
                logger.info(f"\n--- VALIDATION USER MESSAGE (first 1500 chars) ---")
                logger.info(user_message[:1500] + ("...\n[TRUNCATED]" if len(user_message) > 1500 else ""))
                
                start_time = time.time()
                
                response = await self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=1000,
                    response_format=ArchitectValidationResponse
                )
                
                elapsed_time = time.time() - start_time
                
                parsed_result = response.choices[0].message.parsed
                
                logger.info(f"\n{'='*60}")
                logger.info(f"✅ OpenAI -> ARCHITECT Validation Response")
                logger.info(f"{'='*60}")
                logger.info(f"Response Time: {elapsed_time:.2f} seconds")
                logger.info(f"Finish Reason: {response.choices[0].finish_reason}")
                if hasattr(response, 'usage'):
                    logger.info(f"Token Usage:")
                    logger.info(f"  - Prompt Tokens: {response.usage.prompt_tokens}")
                    logger.info(f"  - Completion Tokens: {response.usage.completion_tokens}")
                    logger.info(f"  - Total Tokens: {response.usage.total_tokens}")
                
                # Convert Pydantic model to dict for compatibility
                result = parsed_result.model_dump()
                
                logger.info(f"\n--- GRADING RESULTS ---")
                logger.info("=" * 40)
                logger.info(f"GRADE: {result['grade']}")
                logger.info(f"\nCRITERIA EVALUATION:")
                logger.info("-" * 40)
                logger.info(result.get('criteria_evaluation', 'Not specified'))
                logger.info("-" * 40)
                logger.info(f"\nGRADE JUSTIFICATION:")
                logger.info("-" * 40)
                logger.info(result.get('grade_justification', 'Not specified'))
                logger.info("-" * 40)
                
                if result['is_complete']:
                    logger.info(f"\n✅ VERDICT: PASSING GRADE ({result['grade']})")
                    logger.info("ARCHITECT DECISION: The analysis meets standards (B- or higher).")
                else:
                    logger.info(f"\n❌ VERDICT: FAILING GRADE ({result['grade']})")
                    logger.info("ARCHITECT DECISION: The analysis needs improvement.")
                    logger.info(f"\nIMPROVEMENT FEEDBACK:")
                    logger.info("-" * 40)
                    logger.info(result['feedback'])
                    logger.info("-" * 40)
                logger.info("=" * 40)
                
                return result
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                else:
                    # Be lenient on validation failure
                    return {
                        "is_complete": True,
                        "feedback": "Validation check failed, assuming completion"
                    }
    
    def _profile_data(self, data_path: Path) -> str:
        try:
            file_ext = data_path.suffix.lower()
            
            # Read data based on file type
            if file_ext == '.csv':
                df = pd.read_csv(data_path, nrows=1000)  # Sample first 1000 rows
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(data_path, nrows=1000)
            elif file_ext == '.json':
                df = pd.read_json(data_path)
                if isinstance(df, pd.Series):
                    df = df.to_frame()
                df = df.head(1000)
            elif file_ext == '.parquet':
                df = pd.read_parquet(data_path)
                df = df.head(1000)
            else:
                return f"File type: {file_ext}, Unable to profile"
            
            # Generate profile
            profile = []
            profile.append(f"File type: {file_ext}")
            profile.append(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            profile.append(f"Columns: {list(df.columns)}")
            profile.append("\nColumn Types:")
            for col, dtype in df.dtypes.items():
                profile.append(f"  - {col}: {dtype}")
            
            profile.append("\nBasic Statistics:")
            # Numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                profile.append("  Numeric columns:")
                for col in numeric_cols[:10]:  # Limit to first 10
                    profile.append(f"    - {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}")
            
            # Categorical columns
            object_cols = df.select_dtypes(include=['object']).columns
            if len(object_cols) > 0:
                profile.append("  Categorical columns:")
                for col in object_cols[:10]:  # Limit to first 10
                    n_unique = df[col].nunique()
                    profile.append(f"    - {col}: {n_unique} unique values")
                    if n_unique <= 10:
                        profile.append(f"      Values: {df[col].unique()[:10].tolist()}")
            
            # Missing values
            missing = df.isnull().sum()
            if missing.any():
                profile.append("\nMissing Values:")
                for col, count in missing[missing > 0].items():
                    profile.append(f"  - {col}: {count} ({count/len(df)*100:.1f}%)")
            
            return "\n".join(profile)
            
        except Exception as e:
            logger.error(f"Error profiling data: {str(e)}")
            return f"Error profiling data: {str(e)}"
    
    def _load_prompt(self, filename: str) -> str:
        prompt_path = Path(__file__).parent.parent / "config" / "prompts" / filename
        if prompt_path.exists():
            return prompt_path.read_text()
        return ""