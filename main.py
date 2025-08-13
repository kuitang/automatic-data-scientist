import os
import sys
import uuid
import shutil
import logging
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import httpx
import asyncio

from agents.architect import ArchitectAgent
from agents.coder import CoderAgent
from executor import PythonExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    logger.error("OPENAI_API_KEY environment variable is not set")
    sys.exit(1)

app = FastAPI(title="Automatic Data Scientist MVP")

MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls', '.json', '.parquet'}
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))  # Configurable, default 3
TOTAL_TIMEOUT = 300  # 5 minutes


@app.post("/analyze")
async def analyze(url: str = Form(...), prompt: str = Form(...)):
    analysis_id = str(uuid.uuid4())
    work_dir = Path(f"/tmp/analysis_{analysis_id}")
    work_dir.mkdir(exist_ok=True)
    
    try:
        # Download and validate file
        data_path = await download_and_validate_file(url, work_dir)
        logger.info(f"Downloaded file to {data_path}")
        
        # Initialize agents
        architect = ArchitectAgent()
        coder = CoderAgent()
        executor = PythonExecutor(work_dir)
        
        # Profile data and create initial requirements
        initial_requirements = await architect.profile_and_plan(data_path, prompt)
        
        # Architect-Coder iteration loop
        iteration = 0
        last_result = None
        last_code = None
        
        logger.info("\n" + "#"*80)
        logger.info("# STARTING ARCHITECT-CODER ITERATION LOOP")
        logger.info(f"# Max Iterations: {MAX_ITERATIONS}")
        logger.info(f"# Acceptance Criteria Count: {len(initial_requirements['acceptance_criteria'])}")
        logger.info("#"*80 + "\n")
        
        while iteration < MAX_ITERATIONS:
            iteration += 1
            logger.info("\n" + "*"*60)
            logger.info(f"*** ITERATION {iteration}/{MAX_ITERATIONS} ***")
            logger.info("*"*60)
            
            if iteration == 1:
                # First iteration with initial requirements
                logger.info("\n🔍 ITERATION TYPE: Initial code generation")
                logger.info("ITERATION GOAL: Generate first version of analysis script")
                code = await coder.generate_code(initial_requirements['requirements'], data_path)
            else:
                # Subsequent iterations with feedback
                logger.info("\n🔄 ITERATION TYPE: Code revision based on feedback")
                logger.info(f"ITERATION GOAL: Address feedback from previous validation")
                logger.info(f"Previous feedback: {initial_requirements['feedback'][:200]}..." if len(initial_requirements.get('feedback', '')) > 200 else f"Previous feedback: {initial_requirements.get('feedback', '')}")
                code = await coder.revise_code(
                    last_code,
                    initial_requirements['requirements'],
                    initial_requirements['feedback'],
                    data_path
                )
            
            last_code = code
            
            # Execute the generated code
            logger.info("\n⚙️ EXECUTING generated Python script...")
            execution_result = await executor.execute(code, data_path)
            
            if execution_result['success']:
                logger.info("✅ Execution SUCCESSFUL - Proceeding to validation")
                logger.info(f"Output length: {len(execution_result['output'])} characters")
                
                # Validate the results
                logger.info("\n🔍 VALIDATING output against acceptance criteria...")
                validation = await architect.validate_results(
                    execution_result['output'],
                    initial_requirements['acceptance_criteria']
                )
                
                if validation['is_complete']:
                    logger.info("\n" + "="*60)
                    logger.info("🎆 SUCCESS! All acceptance criteria met!")
                    logger.info(f"Analysis completed successfully in {iteration} iteration(s)")
                    logger.info("="*60)
                    return HTMLResponse(content=execution_result['output'])
                else:
                    logger.info("\n⚠️ Validation FAILED - Need another iteration")
                    logger.info(f"Setting feedback for next iteration: {validation['feedback'][:200]}..." if len(validation['feedback']) > 200 else f"Setting feedback for next iteration: {validation['feedback']}")
                    initial_requirements['feedback'] = validation['feedback']
                    last_result = execution_result
            else:
                # Handle execution error
                logger.error(f"\n❌ Execution FAILED with error: {execution_result['error']}")
                logger.info("Setting error as feedback for next iteration")
                initial_requirements['feedback'] = f"Code execution error: {execution_result['error']}"
                last_result = execution_result
        
        # Max iterations reached - return output regardless of acceptance
        logger.warning("\n" + "!"*60)
        logger.warning(f"! MAX ITERATIONS ({MAX_ITERATIONS}) REACHED !")
        logger.warning("! Returning best available output")
        logger.warning("!"*60)
        if last_result and last_result['success']:
            # Return the output with warning that criteria may not be met
            warning_html = f"<div style='background-color: #fff3cd; padding: 10px; margin-bottom: 20px;'><strong>Warning:</strong> Analysis completed but may not meet all criteria (max iterations reached)</div>{last_result['output']}"
            return HTMLResponse(content=warning_html)
        elif last_result and last_result.get('output'):
            # Even if execution failed, return any partial output
            warning_html = f"<div style='background-color: #f8d7da; padding: 10px; margin-bottom: 20px;'><strong>Error:</strong> Analysis encountered errors and may be incomplete (max iterations reached)</div>{last_result.get('output', '')}"
            return HTMLResponse(content=warning_html)
        else:
            # No output at all to return
            error_html = "<div style='background-color: #f8d7da; padding: 10px;'><strong>Error:</strong> Analysis failed to produce any output within iteration limit</div>"
            return HTMLResponse(content=error_html)
            
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if work_dir.exists():
            shutil.rmtree(work_dir)


async def download_and_validate_file(url: str, work_dir: Path) -> Path:
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=30.0, follow_redirects=True)
            response.raise_for_status()
        except httpx.RequestError as e:
            raise HTTPException(status_code=400, detail=f"Failed to download file: {str(e)}")
        
        # Check file size
        if len(response.content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail=f"File exceeds maximum size of {MAX_FILE_SIZE / 1024 / 1024}MB")
        
        # Determine file extension from URL or content-type
        file_ext = None
        url_path = httpx.URL(url).path
        if '.' in url_path:
            file_ext = Path(url_path).suffix.lower()
        
        if file_ext not in ALLOWED_EXTENSIONS:
            # Try to infer from content-type
            content_type = response.headers.get('content-type', '').lower()
            if 'csv' in content_type:
                file_ext = '.csv'
            elif 'excel' in content_type or 'spreadsheet' in content_type:
                file_ext = '.xlsx'
            elif 'json' in content_type:
                file_ext = '.json'
            elif 'parquet' in content_type:
                file_ext = '.parquet'
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}")
        
        # Save file
        file_path = work_dir / f"data{file_ext}"
        file_path.write_bytes(response.content)
        
        return file_path


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)