import os
import sys
import uuid
import shutil
import logging
import json
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import httpx
import asyncio

from agents.architect import ArchitectAgent
from agents.coder import CoderAgent
from executor import PythonExecutor
from limits import MAX_FILE_SIZE, MAX_ITERATIONS, TOTAL_REQUEST_TIMEOUT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    logger.error("OPENAI_API_KEY environment variable is not set")
    sys.exit(1)

app = FastAPI(title="Automatic Data Scientist MVP")

ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls', '.json', '.parquet'}


@app.post("/analyze")
async def analyze(url: str = Form(...), prompt: str = Form(...)):
    analysis_id = str(uuid.uuid4())
    work_dir = Path(f"/tmp/analysis_{analysis_id}")
    work_dir.mkdir(exist_ok=True)
    
    # Set up artifact saving if ARTIFACTS_OUTPUT is set
    artifacts_dir = None
    if os.getenv("ARTIFACTS_OUTPUT"):
        artifacts_base = Path(os.getenv("ARTIFACTS_OUTPUT"))
        artifacts_dir = artifacts_base / analysis_id
        try:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            # Save the user request
            request_file = artifacts_dir / "request.txt"
            request_content = f"URL: {url}\n\nPrompt:\n{prompt}"
            request_file.write_text(request_content)
            logger.info(f"Saving artifacts to: {artifacts_dir}")
        except Exception as e:
            logger.warning(f"Failed to set up artifacts directory: {e}")
            artifacts_dir = None
    
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
        last_validation = None
        
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
            
            # Create iteration directory for artifacts
            iteration_dir = None
            if artifacts_dir:
                iteration_dir = artifacts_dir / f"iteration_{iteration}"
                try:
                    iteration_dir.mkdir(exist_ok=True)
                except Exception as e:
                    logger.warning(f"Failed to create iteration directory: {e}")
                    iteration_dir = None
            
            if iteration == 1:
                # First iteration with initial requirements
                logger.info("\n🔍 ITERATION TYPE: Initial code generation")
                logger.info("ITERATION GOAL: Generate first version of analysis script")
                code = await coder.generate_code(
                    initial_requirements['requirements'],
                    initial_requirements['acceptance_criteria'],
                    data_path
                )
            else:
                # Subsequent iterations with feedback
                logger.info("\n🔄 ITERATION TYPE: Code revision based on feedback")
                logger.info(f"ITERATION GOAL: Address feedback from previous validation")
                logger.info(f"Previous feedback: {initial_requirements['feedback'][:200]}..." if len(initial_requirements.get('feedback', '')) > 200 else f"Previous feedback: {initial_requirements.get('feedback', '')}")
                code = await coder.revise_code(
                    last_code,
                    initial_requirements['requirements'],
                    initial_requirements['acceptance_criteria'],
                    initial_requirements['feedback'],
                    data_path,
                    initial_requirements.get('grade', 'Not provided'),
                    initial_requirements.get('grade_justification', 'Not provided')
                )
            
            last_code = code
            
            # Save the generated script
            if iteration_dir:
                try:
                    script_file = iteration_dir / "script.py"
                    script_file.write_text(code)
                    logger.info(f"Saved script to {script_file}")
                except Exception as e:
                    logger.warning(f"Failed to save script: {e}")
            
            # Execute the generated code
            logger.info("\n⚙️ EXECUTING generated Python script...")
            execution_result = await executor.execute(code, data_path)
            
            if execution_result['success']:
                logger.info("✅ Execution SUCCESSFUL - Proceeding to validation")
                logger.info(f"Output length: {len(execution_result['output'])} characters")
                
                # Save the HTML output
                if iteration_dir:
                    try:
                        output_file = iteration_dir / "output.html"
                        output_file.write_text(execution_result['output'])
                        logger.info(f"Saved HTML output to {output_file}")
                    except Exception as e:
                        logger.warning(f"Failed to save HTML output: {e}")
                
                # Validate the results
                logger.info("\n🔍 VALIDATING output against acceptance criteria...")
                validation = await architect.validate_results(
                    execution_result['output'],
                    initial_requirements['acceptance_criteria'],
                    initial_requirements.get('requirements'),
                    initial_requirements.get('criteria_importance')
                )
                last_validation = validation  # Store for potential use in warning message
                
                # Save the validation/grade
                if iteration_dir:
                    try:
                        grade_data = {
                            "grade": validation.get('grade', 'N/A'),
                            "grade_justification": validation.get('grade_justification', ''),
                            "criteria_evaluation": validation.get('criteria_evaluation', ''),
                            "is_complete": validation['is_complete'],
                            "feedback": validation['feedback']
                        }
                        grade_file = iteration_dir / "grade.json"
                        grade_file.write_text(json.dumps(grade_data, indent=2))
                        logger.info(f"Saved grade to {grade_file}: {grade_data['grade']}")
                    except Exception as e:
                        logger.warning(f"Failed to save grade: {e}")
                
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
                    initial_requirements['grade'] = validation.get('grade', 'Not provided')
                    initial_requirements['grade_justification'] = validation.get('grade_justification', 'Not provided')
                    last_result = execution_result
            else:
                # Handle execution error
                logger.error(f"\n❌ Execution FAILED with error: {execution_result['error']}")
                logger.info("Setting error as feedback for next iteration")
                
                # Save error information
                if iteration_dir:
                    try:
                        error_data = {
                            "execution_failed": True,
                            "error": execution_result['error'],
                            "stderr": execution_result.get('stderr', ''),
                            "partial_output": execution_result.get('output', '')
                        }
                        error_file = iteration_dir / "error.json"
                        error_file.write_text(json.dumps(error_data, indent=2))
                        logger.info(f"Saved error details to {error_file}")
                        
                        # If there's partial output, save it too
                        if execution_result.get('output'):
                            output_file = iteration_dir / "partial_output.html"
                            output_file.write_text(execution_result['output'])
                            logger.info(f"Saved partial output to {output_file}")
                    except Exception as e:
                        logger.warning(f"Failed to save error details: {e}")
                
                initial_requirements['feedback'] = f"Code execution error: {execution_result['error']}"
                last_result = execution_result
        
        # Max iterations reached - return output regardless of acceptance
        logger.warning("\n" + "!"*60)
        logger.warning(f"! MAX ITERATIONS ({MAX_ITERATIONS}) REACHED !")
        logger.warning("! Returning best available output")
        logger.warning("!"*60)
        if last_result and last_result['success']:
            # Build warning message with architect's evaluation
            warning_parts = [
                "<div style='background-color: #fff3cd; border: 1px solid #ffc107; border-radius: 5px; padding: 15px; margin-bottom: 20px;'>",
                "<h3 style='margin-top: 0; color: #856404;'>⚠️ Analysis Completed with Warnings</h3>",
                "<p><strong>Status:</strong> Maximum iterations reached ({} iterations)</p>".format(MAX_ITERATIONS)
            ]
            
            if last_validation:
                grade = last_validation.get('grade', 'N/A')
                grade_color = '#28a745' if grade.startswith(('A', 'B')) else '#dc3545' if grade.startswith(('D', 'F')) else '#ffc107'
                
                warning_parts.extend([
                    "<hr style='margin: 10px 0; border-color: #ffc107;'>",
                    "<h4 style='color: #856404;'>Architect's Final Evaluation:</h4>",
                    f"<p><strong>Grade:</strong> <span style='color: {grade_color}; font-size: 1.2em; font-weight: bold;'>{grade}</span></p>"
                ])
                
                if last_validation.get('grade_justification'):
                    warning_parts.append(f"<p><strong>Justification:</strong> {last_validation['grade_justification']}</p>")
                
                if last_validation.get('criteria_evaluation'):
                    warning_parts.append(f"<div style='background-color: #fff; border-left: 3px solid #ffc107; padding: 10px; margin: 10px 0;'>"
                                       f"<strong>Criteria Evaluation:</strong><br>{last_validation['criteria_evaluation']}"
                                       f"</div>")
                
                if last_validation.get('feedback'):
                    warning_parts.append(f"<div style='background-color: #fff; border-left: 3px solid #dc3545; padding: 10px; margin: 10px 0;'>"
                                       f"<strong>Remaining Issues:</strong><br>{last_validation['feedback']}"
                                       f"</div>")
            
            warning_parts.append("</div>")
            warning_html = "".join(warning_parts) + last_result['output']
            return HTMLResponse(content=warning_html)
        elif last_result and last_result.get('output'):
            # Even if execution failed, return any partial output
            error_parts = [
                "<div style='background-color: #f8d7da; border: 1px solid #dc3545; border-radius: 5px; padding: 15px; margin-bottom: 20px;'>",
                "<h3 style='margin-top: 0; color: #721c24;'>❌ Analysis Failed</h3>",
                "<p><strong>Status:</strong> Maximum iterations reached ({} iterations) with execution errors</p>".format(MAX_ITERATIONS),
                "<p><strong>Last Error:</strong> {}</p>".format(last_result.get('error', 'Unknown error'))
            ]
            
            if last_validation:
                grade = last_validation.get('grade', 'N/A') 
                error_parts.extend([
                    "<hr style='margin: 10px 0; border-color: #dc3545;'>",
                    "<h4 style='color: #721c24;'>Last Successful Validation:</h4>",
                    f"<p><strong>Grade:</strong> {grade}</p>"
                ])
                
                if last_validation.get('feedback'):
                    error_parts.append(f"<p><strong>Issues:</strong> {last_validation['feedback']}</p>")
            
            error_parts.append("</div>")
            warning_html = "".join(error_parts) + last_result.get('output', '')
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
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)