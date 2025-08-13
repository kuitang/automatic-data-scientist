import os
import subprocess
import asyncio
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any

from limits import EXECUTION_TIMEOUT, MEMORY_LIMIT

logger = logging.getLogger(__name__)

class PythonExecutor:
    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.timeout = EXECUTION_TIMEOUT
        self.memory_limit = MEMORY_LIMIT
        
    async def execute(self, code: str, data_path: Path) -> Dict[str, Any]:
        # Save code to a temporary file
        script_path = self.work_dir / "analysis_script.py"
        script_path.write_text(code)
        
        # Prepare command - try python3 first, fallback to python
        python_cmd = "python3" if os.system("which python3 > /dev/null 2>&1") == 0 else "python"
        cmd = [
            python_cmd,
            str(script_path),
            "--data",
            str(data_path)
        ]
        
        # Set environment variables to limit resource usage
        env = os.environ.copy()
        env['MPLBACKEND'] = 'Agg'  # Non-interactive matplotlib backend
        
        try:
            # Run the script with timeout
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.work_dir),
                env=env
            )
            
            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return {
                    'success': False,
                    'error': f'Script execution exceeded timeout of {self.timeout} seconds',
                    'output': None,
                    'stderr': None
                }
            
            # Check return code
            if process.returncode == 0:
                output = stdout.decode('utf-8', errors='replace')
                
                # Validate that output is HTML
                if not self._is_valid_html(output):
                    return {
                        'success': False,
                        'error': 'Output is not valid HTML',
                        'output': output,
                        'stderr': stderr.decode('utf-8', errors='replace') if stderr else None
                    }
                
                logger.info("Script executed successfully")
                return {
                    'success': True,
                    'error': None,
                    'output': output,
                    'stderr': stderr.decode('utf-8', errors='replace') if stderr else None
                }
            else:
                error_msg = stderr.decode('utf-8', errors='replace') if stderr else 'Unknown error'
                logger.error(f"Script execution failed: {error_msg}")
                return {
                    'success': False,
                    'error': error_msg,
                    'output': stdout.decode('utf-8', errors='replace') if stdout else None,
                    'stderr': error_msg
                }
                
        except Exception as e:
            logger.error(f"Unexpected error during script execution: {str(e)}")
            return {
                'success': False,
                'error': f'Unexpected error: {str(e)}',
                'output': None,
                'stderr': None
            }
    
    def _is_valid_html(self, content: str) -> bool:
        # Basic HTML validation
        content_lower = content.lower().strip()
        return (
            content_lower.startswith('<!doctype html') or
            content_lower.startswith('<html') or
            ('<html' in content_lower and ('</html>' in content_lower or '<body' in content_lower))
        )