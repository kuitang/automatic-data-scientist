import pytest
import asyncio
from pathlib import Path
import tempfile
import shutil

import sys
sys.path.append(str(Path(__file__).parent.parent))

from executor import PythonExecutor


class TestPythonExecutor:
    
    @pytest.fixture
    def work_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def executor(self, work_dir):
        return PythonExecutor(work_dir)
    
    @pytest.fixture
    def sample_data_file(self, work_dir):
        data_file = work_dir / "data.csv"
        data_file.write_text("name,value\nA,1\nB,2\nC,3")
        return data_file
    
    @pytest.mark.asyncio
    async def test_successful_execution(self, executor, sample_data_file):
        code = """
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True)
args = parser.parse_args()

print("# Test Output")
print("This is a test markdown document.")
print("## Results")
print("The analysis was successful.")
"""
        result = await executor.execute(code, sample_data_file)
        
        assert result['success'] == True
        assert result['error'] is None
        assert "# Test Output" in result['output']
        assert "markdown" in result['output'].lower() or "results" in result['output'].lower()
    
    @pytest.mark.asyncio
    async def test_execution_with_error(self, executor, sample_data_file):
        code = """
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True)
args = parser.parse_args()

# This will cause an error
undefined_variable
"""
        result = await executor.execute(code, sample_data_file)
        
        assert result['success'] == False
        assert "NameError" in result['error'] or "undefined_variable" in result['error']
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, executor, sample_data_file):
        code = """
import argparse
import time
parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True)
args = parser.parse_args()

# This will exceed the timeout
time.sleep(35)
print("<html><body>Should not reach here</body></html>")
"""
        # Temporarily reduce timeout for testing
        executor.timeout = 2
        
        result = await executor.execute(code, sample_data_file)
        
        assert result['success'] == False
        assert "timeout" in result['error'].lower()
    
    @pytest.mark.asyncio
    async def test_invalid_html_detection(self, executor, sample_data_file):
        code = """
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True)
args = parser.parse_args()

print("# Analysis Results")
print("This is valid markdown output")
"""
        result = await executor.execute(code, sample_data_file)
        
        # Markdown validation was removed, so any output is now accepted
        assert result['success'] == True
        assert "Analysis Results" in result['output']
    
    def test_markdown_output_accepted(self, executor):
        # Test that various markdown outputs are accepted (no validation)
        # Since we removed validation, we just verify the executor doesn't have _is_valid_html
        assert not hasattr(executor, '_is_valid_html'), \
            "HTML validation method should be removed"
    
    @pytest.mark.asyncio
    async def test_script_with_pandas(self, executor, sample_data_file):
        code = """
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True)
args = parser.parse_args()

df = pd.read_csv(args.data)

print("<!DOCTYPE html>")
print("<html><body>")
print(f"<p>Data shape: {df.shape}</p>")
print(f"<p>Columns: {list(df.columns)}</p>")
print("</body></html>")
"""
        result = await executor.execute(code, sample_data_file)
        
        assert result['success'] == True
        assert "Data shape: (3, 2)" in result['output']
        assert "['name', 'value']" in result['output'] or "name" in result['output']