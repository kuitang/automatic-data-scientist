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

print("<!DOCTYPE html>")
print("<html><body>")
print("<h1>Test Output</h1>")
print("</body></html>")
"""
        result = await executor.execute(code, sample_data_file)
        
        assert result['success'] == True
        assert result['error'] is None
        assert "<h1>Test Output</h1>" in result['output']
        assert executor._is_valid_html(result['output'])
    
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

print("This is not HTML")
print("Just plain text output")
"""
        result = await executor.execute(code, sample_data_file)
        
        assert result['success'] == False
        assert "not valid HTML" in result['error']
    
    def test_html_validation(self, executor):
        # Test valid HTML with various structures
        assert executor._is_valid_html("<!DOCTYPE html><html><body>Test</body></html>"), \
            "Should accept complete HTML with DOCTYPE"
        assert executor._is_valid_html("<html><body>Test</body></html>"), \
            "Should accept HTML without DOCTYPE"
        assert executor._is_valid_html("  <HTML><BODY>Test</BODY></HTML>  "), \
            "Should accept HTML with whitespace and uppercase tags"
        
        # Test edge cases that should be valid according to the implementation
        assert executor._is_valid_html("<html><body><h1>Title</h1><p>Text</p></body></html>"), \
            "Should accept HTML with nested elements"
        assert executor._is_valid_html("<html>\n<body>\nContent\n</body>\n</html>"), \
            "Should accept HTML with newlines"
        
        # The implementation accepts <html> with either </html> OR <body
        assert executor._is_valid_html("<html>Missing body tag</html>"), \
            "Implementation accepts HTML with </html> even without body"
        assert executor._is_valid_html("<html><body>No closing tags"), \
            "Implementation accepts HTML with <html and <body even without closing"
        
        # Test invalid cases
        assert not executor._is_valid_html("Just plain text"), \
            "Should reject plain text without HTML tags"
        assert not executor._is_valid_html("{ 'json': 'data' }"), \
            "Should reject JSON data"
        assert not executor._is_valid_html("<div>Partial HTML</div>"), \
            "Should reject partial HTML without html/body tags"
        assert not executor._is_valid_html(""), \
            "Should reject empty string"
        assert not executor._is_valid_html("<body>Missing html tag</body>"), \
            "Should reject HTML missing html wrapper"
        
        # Test that the function correctly identifies HTML starting patterns
        assert executor._is_valid_html("<!doctype html>minimal"), \
            "Should accept content starting with doctype"
        assert executor._is_valid_html("<html>minimal"), \
            "Should accept content starting with <html"
        # Note: Implementation accepts <html anywhere if it has </html> or <body
        assert executor._is_valid_html("Some text <html>later</html>"), \
            "Implementation accepts <html anywhere if it has </html>"
        assert executor._is_valid_html("prefix <html><body>content"), \
            "Implementation accepts <html anywhere if it has <body"
    
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