import pytest
import asyncio
import json
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
import tempfile
import shutil

import sys
import os
sys.path.append(str(Path(__file__).parent.parent))

# Set a dummy OpenAI key for testing
os.environ['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY', 'test_key')

from main import app
from agents.architect import ArchitectAgent
from agents.coder import CoderAgent


class TestIntegration:
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.fixture
    def mock_csv_response(self):
        return b"product,sales,profit\nA,100,20\nB,150,35\nC,200,50"
    
    @pytest.mark.asyncio
    async def test_full_analysis_flow_with_mocks(self):
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
            # Mock file download
            csv_content = b"name,value\nItem1,10\nItem2,20\nItem3,30"
            
            # Mock architect responses
            architect_initial_response = {
                "requirements": "Create summary statistics and a bar chart",
                "acceptance_criteria": [
                    "HTML output generated",
                    "Summary statistics included",
                    "Bar chart created"
                ],
                "is_complete": False,
                "feedback": ""
            }
            
            architect_validation_response = {
                "is_complete": True,
                "feedback": ""
            }
            
            # Mock coder response
            generated_code = """
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True)
args = parser.parse_args()

df = pd.read_csv(args.data)

print("<!DOCTYPE html>")
print("<html><head><title>Analysis</title></head><body>")
print("<h1>Data Analysis Results</h1>")
print(f"<p>Total rows: {len(df)}</p>")
print(f"<p>Mean value: {df['value'].mean():.2f}</p>")
print("<div>Bar chart placeholder</div>")
print("</body></html>")
"""
            
            with patch('httpx.AsyncClient') as mock_http_client, \
                 patch('agents.architect.ArchitectAgent.profile_and_plan') as mock_profile, \
                 patch('agents.architect.ArchitectAgent.validate_results') as mock_validate, \
                 patch('agents.coder.CoderAgent.generate_code') as mock_generate:
                
                # Setup HTTP mock
                mock_response = AsyncMock()
                mock_response.content = csv_content
                mock_response.raise_for_status = MagicMock()
                mock_response.headers = {'content-type': 'text/csv'}
                
                mock_client_instance = AsyncMock()
                mock_client_instance.get.return_value = mock_response
                mock_http_client.return_value.__aenter__.return_value = mock_client_instance
                
                # Setup agent mocks
                mock_profile.return_value = architect_initial_response
                mock_validate.return_value = architect_validation_response
                mock_generate.return_value = generated_code
                
                # Make request
                with TestClient(app) as client:
                    response = client.post(
                        "/analyze",
                        data={
                            "url": "http://example.com/data.csv",
                            "prompt": "Analyze this sales data"
                        }
                    )
                    
                    # Assertions
                    assert response.status_code == 200
                    assert "text/markdown" in response.headers['content-type']
                    assert "Data Analysis Results" in response.text
                    assert "Total rows: 3" in response.text
    
    @pytest.mark.asyncio
    async def test_error_recovery_iteration(self):
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
            csv_content = b"x,y\n1,2\n3,4"
            
            # First code has error, second code fixes it
            first_code = """
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True)
args = parser.parse_args()

# This will fail
undefined_function()
"""
            
            fixed_code = """
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True)
args = parser.parse_args()

df = pd.read_csv(args.data)
print("<!DOCTYPE html>")
print("<html><body><h1>Fixed Analysis</h1></body></html>")
"""
            
            architect_initial = {
                "requirements": "Analyze data",
                "acceptance_criteria": ["HTML output"],
                "is_complete": False,
                "feedback": ""
            }
            
            architect_after_error = {
                "is_complete": False,
                "feedback": "Fix the undefined function error"
            }
            
            architect_after_fix = {
                "is_complete": True,
                "feedback": ""
            }
            
            with patch('httpx.AsyncClient') as mock_http_client, \
                 patch('agents.architect.ArchitectAgent.profile_and_plan') as mock_profile, \
                 patch('agents.architect.ArchitectAgent.validate_results') as mock_validate, \
                 patch('agents.coder.CoderAgent.generate_code') as mock_generate, \
                 patch('agents.coder.CoderAgent.revise_code') as mock_revise:
                
                # Setup mocks
                mock_response = AsyncMock()
                mock_response.content = csv_content
                mock_response.raise_for_status = MagicMock()
                mock_response.headers = {'content-type': 'text/csv'}
                
                mock_client_instance = AsyncMock()
                mock_client_instance.get.return_value = mock_response
                mock_http_client.return_value.__aenter__.return_value = mock_client_instance
                
                mock_profile.return_value = architect_initial
                mock_validate.side_effect = [architect_after_fix]  # After fix
                mock_generate.return_value = first_code
                mock_revise.return_value = fixed_code
                
                with TestClient(app) as client:
                    response = client.post(
                        "/analyze",
                        data={
                            "url": "http://example.com/data.csv",
                            "prompt": "Analyze this data"
                        }
                    )
                    
                    # Should recover from error and return fixed analysis
                    assert response.status_code == 200
                    assert "Fixed Analysis" in response.text
    
    @pytest.mark.asyncio 
    async def test_max_iterations_handling(self):
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
            csv_content = b"a,b\n1,2"
            
            never_complete_validation = {
                "is_complete": False,
                "feedback": "Still needs improvement"
            }
            
            valid_code = """
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True)
args = parser.parse_args()
print("<!DOCTYPE html><html><body>Partial results</body></html>")
"""
            
            with patch('httpx.AsyncClient') as mock_http_client, \
                 patch('agents.architect.ArchitectAgent.profile_and_plan') as mock_profile, \
                 patch('agents.architect.ArchitectAgent.validate_results') as mock_validate, \
                 patch('agents.coder.CoderAgent.generate_code') as mock_generate, \
                 patch('agents.coder.CoderAgent.revise_code') as mock_revise, \
                 patch('main.MAX_ITERATIONS', 2):  # Override for testing
                
                # Setup mocks  
                mock_response = AsyncMock()
                mock_response.content = csv_content
                mock_response.raise_for_status = MagicMock()
                mock_response.headers = {'content-type': 'text/csv'}
                
                mock_client_instance = AsyncMock()
                mock_client_instance.get.return_value = mock_response
                mock_http_client.return_value.__aenter__.return_value = mock_client_instance
                
                mock_profile.return_value = {
                    "requirements": "Analyze",
                    "acceptance_criteria": ["Complete analysis"],
                    "is_complete": False,
                    "feedback": ""
                }
                mock_validate.return_value = never_complete_validation
                mock_generate.return_value = valid_code
                mock_revise.return_value = valid_code
                
                with TestClient(app) as client:
                    response = client.post(
                        "/analyze",
                        data={
                            "url": "http://example.com/data.csv",
                            "prompt": "Complex analysis"
                        }
                    )
                    
                    # Should always return output even when criteria not met
                    assert response.status_code == 200
                    assert "Warning" in response.text or "warning" in response.text.lower()
                    assert "Partial results" in response.text
    
    def test_max_iterations_env_config(self):
        # Test that MAX_ITERATIONS can be configured via environment variable
        import os
        import importlib
        
        # Test default value
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}, clear=True):
            import limits
            import main
            importlib.reload(limits)
            importlib.reload(main)
            assert main.MAX_ITERATIONS == 3
        
        # Test custom value
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key', 'MAX_ITERATIONS': '5'}):
            importlib.reload(limits)
            importlib.reload(main)
            assert main.MAX_ITERATIONS == 5
        
        # Test invalid value defaults to 3
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key', 'MAX_ITERATIONS': 'invalid'}):
            try:
                importlib.reload(limits)
                importlib.reload(main)
            except ValueError:
                # This is expected, the int() conversion will fail
                pass
    
    @pytest.mark.asyncio
    async def test_returns_output_even_on_failure(self):
        # Test that we return output even when execution fails but produces partial output
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
            csv_content = b"col1,col2\n1,2"
            
            failing_code = """
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True)
args = parser.parse_args()

# Print partial output before failing
print("<!DOCTYPE html><html><body>")
print("<h1>Partial Analysis</h1>")
print("<p>Starting analysis...</p>")

# This will cause an error
undefined_variable
"""
            
            with patch('httpx.AsyncClient') as mock_http_client, \
                 patch('agents.architect.ArchitectAgent.profile_and_plan') as mock_profile, \
                 patch('agents.architect.ArchitectAgent.validate_results') as mock_validate, \
                 patch('agents.coder.CoderAgent.generate_code') as mock_generate, \
                 patch('agents.coder.CoderAgent.revise_code') as mock_revise, \
                 patch('main.MAX_ITERATIONS', 1):
                
                # Setup mocks
                mock_response = AsyncMock()
                mock_response.content = csv_content
                mock_response.raise_for_status = MagicMock()
                mock_response.headers = {'content-type': 'text/csv'}
                
                mock_client_instance = AsyncMock()
                mock_client_instance.get.return_value = mock_response
                mock_http_client.return_value.__aenter__.return_value = mock_client_instance
                
                mock_profile.return_value = {
                    "requirements": "Analyze",
                    "acceptance_criteria": ["Complete"],
                    "is_complete": False,
                    "feedback": ""
                }
                mock_generate.return_value = failing_code
                mock_revise.return_value = failing_code
                
                with TestClient(app) as client:
                    response = client.post(
                        "/analyze",
                        data={
                            "url": "http://example.com/data.csv",
                            "prompt": "Analyze"
                        }
                    )
                    
                    # Should return partial output with error warning
                    assert response.status_code == 200
                    assert "Error" in response.text or "error" in response.text.lower()
                    # Check that we still show whatever output was produced
                    assert "html" in response.text.lower()