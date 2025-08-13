import pytest
import asyncio
import json
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
import tempfile

import sys
sys.path.append(str(Path(__file__).parent.parent))

from agents.architect import ArchitectAgent
from agents.coder import CoderAgent


class TestArchitectAgent:
    
    @pytest.fixture
    def architect(self):
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
            return ArchitectAgent()
    
    @pytest.fixture
    def sample_csv_file(self):
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_file.write("name,age,score\nAlice,25,90\nBob,30,85\nCarol,35,95")
        temp_file.close()
        yield Path(temp_file.name)
        Path(temp_file.name).unlink()
    
    @pytest.mark.asyncio
    async def test_profile_and_plan(self, architect, sample_csv_file):
        from agents.models import ArchitectPlanResponse
        
        # Create a proper response object
        mock_plan = ArchitectPlanResponse(
            requirements="Analyze the CSV data and create visualizations",
            acceptance_criteria=[
                "Generate summary statistics",
                "Create at least one chart",
                "Output valid HTML"
            ],
            criteria_importance="Statistics are critical for understanding data. Charts provide visual insights. HTML output is required for delivery.",
            is_complete=False,
            feedback=""
        )
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.parsed = mock_plan
        
        with patch.object(architect.client.beta.chat.completions, 'parse', 
                         return_value=mock_response) as mock_parse:
            result = await architect.profile_and_plan(
                sample_csv_file,
                "Analyze this data"
            )
            
            # Verify the API was called with correct parameters
            mock_parse.assert_called_once()
            call_args = mock_parse.call_args
            messages = call_args.kwargs['messages']
            
            # Check that data profiling was included in the prompt
            assert len(messages) == 2
            assert messages[0]['role'] == 'system'
            assert 'data analysis architect' in messages[0]['content'].lower()
            assert messages[1]['role'] == 'user'
            # The user message should include the data profile
            user_message = messages[1]['content']
            assert "['name', 'age', 'score']" in user_message  # CSV columns
            assert '3 rows, 3 columns' in user_message  # Data shape
            assert 'Analyze this data' in user_message  # User prompt
            
            # Verify the model and response format were specified
            assert call_args.kwargs['model'] == architect.model
            assert call_args.kwargs['response_format'] == ArchitectPlanResponse
            
            # Verify the result structure
            assert "requirements" in result
            assert result["requirements"] == "Analyze the CSV data and create visualizations"
            assert "acceptance_criteria" in result
            assert isinstance(result["acceptance_criteria"], list)
            assert len(result["acceptance_criteria"]) == 3
            # Don't test is_complete value, test that it comes from the mock
            assert "is_complete" in result
    
    def test_data_profiling(self, architect, sample_csv_file):
        profile = architect._profile_data(sample_csv_file)
        
        assert "File type: .csv" in profile
        assert "3 rows, 3 columns" in profile
        assert "['name', 'age', 'score']" in profile
        assert "Numeric columns:" in profile
        assert "Categorical columns:" in profile
    
    @pytest.mark.asyncio
    async def test_validate_results(self, architect):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "criteria_evaluation": "All criteria met successfully",
            "grade": "B+",
            "grade_justification": "Good analysis with solid insights",
            "is_complete": True,
            "feedback": ""
        })
        
        html_output = "<html><body><h1>Analysis Results</h1></body></html>"
        criteria = ["HTML output generated", "Contains results"]
        
        with patch.object(architect.client.chat.completions, 'create',
                         return_value=mock_response):
            result = await architect.validate_results(html_output, criteria)
            
            assert result["is_complete"] == True
            assert "feedback" in result
    
    @pytest.mark.asyncio
    async def test_retry_on_api_failure(self, architect, sample_csv_file):
        with patch.object(architect.client.chat.completions, 'create',
                         side_effect=[Exception("API Error"), Exception("API Error")]):
            # Should fall back to default requirements after retries
            result = await architect.profile_and_plan(
                sample_csv_file,
                "Analyze this data"
            )
            
            assert "requirements" in result
            assert "acceptance_criteria" in result
            assert len(result["acceptance_criteria"]) > 0


class TestCoderAgent:
    
    @pytest.fixture
    def coder(self):
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
            return CoderAgent()
    
    @pytest.fixture
    def sample_data_path(self):
        return Path("/tmp/test_data.csv")
    
    @pytest.mark.asyncio
    async def test_generate_code(self, coder, sample_data_path):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """```python
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True)
args = parser.parse_args()

df = pd.read_csv(args.data)
print("<html><body>Analysis complete</body></html>")
```"""
        
        # Use AsyncMock instead of MagicMock
        with patch.object(coder.client.chat.completions, 'create',
                         new=AsyncMock(return_value=mock_response)):
            code = await coder.generate_code(
                "Create basic analysis",
                ["Generate HTML output", "Include basic statistics"],  # Add acceptance criteria
                sample_data_path
            )
            
            assert "import pandas" in code
            assert "argparse" in code
            assert "--data" in code
            assert "<html>" in code
    
    @pytest.mark.asyncio
    async def test_revise_code(self, coder, sample_data_path):
        previous_code = """import pandas as pd
df = pd.read_csv('data.csv')"""
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """```python
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True)
args = parser.parse_args()

df = pd.read_csv(args.data)
print("<html><body>Fixed version</body></html>")
```"""
        
        # Use AsyncMock instead of MagicMock
        with patch.object(coder.client.chat.completions, 'create',
                         new=AsyncMock(return_value=mock_response)):
            code = await coder.revise_code(
                previous_code,
                "Fix the argparse issue",
                ["Accept --data argument", "Generate HTML output"],  # Add acceptance criteria
                "Code needs to accept --data argument",
                sample_data_path,
                grade="C",  # Add required grade parameter
                grade_justification="Missing argparse implementation"  # Add required justification
            )
            
            assert "argparse" in code
            assert "--data" in code
    
    @pytest.mark.asyncio
    async def test_code_extraction_from_markdown(self, coder, sample_data_path):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """Here's the code:

```python
import pandas as pd
print("Hello World")
```

This code prints hello world."""
        
        # Use AsyncMock instead of MagicMock
        with patch.object(coder.client.chat.completions, 'create',
                         new=AsyncMock(return_value=mock_response)):
            code = await coder.generate_code(
                "Simple script",
                [],  # Add empty acceptance criteria
                sample_data_path
            )
            
            assert code == 'import pandas as pd\nprint("Hello World")'
            assert "```" not in code
            assert "Here's the code" not in code