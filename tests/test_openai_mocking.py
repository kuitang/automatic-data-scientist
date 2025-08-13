"""Comprehensive unit tests with mocked OpenAI API responses."""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock, Mock
import sys

sys.path.append(str(Path(__file__).parent.parent))

from agents.architect import ArchitectAgent
from agents.coder import CoderAgent
from agents.models import (
    ArchitectPlanResponse,
    ArchitectValidationResponse,
    CoderResponse
)


class TestOpenAIMocking:
    """Test OpenAI API response mocking for agent workflows."""
    
    @pytest.fixture
    def mock_openai_response(self):
        """Create a properly structured OpenAI API response."""
        def _create_response(content, model="gpt-4", usage=None):
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message = MagicMock()
            response.choices[0].message.content = content
            response.choices[0].finish_reason = "stop"
            response.model = model
            
            if usage:
                response.usage = MagicMock()
                response.usage.prompt_tokens = usage.get('prompt_tokens', 500)
                response.usage.completion_tokens = usage.get('completion_tokens', 200)
                response.usage.total_tokens = usage.get('total_tokens', 700)
            
            return response
        return _create_response
    
    @pytest.fixture
    def mock_structured_response(self):
        """Create a properly structured response for beta.chat.completions.parse."""
        def _create_response(parsed_obj, model="gpt-4", usage=None):
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message = MagicMock()
            response.choices[0].message.parsed = parsed_obj
            response.choices[0].finish_reason = "stop"
            response.model = model
            
            if usage:
                response.usage = MagicMock()
                response.usage.prompt_tokens = usage.get('prompt_tokens', 500)
                response.usage.completion_tokens = usage.get('completion_tokens', 200)
                response.usage.total_tokens = usage.get('total_tokens', 700)
            
            return response
        return _create_response
    
    @pytest.fixture
    def sample_csv_file(self):
        """Create a sample CSV file for testing."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_file.write("product,category,price,quantity,rating\n")
        temp_file.write("Laptop,Electronics,1200.50,15,4.5\n")
        temp_file.write("Phone,Electronics,800.00,25,4.2\n")
        temp_file.write("Desk,Furniture,350.00,10,4.0\n")
        temp_file.write("Chair,Furniture,150.00,30,3.8\n")
        temp_file.write("Monitor,Electronics,400.00,20,4.3\n")
        temp_file.close()
        yield Path(temp_file.name)
        Path(temp_file.name).unlink()


class TestArchitectAgentMocking(TestOpenAIMocking):
    """Test Architect agent with mocked OpenAI responses."""
    
    @pytest.fixture
    def architect(self):
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
            return ArchitectAgent()
    
    @pytest.mark.asyncio
    async def test_profile_and_plan_success(self, architect, sample_csv_file, mock_structured_response):
        """Test successful profile and plan with realistic OpenAI response."""
        
        # Create a proper Pydantic model instance
        plan_response = ArchitectPlanResponse(
            requirements="Analyze the product sales data to identify trends and patterns. Generate summary statistics for each category, identify top-performing products, and analyze price-quantity relationships.",
            acceptance_criteria=[
                "Calculate and display summary statistics for sales by category",
                "Identify top 3 products by rating and revenue",
                "Create visualizations showing price distribution and category breakdown",
                "Analyze correlation between price, quantity, and rating"
            ],
            criteria_importance="Category statistics are essential for understanding market segments. Top products show performance leaders. Visualizations provide quick insights. Correlations reveal pricing strategies.",
            is_complete=False,
            feedback=""
        )
        
        mock_response = mock_structured_response(
            plan_response,
            usage={'prompt_tokens': 450, 'completion_tokens': 250, 'total_tokens': 700}
        )
        
        with patch.object(architect.client.beta.chat.completions, 'parse',
                         return_value=mock_response) as mock_parse:
            result = await architect.profile_and_plan(
                sample_csv_file,
                "Analyze product sales performance and identify opportunities"
            )
            
            # Verify the API was called
            mock_parse.assert_called_once()
            call_args = mock_parse.call_args
            
            # Check the model and format
            assert call_args.kwargs['model'] == architect.model
            assert call_args.kwargs['response_format'] == ArchitectPlanResponse
            
            # Check the messages structure
            messages = call_args.kwargs['messages']
            assert len(messages) == 2
            assert messages[0]['role'] == 'system'
            assert messages[1]['role'] == 'user'
            assert 'data analysis architect' in messages[0]['content'].lower()
            assert 'product' in messages[1]['content']  # Our CSV has product data
            
            # Verify the result
            assert result['requirements'] == plan_response.requirements
            assert len(result['acceptance_criteria']) == 4
            assert result['is_complete'] == False
            assert 'Category statistics' in result['criteria_importance']
    
    @pytest.mark.asyncio
    async def test_validate_results_passing_grade(self, architect, mock_structured_response):
        """Test validation with a passing grade (B or higher)."""
        
        validation_response = ArchitectValidationResponse(
            criteria_evaluation="All acceptance criteria have been successfully met. The analysis provides comprehensive statistics, clear visualizations, and actionable insights.",
            grade="B+",
            grade_justification="Strong analysis with detailed category breakdowns, effective visualizations, and solid statistical insights. Minor improvements could include deeper trend analysis.",
            is_complete=True,
            feedback=""
        )
        
        mock_response = mock_structured_response(validation_response)
        
        html_output = """
        <html>
        <body>
            <h1>Product Sales Analysis</h1>
            <h2>Summary Statistics by Category</h2>
            <table>...</table>
            <h2>Top Products</h2>
            <ol>...</ol>
            <svg>...</svg>
        </body>
        </html>
        """
        
        criteria = [
            "Calculate and display summary statistics for sales by category",
            "Identify top 3 products by rating and revenue",
            "Create visualizations showing price distribution"
        ]
        
        with patch.object(architect.client.beta.chat.completions, 'parse',
                         return_value=mock_response) as mock_parse:
            result = await architect.validate_results(html_output, criteria)
            
            # Verify API call
            mock_parse.assert_called_once()
            messages = mock_parse.call_args.kwargs['messages']
            
            # Check message structure
            assert messages[0]['role'] == 'system'
            assert 'grading' in messages[0]['content'].lower()
            assert messages[1]['role'] == 'user'
            
            # Verify result
            assert result['is_complete'] == True
            assert result['grade'] == 'B+'
            assert 'comprehensive statistics' in result['criteria_evaluation']
    
    @pytest.mark.asyncio
    async def test_validate_results_failing_grade(self, architect, mock_structured_response):
        """Test validation with a failing grade (below B-)."""
        
        validation_response = ArchitectValidationResponse(
            criteria_evaluation="Only basic statistics provided. Missing key visualizations and correlation analysis.",
            grade="C",
            grade_justification="Analysis lacks depth and misses several acceptance criteria. Need more comprehensive statistical analysis.",
            is_complete=False,
            feedback="Add correlation matrix, improve visualizations with proper labels, include confidence intervals for statistics"
        )
        
        mock_response = mock_structured_response(validation_response)
        
        html_output = "<html><body>Basic stats only</body></html>"
        criteria = ["Deep statistical analysis", "Multiple visualizations", "Correlation matrix"]
        
        with patch.object(architect.client.beta.chat.completions, 'parse',
                         return_value=mock_response):
            result = await architect.validate_results(html_output, criteria)
            
            assert result['is_complete'] == False
            assert result['grade'] == 'C'
            assert 'correlation matrix' in result['feedback']
    
    @pytest.mark.asyncio
    async def test_api_retry_mechanism(self, architect, sample_csv_file, mock_structured_response):
        """Test retry mechanism when API calls fail."""
        
        # All calls fail, should fall back to default
        side_effects = [
            Exception("API rate limit exceeded"),
            Exception("Connection timeout"),
            Exception("Service unavailable")
        ]
        
        with patch.object(architect.client.beta.chat.completions, 'parse',
                         side_effect=side_effects) as mock_parse:
            result = await architect.profile_and_plan(
                sample_csv_file,
                "Analyze this data"
            )
            
            # Should have tried 3 times
            assert mock_parse.call_count == 3
            
            # Should get default fallback after max retries
            assert 'requirements' in result
            assert 'acceptance_criteria' in result
            
            # Verify it's the default fallback, not from API
            assert isinstance(result['acceptance_criteria'], list)
            assert len(result['acceptance_criteria']) > 0
            # The default should include basic analysis requirements
            assert any('statistics' in criterion.lower() or 'summary' in criterion.lower() 
                      for criterion in result['acceptance_criteria']), \
                "Default fallback should include basic statistical analysis"
            assert result['is_complete'] == False  # Default is always incomplete
    
    @pytest.mark.asyncio
    async def test_data_profiling_accuracy(self, architect, sample_csv_file):
        """Test that data profiling extracts correct information."""
        profile = architect._profile_data(sample_csv_file)
        
        # Check basic info
        assert "File type: .csv" in profile
        assert "5 rows, 5 columns" in profile
        assert "['product', 'category', 'price', 'quantity', 'rating']" in profile
        
        # Check numeric columns detection
        assert "Numeric columns:" in profile
        assert "price:" in profile
        assert "quantity:" in profile
        assert "rating:" in profile
        
        # Check categorical columns
        assert "Categorical columns:" in profile
        assert "product:" in profile
        assert "category:" in profile


class TestCoderAgentMocking(TestOpenAIMocking):
    """Test Coder agent with mocked OpenAI responses."""
    
    @pytest.fixture
    def coder(self):
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
            return CoderAgent()
    
    @pytest.fixture
    def sample_data_path(self):
        return Path("/tmp/test_data.csv")
    
    @pytest.mark.asyncio
    async def test_generate_code_with_complete_script(self, coder, sample_data_path, mock_openai_response):
        """Test code generation with a complete Python script response."""
        
        python_code = '''#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from io import BytesIO
import base64

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Analyze sales data')
    parser.add_argument('--data', required=True, help='Path to data file')
    args = parser.parse_args()
    
    # Read data
    df = pd.read_csv(args.data)
    
    # Generate analysis
    summary_stats = df.describe()
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    df.groupby('category')['price'].mean().plot(kind='bar', ax=ax)
    ax.set_title('Average Price by Category')
    ax.set_xlabel('Category')
    ax.set_ylabel('Average Price ($)')
    
    # Convert plot to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    # Generate HTML output
    html = f"""
    <html>
    <head><title>Sales Analysis Report</title></head>
    <body>
        <h1>Sales Data Analysis</h1>
        <h2>Summary Statistics</h2>
        {summary_stats.to_html()}
        <h2>Price Analysis by Category</h2>
        <img src="data:image/png;base64,{img_base64}" />
    </body>
    </html>
    """
    
    print(html)

if __name__ == '__main__':
    main()
'''
        
        # Mock response with code in markdown
        mock_response = mock_openai_response(
            f"```python\n{python_code}\n```",
            usage={'prompt_tokens': 300, 'completion_tokens': 500, 'total_tokens': 800}
        )
        
        with patch.object(coder.client.chat.completions, 'create',
                         new=AsyncMock(return_value=mock_response)) as mock_create:
            
            requirements = "Analyze sales data and create price visualizations"
            code = await coder.generate_code(requirements, [], sample_data_path)
            
            # Verify API was called
            mock_create.assert_called_once()
            messages = mock_create.call_args.kwargs['messages']
            
            # Check message structure
            assert len(messages) == 2
            assert messages[0]['role'] == 'system'
            assert 'Python data analysis code generator' in messages[0]['content']
            assert messages[1]['role'] == 'user'
            
            # Verify code extraction from markdown
            assert '```' not in code
            assert 'import pandas as pd' in code
            assert 'argparse' in code
            assert '--data' in code
            assert 'base64' in code
            assert '<html>' in code
    
    @pytest.mark.asyncio
    async def test_revise_code_with_feedback(self, coder, sample_data_path, mock_openai_response):
        """Test code revision based on feedback."""
        
        previous_code = '''import pandas as pd
df = pd.read_csv('hardcoded_path.csv')
print(df.head())'''
        
        revised_code = '''import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True)
args = parser.parse_args()

df = pd.read_csv(args.data)

# Generate proper HTML output
html = f"""
<html>
<body>
    <h1>Data Analysis</h1>
    <pre>{df.describe().to_html()}</pre>
</body>
</html>
"""
print(html)'''
        
        mock_response = mock_openai_response(
            f"```python\n{revised_code}\n```"
        )
        
        with patch.object(coder.client.chat.completions, 'create',
                         new=AsyncMock(return_value=mock_response)) as mock_create:
            
            code = await coder.revise_code(
                previous_code,
                "Analyze data and output HTML",
                [],  # Add empty acceptance criteria
                "Code uses hardcoded path instead of argparse. No HTML output generated.",
                sample_data_path,
                grade="D",
                grade_justification="Missing argparse and HTML output"
            )
            
            # Verify the revision call
            mock_create.assert_called_once()
            messages = mock_create.call_args.kwargs['messages']
            
            # Check that feedback was included
            assert 'hardcoded path' in messages[1]['content']
            assert previous_code in messages[1]['content']
            
            # Verify revised code
            assert 'argparse' in code
            assert '--data' in code
            assert '<html>' in code
            assert 'hardcoded_path.csv' not in code
    
    @pytest.mark.asyncio
    async def test_code_extraction_variations(self, coder, sample_data_path, mock_openai_response):
        """Test extraction of code from various markdown formats."""
        
        test_cases = [
            # Python-specific code block
            ("```python\nprint('test')\n```", "print('test')"),
            # Generic code block
            ("```\nprint('generic')\n```", "print('generic')"),
            # Code with explanation
            ("Here's the code:\n```python\nprint('explained')\n```\nThis prints explained.", "print('explained')"),
            # No code block
            ("print('no block')", "print('no block')"),
            # Multiple code blocks (should get first)
            ("```python\nfirst_block()\n```\nSome text\n```python\nsecond_block()\n```", "first_block()")
        ]
        
        for input_text, expected_output in test_cases:
            mock_response = mock_openai_response(input_text)
            
            with patch.object(coder.client.chat.completions, 'create',
                             new=AsyncMock(return_value=mock_response)):
                code = await coder.generate_code("Test", [], sample_data_path)
                assert code.strip() == expected_output.strip()
    
    @pytest.mark.asyncio
    async def test_retry_on_failure(self, coder, sample_data_path):
        """Test that coder retries on API failures."""
        
        # Create a sequence of failures then success
        side_effects = [
            Exception("Rate limit"),
            Exception("Timeout"),
            AsyncMock(return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content="print('success')"))],
                usage=MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)
            ))()
        ]
        
        with patch.object(coder.client.chat.completions, 'create',
                         side_effect=side_effects) as mock_create:
            code = await coder.generate_code("Simple test", [], sample_data_path)
            
            # Should have tried 3 times
            assert mock_create.call_count == 3
            assert code == "print('success')"
    
    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, coder, sample_data_path):
        """Test that coder raises exception after max retries."""
        
        with patch.object(coder.client.chat.completions, 'create',
                         side_effect=Exception("Persistent failure")):
            with pytest.raises(Exception) as exc_info:
                await coder.generate_code("Test", [], sample_data_path)
            
            assert "Failed to generate code after" in str(exc_info.value)


class TestIntegratedWorkflow(TestOpenAIMocking):
    """Test integrated workflow with both agents."""
    
    @pytest.fixture
    def architect(self):
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
            return ArchitectAgent()
    
    @pytest.fixture
    def coder(self):
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
            return CoderAgent()
    
    @pytest.mark.asyncio
    async def test_full_analysis_workflow(self, architect, coder, sample_csv_file, 
                                         mock_structured_response, mock_openai_response):
        """Test complete workflow from planning to code generation to validation."""
        
        # Step 1: Architect plans
        plan_response = ArchitectPlanResponse(
            requirements="Analyze product data for insights",
            acceptance_criteria=[
                "Generate summary statistics",
                "Create price distribution chart",
                "Identify top products"
            ],
            criteria_importance="Statistics provide baseline understanding. Charts visualize patterns. Top products show leaders.",
            is_complete=False,
            feedback=""
        )
        
        with patch.object(architect.client.beta.chat.completions, 'parse',
                         return_value=mock_structured_response(plan_response)):
            plan = await architect.profile_and_plan(sample_csv_file, "Analyze products")
        
        assert len(plan['acceptance_criteria']) == 3
        
        # Step 2: Coder generates code
        generated_code = '''import pandas as pd
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True)
args = parser.parse_args()
df = pd.read_csv(args.data)
print("<html><body>Analysis complete</body></html>")'''
        
        with patch.object(coder.client.chat.completions, 'create',
                         new=AsyncMock(return_value=mock_openai_response(f"```python\n{generated_code}\n```"))):
            code = await coder.generate_code(plan['requirements'], plan['acceptance_criteria'], sample_csv_file)
        
        assert 'argparse' in code
        assert '<html>' in code
        
        # Step 3: Architect validates (finds issues)
        validation_fail = ArchitectValidationResponse(
            criteria_evaluation="Missing visualizations and detailed statistics",
            grade="C+",
            grade_justification="Basic structure present but lacks required analysis depth",
            is_complete=False,
            feedback="Add matplotlib charts and detailed statistics"
        )
        
        with patch.object(architect.client.beta.chat.completions, 'parse',
                         return_value=mock_structured_response(validation_fail)):
            validation = await architect.validate_results("<html>basic</html>", plan['acceptance_criteria'])
        
        assert validation['is_complete'] == False
        assert 'matplotlib' in validation['feedback']
        
        # Step 4: Coder revises
        revised_code = '''import pandas as pd
import matplotlib.pyplot as plt
import argparse
# ... complete analysis code ...
print("<html><body>Complete analysis with charts</body></html>")'''
        
        with patch.object(coder.client.chat.completions, 'create',
                         new=AsyncMock(return_value=mock_openai_response(f"```python\n{revised_code}\n```"))):
            code = await coder.revise_code(
                generated_code,
                plan['requirements'],
                plan['acceptance_criteria'],
                validation['feedback'],
                sample_csv_file,
                grade=validation.get('grade', 'C'),
                grade_justification=validation.get('grade_justification', 'Needs improvement')
            )
        
        assert 'matplotlib' in code
        
        # Step 5: Final validation passes
        validation_pass = ArchitectValidationResponse(
            criteria_evaluation="All criteria met with comprehensive analysis",
            grade="B+",
            grade_justification="Solid analysis with good visualizations and insights",
            is_complete=True,
            feedback=""
        )
        
        with patch.object(architect.client.beta.chat.completions, 'parse',
                         return_value=mock_structured_response(validation_pass)):
            final_validation = await architect.validate_results(
                "<html>Complete analysis</html>",
                plan['acceptance_criteria']
            )
        
        assert final_validation['is_complete'] == True
        assert final_validation['grade'] == 'B+'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])