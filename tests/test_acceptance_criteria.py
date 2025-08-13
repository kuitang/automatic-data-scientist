"""Unit tests for acceptance criteria functionality in coder agent."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.coder import CoderAgent


class TestCoderAcceptanceCriteria:
    """Test that coder agent properly handles acceptance criteria."""
    
    @pytest.fixture
    def coder_agent(self):
        """Create a coder agent instance with mocked OpenAI client."""
        with patch('agents.coder.openai.AsyncOpenAI'):
            agent = CoderAgent()
            agent.client = Mock()
            agent.client.chat = Mock()
            agent.client.chat.completions = Mock()
            agent.client.chat.completions.create = AsyncMock()
            return agent
    
    @pytest.fixture
    def sample_data_path(self, tmp_path):
        """Create a temporary CSV file for testing."""
        data_file = tmp_path / "test_data.csv"
        data_file.write_text("col1,col2\n1,2\n3,4")
        return data_file
    
    @pytest.mark.asyncio
    async def test_generate_code_includes_acceptance_criteria(self, coder_agent, sample_data_path):
        """Test that generate_code includes acceptance criteria in the prompt."""
        # Setup mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "import pandas as pd\nprint('test')"
        mock_response.choices[0].finish_reason = "stop"
        coder_agent.client.chat.completions.create.return_value = mock_response
        
        requirements = "Analyze the data and create visualizations"
        acceptance_criteria = [
            "Calculate summary statistics",
            "Create at least 2 plots",
            "Include correlation analysis"
        ]
        
        # Call generate_code
        result = await coder_agent.generate_code(requirements, acceptance_criteria, sample_data_path)
        
        # Verify the API was called
        assert coder_agent.client.chat.completions.create.called
        
        # Get the messages sent to the API
        call_args = coder_agent.client.chat.completions.create.call_args
        messages = call_args[1]['messages']
        
        # Check that acceptance criteria are in the user message
        user_message = messages[1]['content']
        assert "Acceptance Criteria (must be met):" in user_message
        assert "Calculate summary statistics" in user_message
        assert "Create at least 2 plots" in user_message
        assert "Include correlation analysis" in user_message
        
        # Verify the code was returned
        assert result == "import pandas as pd\nprint('test')"
    
    @pytest.mark.asyncio
    async def test_revise_code_includes_acceptance_criteria(self, coder_agent, sample_data_path):
        """Test that revise_code includes acceptance criteria in the prompt."""
        # Setup mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "import pandas as pd\nprint('revised')"
        mock_response.choices[0].finish_reason = "stop"
        coder_agent.client.chat.completions.create.return_value = mock_response
        
        previous_code = "import pandas as pd\nprint('original')"
        requirements = "Analyze the data and create visualizations"
        acceptance_criteria = [
            "Calculate summary statistics",
            "Create at least 2 plots",
            "Include correlation analysis"
        ]
        feedback = "Missing correlation analysis and second plot"
        
        # Call revise_code with required grade parameters
        result = await coder_agent.revise_code(
            previous_code, requirements, acceptance_criteria, feedback, sample_data_path,
            grade="C+",
            grade_justification="Missing correlation analysis"
        )
        
        # Verify the API was called
        assert coder_agent.client.chat.completions.create.called
        
        # Get the messages sent to the API
        call_args = coder_agent.client.chat.completions.create.call_args
        messages = call_args[1]['messages']
        
        # Check that acceptance criteria are in the user message
        user_message = messages[1]['content']
        assert "Acceptance Criteria (must be met):" in user_message
        assert "Calculate summary statistics" in user_message
        assert "Create at least 2 plots" in user_message
        assert "Include correlation analysis" in user_message
        assert "Missing correlation analysis and second plot" in user_message
        
        # Verify the revised code was returned
        assert result == "import pandas as pd\nprint('revised')"
    
    @pytest.mark.asyncio
    async def test_acceptance_criteria_formatting(self, coder_agent, sample_data_path):
        """Test that acceptance criteria are properly formatted as bullet points."""
        # Setup mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "print('test')"
        mock_response.choices[0].finish_reason = "stop"
        coder_agent.client.chat.completions.create.return_value = mock_response
        
        requirements = "Test requirements"
        acceptance_criteria = [
            "First criterion",
            "Second criterion",
            "Third criterion"
        ]
        
        # Call generate_code
        await coder_agent.generate_code(requirements, acceptance_criteria, sample_data_path)
        
        # Get the user message
        call_args = coder_agent.client.chat.completions.create.call_args
        messages = call_args[1]['messages']
        user_message = messages[1]['content']
        
        # Check formatting
        assert "- First criterion" in user_message
        assert "- Second criterion" in user_message
        assert "- Third criterion" in user_message
    
    @pytest.mark.asyncio
    async def test_empty_acceptance_criteria(self, coder_agent, sample_data_path):
        """Test handling of empty acceptance criteria list."""
        # Setup mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "print('test')"
        mock_response.choices[0].finish_reason = "stop"
        coder_agent.client.chat.completions.create.return_value = mock_response
        
        requirements = "Test requirements"
        acceptance_criteria = []
        
        # Call generate_code
        result = await coder_agent.generate_code(requirements, acceptance_criteria, sample_data_path)
        
        # Should still work without error
        assert result == "print('test')"
        
        # Get the user message
        call_args = coder_agent.client.chat.completions.create.call_args
        messages = call_args[1]['messages']
        user_message = messages[1]['content']
        
        # Verify that empty criteria are handled properly
        assert "Acceptance Criteria (must be met):" in user_message
        # Should not have any bullet points since criteria list is empty
        assert "- " not in user_message.split("Acceptance Criteria (must be met):")[1].split("\n")[0:3], \
            "Should not have bullet points immediately after criteria header when list is empty"
        # Verify the requirements are still included
        assert "Test requirements" in user_message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])