"""Test that coder revision properly filters images from previous output."""

import pytest
import sys
import os
from unittest.mock import Mock, AsyncMock, patch
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set dummy API key before importing
os.environ['OPENAI_API_KEY'] = 'test-key'

from agents.coder import CoderAgent
from agents.architect import strip_base64_images


class TestCoderRevisionFiltering:
    """Test that images are filtered when passing output to coder revision."""
    
    def test_strip_base64_images_removes_img_tags(self):
        """Test that base64 images in img tags are removed."""
        html = '''
        <html>
        <body>
        <h1>Analysis Results</h1>
        <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA...">
        <p>Some text</p>
        <img src='data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD...'>
        </body>
        </html>
        '''
        
        filtered = strip_base64_images(html)
        
        assert "iVBORw0KGgoAAAANSUhEUgAAAAUA" not in filtered
        assert "/9j/4AAQSkZJRgABAQEAYABgAAD" not in filtered
        assert "[BASE64_IMAGE_REMOVED]" in filtered
        assert "<h1>Analysis Results</h1>" in filtered
        assert "<p>Some text</p>" in filtered
    
    def test_strip_base64_images_removes_svg(self):
        """Test that inline SVG elements are removed."""
        html = '''
        <html>
        <body>
        <h1>Chart</h1>
        <svg width="100" height="100">
            <circle cx="50" cy="50" r="40" stroke="green" stroke-width="4" fill="yellow" />
            <text>Some SVG text</text>
        </svg>
        <p>Analysis complete</p>
        </body>
        </html>
        '''
        
        filtered = strip_base64_images(html)
        
        assert "circle cx=" not in filtered
        assert "Some SVG text" not in filtered
        assert "<svg>[SVG_REMOVED]</svg>" in filtered
        assert "<h1>Chart</h1>" in filtered
        assert "<p>Analysis complete</p>" in filtered
    
    def test_strip_base64_images_removes_background_images(self):
        """Test that base64 images in CSS are removed."""
        html = '''
        <div style="background-image: url(data:image/png;base64,ABC123...);">
            <p>Content with background</p>
        </div>
        <span style='background: url("data:image/jpeg;base64,XYZ789...");'>Text</span>
        '''
        
        filtered = strip_base64_images(html)
        
        assert "ABC123" not in filtered
        assert "XYZ789" not in filtered
        assert "url([BASE64_IMAGE_REMOVED])" in filtered
        assert "<p>Content with background</p>" in filtered
    
    def test_strip_base64_preserves_regular_images(self):
        """Test that regular image URLs are preserved."""
        html = '''
        <img src="https://example.com/image.png" alt="Chart">
        <img src="/static/logo.jpg">
        <div style="background-image: url('/static/bg.png');">Content</div>
        '''
        
        filtered = strip_base64_images(html)
        
        assert "https://example.com/image.png" in filtered
        assert "/static/logo.jpg" in filtered
        assert "url('/static/bg.png')" in filtered
    
    @pytest.mark.asyncio
    async def test_coder_revise_code_with_filtered_output(self):
        """Test that coder.revise_code properly filters previous output."""
        coder = CoderAgent()
        
        # Mock the OpenAI client
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "print('Fixed code')"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = Mock(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        
        coder.client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # HTML with images that should be filtered
        previous_output = '''
        <html>
        <body>
        <h1>Data Analysis Report</h1>
        <p>Mean: 42.5</p>
        <img src="data:image/png;base64,LONGBASE64STRING...">
        <svg><rect width="100" height="100"/></svg>
        <p>Conclusion: Data shows trend</p>
        </body>
        </html>
        '''
        
        # Call revise_code with previous output
        from pathlib import Path
        result = await coder.revise_code(
            previous_code="print('Original code')",
            requirements="Analyze data",
            acceptance_criteria=["Show statistics"],
            feedback="Add median calculation",
            data_path=Path("/tmp/test.csv"),
            grade="C+",
            grade_justification="Missing key statistics",
            previous_output=previous_output
        )
        
        # Check that the API was called
        assert coder.client.chat.completions.create.called
        
        # Get the actual prompt sent to the API
        call_args = coder.client.chat.completions.create.call_args
        messages = call_args[1]['messages']
        user_message = messages[1]['content']
        
        # Verify images were filtered from the prompt
        assert "LONGBASE64STRING" not in user_message
        assert "rect width=" not in user_message
        assert "[BASE64_IMAGE_REMOVED]" in user_message or "[SVG_REMOVED]" in user_message
        
        # Verify important content was preserved
        assert "Mean: 42.5" in user_message
        assert "Data shows trend" in user_message
    
    @pytest.mark.asyncio
    async def test_coder_revise_without_previous_output_backwards_compat(self):
        """Test that coder.revise_code works without previous_output (backwards compatibility)."""
        coder = CoderAgent()
        
        # Mock the OpenAI client
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "print('Fixed code')"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = Mock(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        
        coder.client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Call revise_code WITHOUT previous_output
        from pathlib import Path
        result = await coder.revise_code(
            previous_code="print('Original code')",
            requirements="Analyze data",
            acceptance_criteria=["Show statistics"],
            feedback="Add median calculation",
            data_path=Path("/tmp/test.csv"),
            grade="C+",
            grade_justification="Missing key statistics"
            # Note: no previous_output parameter
        )
        
        # Should still work
        assert result == "print('Fixed code')"
        assert coder.client.chat.completions.create.called
    
    def test_filtering_handles_malformed_html(self):
        """Test that filtering handles malformed or incomplete HTML."""
        malformed_html = '''
        <h1>Report
        <img src="data:image/png;base64,ABC
        <p>Text<svg><circle
        '''
        
        # Should not crash
        filtered = strip_base64_images(malformed_html)
        assert filtered is not None
        assert "Report" in filtered
    
    def test_filtering_preserves_plotly_json(self):
        """Test that Plotly JSON data is preserved (not mistaken for base64)."""
        html_with_plotly = '''
        <script type="application/json" id="plotly-data">
        {"data": [{"x": [1,2,3], "y": [4,5,6]}]}
        </script>
        <div id="plotly-div"></div>
        '''
        
        filtered = strip_base64_images(html_with_plotly)
        
        # Plotly data should be preserved
        assert '"data":' in filtered
        assert '"x": [1,2,3]' in filtered
        assert 'plotly-div' in filtered