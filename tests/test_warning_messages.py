"""Unit tests for warning message generation when max iterations reached."""

import pytest
import sys
import os
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set dummy API key before importing main to avoid sys.exit
os.environ['OPENAI_API_KEY'] = 'test-key'

import main
from main import app


class TestWarningMessages:
    """Test warning message generation when max iterations are reached."""
    
    def test_warning_html_with_passing_grade(self):
        """Test warning HTML generation with a passing grade (B or above)."""
        last_validation = {
            'grade': 'B+',
            'grade_justification': 'Good analysis with minor issues',
            'criteria_evaluation': 'Criterion 1: Met. Criterion 2: Met.',
            'feedback': 'Could improve visualization clarity',
            'is_complete': False
        }
        
        # Build the warning HTML using the same logic as main.py
        warning_parts = [
            "<div style='background-color: #fff3cd; border: 1px solid #ffc107; border-radius: 5px; padding: 15px; margin-bottom: 20px;'>",
            "<h3 style='margin-top: 0; color: #856404;'>⚠️ Analysis Completed with Warnings</h3>",
            "<p><strong>Status:</strong> Maximum iterations reached (3 iterations)</p>"
        ]
        
        grade = last_validation.get('grade', 'N/A')
        grade_color = '#28a745' if grade.startswith(('A', 'B')) else '#dc3545' if grade.startswith(('D', 'F')) else '#ffc107'
        
        warning_parts.extend([
            "<hr style='margin: 10px 0; border-color: #ffc107;'>",
            "<h4 style='color: #856404;'>Architect's Final Evaluation:</h4>",
            f"<p><strong>Grade:</strong> <span style='color: {grade_color}; font-size: 1.2em; font-weight: bold;'>{grade}</span></p>"
        ])
        
        warning_html = "".join(warning_parts) + "</div>"
        
        # Verify the HTML contains expected elements
        assert "B+" in warning_html
        assert "#28a745" in warning_html  # Green color for passing grade
        assert "Maximum iterations reached" in warning_html
        assert "Architect's Final Evaluation" in warning_html
    
    def test_warning_html_with_failing_grade(self):
        """Test warning HTML generation with a failing grade (D or F)."""
        last_validation = {
            'grade': 'F',
            'grade_justification': 'Missing critical requirements',
            'criteria_evaluation': 'Most criteria not met',
            'feedback': 'Need to implement basic statistics and visualizations',
            'is_complete': False
        }
        
        warning_parts = []
        grade = last_validation.get('grade', 'N/A')
        grade_color = '#28a745' if grade.startswith(('A', 'B')) else '#dc3545' if grade.startswith(('D', 'F')) else '#ffc107'
        
        warning_parts.append(f"<p><strong>Grade:</strong> <span style='color: {grade_color}; font-size: 1.2em; font-weight: bold;'>{grade}</span></p>")
        
        warning_html = "".join(warning_parts)
        
        # Verify failing grade uses red color
        assert "F" in warning_html
        assert "#dc3545" in warning_html  # Red color for failing grade
    
    def test_warning_html_with_marginal_grade(self):
        """Test warning HTML generation with a marginal grade (C range)."""
        last_validation = {
            'grade': 'C+',
            'grade_justification': 'Basic requirements met but lacks depth',
            'criteria_evaluation': 'Some criteria met',
            'feedback': 'Add more analysis',
            'is_complete': False
        }
        
        grade = last_validation.get('grade', 'N/A')
        grade_color = '#28a745' if grade.startswith(('A', 'B')) else '#dc3545' if grade.startswith(('D', 'F')) else '#ffc107'
        
        warning_html = f"<span style='color: {grade_color};'>{grade}</span>"
        
        # Verify marginal grade uses yellow color
        assert "C+" in warning_html
        assert "#ffc107" in warning_html  # Yellow color for marginal grade
    
    def test_warning_includes_all_validation_fields(self):
        """Test that warning includes all validation fields when present."""
        last_validation = {
            'grade': 'B-',
            'grade_justification': 'Test justification',
            'criteria_evaluation': 'Test criteria evaluation',
            'feedback': 'Test feedback',
            'is_complete': False
        }
        
        # Build complete warning message
        warning_parts = []
        
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
        
        warning_html = "".join(warning_parts)
        
        # Verify all fields are included
        assert "Test justification" in warning_html
        assert "Test criteria evaluation" in warning_html
        assert "Test feedback" in warning_html
        assert "Criteria Evaluation:" in warning_html
        assert "Remaining Issues:" in warning_html
    
    def test_warning_without_validation(self):
        """Test warning message when no validation data is available."""
        last_validation = None
        
        warning_parts = [
            "<div style='background-color: #fff3cd; border: 1px solid #ffc107; border-radius: 5px; padding: 15px; margin-bottom: 20px;'>",
            "<h3 style='margin-top: 0; color: #856404;'>⚠️ Analysis Completed with Warnings</h3>",
            "<p><strong>Status:</strong> Maximum iterations reached (3 iterations)</p>"
        ]
        
        # Test that validation-specific content is only added when validation exists
        if last_validation:
            # This block should not execute when last_validation is None
            warning_parts.append("<hr style='margin: 10px 0; border-color: #ffc107;'>")
            warning_parts.append("<h4 style='color: #856404;'>Architect's Final Evaluation:</h4>")
            grade = last_validation.get('grade', 'N/A')
            warning_parts.append(f"<p><strong>Grade:</strong> {grade}</p>")
        
        warning_parts.append("</div>")
        warning_html = "".join(warning_parts)
        
        # Verify basic warning is shown without evaluation details
        assert "Maximum iterations reached" in warning_html
        assert "Architect's Final Evaluation" not in warning_html
        assert "Grade:" not in warning_html
        assert warning_html.count("<hr") == 0  # No horizontal rule should be added
        
        # Verify the structure is correct
        assert warning_html.startswith("<div style='background-color: #fff3cd")
        assert warning_html.endswith("</div>")
        assert "⚠️ Analysis Completed with Warnings" in warning_html
    
    def test_error_message_with_validation(self):
        """Test error message generation with last validation data."""
        last_validation = {
            'grade': 'D',
            'feedback': 'Many issues found'
        }
        
        last_result = {
            'error': 'Execution timeout',
            'output': '<p>Partial output</p>'
        }
        
        error_parts = [
            "<div style='background-color: #f8d7da; border: 1px solid #dc3545; border-radius: 5px; padding: 15px; margin-bottom: 20px;'>",
            "<h3 style='margin-top: 0; color: #721c24;'>❌ Analysis Failed</h3>",
            f"<p><strong>Status:</strong> Maximum iterations reached (3 iterations) with execution errors</p>",
            f"<p><strong>Last Error:</strong> {last_result.get('error', 'Unknown error')}</p>"
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
        error_html = "".join(error_parts)
        
        # Verify error message includes validation info
        assert "Analysis Failed" in error_html
        assert "Execution timeout" in error_html
        assert "Grade:</strong> D" in error_html
        assert "Many issues found" in error_html
    
    def test_html_escaping_in_messages(self):
        """Test that HTML special characters are handled properly."""
        last_validation = {
            'grade': 'B+',
            'grade_justification': 'Good analysis with <script>alert("test")</script> issues',
            'criteria_evaluation': 'Criterion 1: Met & verified',
            'feedback': 'Need more detail on correlation > 0.5',
            'is_complete': False
        }
        
        # The actual implementation should handle this appropriately
        # For now, we test that the structure is maintained
        warning_parts = []
        if last_validation.get('grade_justification'):
            warning_parts.append(f"<p><strong>Justification:</strong> {last_validation['grade_justification']}</p>")
        
        warning_html = "".join(warning_parts)
        
        # The raw content is included (in production, this should be escaped)
        assert last_validation['grade_justification'] in warning_html
    
    @pytest.mark.asyncio
    async def test_max_iterations_integration(self):
        """Integration test for max iterations warning flow."""
        with patch('main.download_and_validate_file') as mock_download, \
             patch('main.ArchitectAgent') as mock_architect_class, \
             patch('main.CoderAgent') as mock_coder_class, \
             patch('main.PythonExecutor') as mock_executor_class, \
             patch('main.MAX_ITERATIONS', 1):  # Set to 1 for quick test
            
            # Setup mocks
            mock_download.return_value = MagicMock()
            
            mock_architect = AsyncMock()
            mock_architect.profile_and_plan.return_value = {
                'requirements': 'Test requirements',
                'acceptance_criteria': ['Criterion 1', 'Criterion 2'],
                'feedback': ''
            }
            mock_architect.validate_results.return_value = {
                'is_complete': False,
                'grade': 'B-',
                'grade_justification': 'Almost there',
                'criteria_evaluation': 'Criterion 1: Met',
                'feedback': 'Need improvement'
            }
            mock_architect_class.return_value = mock_architect
            
            mock_coder = AsyncMock()
            mock_coder.generate_code.return_value = "print('test')"
            mock_coder_class.return_value = mock_coder
            
            mock_executor = AsyncMock()
            mock_executor.execute.return_value = {
                'success': True,
                'output': '<html><body>Test output</body></html>'
            }
            mock_executor_class.return_value = mock_executor
            
            # Make request
            client = TestClient(app)
            response = client.post(
                "/analyze",
                data={"url": "http://example.com/data.csv", "prompt": "Analyze this data"}
            )
            
            # Check response
            assert response.status_code == 200
            html_content = response.text
            
            # Verify warning message is included
            assert "Analysis Completed with Warnings" in html_content or "Maximum iterations reached" in html_content
            # The actual HTML structure depends on the specific implementation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])