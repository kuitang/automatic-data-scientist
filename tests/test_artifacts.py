"""Test suite for artifact saving functionality."""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

# Set a dummy API key before importing main to prevent sys.exit
os.environ['OPENAI_API_KEY'] = 'test-key-for-testing'

from main import app


class TestArtifactSaving:
    """Test artifact saving functionality when ARTIFACTS_OUTPUT is set."""
    
    @pytest.fixture
    def temp_artifacts_dir(self):
        """Create a temporary directory for artifacts."""
        temp_dir = tempfile.mkdtemp(prefix="test_artifacts_")
        yield temp_dir
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_env_artifacts(self, temp_artifacts_dir, monkeypatch):
        """Mock ARTIFACTS_OUTPUT environment variable."""
        monkeypatch.setenv("ARTIFACTS_OUTPUT", temp_artifacts_dir)
        return temp_artifacts_dir
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.mark.asyncio
    async def test_artifacts_dir_created_when_env_set(self, mock_env_artifacts):
        """Test that artifacts directory is created when ARTIFACTS_OUTPUT is set."""
        from main import analyze
        
        with patch('main.download_and_validate_file', new_callable=AsyncMock) as mock_download:
            with patch('main.ArchitectAgent') as mock_architect_class:
                with patch('main.CoderAgent') as mock_coder_class:
                    with patch('main.PythonExecutor') as mock_executor_class:
                        # Setup mocks
                        mock_download.return_value = Path("/tmp/test_data.csv")
                        
                        mock_architect = AsyncMock()
                        mock_architect.profile_and_plan.return_value = {
                            'requirements': 'Test requirements',
                            'acceptance_criteria': ['Criterion 1'],
                            'feedback': ''
                        }
                        mock_architect.validate_results.return_value = {
                            'is_complete': True,
                            'feedback': '',
                            'grade': 'A',
                            'grade_justification': 'Excellent',
                            'criteria_evaluation': 'All met'
                        }
                        mock_architect_class.return_value = mock_architect
                        
                        mock_coder = AsyncMock()
                        mock_coder.generate_code.return_value = "print('test')"
                        mock_coder_class.return_value = mock_coder
                        
                        mock_executor = AsyncMock()
                        mock_executor.execute.return_value = {
                            'success': True,
                            'output': '<html><body>Test</body></html>',
                            'error': None
                        }
                        mock_executor_class.return_value = mock_executor
                        
                        # Call analyze
                        from fastapi import Form
                        with patch('uuid.uuid4', return_value='test-uuid-123'):
                            response = await analyze(
                                url="http://example.com/data.csv",
                                prompt="Test analysis"
                            )
                        
                        # Check artifacts directory was created
                        artifacts_path = Path(mock_env_artifacts) / "test-uuid-123"
                        assert artifacts_path.exists()
                        
                        # Check request.txt was created
                        request_file = artifacts_path / "request.txt"
                        assert request_file.exists()
                        content = request_file.read_text()
                        assert "http://example.com/data.csv" in content
                        assert "Test analysis" in content
    
    @pytest.mark.asyncio
    async def test_iteration_artifacts_saved(self, mock_env_artifacts):
        """Test that iteration-specific artifacts are saved."""
        from main import analyze
        
        with patch('main.download_and_validate_file', new_callable=AsyncMock) as mock_download:
            with patch('main.ArchitectAgent') as mock_architect_class:
                with patch('main.CoderAgent') as mock_coder_class:
                    with patch('main.PythonExecutor') as mock_executor_class:
                        # Setup mocks
                        mock_download.return_value = Path("/tmp/test_data.csv")
                        
                        mock_architect = AsyncMock()
                        mock_architect.profile_and_plan.return_value = {
                            'requirements': 'Test requirements',
                            'acceptance_criteria': ['Criterion 1', 'Criterion 2'],
                            'feedback': ''
                        }
                        # First validation fails, second succeeds
                        mock_architect.validate_results.side_effect = [
                            {
                                'is_complete': False,
                                'feedback': 'Need improvements',
                                'grade': 'C',
                                'grade_justification': 'Needs work',
                                'criteria_evaluation': 'Partially met'
                            },
                            {
                                'is_complete': True,
                                'feedback': 'Good job',
                                'grade': 'B+',
                                'grade_justification': 'Well done',
                                'criteria_evaluation': 'All criteria met'
                            }
                        ]
                        mock_architect_class.return_value = mock_architect
                        
                        mock_coder = AsyncMock()
                        mock_coder.generate_code.return_value = "print('first attempt')"
                        mock_coder.revise_code.return_value = "print('revised code')"
                        mock_coder_class.return_value = mock_coder
                        
                        mock_executor = AsyncMock()
                        mock_executor.execute.side_effect = [
                            {
                                'success': True,
                                'output': '<html><body>First</body></html>',
                                'error': None
                            },
                            {
                                'success': True,
                                'output': '<html><body>Revised</body></html>',
                                'error': None
                            }
                        ]
                        mock_executor_class.return_value = mock_executor
                        
                        # Call analyze with controlled UUID
                        with patch('uuid.uuid4', return_value='test-uuid-456'):
                            response = await analyze(
                                url="http://example.com/data.csv",
                                prompt="Test analysis"
                            )
                        
                        artifacts_path = Path(mock_env_artifacts) / "test-uuid-456"
                        
                        # Check iteration 1 artifacts
                        iter1_path = artifacts_path / "iteration_1"
                        assert iter1_path.exists()
                        
                        script1 = iter1_path / "script.py"
                        assert script1.exists()
                        assert "first attempt" in script1.read_text()
                        
                        output1 = iter1_path / "output.html"
                        assert output1.exists()
                        assert "First" in output1.read_text()
                        
                        grade1 = iter1_path / "grade.json"
                        assert grade1.exists()
                        grade1_data = json.loads(grade1.read_text())
                        assert grade1_data['grade'] == 'C'
                        assert grade1_data['is_complete'] == False
                        
                        # Check iteration 2 artifacts
                        iter2_path = artifacts_path / "iteration_2"
                        assert iter2_path.exists()
                        
                        script2 = iter2_path / "script.py"
                        assert script2.exists()
                        assert "revised code" in script2.read_text()
                        
                        output2 = iter2_path / "output.html"
                        assert output2.exists()
                        assert "Revised" in output2.read_text()
                        
                        grade2 = iter2_path / "grade.json"
                        assert grade2.exists()
                        grade2_data = json.loads(grade2.read_text())
                        assert grade2_data['grade'] == 'B+'
                        assert grade2_data['is_complete'] == True
    
    @pytest.mark.asyncio
    async def test_error_artifacts_saved(self, mock_env_artifacts):
        """Test that error information is saved when execution fails."""
        from main import analyze
        
        with patch('main.download_and_validate_file', new_callable=AsyncMock) as mock_download:
            with patch('main.ArchitectAgent') as mock_architect_class:
                with patch('main.CoderAgent') as mock_coder_class:
                    with patch('main.PythonExecutor') as mock_executor_class:
                        # Setup mocks
                        mock_download.return_value = Path("/tmp/test_data.csv")
                        
                        mock_architect = AsyncMock()
                        mock_architect.profile_and_plan.return_value = {
                            'requirements': 'Test requirements',
                            'acceptance_criteria': ['Criterion 1'],
                            'feedback': ''
                        }
                        mock_architect_class.return_value = mock_architect
                        
                        mock_coder = AsyncMock()
                        mock_coder.generate_code.return_value = "print('error code')"
                        mock_coder.revise_code.return_value = "print('fixed code')"
                        mock_coder_class.return_value = mock_coder
                        
                        mock_executor = AsyncMock()
                        mock_executor.execute.side_effect = [
                            {
                                'success': False,
                                'output': '<html>Partial</html>',
                                'error': 'ImportError: missing module',
                                'stderr': 'Traceback...'
                            },
                            {
                                'success': True,
                                'output': '<html><body>Fixed</body></html>',
                                'error': None
                            }
                        ]
                        mock_executor_class.return_value = mock_executor
                        
                        mock_architect.validate_results.return_value = {
                            'is_complete': True,
                            'feedback': '',
                            'grade': 'A',
                            'grade_justification': 'Fixed',
                            'criteria_evaluation': 'All met after fix'
                        }
                        
                        # Call analyze
                        with patch('uuid.uuid4', return_value='test-uuid-789'):
                            response = await analyze(
                                url="http://example.com/data.csv",
                                prompt="Test analysis"
                            )
                        
                        artifacts_path = Path(mock_env_artifacts) / "test-uuid-789"
                        
                        # Check iteration 1 error artifacts
                        iter1_path = artifacts_path / "iteration_1"
                        assert iter1_path.exists()
                        
                        error_file = iter1_path / "error.json"
                        assert error_file.exists()
                        error_data = json.loads(error_file.read_text())
                        assert error_data['execution_failed'] == True
                        assert 'ImportError' in error_data['error']
                        assert 'Traceback' in error_data['stderr']
                        
                        partial_output = iter1_path / "partial_output.html"
                        assert partial_output.exists()
                        assert "Partial" in partial_output.read_text()
    
    @pytest.mark.asyncio
    async def test_no_artifacts_when_env_not_set(self):
        """Test that no artifacts are saved when ARTIFACTS_OUTPUT is not set."""
        from main import analyze
        
        # Ensure ARTIFACTS_OUTPUT is not set
        if 'ARTIFACTS_OUTPUT' in os.environ:
            del os.environ['ARTIFACTS_OUTPUT']
        
        with patch('main.download_and_validate_file', new_callable=AsyncMock) as mock_download:
            with patch('main.ArchitectAgent') as mock_architect_class:
                with patch('main.CoderAgent') as mock_coder_class:
                    with patch('main.PythonExecutor') as mock_executor_class:
                        # Setup minimal mocks
                        mock_download.return_value = Path("/tmp/test_data.csv")
                        
                        mock_architect = AsyncMock()
                        mock_architect.profile_and_plan.return_value = {
                            'requirements': 'Test',
                            'acceptance_criteria': ['Test'],
                            'feedback': ''
                        }
                        mock_architect.validate_results.return_value = {
                            'is_complete': True,
                            'feedback': '',
                            'grade': 'A'
                        }
                        mock_architect_class.return_value = mock_architect
                        
                        mock_coder = AsyncMock()
                        mock_coder.generate_code.return_value = "print('test')"
                        mock_coder_class.return_value = mock_coder
                        
                        mock_executor = AsyncMock()
                        mock_executor.execute.return_value = {
                            'success': True,
                            'output': '<html>Test</html>',
                            'error': None
                        }
                        mock_executor_class.return_value = mock_executor
                        
                        test_uuid = 'no-artifacts-uuid'
                        with patch('uuid.uuid4', return_value=test_uuid):
                            response = await analyze(
                                url="http://example.com/data.csv",
                                prompt="Test"
                            )
                        
                        # Verify no artifacts directory was created
                        assert response.status_code == 200
                        
                        # Check common artifact locations to ensure no directory was created
                        import tempfile
                        possible_locations = [
                            Path(f"/tmp/no-artifacts-uuid"),
                            Path(tempfile.gettempdir()) / "no-artifacts-uuid",
                            Path(".") / "no-artifacts-uuid"
                        ]
                        for location in possible_locations:
                            assert not location.exists(), f"Artifacts directory should not exist at {location}"
    
    @pytest.mark.asyncio
    async def test_artifacts_saving_failure_does_not_break_analysis(self, mock_env_artifacts):
        """Test that artifact saving failures don't interrupt the main analysis."""
        from main import analyze
        
        # Make artifacts directory read-only to cause save failures
        artifacts_base = Path(mock_env_artifacts)
        artifacts_base.chmod(0o444)
        
        try:
            with patch('main.download_and_validate_file', new_callable=AsyncMock) as mock_download:
                with patch('main.ArchitectAgent') as mock_architect_class:
                    with patch('main.CoderAgent') as mock_coder_class:
                        with patch('main.PythonExecutor') as mock_executor_class:
                            # Setup mocks
                            mock_download.return_value = Path("/tmp/test_data.csv")
                            
                            mock_architect = AsyncMock()
                            mock_architect.profile_and_plan.return_value = {
                                'requirements': 'Test',
                                'acceptance_criteria': ['Test'],
                                'feedback': ''
                            }
                            mock_architect.validate_results.return_value = {
                                'is_complete': True,
                                'feedback': '',
                                'grade': 'A',
                                'grade_justification': 'Good',
                                'criteria_evaluation': 'Met'
                            }
                            mock_architect_class.return_value = mock_architect
                            
                            mock_coder = AsyncMock()
                            mock_coder.generate_code.return_value = "print('test')"
                            mock_coder_class.return_value = mock_coder
                            
                            mock_executor = AsyncMock()
                            mock_executor.execute.return_value = {
                                'success': True,
                                'output': '<html>Success despite save failure</html>',
                                'error': None
                            }
                            mock_executor_class.return_value = mock_executor
                            
                            # Analysis should complete successfully despite artifact save failures
                            with patch('uuid.uuid4', return_value='fail-save-uuid'):
                                response = await analyze(
                                    url="http://example.com/data.csv",
                                    prompt="Test"
                                )
                            
                            # Verify analysis completed successfully
                            assert response.status_code == 200
                            assert b"Success despite save failure" in response.body
        finally:
            # Restore permissions
            artifacts_base.chmod(0o755)
    
    @pytest.mark.asyncio
    async def test_max_iterations_artifacts(self, mock_env_artifacts):
        """Test that all iteration artifacts are saved when max iterations is reached."""
        from main import analyze
        
        with patch('main.MAX_ITERATIONS', 2):  # Limit to 2 iterations for testing
            with patch('main.download_and_validate_file', new_callable=AsyncMock) as mock_download:
                with patch('main.ArchitectAgent') as mock_architect_class:
                    with patch('main.CoderAgent') as mock_coder_class:
                        with patch('main.PythonExecutor') as mock_executor_class:
                            # Setup mocks
                            mock_download.return_value = Path("/tmp/test_data.csv")
                            
                            mock_architect = AsyncMock()
                            mock_architect.profile_and_plan.return_value = {
                                'requirements': 'Test',
                                'acceptance_criteria': ['Never satisfied'],
                                'feedback': ''
                            }
                            # Always return incomplete
                            mock_architect.validate_results.return_value = {
                                'is_complete': False,
                                'feedback': 'Still not good enough',
                                'grade': 'D',
                                'grade_justification': 'Poor',
                                'criteria_evaluation': 'Not met'
                            }
                            mock_architect_class.return_value = mock_architect
                            
                            mock_coder = AsyncMock()
                            mock_coder.generate_code.return_value = "print('attempt 1')"
                            mock_coder.revise_code.return_value = "print('attempt 2')"
                            mock_coder_class.return_value = mock_coder
                            
                            mock_executor = AsyncMock()
                            mock_executor.execute.return_value = {
                                'success': True,
                                'output': '<html>Never good enough</html>',
                                'error': None
                            }
                            mock_executor_class.return_value = mock_executor
                            
                            with patch('uuid.uuid4', return_value='max-iter-uuid'):
                                response = await analyze(
                                    url="http://example.com/data.csv",
                                    prompt="Test"
                                )
                            
                            artifacts_path = Path(mock_env_artifacts) / "max-iter-uuid"
                            
                            # Check both iterations were saved
                            assert (artifacts_path / "iteration_1").exists()
                            assert (artifacts_path / "iteration_2").exists()
                            
                            # Verify warning in response about max iterations
                            assert b"Maximum iterations reached" in response.body