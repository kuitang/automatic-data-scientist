import pytest
import asyncio
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, AsyncMock, MagicMock
import httpx

import sys
sys.path.append(str(Path(__file__).parent.parent))

from main import download_and_validate_file


class TestFileHandling:
    
    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    async def test_download_csv_file(self, temp_dir):
        csv_content = b"name,age,score\nAlice,25,90\nBob,30,85"
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = AsyncMock()
            mock_response.content = csv_content
            mock_response.raise_for_status = MagicMock()
            mock_response.headers = {'content-type': 'text/csv'}
            
            mock_client_instance = AsyncMock()
            mock_client_instance.get.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            result = await download_and_validate_file(
                "http://example.com/data.csv",
                temp_dir
            )
            
            assert result.exists()
            assert result.suffix == '.csv'
            assert result.read_bytes() == csv_content
    
    @pytest.mark.asyncio
    async def test_file_size_validation(self, temp_dir):
        large_content = b"x" * (101 * 1024 * 1024)  # 101MB
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = AsyncMock()
            mock_response.content = large_content
            mock_response.raise_for_status = MagicMock()
            
            mock_client_instance = AsyncMock()
            mock_client_instance.get.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            with pytest.raises(Exception) as exc_info:
                await download_and_validate_file(
                    "http://example.com/large.csv",
                    temp_dir
                )
            
            # HTTPException detail is in the detail attribute, not in str()
            assert "100" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_supported_file_types(self, temp_dir):
        test_cases = [
            ("data.csv", b"col1,col2\n1,2", ".csv"),
            ("data.xlsx", b"excel_content", ".xlsx"),
            ("data.json", b'{"key": "value"}', ".json"),
            ("data.parquet", b"parquet_content", ".parquet"),
        ]
        
        for filename, content, expected_ext in test_cases:
            with patch('httpx.AsyncClient') as mock_client:
                mock_response = AsyncMock()
                mock_response.content = content
                mock_response.raise_for_status = MagicMock()
                mock_response.headers = {}
                
                mock_client_instance = AsyncMock()
                mock_client_instance.get.return_value = mock_response
                mock_client.return_value.__aenter__.return_value = mock_client_instance
                
                result = await download_and_validate_file(
                    f"http://example.com/{filename}",
                    temp_dir
                )
                
                assert result.suffix == expected_ext
    
    @pytest.mark.asyncio
    async def test_unsupported_file_type(self, temp_dir):
        with patch('httpx.AsyncClient') as mock_client:
            mock_response = AsyncMock()
            mock_response.content = b"text content"
            mock_response.raise_for_status = MagicMock()
            mock_response.headers = {}
            
            mock_client_instance = AsyncMock()
            mock_client_instance.get.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            with pytest.raises(Exception) as exc_info:
                await download_and_validate_file(
                    "http://example.com/file.txt",
                    temp_dir
                )
            
            # HTTPException detail is in the detail attribute
            assert "Unsupported" in str(exc_info.value.detail) or ".csv" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_network_error_handling(self, temp_dir):
        with patch('httpx.AsyncClient') as mock_client:
            mock_client_instance = AsyncMock()
            mock_client_instance.get.side_effect = httpx.RequestError("Connection failed")
            mock_client.return_value.__aenter__.return_value = mock_client_instance
            
            with pytest.raises(Exception) as exc_info:
                await download_and_validate_file(
                    "http://example.com/data.csv",
                    temp_dir
                )
            
            # HTTPException detail is in the detail attribute
            assert "Failed to download" in str(exc_info.value.detail) or "Connection" in str(exc_info.value.detail)