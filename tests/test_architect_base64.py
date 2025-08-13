"""Unit tests for base64 filtering in architect agent."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.architect import strip_base64_images


class TestBase64Filtering:
    """Test base64 image filtering functionality."""
    
    def test_strip_img_tag_base64(self):
        """Test filtering base64 from img tags."""
        html = '<img src="data:image/png;base64,iVBORw0KGgoAAAA..." alt="chart">'
        result = strip_base64_images(html)
        assert result == '<img src="[BASE64_IMAGE_REMOVED]" alt="chart">'
    
    def test_strip_css_background_base64(self):
        """Test filtering base64 from CSS background-image."""
        html = '<div style="background-image: url(data:image/jpeg;base64,/9j/4AAQ...);">'
        result = strip_base64_images(html)
        assert result == '<div style="background-image: url([BASE64_IMAGE_REMOVED]);">'
    
    def test_preserve_normal_images(self):
        """Test that normal image URLs are preserved."""
        html = '''
        <img src="https://example.com/image.png" alt="external">
        <img src="/static/local.jpg" alt="local">
        '''
        result = strip_base64_images(html)
        assert result == html
    
    def test_mixed_content(self):
        """Test filtering with both base64 and normal images."""
        html = '''
        <img src="https://example.com/logo.png" alt="logo">
        <img src="data:image/png;base64,ABCDEF" alt="chart">
        <img src="/static/icon.svg" alt="icon">
        '''
        expected = '''
        <img src="https://example.com/logo.png" alt="logo">
        <img src="[BASE64_IMAGE_REMOVED]" alt="chart">
        <img src="/static/icon.svg" alt="icon">
        '''
        result = strip_base64_images(html)
        assert result == expected
    
    def test_multiple_base64_images(self):
        """Test filtering multiple base64 images."""
        html = '''
        <div>
            <img src="data:image/png;base64,ABC123" class="chart1">
            <p>Text between images</p>
            <img src="data:image/jpeg;base64,XYZ789" class="chart2">
        </div>
        '''
        expected = '''
        <div>
            <img src="[BASE64_IMAGE_REMOVED]" class="chart1">
            <p>Text between images</p>
            <img src="[BASE64_IMAGE_REMOVED]" class="chart2">
        </div>
        '''
        result = strip_base64_images(html)
        assert result == expected
    
    def test_case_insensitive(self):
        """Test case insensitive matching."""
        html = '''
        <IMG SRC="DATA:IMAGE/PNG;BASE64,ABC" ALT="test1">
        <img Src="Data:Image/Jpeg;Base64,DEF" Alt="test2">
        '''
        expected = '''
        <IMG SRC="[BASE64_IMAGE_REMOVED]" ALT="test1">
        <img Src="[BASE64_IMAGE_REMOVED]" Alt="test2">
        '''
        result = strip_base64_images(html)
        assert result == expected
    
    def test_svg_with_base64(self):
        """Test filtering SVG elements containing base64."""
        html = '<svg width="100"><image href="data:image/png;base64,ABC"/></svg>'
        result = strip_base64_images(html)
        assert result == '<svg>[SVG_REMOVED]</svg>'
    
    def test_svg_without_base64(self):
        """Test filtering regular SVG elements without base64."""
        html = '<svg width="100" height="100"><circle cx="50" cy="50" r="40"/></svg>'
        result = strip_base64_images(html)
        assert result == '<svg>[SVG_REMOVED]</svg>'
    
    def test_inline_svg_chart(self):
        """Test filtering inline SVG chart without base64."""
        html = '''
        <svg viewBox="0 0 200 100">
            <rect x="10" y="10" width="30" height="80" fill="blue"/>
            <rect x="50" y="20" width="30" height="70" fill="red"/>
            <text x="10" y="95">Chart</text>
        </svg>
        '''
        result = strip_base64_images(html)
        assert '<svg>[SVG_REMOVED]</svg>' in result
    
    def test_multiple_svg_elements(self):
        """Test filtering multiple SVG elements."""
        html = '''
        <div>
            <svg id="chart1"><rect width="100" height="50"/></svg>
            <p>Some text</p>
            <svg id="chart2"><circle r="25"/></svg>
        </div>
        '''
        expected = '''
        <div>
            <svg>[SVG_REMOVED]</svg>
            <p>Some text</p>
            <svg>[SVG_REMOVED]</svg>
        </div>
        '''
        result = strip_base64_images(html)
        assert result == expected
    
    def test_nested_svg_elements(self):
        """Test filtering nested SVG elements."""
        html = '''
        <div class="chart-container">
            <svg xmlns="http://www.w3.org/2000/svg" width="400" height="300">
                <g transform="translate(10,10)">
                    <rect width="100" height="200" fill="green"/>
                    <text x="50" y="100">Data</text>
                </g>
            </svg>
        </div>
        '''
        result = strip_base64_images(html)
        assert '<svg>[SVG_REMOVED]</svg>' in result
        assert '<rect' not in result
        assert '<text' not in result
    
    def test_object_embed_tags(self):
        """Test filtering object and embed tags with base64."""
        html1 = '<object data="data:image/png;base64,ABC" type="image/png"></object>'
        result1 = strip_base64_images(html1)
        assert result1 == '<object data="[BASE64_IMAGE_REMOVED]" type="image/png"></object>'
        
        html2 = '<embed src="data:image/png;base64,XYZ" type="image/png">'
        result2 = strip_base64_images(html2)
        assert result2 == '<embed src="[BASE64_IMAGE_REMOVED]" type="image/png">'
    
    def test_quotes_handling(self):
        """Test handling of single and double quotes."""
        html1 = "<img src='data:image/png;base64,ABC' alt='test'>"
        result1 = strip_base64_images(html1)
        assert result1 == "<img src='[BASE64_IMAGE_REMOVED]' alt='test'>"
        
        html2 = '<img src="data:image/png;base64,ABC" alt="test">'
        result2 = strip_base64_images(html2)
        assert result2 == '<img src="[BASE64_IMAGE_REMOVED]" alt="test">'
    
    def test_empty_input(self):
        """Test handling of empty input."""
        assert strip_base64_images("") == ""
    
    def test_no_images(self):
        """Test HTML with no images."""
        html = "<div><p>Just text</p><span>No images</span></div>"
        result = strip_base64_images(html)
        assert result == html
    
    def test_malformed_base64(self):
        """Test handling of malformed base64 data."""
        html = '<img src="data:image/png;base64,!!!INVALID!!!" alt="bad">'
        result = strip_base64_images(html)
        assert result == '<img src="[BASE64_IMAGE_REMOVED]" alt="bad">'
    
    def test_real_world_example(self):
        """Test with a more realistic HTML example."""
        html = '''
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                .chart { background-image: url(data:image/png;base64,ABCDEF); }
            </style>
        </head>
        <body>
            <h1>Data Analysis Report</h1>
            <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ" alt="chart1">
            <p>Some analysis text here</p>
            <div style="background-image: url('data:image/jpeg;base64,/9j/4AAQ');">
                Background image div
            </div>
            <img src="https://cdn.example.com/logo.png" alt="logo">
        </body>
        </html>
        '''
        
        result = strip_base64_images(html)
        
        # Check that base64 images are removed
        assert "data:image/png;base64,ABCDEF" not in result
        assert "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ" not in result
        assert "data:image/jpeg;base64,/9j/4AAQ" not in result
        
        # Check that placeholders are added
        assert "[BASE64_IMAGE_REMOVED]" in result
        
        # Check that normal URLs are preserved
        assert "https://cdn.example.com/logo.png" in result
        
        # Check that text content is preserved
        assert "Data Analysis Report" in result
        assert "Some analysis text here" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])