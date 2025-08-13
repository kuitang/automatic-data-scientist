import pytest
import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Actual implementation
def strip_base64_images(html_content: str) -> str:
    """Strip base64-encoded images from HTML and replace with placeholders.
    
    This helps reduce token usage when sending HTML to OpenAI, since the
    model cannot process images anyway.
    """
    # Pattern to match base64 image data URIs in img src attributes
    # Matches: src="data:image/[type];base64,[data]" or src='...'
    img_base64_pattern = r'(<img[^>]*\s+src=[\"\'])data:image/[^;]+;base64,[^\"\']*([\"\'][^>]*>)'
    
    # Replace with placeholder keeping the img tag structure
    html_content = re.sub(
        img_base64_pattern,
        r'\1[BASE64_IMAGE_REMOVED]\2',
        html_content,
        flags=re.IGNORECASE | re.DOTALL
    )
    
    # Pattern to match base64 data in style attributes (background-image)
    # Matches: url(data:image/[type];base64,[data])
    style_base64_pattern = r'url\(["\']?data:image/[^;]+;base64,[^)"\']*([\"\'])?\)'
    
    # Replace with placeholder
    html_content = re.sub(
        style_base64_pattern,
        'url([BASE64_IMAGE_REMOVED])',
        html_content,
        flags=re.IGNORECASE | re.DOTALL
    )
    
    # Pattern to match inline SVG elements with base64 data
    # This is less common but can occur
    svg_base64_pattern = r'<svg[^>]*>[\s\S]*?data:image/[^;]+;base64,[^"\']*([\"\'])?[\s\S]*?</svg>'
    
    # Replace entire SVG if it contains base64
    html_content = re.sub(
        svg_base64_pattern,
        '<svg>[SVG_WITH_BASE64_REMOVED]</svg>',
        html_content,
        flags=re.IGNORECASE | re.DOTALL
    )
    
    # Also handle object or embed tags with base64 data
    object_base64_pattern = r'(<(?:object|embed)[^>]*\s+(?:src|data)=[\"\'])data:image/[^;]+;base64,[^\"\']*([\"\'][^>]*>)'
    
    html_content = re.sub(
        object_base64_pattern,
        r'\1[BASE64_IMAGE_REMOVED]\2',
        html_content,
        flags=re.IGNORECASE | re.DOTALL
    )
    
    return html_content


class TestBase64ImageFilter:
    """Test cases for stripping base64-encoded images from HTML."""
    
    def test_simple_img_tag_with_base64(self):
        """Test filtering a simple img tag with base64 data."""
        html_input = '<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==" alt="test">'
        expected = '<img src="[BASE64_IMAGE_REMOVED]" alt="test">'
        result = strip_base64_images(html_input)
        assert result == expected, f"Expected: {expected}\nGot: {result}"
    
    def test_img_tag_with_single_quotes(self):
        """Test filtering img tag with single quotes."""
        html_input = "<img src='data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCwAA8A/9k=' alt='test'>"
        expected = "<img src='[BASE64_IMAGE_REMOVED]' alt='test'>"
        result = strip_base64_images(html_input)
        assert result == expected, f"Expected: {expected}\nGot: {result}"
    
    def test_multiple_img_tags(self):
        """Test filtering multiple img tags with base64 data."""
        html_input = '''
        <div>
            <img src="data:image/png;base64,ABC123DEF456" class="chart1">
            <p>Some text between images</p>
            <img src="data:image/jpeg;base64,XYZ789GHI012" class="chart2">
        </div>
        '''
        expected = '''
        <div>
            <img src="[BASE64_IMAGE_REMOVED]" class="chart1">
            <p>Some text between images</p>
            <img src="[BASE64_IMAGE_REMOVED]" class="chart2">
        </div>
        '''
        result = strip_base64_images(html_input)
        assert result == expected, f"Expected: {expected}\nGot: {result}"
    
    def test_css_background_image_base64(self):
        """Test filtering base64 in CSS background-image property."""
        html_input = '<div style="background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==); width: 100px;">'
        expected = '<div style="background-image: url([BASE64_IMAGE_REMOVED]); width: 100px;">'
        result = strip_base64_images(html_input)
        assert result == expected, f"Expected: {expected}\nGot: {result}"
    
    def test_css_background_with_quotes(self):
        """Test filtering base64 in CSS with quotes."""
        html_input = '''<div style="background-image: url('data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD');">Content</div>'''
        expected = '''<div style="background-image: url([BASE64_IMAGE_REMOVED]);">Content</div>'''
        result = strip_base64_images(html_input)
        assert result == expected, f"Expected: {expected}\nGot: {result}"
    
    def test_inline_svg_with_base64(self):
        """Test filtering SVG elements containing base64 data."""
        html_input = '''<svg width="100" height="100">
            <image href="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==" />
        </svg>'''
        expected = '<svg>[SVG_WITH_BASE64_REMOVED]</svg>'
        result = strip_base64_images(html_input).strip()
        assert result == expected, f"Expected: {expected}\nGot: {result}"
    
    def test_object_tag_with_base64(self):
        """Test filtering object tag with base64 data."""
        html_input = '<object data="data:image/png;base64,ABC123" type="image/png"></object>'
        expected = '<object data="[BASE64_IMAGE_REMOVED]" type="image/png"></object>'
        result = strip_base64_images(html_input)
        assert result == expected, f"Expected: {expected}\nGot: {result}"
    
    def test_embed_tag_with_base64(self):
        """Test filtering embed tag with base64 data."""
        html_input = '<embed src="data:image/png;base64,XYZ789" type="image/png">'
        expected = '<embed src="[BASE64_IMAGE_REMOVED]" type="image/png">'
        result = strip_base64_images(html_input)
        assert result == expected, f"Expected: {expected}\nGot: {result}"
    
    def test_preserve_non_base64_images(self):
        """Test that normal image URLs are not affected."""
        html_input = '''
        <img src="https://example.com/image.png" alt="external">
        <img src="/static/chart.jpg" alt="local">
        <img src="relative/path/image.gif" alt="relative">
        '''
        expected = html_input
        result = strip_base64_images(html_input)
        assert result == expected, f"Non-base64 images should not be modified\nExpected: {expected}\nGot: {result}"
    
    def test_mixed_content(self):
        """Test filtering in HTML with both base64 and normal images."""
        html_input = '''
        <html>
        <body>
            <img src="https://example.com/logo.png" alt="logo">
            <img src="data:image/png;base64,ABCDEFGHIJKLMNOP" alt="chart">
            <div style="background-image: url(data:image/jpeg;base64,QRSTUVWXYZ);">
                <img src="/static/icon.svg" alt="icon">
            </div>
        </body>
        </html>
        '''
        expected = '''
        <html>
        <body>
            <img src="https://example.com/logo.png" alt="logo">
            <img src="[BASE64_IMAGE_REMOVED]" alt="chart">
            <div style="background-image: url([BASE64_IMAGE_REMOVED]);">
                <img src="/static/icon.svg" alt="icon">
            </div>
        </body>
        </html>
        '''
        result = strip_base64_images(html_input)
        assert result == expected, f"Expected: {expected}\nGot: {result}"
    
    def test_case_insensitive_matching(self):
        """Test that filtering works regardless of case."""
        html_input = '''
        <IMG SRC="DATA:IMAGE/PNG;BASE64,ABC123" ALT="test1">
        <img Src="Data:Image/Jpeg;Base64,DEF456" Alt="test2">
        '''
        expected = '''
        <IMG SRC="[BASE64_IMAGE_REMOVED]" ALT="test1">
        <img Src="[BASE64_IMAGE_REMOVED]" Alt="test2">
        '''
        result = strip_base64_images(html_input)
        assert result == expected, f"Expected: {expected}\nGot: {result}"
    
    def test_malformed_base64_still_filtered(self):
        """Test that even malformed base64 data is filtered."""
        html_input = '<img src="data:image/png;base64,!!!INVALID_BASE64!!!" alt="bad">'
        expected = '<img src="[BASE64_IMAGE_REMOVED]" alt="bad">'
        result = strip_base64_images(html_input)
        assert result == expected, f"Expected: {expected}\nGot: {result}"
    
    def test_empty_html(self):
        """Test that empty HTML is handled correctly."""
        html_input = ""
        expected = ""
        result = strip_base64_images(html_input)
        assert result == expected, "Empty HTML should remain empty"
    
    def test_no_images(self):
        """Test HTML with no images at all."""
        html_input = "<div><p>Just text content</p><span>No images here</span></div>"
        expected = html_input
        result = strip_base64_images(html_input)
        assert result == expected, "HTML without images should not be modified"


if __name__ == "__main__":
    # Run all tests
    test_instance = TestBase64ImageFilter()
    test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
    
    print("Running base64 filter tests with placeholder implementation...")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_method in test_methods:
        try:
            getattr(test_instance, test_method)()
            print(f"✓ {test_method} - PASSED")
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_method} - FAILED")
            print(f"  {str(e)}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_method} - ERROR: {str(e)}")
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(test_methods)} tests")
    
    if failed > 0:
        print("\nAs expected, tests are failing with the placeholder implementation.")
        print("Next step: Add the actual implementation and verify tests pass.")