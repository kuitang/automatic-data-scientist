"""Test suite for Pydantic models."""

import pytest
from pydantic import ValidationError
from agents.models import (
    ArchitectPlanResponse,
    ArchitectValidationResponse,
    CoderResponse
)


class TestArchitectPlanResponse:
    """Test ArchitectPlanResponse model validation."""
    
    def test_valid_plan_response(self):
        """Test creating a valid plan response."""
        response = ArchitectPlanResponse(
            requirements="Analyze dataset and create visualizations",
            acceptance_criteria=["Generate HTML output", "Include charts"],
            criteria_importance="HTML output is critical for delivery. Charts help visualize patterns.",
            is_complete=False,
            feedback=""
        )
        assert response.requirements == "Analyze dataset and create visualizations"
        assert len(response.acceptance_criteria) == 2
        assert response.criteria_importance == "HTML output is critical for delivery. Charts help visualize patterns."
        assert response.is_complete is False
        assert response.feedback == ""
    
    def test_plan_response_with_defaults(self):
        """Test plan response with default values."""
        response = ArchitectPlanResponse(
            requirements="Basic analysis",
            acceptance_criteria=["Create report"],
            criteria_importance="Report generation is the primary requirement."
        )
        assert response.is_complete is False
        assert response.feedback == ""
    
    def test_invalid_plan_response_missing_required(self):
        """Test that missing required fields raise validation error."""
        with pytest.raises(ValidationError) as exc_info:
            ArchitectPlanResponse(requirements="Test")
        assert "acceptance_criteria" in str(exc_info.value)
        
        with pytest.raises(ValidationError) as exc_info:
            ArchitectPlanResponse(acceptance_criteria=["Test"])
        assert "requirements" in str(exc_info.value)
    
    def test_plan_response_type_validation(self):
        """Test type validation for plan response fields."""
        with pytest.raises(ValidationError):
            ArchitectPlanResponse(
                requirements="Valid",
                acceptance_criteria="Should be a list",  # Wrong type
                criteria_importance="Important",
                is_complete=False
            )
        
        with pytest.raises(ValidationError):
            ArchitectPlanResponse(
                requirements="Valid",
                acceptance_criteria=["Valid"],
                criteria_importance="Important",
                is_complete="not a boolean"  # Wrong type
            )
    
    def test_max_acceptance_criteria(self):
        """Test that acceptance criteria is limited to 5 items."""
        with pytest.raises(ValidationError) as exc_info:
            ArchitectPlanResponse(
                requirements="Analysis",
                acceptance_criteria=["C1", "C2", "C3", "C4", "C5", "C6"],  # Too many
                criteria_importance="All important"
            )
        assert "at most 5 items" in str(exc_info.value) or "too_long" in str(exc_info.value)


class TestArchitectValidationResponse:
    """Test ArchitectValidationResponse model validation."""
    
    def test_valid_validation_response_complete(self):
        """Test validation response when criteria are met."""
        response = ArchitectValidationResponse(
            criteria_evaluation="All criteria met excellently",
            grade="A",
            grade_justification="Exceptional analysis with deep insights",
            is_complete=True,
            feedback="Minor improvements could include additional visualizations"
        )
        assert response.is_complete is True
        assert response.grade == "A"
        assert "Exceptional" in response.grade_justification
    
    def test_valid_validation_response_incomplete(self):
        """Test validation response when criteria are not met."""
        response = ArchitectValidationResponse(
            criteria_evaluation="Basic statistics present but lacks depth",
            grade="C",
            grade_justification="Adequate but missing key insights",
            is_complete=False,
            feedback="Need deeper statistical analysis and correlation matrix"
        )
        assert response.is_complete is False
        assert response.grade == "C"
        assert "correlation matrix" in response.feedback
    
    def test_invalid_validation_response_missing_required(self):
        """Test that missing required fields raise validation error."""
        with pytest.raises(ValidationError) as exc_info:
            ArchitectValidationResponse(
                is_complete=True,
                grade="B"
            )
        # Multiple fields missing
        error_str = str(exc_info.value)
        assert "criteria_evaluation" in error_str or "feedback" in error_str
    
    def test_validation_response_type_validation(self):
        """Test type validation for validation response fields."""
        with pytest.raises(ValidationError):
            ArchitectValidationResponse(
                criteria_evaluation="Valid",
                grade="A",
                grade_justification="Valid",
                is_complete="not a boolean",
                feedback="Valid feedback"
            )
    
    def test_grade_validation(self):
        """Test that only valid letter grades are accepted."""
        # Valid grade
        response = ArchitectValidationResponse(
            criteria_evaluation="Good work",
            grade="B+",
            grade_justification="Solid analysis",
            is_complete=True,
            feedback="Good job"
        )
        assert response.grade == "B+"
        
        # Invalid grade
        with pytest.raises(ValidationError) as exc_info:
            ArchitectValidationResponse(
                criteria_evaluation="Poor",
                grade="E",  # Invalid grade
                grade_justification="Bad",
                is_complete=False,
                feedback="Needs work"
            )
        assert "Grade must be one of" in str(exc_info.value)
    
    def test_grade_threshold_validation(self):
        """Test that is_complete matches grade threshold."""
        # B- should be complete
        response = ArchitectValidationResponse(
            criteria_evaluation="Adequate",
            grade="B-",
            grade_justification="Meets minimum standards",
            is_complete=True,
            feedback="Acceptable"
        )
        assert response.is_complete is True
        
        # C+ should not be complete
        with pytest.raises(ValidationError) as exc_info:
            ArchitectValidationResponse(
                criteria_evaluation="Below standards",
                grade="C+",
                grade_justification="Not quite there",
                is_complete=True,  # Wrong - should be False for C+
                feedback="Needs improvement"
            )
        assert "is_complete must be False for grade C+" in str(exc_info.value)


class TestCoderResponse:
    """Test CoderResponse model validation."""
    
    def test_valid_coder_response_with_explanation(self):
        """Test creating a valid coder response with explanation."""
        code = """import pandas as pd
def analyze(data):
    return data.describe()"""
        
        response = CoderResponse(
            code=code,
            explanation="Simple statistical analysis function"
        )
        assert response.code == code
        assert response.explanation == "Simple statistical analysis function"
    
    def test_valid_coder_response_without_explanation(self):
        """Test coder response with default explanation."""
        code = "print('Hello, World!')"
        response = CoderResponse(code=code)
        assert response.code == code
        assert response.explanation == ""
    
    def test_invalid_coder_response_missing_code(self):
        """Test that missing code field raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            CoderResponse(explanation="Some explanation")
        assert "code" in str(exc_info.value)
    
    def test_coder_response_type_validation(self):
        """Test type validation for coder response fields."""
        with pytest.raises(ValidationError):
            CoderResponse(
                code=12345,  # Should be string
                explanation="Valid"
            )
        
        with pytest.raises(ValidationError):
            CoderResponse(
                code="Valid code",
                explanation={"not": "a string"}  # Should be string
            )
    
    def test_coder_response_with_multiline_code(self):
        """Test coder response with realistic multiline code."""
        code = """#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    args = parser.parse_args()
    
    df = pd.read_csv(args.data)
    print(df.describe().to_html())

if __name__ == '__main__':
    main()
"""
        response = CoderResponse(code=code)
        assert "argparse" in response.code
        assert "pd.read_csv" in response.code