"""Pydantic models for structured OpenAI API responses."""

from pydantic import BaseModel, Field, validator
from typing import List, Dict


class ArchitectPlanResponse(BaseModel):
    """Response model for Architect's initial planning phase."""
    requirements: str = Field(
        description="Detailed analysis requirements for the Python script"
    )
    acceptance_criteria: List[str] = Field(
        description="List of measurable criteria (max 5, focused on data substance and insights)",
        max_length=5
    )
    criteria_importance: str = Field(
        description="Explanation of why each criterion matters and which are most critical"
    )
    is_complete: bool = Field(
        default=False,
        description="Always False for initial planning"
    )
    feedback: str = Field(
        default="",
        description="Empty for initial planning"
    )


class ArchitectValidationResponse(BaseModel):
    """Response model for Architect's validation phase."""
    criteria_evaluation: str = Field(
        description="Detailed evaluation of how well each criterion was met"
    )
    grade: str = Field(
        description="Letter grade (A+, A, A-, B+, B, B-, C+, C, C-, D, F)"
    )
    grade_justification: str = Field(
        description="Explanation of why this grade was assigned"
    )
    is_complete: bool = Field(
        description="True if grade is B- or higher"
    )
    feedback: str = Field(
        description="Specific feedback on what needs to be fixed or improved for a better grade"
    )
    
    @validator('grade')
    def validate_grade(cls, v):
        """Ensure grade is valid."""
        valid_grades = ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D', 'F']
        if v not in valid_grades:
            raise ValueError(f"Grade must be one of {valid_grades}")
        return v
    
    @validator('is_complete', always=True)
    def check_grade_threshold(cls, v, values):
        """Ensure is_complete matches grade threshold (B- or higher)."""
        if 'grade' in values:
            passing_grades = ['A+', 'A', 'A-', 'B+', 'B', 'B-']
            should_be_complete = values['grade'] in passing_grades
            if v != should_be_complete:
                raise ValueError(f"is_complete must be {should_be_complete} for grade {values['grade']}")
        return v


class CoderResponse(BaseModel):
    """Response model for Coder agent code generation."""
    code: str = Field(
        description="Complete Python script that implements the requirements"
    )
    explanation: str = Field(
        default="",
        description="Brief explanation of the approach taken"
    )