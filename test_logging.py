#!/usr/bin/env python3
"""
Test script to verify the comprehensive logging for OpenAI interactions.
This script creates a simple test dataset and runs the analysis pipeline.
"""

import os
import sys
import logging
import asyncio
from pathlib import Path
import pandas as pd
import tempfile

# Setup detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_output.log'),
        logging.StreamHandler()
    ]
)

# Import the agents
from agents.architect import ArchitectAgent
from agents.coder import CoderAgent
from executor import PythonExecutor

async def test_logging():
    """Test the logging functionality with a simple dataset."""
    
    # Create a test dataset
    test_data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=30),
        'sales': [100 + i * 5 + (i % 7) * 10 for i in range(30)],
        'category': ['A', 'B', 'C'] * 10,
        'region': ['North', 'South', 'East', 'West'] * 7 + ['North', 'South']
    })
    
    # Save test data to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        test_path = Path(f.name)
        test_data.to_csv(test_path, index=False)
    
    work_dir = Path(tempfile.mkdtemp(prefix="test_analysis_"))
    
    try:
        print("\n" + "="*80)
        print("STARTING TEST OF LOGGING SYSTEM")
        print("="*80)
        print(f"Test data saved to: {test_path}")
        print(f"Working directory: {work_dir}")
        print("\nCheck test_output.log for detailed logging")
        print("="*80 + "\n")
        
        # Initialize agents
        architect = ArchitectAgent()
        coder = CoderAgent()
        executor = PythonExecutor(work_dir)
        
        # Test user prompt
        user_prompt = "Create a comprehensive analysis with summary statistics, trends over time, and breakdown by category and region"
        
        # Run the architect to create requirements
        print("\n📋 PHASE 1: Architect creating requirements...")
        requirements = await architect.profile_and_plan(test_path, user_prompt)
        
        print("\n📋 Requirements created:")
        print(f"  - {len(requirements['acceptance_criteria'])} acceptance criteria")
        
        # Generate initial code
        print("\n💻 PHASE 2: Coder generating initial script...")
        code = await coder.generate_code(requirements['requirements'], test_path)
        
        print(f"\n💻 Generated code: {len(code)} characters")
        
        # Execute the code
        print("\n⚙️ PHASE 3: Executing generated script...")
        execution_result = await executor.execute(code, test_path)
        
        if execution_result['success']:
            print("✅ Execution successful!")
            
            # Validate results
            print("\n🔍 PHASE 4: Architect validating results...")
            validation = await architect.validate_results(
                execution_result['output'],
                requirements['acceptance_criteria']
            )
            
            if validation['is_complete']:
                print("\n🎉 SUCCESS: All acceptance criteria met!")
            else:
                print(f"\n⚠️ Validation failed: {validation['feedback'][:100]}...")
                
                # Test revision
                print("\n🔄 PHASE 5: Testing code revision...")
                revised_code = await coder.revise_code(
                    code,
                    requirements['requirements'],
                    validation['feedback'],
                    test_path
                )
                print(f"📝 Revised code: {len(revised_code)} characters")
        else:
            print(f"❌ Execution failed: {execution_result['error']}")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if test_path.exists():
            test_path.unlink()
        if work_dir.exists():
            import shutil
            shutil.rmtree(work_dir)
        
        print("\n" + "="*80)
        print("TEST COMPLETE - Check test_output.log for detailed logs")
        print("="*80)

if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    # Run the test
    asyncio.run(test_logging())