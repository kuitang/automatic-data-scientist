# Automatic Data Scientist MVP Implementation

## Quick Summary
Build a local-only FastAPI backend that uses two OpenAI LLM agents (Architect and Coder) to automatically analyze datasets. The Architect agent examines data, plans analyses, and validates results. The Coder agent generates Python scripts. They iterate up to 10 times until the analysis meets acceptance criteria, then return an HTML report with charts and tables.

## Core Specification

### System Flow
1. User POSTs a dataset URL and analysis prompt to `/analyze`
2. System downloads file to `/tmp/analysis_{uuid}/`
3. Architect profiles the data and creates an analysis plan
4. Architect-Coder loop (max 10 iterations):
   - Architect provides requirements → Coder generates Python → Execute with 30s timeout → Architect validates
5. Returns HTML report or error

### Key Constraints
- **Local-only**: Single user, no sandboxing needed for MVP
- **File limits**: 100MB max, supports CSV/Excel/JSON/Parquet
- **Execution**: 30-second timeout, 2GB memory limit
- **Packages**: pandas, numpy, matplotlib, seaborn, scipy, sklearn, plotly only
- **Output**: HTML with embedded SVG/base64 charts to stdout

### Project Structure
```
project/
├── main.py                 # FastAPI app
├── agents/
│   ├── architect.py       # Architect agent logic
│   └── coder.py          # Coder agent logic
├── config/
│   ├── prompts/          # External prompt files
│   │   ├── architect_initial.txt
│   │   ├── architect_feedback.txt
│   │   ├── coder_initial.txt
│   │   └── coder_revision.txt
│   └── models.yaml       # OpenAI model configs
├── executor.py           # Python subprocess execution
├── requirements.txt      # Dependencies
└── tests/               # Test files
```

### API Endpoint
**POST /analyze** (URL-encoded form)
- Input: `url` (string), `prompt` (string)  
- Output: HTML document or JSON error
- Timeout: 5 minutes total

### Agent Interfaces

**Architect Output** (structured JSON):
```json
{
  "requirements": "string describing what to build",
  "acceptance_criteria": ["criterion_1", "criterion_2"],
  "is_complete": boolean,
  "feedback": "string with specific fixes needed"
}
```

**Coder Output**: Complete Python script as string that:
- Reads data from command-line argument path
- Outputs valid HTML to stdout
- Embeds all charts as SVG or base64 PNG
- No network access or file writes (except temp files for plotting)

### Implementation Priorities
1. Start with basic file download and validation
2. Implement Coder agent with simple, fixed requirements first
3. Add Architect agent and iteration loop
4. Add domain detection (financial, social science, general)
5. Implement proper cleanup and error handling

### Critical Implementation Details
- Store all files in `/tmp/analysis_{uuid}/` and clean up after completion
- Pass data file path as `--data` argument to generated Python scripts  
- Log all LLM interactions, generated code, and execution results
- Use environment variables for OpenAI API credentials
- Implement 3x retry with exponential backoff for LLM API failures
- If 10 iterations hit without success, return best attempt with warning

### Testing Approach
- Unit test file handling and prompt construction
- Integration test with mock LLM responses  
- End-to-end test with pre-recorded LLM responses for deterministic testing
- Test error recovery through at least one code correction cycle

### Success Criteria
- Analyze a simple CSV with basic prompt
- Generate valid HTML with at least one chart
- Complete within 5 minutes
- Recover from one code error via iteration
