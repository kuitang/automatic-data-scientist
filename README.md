# Automatic Data Scientist MVP

An automated data analysis system that uses OpenAI LLM agents to analyze datasets and generate HTML reports with visualizations.

## Features

- **Dual-Agent Architecture**: Architect agent plans analysis, Coder agent generates Python scripts
- **Iterative Refinement**: Up to 10 iterations to meet acceptance criteria
- **Multiple Data Formats**: Supports CSV, Excel, JSON, and Parquet files
- **Automated Visualizations**: Generates charts using matplotlib, seaborn, and plotly
- **HTML Reports**: Outputs complete HTML documents with embedded visualizations
- **Error Recovery**: Automatically fixes code errors through iteration

## Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd automatic-data-scientist
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set your OpenAI API key:
```bash
export OPENAI_API_KEY=your_api_key_here
```

### Running the Application

Start the FastAPI server:
```bash
python3 main.py
```

With custom settings:
```bash
# Set max iterations to 10
export MAX_ITERATIONS=10
python3 main.py

# Save artifacts for debugging
export ARTIFACTS_OUTPUT=/path/to/output
python3 main.py

# Both settings
export MAX_ITERATIONS=10
export ARTIFACTS_OUTPUT=/path/to/output
python3 main.py
```

The server will start on http://localhost:8000

### Usage

Send a POST request to `/analyze` with:
- `url`: URL to your dataset (CSV, Excel, JSON, or Parquet)
- `prompt`: Natural language description of the analysis you want

Quick test with Iris dataset:
```bash
curl -X POST http://localhost:8000/analyze \
  -F "url=https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv" \
  -F "prompt=Perform exploratory data analysis with visualizations" \
  -o iris_report.html
```

## Example Scripts

The `examples/` directory contains ready-to-run scripts:

- **`analyze_nvidia.py`** - Analyzes NVIDIA stock data with investment strategy comparisons
- **`simple_test.py`** - Quick test using the Iris dataset

Run an example:
```bash
# Make sure the server is running first
python main.py

# In another terminal, run an example
python examples/analyze_nvidia.py
```

## Testing

Run the test suite:
```bash
pip install pytest pytest-asyncio
pytest -v
```

Run specific tests:
```bash
pytest tests/test_executor.py -v
pytest tests/test_agents.py::TestArchitectAgent -v
```

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌────────────┐
│   FastAPI   │────▶│   Architect  │────▶│   Coder    │
│   Endpoint  │     │     Agent    │     │   Agent    │
└─────────────┘     └──────────────┘     └────────────┘
                            │                     │
                            ▼                     ▼
                    ┌──────────────┐     ┌────────────┐
                    │   Validate   │◀────│  Execute   │
                    │   Results    │     │   Python   │
                    └──────────────┘     └────────────┘
```

### Components

- **main.py**: FastAPI application with `/analyze` endpoint
- **agents/architect.py**: Plans analysis and validates results
- **agents/coder.py**: Generates and revises Python code
- **executor.py**: Safely executes generated Python scripts
- **config/prompts/**: LLM prompt templates
- **tests/**: Comprehensive test suite

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `MAX_ITERATIONS`: Maximum iteration attempts (default: 3)
- `ARTIFACTS_OUTPUT`: Directory to save artifacts/debug output

### Model Configuration

Edit `config/models.yaml` to change the OpenAI models used:
```yaml
architect_model: gpt-4-turbo-preview
coder_model: gpt-4-turbo-preview
```

### Limits

- Maximum file size: 100MB
- Execution timeout: 30 seconds per script
- Maximum iterations: 10
- Total request timeout: 5 minutes

## Supported Data Formats

- CSV (.csv)
- Excel (.xlsx, .xls)
- JSON (.json)
- Parquet (.parquet)

## Development

### Project Structure

```
automatic-data-scientist/
├── main.py                 # FastAPI application
├── agents/
│   ├── architect.py       # Architect agent
│   └── coder.py          # Coder agent
├── executor.py           # Python script executor
├── config/
│   ├── models.yaml       # Model configuration
│   └── prompts/         # Prompt templates
├── tests/               # Test suite
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

### Adding New Features

1. Create a feature branch
2. Write tests first (TDD approach)
3. Implement the feature
4. Ensure all tests pass
5. Update documentation

## Security Notes

- This is designed for local use only
- No sandboxing is implemented in the MVP
- Do not expose to public internet without proper security measures
- Generated scripts have full system access

## License

[Your License Here]

## Contributing

[Contributing guidelines]