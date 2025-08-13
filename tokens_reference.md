# Token and Temperature Settings Reference

## Current Settings in Codebase

- **Temperature**: 0.3
- **Max Output Tokens**: 
  - Architect (planning): 2000
  - Architect (validation): 1000
  - Coder (generation/revision): 4000

## GPT-5 Specifications (2025)

### Token Limits
- **Maximum Input Tokens**: 272,000
- **Maximum Output Tokens**: 128,000
- **Total Context Window**: 400,000 tokens
- **Note**: The output tokens include "invisible reasoning tokens" when using GPT-5's reasoning capabilities

### Model Variants
- **gpt-5**: $1.25/1M input tokens, $10/1M output tokens
- **gpt-5-mini**: $0.25/1M input tokens, $2/1M output tokens  
- **gpt-5-nano**: $0.05/1M input tokens, $0.40/1M output tokens

### New Features
- Supports `reasoning_effort` parameter (minimal, low, medium, high)
- Supports `verbosity` parameter
- 100% reliability with structured outputs (JSON Schema)
- Built-in tools: web search, file search, image generation

## Temperature Best Practices

### Why Temperature = 0.3?

1. **Optimal for Code Generation**
   - Provides balance between deterministic output and coherent code
   - Research shows 0.2-0.3 range ideal for technical tasks requiring precision
   - Low temperature ensures consistent, reproducible outputs

2. **Structured Output Benefits**
   - Since agents generate Python scripts and use Pydantic models for validation
   - Low temperature ensures reliable structured responses
   - Reduces unexpected or creative variations that could break parsing

3. **Research Findings**
   - For fact-based tasks like data analysis, temperatures close to 0 are recommended
   - Studies show temperature 0.1-0.3 with high top_p (0.9) yields best results for code
   - 0.3 setting allows slight variation while maintaining high reliability

### Temperature Guidelines

- **0.0-0.2**: Maximum determinism, fact-based Q&A, technical documentation
- **0.2-0.4**: Code generation, structured data, analysis tasks (current setting: 0.3)
- **0.5-0.7**: Balanced creativity and coherence
- **0.8-1.0**: Creative writing, brainstorming
- **1.0-2.0**: Maximum randomness and creativity

**Important**: Don't use both temperature and top_p simultaneously. OpenAI recommends using only one at a time.

## Max Output Tokens Analysis

### Current Usage vs. Capacity

| Agent Task | Current Setting | GPT-5 Capacity | Usage % |
|------------|----------------|----------------|---------|
| Architect Planning | 2000 | 128,000 | 1.6% |
| Architect Validation | 1000 | 128,000 | 0.8% |
| Coder Generation | 4000 | 128,000 | 3.1% |

### Considerations

1. **Cost Management**
   - Current settings keep costs reasonable
   - GPT-5 output tokens are expensive ($10/1M tokens)
   - 4000 tokens ≈ $0.04 per code generation

2. **Underutilization**
   - Using only ~3% of GPT-5's output capacity
   - Complex data analysis scripts could benefit from more tokens
   - Consider 8000-16000 for complex analyses

3. **Performance Impact**
   - Structured JSON outputs reduce processing time by up to 60%
   - Boost response relevancy by 35%
   - Grammar-based decoding ensures syntactic correctness

## Recommendations for GPT-5 Optimization

1. **Keep Temperature at 0.3**
   - Already optimal for code generation tasks
   - Provides right balance for structured outputs

2. **Consider Increasing Max Output Tokens**
   - For complex analyses: 8000-12000 tokens
   - For simple scripts: keep at 4000
   - Monitor costs vs. completeness tradeoff

3. **Utilize New GPT-5 Features**
   - Add `reasoning_effort` parameter for complex logic
   - Use native structured output support (100% reliability)
   - Consider parallel tool calling for efficiency

4. **Implement Dynamic Token Allocation**
   - Simple tasks: 2000-4000 tokens
   - Medium complexity: 6000-10000 tokens
   - Complex multi-step analyses: 12000-20000 tokens

## Best Practices Summary

- **Temperature 0.3**: Optimal for code generation and structured outputs
- **Structured Outputs**: Use Pydantic models with `response_format` for reliability
- **Token Management**: Balance between cost and completeness
- **Validation**: Always validate structured outputs with JSON Schema
- **Error Handling**: Account for token limit errors (272k input limit)