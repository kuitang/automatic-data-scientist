# Frontend Log Streaming Implementation Plan

## Architecture Decision: SSE over WebSockets

Use Server-Sent Events (SSE) for log streaming - unidirectional, built-in reconnection, simpler than WebSockets.

## Session Isolation

```
POST /analyze → Returns {analysis_id: uuid, stream_url: string}
GET /stream/{analysis_id} → SSE stream for that session only
```

## Log Messages to Send

Only send architect evaluation progress:
- Iteration start (1/10, 2/10, etc.)
- Data profiling status
- Requirements generated
- Acceptance criteria list
- Validation result (pass/fail)
- Feedback for failed iterations
- Final success/failure

## Message Structure

```python
# Architect-focused log types
{
    "type": "iteration_start",
    "iteration": 1,
    "total": 10
}

{
    "type": "requirements",
    "text": "Generate visualizations for..."
}

{
    "type": "criteria",
    "items": ["Chart showing X", "Table with Y", "..."]
}

{
    "type": "validation",
    "passed": false,
    "feedback": "Missing correlation matrix..."
}

{
    "type": "complete",
    "success": true,
    "iterations_used": 3
}
```

## Frontend Display

Simple progress log showing:
- Iteration counter
- Current phase (profiling → requirements → executing → validating)
- Pass/fail for each iteration with architect feedback
- Collapsible criteria checklist

## Dependencies

Backend: `sse-starlette==1.8.2`