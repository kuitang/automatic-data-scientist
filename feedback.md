# Architect-Coder Convergence Issues and Solutions

## Problem Analysis

The system gets stuck in a feedback loop where the architect keeps giving the same feedback but the coder doesn't make meaningful changes. This happens because of a fundamental information asymmetry.

### Core Issues

1. **Information Asymmetry**
   - The coder cannot see the HTML output their code produced
   - The architect sees the output and provides feedback
   - The coder must guess what went wrong based only on abstract feedback text

2. **Context Loss in Revision**
   - When revising, the coder receives:
     - Previous code
     - Grade and feedback text
     - Original requirements
   - But the coder DOESN'T receive:
     - What the output actually looked like
     - Which analyses were successfully completed
     - Specific examples of what's missing

3. **Vague Feedback Loop**
   - Architect provides high-level feedback like "needs more statistical depth"
   - Coder doesn't know if they already calculated statistics (but displayed them poorly) or if they're completely missing
   - The same feedback repeats because the coder can't understand the specific gap

4. **No Convergence Detection**
   - System doesn't detect when it's stuck (producing similar code/feedback repeatedly)
   - No mechanism to break out of repetitive cycles

## Solution: Quick Fix Implementation

### Phase 1: Give Coder Visibility (Quick Fix)

The most impactful immediate change is to show the coder what their previous code actually produced:

1. **Extract and filter the HTML output** (remove images like we do for architect)
2. **Pass filtered output to coder during revision**
3. **Update revision prompt to include this context**

This allows the coder to:
- See exactly what analyses were performed
- Understand what's missing or incorrect
- Make targeted fixes instead of guessing

### Implementation Steps

1. **Move image filtering to shared location** (agents/utils.py)
   - Currently `strip_base64_images()` is only in architect.py
   - Both agents need this functionality

2. **Update coder.revise_code() signature**
   - Add `previous_output: Optional[str]` parameter
   - Filter images before including in prompt

3. **Update main.py**
   - Pass filtered output to coder.revise_code()
   - Use same filtering as architect

4. **Update revision prompt**
   - Include previous output in context
   - Help coder understand what was produced vs. expected

## Future Improvements (Phase 2)

### Structured Feedback
- Make architect provide specific lists:
  - Sections present in output
  - Sections missing from output
  - Specific calculations/methods needed

### Convergence Detection
- Track feedback similarity across iterations
- Detect when system is stuck
- Trigger more aggressive interventions

### Example-Based Feedback
- Architect quotes specific text/numbers from output
- Provides concrete examples of fixes needed
- Suggests specific code snippets

## Testing Strategy

### Test Coverage Required
1. **Image filtering works correctly**
   - Base64 images removed
   - SVG content removed
   - HTML structure preserved

2. **Filtered output passed to coder**
   - Previous output included in revision
   - Images properly filtered
   - No tokens wasted on image data

3. **Convergence improvement**
   - Coder makes different changes when given output context
   - Feedback loop breaks with better information

## Success Metrics

- Fewer iterations needed to reach acceptance
- Less repetitive feedback across iterations
- Coder produces meaningfully different code each iteration
- System reaches B- grade more consistently