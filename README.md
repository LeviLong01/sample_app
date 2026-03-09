# Sample App - Strands Agent with MLflow Evaluation

This sample application demonstrates the issue described in [mlflow/mlflow#21453](https://github.com/mlflow/mlflow/pull/21453) by allowing you to toggle between the current MLflow release and the patched branch.

## The Issue

MLflow's `ToolCallCorrectness` and `ToolCallEfficiency` scorers fail when evaluating Strands SDK traces due to incorrect tool input formatting in the autolog processor. The Strands autolog appends tool inputs to a list instead of storing them as a dict, causing Pydantic validation errors.

## Reproducing the Issue

This app lets you compare behavior between:
- **Current MLflow release**: Shows the validation error when running tool call evaluation
- **Patched branch** (`genai-otel-semantic-conventions`): Demonstrates the fix working correctly

## Setup

1. Install dependencies with uv:
```bash
uv sync
```

2. Create `.env` file from template:
```bash
cp .env.example .env
```

3. Add your Anthropic API key to `.env`:
```
ANTHROPIC_API_KEY=your_actual_key_here
```

## Run Evaluation

```bash
uv run python strands_evaluate.py
```

This will:
- Generate a trace using the Strands agent with calculator and current_time tools
- Evaluate the trace with ToolCallCorrectness, ToolCallEfficiency, and RelevanceToQuery scorers
- Display evaluation scores (1.0 = perfect)

## What's Being Tested

Test case: "What is the current time and what is the sum of it's hour and minutes?"

Expected behavior:
- Agent calls `current_time` tool
- Agent calls `calculator` tool to sum hour + minutes
- Both tools are called correctly with appropriate arguments

## Evaluation Metrics

- **tool_call_correctness**: Are the right tools called with correct arguments?
- **tool_call_efficiency**: Are tool calls redundant or duplicated?
- **relevance_to_query**: Does the response answer the user's question?
