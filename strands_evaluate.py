"""
Multi evaluation: generate traces for 10 test cases with tool calls, then evaluate them
"""
import os
import time
import uuid
import mlflow
import pandas as pd
from strands_agent import run_calculation
from mlflow.genai.scorers import ToolCallCorrectness, ToolCallEfficiency, RelevanceToQuery
from dotenv import load_dotenv

load_dotenv()

TEST_CASES = [
    "What is the current time and what is the sum of it's hour and minutes?",
]

async def multi_evaluation():
    """Generate traces for 10 test cases, then evaluate them"""
    mlflow.set_tracking_uri("./mlruns")
    mlflow.strands.autolog()
    
    experiment_name = "sample_app_evaluation"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="strands_multi_eval_run") as run:
        run_id = run.info.run_id
        print(f"🔄 Step 1: Generating traces for {len(TEST_CASES)} test cases...")

        traces_list = []
        expectations_list = []
        
        for i, question in enumerate(TEST_CASES, 1):
            print(f"   [{i}/{len(TEST_CASES)}] Processing: {question}")
            
            await run_calculation({"question": question})
            
            traces = mlflow.search_traces(run_id=run_id, order_by=["timestamp DESC"])
            trace = mlflow.get_trace(traces.iloc[0].trace_id)
            
            traces_list.append(trace)
            expectations_list.append({
                "expected_tool_calls": [
                    {"name": "calculator"},
                    {"name": "current_time"}
                ]
            })

        print("🔄 Step 2: Evaluating all traces...")
        
        trace_data = pd.DataFrame({
            "trace": traces_list,
            "expectations": expectations_list
        })
        
        anthropic_model = os.getenv("ANTHROPIC_MODEL")
        tool_correctness = ToolCallCorrectness(model=f"anthropic:/{anthropic_model}")
        tool_efficiency = ToolCallEfficiency(model=f"anthropic:/{anthropic_model}")
        relevance_judge = RelevanceToQuery(model=f"anthropic:/{anthropic_model}")
        
        results = mlflow.genai.evaluate(
            data=trace_data,
            scorers=[tool_correctness, tool_efficiency, relevance_judge]
        )
        
        print("✅ Evaluation completed!")
        print(f"   Check results in MLflow UI: http://localhost:5001")
        
        if hasattr(results, 'metrics'):
            print("\n📊 Results:")
            for metric, value in results.metrics.items():
                print(f"   {metric}: {value}")
    
    return results

if __name__ == "__main__":
    import asyncio
    print("Starting Strands trace evaluation...")
    asyncio.run(multi_evaluation())
