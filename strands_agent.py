"""
Simple Strands agent with calculator tool
"""
import os
from strands import Agent
from strands.agent import AgentResult
from strands.models.anthropic import AnthropicModel
from strands_tools import calculator, current_time
from dotenv import load_dotenv

load_dotenv()

async def run_calculation(inputs: dict) -> AgentResult:
    """Run a calculation using Strands agent with Anthropic"""
    messages = inputs.get("messages")
    if messages is None:
        question = inputs.get("question", "")
        messages = question
    
    model = AnthropicModel(
        client_args={
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
        },
        max_tokens=1024,
        model_id=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929"),
    )
    
    agent = Agent(model=model, tools=[calculator, current_time])
    response = await agent.invoke_async(messages)
    return response

if __name__ == "__main__":
    test_question = "What is 25 + 17?"
    result = run_calculation({"question": test_question})
    print(f"Question: {test_question}")
    print(f"Answer: {result}")
