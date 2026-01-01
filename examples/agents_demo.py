"""
Example Agents demonstrating the AI Agent Framework capabilities.

This module contains two reference agents:
1. Research Agent - Performs multi-step research workflows
2. Data Processing Agent - Handles data transformation pipelines
"""

import time
import random
from typing import Any, Dict, List

# Import from our framework
from framework import (
    Agent,
    Flow,
    FunctionTask,
    ToolTask,
    LLMTask,
    ConditionalTask,
    tool,
    tool_registry,
    LogLevel,
    setup_logging
)


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

@tool(
    name="web_search",
    description="Search the web for information",
    parameters={"query": {"type": "string", "description": "Search query"}},
    tags=["search", "web"]
)
def web_search(query: str) -> Dict[str, Any]:
    """Simulated web search tool."""
    # Simulate API call delay
    time.sleep(0.1)
    return {
        "query": query,
        "results": [
            {"title": f"Result 1 for '{query}'", "snippet": "This is a sample result..."},
            {"title": f"Result 2 for '{query}'", "snippet": "Another relevant result..."},
            {"title": f"Result 3 for '{query}'", "snippet": "More information here..."},
        ],
        "total_results": 3
    }


@tool(
    name="summarize_text",
    description="Summarize a piece of text",
    parameters={"text": {"type": "string", "description": "Text to summarize"}},
    tags=["nlp", "summarization"]
)
def summarize_text(text: str) -> str:
    """Simulated text summarization tool."""
    # In production, this would call an LLM
    time.sleep(0.1)
    words = text.split()
    if len(words) > 20:
        return " ".join(words[:20]) + "... [summarized]"
    return text + " [summarized]"


@tool(
    name="extract_entities",
    description="Extract named entities from text",
    parameters={"text": {"type": "string", "description": "Text to analyze"}},
    tags=["nlp", "ner"]
)
def extract_entities(text: str) -> Dict[str, List[str]]:
    """Simulated entity extraction tool."""
    time.sleep(0.05)
    return {
        "persons": ["John Doe", "Jane Smith"],
        "organizations": ["Acme Corp", "Tech Inc"],
        "locations": ["New York", "San Francisco"],
        "dates": ["2024", "January"]
    }


@tool(
    name="sentiment_analysis",
    description="Analyze sentiment of text",
    parameters={"text": {"type": "string", "description": "Text to analyze"}},
    tags=["nlp", "sentiment"]
)
def sentiment_analysis(text: str) -> Dict[str, Any]:
    """Simulated sentiment analysis tool."""
    time.sleep(0.05)
    sentiment_score = random.uniform(-1, 1)
    return {
        "score": sentiment_score,
        "label": "positive" if sentiment_score > 0.2 else "negative" if sentiment_score < -0.2 else "neutral",
        "confidence": random.uniform(0.7, 0.99)
    }


@tool(
    name="data_transform",
    description="Transform data according to schema",
    parameters={
        "data": {"type": "object", "description": "Input data"},
        "operation": {"type": "string", "description": "Transform operation"}
    },
    tags=["data", "transform"]
)
def data_transform(data: Dict, operation: str = "normalize") -> Dict[str, Any]:
    """Simulated data transformation tool."""
    time.sleep(0.05)
    return {
        "original": data,
        "transformed": {k: str(v).upper() for k, v in data.items()},
        "operation": operation
    }


@tool(
    name="validate_data",
    description="Validate data against schema",
    parameters={"data": {"type": "object", "description": "Data to validate"}},
    tags=["data", "validation"]
)
def validate_data(data: Dict) -> Dict[str, Any]:
    """Simulated data validation tool."""
    time.sleep(0.02)
    return {
        "valid": True,
        "errors": [],
        "warnings": ["Field 'optional_field' not present"],
        "data": data
    }


# =============================================================================
# REFERENCE AGENT 1: RESEARCH AGENT
# =============================================================================

def create_research_agent() -> Agent:
    """
    Create a Research Agent that performs multi-step research workflows.
    
    Flow:
    1. Search for information
    2. Extract entities (parallel with summarization)
    3. Summarize findings
    4. Analyze sentiment
    5. Generate report
    """
    agent = Agent(
        name="ResearchAgent",
        description="An agent that performs automated research tasks"
    )
    
    # Create the research flow
    flow = agent.create_flow(
        name="research_workflow",
        description="Multi-step research pipeline",
        max_workers=4
    )
    
    # Task 1: Search for information
    search_task = FunctionTask(
        name="search",
        func=lambda ctx: tool_registry.execute("web_search", {"query": ctx.get("query", "AI agents")}),
        description="Search the web for relevant information",
        max_retries=2
    )
    
    # Task 2: Extract entities from search results
    extract_task = FunctionTask(
        name="extract_entities",
        func=lambda ctx: tool_registry.execute(
            "extract_entities", 
            {"text": str(ctx.get("search_result", {}).get("results", []))}
        ),
        description="Extract named entities from search results"
    )
    
    # Task 3: Summarize the findings
    summarize_task = FunctionTask(
        name="summarize",
        func=lambda ctx: tool_registry.execute(
            "summarize_text",
            {"text": str(ctx.get("search_result", {}))}
        ),
        description="Summarize the search results"
    )
    
    # Task 4: Analyze sentiment
    sentiment_task = FunctionTask(
        name="analyze_sentiment",
        func=lambda ctx: tool_registry.execute(
            "sentiment_analysis",
            {"text": ctx.get("summarize_result", "")}
        ),
        description="Analyze sentiment of summarized content"
    )
    
    # Task 5: Generate final report
    report_task = FunctionTask(
        name="generate_report",
        func=lambda ctx: {
            "query": ctx.get("query"),
            "search_results": ctx.get("search_result", {}).get("total_results", 0),
            "entities": ctx.get("extract_entities_result", {}),
            "summary": ctx.get("summarize_result", ""),
            "sentiment": ctx.get("analyze_sentiment_result", {}),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        description="Generate final research report"
    )
    
    # Add tasks to flow
    flow.add_tasks(search_task, extract_task, summarize_task, sentiment_task, report_task)
    
    # Define dependencies (DAG structure):
    #     search
    #    /      \
    # extract  summarize
    #            \
    #          sentiment
    #              \
    #            report (depends on all)
    
    flow.add_dependency("extract_entities", "search")
    flow.add_dependency("summarize", "search")
    flow.add_dependency("analyze_sentiment", "summarize")
    flow.add_dependency("generate_report", "extract_entities")
    flow.add_dependency("generate_report", "analyze_sentiment")
    
    return agent


# =============================================================================
# REFERENCE AGENT 2: DATA PROCESSING AGENT
# =============================================================================

def create_data_processing_agent() -> Agent:
    """
    Create a Data Processing Agent that handles ETL-like workflows.
    
    Flow:
    1. Validate input data
    2. Transform data
    3. Process based on condition
    4. Output results
    """
    agent = Agent(
        name="DataProcessingAgent",
        description="An agent that processes and transforms data"
    )
    
    # Create the processing flow
    flow = agent.create_flow(
        name="data_pipeline",
        description="Data processing pipeline",
        max_workers=2
    )
    
    # Task 1: Validate input
    validate_task = FunctionTask(
        name="validate_input",
        func=lambda ctx: tool_registry.execute(
            "validate_data",
            {"data": ctx.get("input_data", {})}
        ),
        description="Validate input data",
        max_retries=1
    )
    
    # Task 2: Transform data
    transform_task = FunctionTask(
        name="transform_data",
        func=lambda ctx: tool_registry.execute(
            "data_transform",
            {
                "data": ctx.get("validate_input_result", {}).get("data", {}),
                "operation": ctx.get("transform_operation", "normalize")
            }
        ),
        description="Transform validated data"
    )
    
    # Task 3: Conditional processing
    conditional_task = ConditionalTask(
        name="check_size",
        condition=lambda ctx: len(ctx.get("transform_data_result", {}).get("transformed", {})) > 2,
        true_task="heavy_processing",
        false_task="light_processing",
        description="Check if data needs heavy or light processing"
    )
    
    # Task 4a: Heavy processing (for large data)
    heavy_task = FunctionTask(
        name="heavy_processing",
        func=lambda ctx: {
            "type": "heavy",
            "data": ctx.get("transform_data_result", {}),
            "processing_time": time.time()
        },
        description="Heavy data processing"
    )
    
    # Task 4b: Light processing (for small data)
    light_task = FunctionTask(
        name="light_processing",
        func=lambda ctx: {
            "type": "light",
            "data": ctx.get("transform_data_result", {}),
            "processing_time": time.time()
        },
        description="Light data processing"
    )
    
    # Task 5: Final output
    output_task = FunctionTask(
        name="output_results",
        func=lambda ctx: {
            "status": "completed",
            "validation": ctx.get("validate_input_result", {}),
            "transformation": ctx.get("transform_data_result", {}),
            "processing": ctx.get("heavy_processing_result") or ctx.get("light_processing_result"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        description="Output final results"
    )
    
    # Add tasks (sequential for this pipeline)
    flow.add_tasks(validate_task, transform_task, output_task)
    
    # Chain: validate -> transform -> output
    flow.chain("validate_input", "transform_data", "output_results")
    
    return agent


# =============================================================================
# DEMO RUNNER
# =============================================================================

def run_research_demo():
    """Run the Research Agent demo."""
    print("\n" + "="*60)
    print("RESEARCH AGENT DEMO")
    print("="*60)
    
    agent = create_research_agent()
    
    # Execute research workflow
    result = agent.run_flow(
        "research_workflow",
        context={"query": "artificial intelligence applications in healthcare"}
    )
    
    print(f"\nFlow Status: {result.status.value}")
    print(f"Execution Time: {result.execution_time:.2f}s")
    print(f"Success: {result.success}")
    
    if result.success:
        report = result.task_results.get("generate_report")
        if report and report.output:
            print("\n--- Research Report ---")
            for key, value in report.output.items():
                print(f"  {key}: {value}")
    else:
        print(f"Errors: {result.errors}")
    
    # Show metrics
    print("\n--- Agent Metrics ---")
    metrics = agent.get_metrics()
    print(f"  Total Tools: {metrics['tools']}")
    print(f"  Memory Stats: {metrics['memory_stats']}")


def run_data_processing_demo():
    """Run the Data Processing Agent demo."""
    print("\n" + "="*60)
    print("DATA PROCESSING AGENT DEMO")
    print("="*60)
    
    agent = create_data_processing_agent()
    
    # Sample input data
    input_data = {
        "name": "Sample Record",
        "value": 42,
        "category": "test",
        "active": True
    }
    
    # Execute data pipeline
    result = agent.run_flow(
        "data_pipeline",
        context={
            "input_data": input_data,
            "transform_operation": "uppercase"
        }
    )
    
    print(f"\nFlow Status: {result.status.value}")
    print(f"Execution Time: {result.execution_time:.2f}s")
    print(f"Success: {result.success}")
    
    if result.success:
        output = result.task_results.get("output_results")
        if output and output.output:
            print("\n--- Processing Results ---")
            print(f"  Status: {output.output.get('status')}")
            print(f"  Timestamp: {output.output.get('timestamp')}")
            print(f"  Transformed Data: {output.output.get('transformation', {}).get('transformed')}")
    else:
        print(f"Errors: {result.errors}")


def main():
    """Main entry point for demos."""
    print("AI Agent Framework - Reference Agents Demo")
    print("=========================================")
    
    # Run Research Agent demo
    run_research_demo()
    
    # Run Data Processing Agent demo
    run_data_processing_demo()
    
    print("\n" + "="*60)
    print("DEMO COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()
