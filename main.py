# -*- coding: utf-8 -*-
"""Main entry point for Multi-Agent Verification System"""

from typing import List, Dict, Any
from langgraph.graph import StateGraph, END
from config import get_config, set_model_provider
from agents import (
    AgentState, plaintiff_agent, critic_agent, librarian_agent,
    jury_node, judge_agent, update_memory, final_agent
)
from utils import (
    print_header, get_logger, reset_logger, Colors,
    print_summary
)


def create_graph():
    """Create and compile the agent workflow graph"""
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("plaintiff", plaintiff_agent)
    graph.add_node("critic", critic_agent)
    graph.add_node("librarian", librarian_agent)
    graph.add_node("jury", jury_node)
    graph.add_node("judge", judge_agent)
    graph.add_node("memory", update_memory)
    
    # Set entry point
    graph.set_entry_point("plaintiff")
    
    # Add edges
    graph.add_edge("plaintiff", "critic")
    graph.add_edge("critic", "librarian")
    graph.add_edge("librarian", "jury")
    graph.add_edge("jury", "judge")
    graph.add_edge("judge", "memory")
    graph.add_node("final", final_agent)
    graph.add_edge("final", END)
    
    # Conditional routing
    def should_continue(state):
        config = get_config()
        # Enforce a minimum number of iterations (at least 2)
        min_iterations = 2
        if state.get("iteration", 0) < min_iterations:
            return "plaintiff"

        if state["stop"] or state["iteration"] >= state["max_iterations"]:
            return "end"

        # Early stopping if high confidence
        if state["rewards"]:
            avg_reward = sum(state["rewards"]) / len(state["rewards"])
            if avg_reward > 0.8:
                return "end"

        return "plaintiff"
    
    graph.add_conditional_edges(
        "memory",
        should_continue,
        {"plaintiff": "plaintiff", "end": "final"}
    )
    
    return graph.compile()


def initialize_state(topic: str) -> AgentState:
    """Initialize agent state"""
    config = get_config()
    return AgentState(
        topic=topic,
        plaintiff_answer="",
        critic_answer="",
        librarian_sources=[],
        jury_opinions=[],
        judge_verdict="",
        final_answer="",
        history=[],
        summary_memory="",
        iteration=0,
        max_iterations=config["max_iterations"],
        stop=False,
        metrics={},
        rewards=[],
        generated_queries=[]
    )


def process_topic(topic: str, app) -> Dict[str, Any]:
    """Process a single topic through the verification system"""
    print_header(f"Processing: {topic}")
    
    state = initialize_state(topic)
    result = app.invoke(state)
    
    return {
        "topic": topic,
        "final_answer": result.get("final_answer", result.get("plaintiff_answer", "")),
        "iterations": result["iteration"],
        "judge_verdict": result["judge_verdict"],
        "rewards": result["rewards"],
    }


def run_system(topics: List[str], provider: str = None, model: str = None):
    """Run the verification system on multiple topics"""
    
    # Configure LLM provider
    if provider:
        set_model_provider(provider, model)
    
    config = get_config()
    print_header("MULTI-AGENT VERIFICATION SYSTEM")
    print(f"Provider: {Colors.YELLOW}{config['model_provider']}{Colors.ENDC}")
    print(f"Model: {Colors.YELLOW}{config[config['model_provider']]['model']}{Colors.ENDC}")
    print(f"Max Iterations: {Colors.YELLOW}{config['max_iterations']}{Colors.ENDC}")
    print()
    
    # Create workflow
    app = create_graph()
    
    # Reset logger
    reset_logger()
    logger = get_logger()
    
    # Process each topic
    results = []
    for topic in topics:
        result = process_topic(topic, app)
        results.append(result)
    
    # Print final answer with full content
    print_header("FINAL VERIFICATION RESULTS")
    for result in results:
        print(f"{Colors.BOLD}{Colors.CYAN}Topic: {result['topic']}{Colors.ENDC}")
        print(f"{Colors.GREEN}{'='*80}{Colors.ENDC}")
        print(result["final_answer"])
        print(f"{Colors.GREEN}{'='*80}{Colors.ENDC}")
        print(f"{Colors.YELLOW}Iterations: {result['iterations']}{Colors.ENDC}\n")
    
    # Save results
    if results:
        final_answer = "\n\n".join([
            f"TOPIC: {r['topic']}\n\n{r['final_answer']}"
            for r in results
        ])
        logger.save_results(final_answer)
    
    # Print statistics
    stats = logger.get_stats()
    logger.print_detailed_summary()
    
    return results


def main():
    """Main entry point"""
    
    # Example topics to process
    topics = [
        "Верно ли утверждение, что все простые числа являются нечётными? Дайте развернутый ответ с доказательством"
    ]
    
    # Run with default provider (Yandex)
    # To use OpenRouter instead, uncomment:
    # run_system(topics, provider="openrouter", model="meta-llama/llama-2-7b-chat")
    
    run_system(topics)


if __name__ == "__main__":
    main()
