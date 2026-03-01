# -*- coding: utf-8 -*-
"""Utility functions for the system"""

import time
import json
from typing import Dict, Any, List
from config import get_config

# ANSI colors for pretty output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def estimate_tokens(text: str) -> int:
    """Rough token estimation (1 token ≈ 4 chars in Russian/English)"""
    return len(text) // 4 + len(text.split())

def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.ENDC}\n")

def print_agent_output(agent_name: str, output: str, iteration: int):
    """Print agent output with formatting"""
    tokens = estimate_tokens(output)
    print(f"\n{Colors.BOLD}{Colors.CYAN}[Iteration {iteration}] {agent_name.upper()}{Colors.ENDC}")
    print(f"{Colors.YELLOW}📊 Output tokens: {tokens}{Colors.ENDC}")
    print(f"{Colors.GREEN}{'-'*80}{Colors.ENDC}")
    print(output)
    print(f"{Colors.GREEN}{'-'*80}{Colors.ENDC}")

def print_summary(total_tokens: int, num_calls: int):
    """Print execution summary"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'EXECUTION SUMMARY'.center(80)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"{Colors.YELLOW}Total Tokens Used: {total_tokens}{Colors.ENDC}")
    print(f"{Colors.YELLOW}Total LLM Calls: {num_calls}{Colors.ENDC}")
    print()

class ConversationLogger:
    """Log all agent interactions"""
    
    def __init__(self):
        self.conversations = []
        self.agent_outputs = {}
        self.start_time = time.time()
    
    def log_agent(self, agent_name: str, iteration: int, input_text: str, output_text: str):
        """Log agent interaction"""
        tokens_in = estimate_tokens(input_text)
        tokens_out = estimate_tokens(output_text)
        
        self.conversations.append({
            "agent": agent_name,
            "iteration": iteration,
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "timestamp": time.time()
        })
        
        key = f"{agent_name}_iter{iteration}"
        self.agent_outputs[key] = output_text
    
    def save_results(self, final_answer: str, results_dir: str = "."):
        """Save all results"""
        config = get_config()
        output_cfg = config["output"]
        
        # Save text file
        if output_cfg["save_txt"]:
            txt_path = f"{results_dir}/{output_cfg['txt_filename']}"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("MULTI-AGENT VERIFICATION SYSTEM - FINAL RESULTS\n")
                f.write("="*80 + "\n\n")
                f.write(final_answer)
                f.write("\n\n" + "="*80 + "\n")
                f.write("CONVERSATION LOG\n")
                f.write("="*80 + "\n\n")
                for conv in self.conversations:
                    f.write(f"[{conv['agent']}] Iteration {conv['iteration']}\n")
                    f.write(f"  Input: {conv['input_tokens']} tokens\n")
                    f.write(f"  Output: {conv['output_tokens']} tokens\n\n")
            
            print(f"✅ Results saved to: {txt_path}")
        
        # Save JSON log
        if output_cfg["save_json"]:
            json_path = f"{results_dir}/{output_cfg['json_filename']}"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "summary": {
                        "total_calls": len(self.conversations),
                        "total_input_tokens": sum(c["input_tokens"] for c in self.conversations),
                        "total_output_tokens": sum(c["output_tokens"] for c in self.conversations),
                        "execution_time": time.time() - self.start_time,
                    },
                    "calls": self.conversations
                }, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Log saved to: {json_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.conversations:
            return {}
        
        return {
            "total_calls": len(self.conversations),
            "total_input_tokens": sum(c["input_tokens"] for c in self.conversations),
            "total_output_tokens": sum(c["output_tokens"] for c in self.conversations),
            "total_tokens": sum(c["input_tokens"] + c["output_tokens"] for c in self.conversations),
            "execution_time": time.time() - self.start_time,
            "avg_tokens_per_call": sum(c["input_tokens"] + c["output_tokens"] for c in self.conversations) / len(self.conversations) if self.conversations else 0,
        }
    
    def print_detailed_summary(self):
        """Print detailed summary"""
        stats = self.get_stats()
        
        print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.HEADER}{'DETAILED STATISTICS'.center(80)}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.ENDC}")
        print(f"{Colors.CYAN}Total LLM Calls: {stats['total_calls']}{Colors.ENDC}")
        print(f"{Colors.CYAN}Input Tokens: {stats['total_input_tokens']}{Colors.ENDC}")
        print(f"{Colors.CYAN}Output Tokens: {stats['total_output_tokens']}{Colors.ENDC}")
        print(f"{Colors.CYAN}Total Tokens: {stats['total_tokens']}{Colors.ENDC}")
        print(f"{Colors.CYAN}Average Tokens/Call: {stats['avg_tokens_per_call']:.0f}{Colors.ENDC}")
        print(f"{Colors.CYAN}Execution Time: {stats['execution_time']:.2f}s{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.ENDC}\n")

# Global logger instance
_logger: ConversationLogger = None

def get_logger() -> ConversationLogger:
    """Get global logger"""
    global _logger
    if _logger is None:
        _logger = ConversationLogger()
    return _logger

def reset_logger():
    """Reset global logger"""
    global _logger
    _logger = ConversationLogger()
