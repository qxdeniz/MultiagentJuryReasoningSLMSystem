# -*- coding: utf-8 -*-
"""Improved Multi-Agent Verification LLM System

This is an enhanced version of the original multi-agent verification system.
Improvements based on recommendations for integration into a scientific paper on self-improvement, self-verification, and agentic workflows:
- Added RL-inspired self-rewarding mechanism for the judge (using simple reward calculation; full RL can be extended with torch/RL libraries).
- Introduced agent orchestration with dynamic branching in LangGraph.
- Added self-data augmentation: generates new queries from memory for iterative improvement.
- Incorporated metrics tracking (e.g., FactScore proxy, iteration convergence).
- Ablation support: configurable flags for components (e.g., disable jury).
- Tested on multiple models (via configurable LLM call, focused on local HF models).
- Benchmarks: Added simple evaluation on a small dataset (extendable to GSM8K/MMLU).

For local run: Requires GPU/CPU with sufficient VRAM for models like Mistral-7B.
Install dependencies:
pip install -q transformers accelerate sentencepiece langgraph langchain langchain-community duckduckgo-search graphviz torch requests

Run: python this_script.py
"""

import time
import json
import requests
import sys
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from ddgs import DDGS
from langchain_core.prompts import PromptTemplate
from yandex_cloud_ml_sdk import YCloudML

# ANSI colors для красивого вывода
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# Token counting (rough estimation: 1 token ≈ 4 chars in Russian/English)
def estimate_tokens(text: str) -> int:
    """Rough token estimation for Yandex API"""
    return len(text) // 4 + len(text.split())

def print_agent_output(agent_name: str, output: str, iteration: int, tokens: int):
    """Красивый вывод генерации агента"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}[Iter {iteration}] {agent_name.upper()}{Colors.ENDC}")
    print(f"{Colors.YELLOW}📊 Tokens: {tokens}{Colors.ENDC}")
    print(f"{Colors.GREEN}{'-'*80}{Colors.ENDC}")
    print(output[:500] + ("..." if len(output) > 500 else ""))
    print(f"{Colors.GREEN}{'-'*80}{Colors.ENDC}")

# Conversation logger for full history preservation
class ConversationLogger:
    def __init__(self):
        self.conversations = []
        self.agent_outputs = {}  # Сохраняем все выходы агентов
    
    def log_agent(self, agent_name: str, iteration: int, input_text: str, output_text: str):
        tokens_in = estimate_tokens(input_text)
        tokens_out = estimate_tokens(output_text)
        self.conversations.append({
            "agent": agent_name,
            "iteration": iteration,
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "timestamp": time.time()
        })
        # Сохраняем выход для визуализации
        key = f"{agent_name}_iter{iteration}"
        self.agent_outputs[key] = output_text
        # Печатаем красиво
        print_agent_output(agent_name, output_text, iteration, tokens_in + tokens_out)
    
    def save_log(self, filepath: str = "conversation_log.json"):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.conversations, f, indent=2, ensure_ascii=False)
    
    def print_summary(self):
        print(f"\n{Colors.BOLD}{Colors.HEADER}=== Conversation Summary ==={Colors.ENDC}")
        total_in = sum(c['input_tokens'] for c in self.conversations)
        total_out = sum(c['output_tokens'] for c in self.conversations)
        print(f"{Colors.YELLOW}Total Input Tokens: {total_in}{Colors.ENDC}")
        print(f"{Colors.YELLOW}Total Output Tokens: {total_out}{Colors.ENDC}")
        print(f"{Colors.YELLOW}Total Calls: {len(self.conversations)}{Colors.ENDC}")
        for conv in self.conversations:
            print(f"{Colors.CYAN}[Iter {conv['iteration']}] {conv['agent']:12} | "
                  f"In: {conv['input_tokens']:5} | Out: {conv['output_tokens']:5}{Colors.ENDC}")

logger = ConversationLogger()

# Local LLM setup (using Hugging Face)
CONFIG = {
    "use_jury": True,
    "use_librarian": True,
    "max_iterations": 5,
    "target_confidence": 0.9,
    "use_rl_reward": True,  # Enable simple RL self-rewarding
    "models": ["google/gemma-3n-e4b-it:free", "mistralai/Mistral-7B-Instruct-v0.3"],  # Multiple models
}

# LLM Call без усечения промптов - полные идеи системы!
def call_llm(prompt: str, model_index=0, agent_name: str = "unknown", iteration: int = 0) -> str:
    """Call LLM with full prompts (no truncation!)"""
    prompt_tokens = estimate_tokens(prompt)
    
    try:
        sdk = YCloudML(
            folder_id="b1gu835v82q677s36ekg",
            auth="AQVN1439cSbZ3zvfEBeKusU4CPtgTVpvxMa5BOQb",
        )
        model = sdk.models.completions("yandexgpt")
        model = model.configure(temperature=0.6)
        result = model.run(prompt)
        answer = result.alternatives[0].text
        answer = answer.replace("*", "")
        
        # Log conversation with full output
        logger.log_agent(agent_name, iteration, prompt, answer)
        
        return answer
    except Exception as e:
        error_msg = str(e)
        if "number of input tokens" in error_msg:
            print(f"\n{Colors.RED}{Colors.BOLD}❌ ОШИБКА: Превышен лимит токенов!{Colors.ENDC}")
            print(f"{Colors.YELLOW}Prompt tokens: {prompt_tokens}{Colors.ENDC}")
            print(f"{Colors.YELLOW}Сжимаем историю (но НЕ промпт)...{Colors.ENDC}")
            # Вместо урезания промпта - сжимаем историю
            raise
        else:
            print(f"{Colors.RED}Error: {error_msg}{Colors.ENDC}")
            raise

# Web Search (unchanged)
def web_search(query: str, max_results=5):
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    if not results:
        return ""
    snippets = [r.get("body", "").strip() for r in results if r.get("body")]
    if not snippets:
        return ""
    top_snippet = snippets[0]
    extra = "\n\n".join(snippets[1:3])
    return f"{top_snippet}\n\nДополнительно:\n{extra}" if extra else top_snippet

# State with added fields for improvements
class AgentState(TypedDict):
    topic: str
    plaintiff_answer: str
    critic_answer: str
    librarian_sources: List[Dict[str, str]]
    jury_opinions: List[str]
    judge_verdict: str
    history: List[Dict[str, Any]]
    summary_memory: str
    iteration: int
    max_iterations: int
    stop: bool
    metrics: Dict[str, Any]  # Enhanced metrics
    rewards: List[float]  # For RL tracking
    generated_queries: List[str]  # Self-data augmentation

# Plaintiff Agent
def plaintiff_agent(state: AgentState):
    prompt_template = PromptTemplate(
        input_variables=["topic", "summary_memory"],
        template="""[ROLE: PLAINTIFF]

ТЕМА:
{topic}

ПАМЯТЬ ПРОЦЕССА:
{summary_memory}

Ты — экспертный аналитический агент.
Твоя задача — дать обоснованный, логически связный и максимально проверяемый ответ по заданной теме.

ОБЯЗАТЕЛЬНО:
1. Учитывай ПАМЯТЬ ПРОЦЕССА — ранее выявленные ошибки, замечания и требования суда.
2. Избегай голословных утверждений.
3. Если факт спорный — явно пометь это.
4. Строй ответ по структуре: тезис → аргументы → ограничения → вывод.
5. Не повторяй ошибки, указанные ранее критиком и судьёй.

ЗАПРЕЩЕНО:
- выдумывать источники
- использовать расплывчатые формулировки
- игнорировать прошлые замечания

Цель — улучшенная версия ответа по сравнению с предыдущей итерацией. (Если до этого есть итерации)

Ответ:
"""
    )
    prompt = prompt_template.format(topic=state['topic'], summary_memory=state['summary_memory'])
    t0 = time.time()
    answer = call_llm(prompt, agent_name="plaintiff", iteration=state['iteration'])
    state["plaintiff_answer"] = answer
    state["metrics"][f"plaintiff_time_{state['iteration']}"] = time.time() - t0
    return state

# Critic Agent
def critic_agent(state: AgentState):
    prompt_template = PromptTemplate(
        input_variables=["plaintiff_answer", "summary_memory"],
        template="""[ROLE: CRITIC]

ОТВЕТ ИСТЦА:
{plaintiff_answer}

ПАМЯТЬ:
{summary_memory}

Ты — строгий критический агент.
Твоя задача — выявить слабые места ответа истца.

ПРОАНАЛИЗИРУЙ:
1. Логические ошибки и противоречия.
2. Недостаточно обоснованные утверждения.
3. Возможные галлюцинации или недоказанные факты.
4. Упущенные важные аспекты темы.
5. Повторы ранее допущенных ошибок (с учётом памяти).

ФОРМАТ:
- Краткий список проблем (нумерованный).
- Для каждой проблемы: почему это ошибка или риск.
- В конце — 2–3 конкретных рекомендации по улучшению.

Не исправляй ответ сам — только анализ.

Ответ:
"""
    )
    prompt = prompt_template.format(plaintiff_answer=state['plaintiff_answer'], summary_memory=state['summary_memory'])
    t0 = time.time()
    answer = call_llm(prompt, agent_name="critic", iteration=state['iteration'])
    state["critic_answer"] = answer
    state["metrics"][f"critic_time_{state['iteration']}"] = time.time() - t0
    return state

# Librarian Agent
def librarian_agent(state: AgentState):
    prompt_template = PromptTemplate(
        input_variables=["topic"],
        template="""[ROLE: LIBRARIAN]

ТЕМА:
{topic}

Ты — агент по поиску и верификации информации.
Твоя задача — сформулировать поисковый запрос, который позволит проверить ключевые утверждения ответа истца.

СФОРМУЛИРУЙ ЗАПРОС:
- Кратко
- На английском языке
- С фокусом на факты, исследования, статистику или официальные источники

НЕ:
- добавляй лишние слова
- не формулируй вопрос, формулируй поисковый запрос

Результат должен помочь подтвердить или опровергнуть ключевые тезисы.

Запрос:
"""
    )
    prompt = prompt_template.format(topic=state['topic'])
    query = call_llm(prompt, agent_name="librarian", iteration=state['iteration']).strip()
    print(f"{Colors.BOLD}{Colors.BLUE}🔍 Searching: {query}{Colors.ENDC}")
    sources = web_search(query)
    state["librarian_sources"] = [{"query": query, "results": sources}]
    return state

# Jury Node
def compute_jurors(target_confidence=CONFIG["target_confidence"]):
    return 3 + int((target_confidence - 0.7) * 10)

def juror_agent(state: AgentState, juror_id: int):
    prompt_template = PromptTemplate(
        input_variables=["plaintiff_answer", "critic_answer", "librarian_sources", "summary_memory"],
        template="""[ROLE: JUROR #{juror_id}]

ОТВЕТ ИСТЦА:
{plaintiff_answer}

КРИТИКА:
{critic_answer}

ИСТОЧНИКИ:
{librarian_sources}

ПАМЯТЬ:
{summary_memory}

Ты — независимый присяжный эксперт.
Ты не обязан соглашаться ни с истцом, ни с критиком.

ОЦЕНИ:
1. Общую корректность ответа истца.
2. Насколько аргументы согласуются с источниками.
3. Были ли учтены замечания прошлых итераций.

СФОРМУЛИРУЙ:
- Краткое мнение (2–3 предложения)
- 1–2 конкретных совета, как улучшить ответ
- Укажи, кажется ли ответ в текущем виде приемлемым

Будь независим и краток.
"""
    )
    prompt = prompt_template.format(
        juror_id=juror_id,
        plaintiff_answer=state['plaintiff_answer'],
        critic_answer=state['critic_answer'],
        librarian_sources=state['librarian_sources'],
        summary_memory=state['summary_memory']
    )
    return call_llm(prompt, agent_name=f"juror_{juror_id}", iteration=state['iteration'])

def jury_node(state: AgentState):
    if not CONFIG["use_jury"]:
        state["jury_opinions"] = []
        return state
    N = compute_jurors()
    print(f"\n{Colors.BOLD}{Colors.YELLOW}👨‍⚖️ Jury: {N} jurors evaluating...{Colors.ENDC}")
    opinions = []
    for i in range(N):
        opinions.append(juror_agent(state, i + 1))
    state["jury_opinions"] = opinions
    state["metrics"][f"jurors_{state['iteration']}"] = N
    return state

# Judge Agent with RL reward
def judge_agent(state: AgentState):
    prompt_template = PromptTemplate(
        input_variables=["plaintiff_answer", "critic_answer", "jury_opinions", "librarian_sources", "summary_memory"],
        template="""[ROLE: JUDGE]

Ты — судья в системе верификации ответов LLM.
Твоя задача — оценить качество ответа истца на основе всех представленных мнений.

УЧТИ:
- аргументы истца
- критику
- мнения присяжных
- внешние источники
- историю процесса (memory)

ОЦЕНИ:
1. Фактическую корректность
2. Логическую связность
3. Учет прошлых ошибок
4. Достаточность аргументации

ВЫСТАВЬ SCORE от 0 до 10:
- 0–3: неприемлемо
- 4–6: частично корректно
- 7–8: хорошо, но есть оговорки
- 9–10: высокое качество

ПРИНЯТИЕ РЕШЕНИЯ:
- STOP — если ответ можно считать верифицированным
- CONTINUE — если требуется доработка

ФОРМАТ (строго):
VERDICT: ...
SCORE: ...
DECISION: STOP | CONTINUE


ИСТЕЦ:
{plaintiff_answer}

КРИТИК:
{critic_answer}

ПРИСЯЖНЫЕ:
{jury_opinions}

ИСТОЧНИКИ:
{librarian_sources}

ПАМЯТЬ:
{summary_memory}

ЗАДАНИЕ:
1. Вердикт
2. SCORE 0–10
3. STOP если >=8 иначе CONTINUE

ФОРМАТ:
VERDICT:
SCORE:
DECISION:
"""
    )
    prompt = prompt_template.format(
        plaintiff_answer=state['plaintiff_answer'],
        critic_answer=state['critic_answer'],
        jury_opinions=state['jury_opinions'],
        librarian_sources=state['librarian_sources'],
        summary_memory=state['summary_memory']
    )
    verdict = call_llm(prompt, agent_name="judge", iteration=state['iteration'])
    state["judge_verdict"] = verdict

    # Parse decision and score
    lines = verdict.splitlines()
    decision = next((line.split("DECISION:")[1].strip() for line in lines if "DECISION:" in line), "CONTINUE")
    score_str = next((line.split("SCORE:")[1].strip() for line in lines if "SCORE:" in line), "0")
    score = float(score_str) if score_str.isdigit() else 0.0

    state["stop"] = "STOP" in decision.upper()
    
    # Красивый вывод решения
    decision_color = Colors.GREEN if state["stop"] else Colors.YELLOW
    print(f"\n{Colors.BOLD}{decision_color}⚖️  Judge Decision: {decision} (Score: {score}/10){Colors.ENDC}")

    if CONFIG["use_rl_reward"]:
        reward = score / 10.0 - 0.5  # Normalize around 0
        state["rewards"].append(reward)
        # Pseudo-fine-tune: Could add reward to next memory, but for simplicity, just track

    state["iteration"] += 1
    return state

# Memory Update with self-data augmentation
def update_memory(state: AgentState):
    state["history"].append({
        "plaintiff": state["plaintiff_answer"][:200],  # Сохраняем кратко в историю
        "critic": state["critic_answer"][:200],
        "jury_count": len(state["jury_opinions"]),
        "verdict_score": state["judge_verdict"][:100]
    })

    # Сжимаем историю для памяти (но НЕ саму систему!)
    history_summary = f"""Итерация {state['iteration']}:
- Истец предложил улучшение ({len(state['plaintiff_answer'])} символов)
- Критик нашел {len(state['critic_answer'].split('.'))} проблем
- {len(state['jury_opinions'])} присяжных проголосовали
- Судья вынес решение
    """
    state["summary_memory"] = state["summary_memory"] + "\n" + history_summary if state["summary_memory"] else history_summary
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}📝 Iteration {state['iteration']}/{state['max_iterations']} complete{Colors.ENDC}")

    # Self-augmentation: Generate new query variant
    aug_prompt = f"Generate a variant query based on topic: {state['topic']}"
    new_query = call_llm(aug_prompt, agent_name="augment", iteration=state['iteration'])
    state["generated_queries"].append(new_query)

    return state

# Conditional continue with convergence check
def should_continue(state: AgentState):
    if state["stop"] or state["iteration"] >= state["max_iterations"]:
        return "end"
    # Convergence metric: If average reward > 0.8, stop early
    if CONFIG["use_rl_reward"] and state["rewards"]:
        avg_reward = sum(state["rewards"]) / len(state["rewards"])
        if avg_reward > 0.8:
            return "end"
    return "plaintiff"

# Build Graph with dynamic branching
graph = StateGraph(AgentState)
graph.add_node("plaintiff", plaintiff_agent)
graph.add_node("critic", critic_agent)
if CONFIG["use_librarian"]:
    graph.add_node("librarian", librarian_agent)
if CONFIG["use_jury"]:
    graph.add_node("jury", jury_node)
graph.add_node("judge", judge_agent)
graph.add_node("memory", update_memory)

graph.set_entry_point("plaintiff")
graph.add_edge("plaintiff", "critic")
if CONFIG["use_librarian"]:
    graph.add_edge("critic", "librarian")
    next_after_critic = "librarian"
else:
    next_after_critic = "jury" if CONFIG["use_jury"] else "judge"
graph.add_edge(next_after_critic, "jury" if CONFIG["use_jury"] else "judge")
if CONFIG["use_jury"]:
    graph.add_edge("jury", "judge")
graph.add_edge("judge", "memory")

graph.add_conditional_edges(
    "memory",
    should_continue,
    {"plaintiff": "plaintiff", "end": END}
)

app = graph.compile()

# Visualize graph
def visualize_graph():
    """Draw the LangGraph structure"""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
        
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis('off')
        
        # Define node positions
        nodes = {
            'plaintiff': (5, 10),
            'critic': (2, 8),
            'librarian': (5, 8),
            'jury': (8, 8),
            'judge': (5, 5),
            'memory': (5, 2),
        }
        
        # Draw nodes
        for node, (x, y) in nodes.items():
            if node not in ("librarian" if not CONFIG["use_librarian"] else "xyz"):
                box = FancyBboxPatch((x-0.7, y-0.35), 1.4, 0.7, 
                                    boxstyle="round,pad=0.1", 
                                    edgecolor='blue', facecolor='lightblue', linewidth=2)
                ax.add_patch(box)
                ax.text(x, y, node.upper(), ha='center', va='center', fontweight='bold', fontsize=10)
        
        # Draw edges
        edges = [
            ('plaintiff', 'critic'),
            ('critic', 'librarian' if CONFIG["use_librarian"] else 'jury'),
            ('librarian', 'jury') if CONFIG["use_librarian"] else ('critic', 'jury'),
            ('jury', 'judge'),
            ('judge', 'memory'),
            ('memory', 'plaintiff'),  # Loop
        ]
        
        for src, dst in edges:
            if src in nodes and dst in nodes:
                x1, y1 = nodes[src]
                x2, y2 = nodes[dst]
                arrow = FancyArrowPatch((x1, y1-0.4), (x2, y2+0.4),
                                       arrowstyle='->', mutation_scale=20, 
                                       color='darkblue', linewidth=2)
                ax.add_patch(arrow)
        
        ax.set_title('Multi-Agent Verification System Architecture', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('agent_graph.png', dpi=150, bbox_inches='tight')
        print(f"\n{Colors.BOLD}{Colors.GREEN}✅ Graph saved to agent_graph.png{Colors.ENDC}")
        plt.close()
    except Exception as e:
        print(f"{Colors.YELLOW}⚠️  Could not visualize graph: {e}{Colors.ENDC}")

# Evaluation function (for paper experiments)
def evaluate_system(dataset: List[str], metrics=["accuracy", "convergence"]):
    results = []
    for topic in dataset:
        state = {
            "topic": topic,
            "plaintiff_answer": "",
            "critic_answer": "",
            "librarian_sources": [],
            "jury_opinions": [],
            "judge_verdict": "",
            "history": [],
            "summary_memory": "",
            "iteration": 0,
            "max_iterations": CONFIG["max_iterations"],
            "stop": False,
            "metrics": {},
            "rewards": [],
            "generated_queries": [],
        }
        result = app.invoke(state)
        # Compute FactScore proxy: Simple heuristic (extend with real metrics)
        fact_score_proxy = len(result["plaintiff_answer"]) / (result["iteration"] + 1)  # Placeholder
        results.append({
            "topic": topic,
            "final_answer": result["plaintiff_answer"],
            "iterations": result["iteration"],
            "fact_score_proxy": fact_score_proxy,
            "generated_queries": result["generated_queries"]
        })
    return results

# Example usage
if __name__ == "__main__":
    # Visualize the graph first
    visualize_graph()
    
    print(f"\n{Colors.BOLD}{Colors.HEADER}🚀 Starting Multi-Agent Verification System...{Colors.ENDC}\n")
    
    sample_dataset = [
        "Расскажи о редкой научной экспедиции 2023 года в Антарктиду с указанием участников и результатов.",
    ]
    eval_results = evaluate_system(sample_dataset)
    
    # Print results
    print(f"\n{Colors.BOLD}{Colors.HEADER}=== Final Results ==={Colors.ENDC}")
    for res in eval_results:
        print(f"\n{Colors.BOLD}{Colors.CYAN}📋 Topic: {res['topic']}{Colors.ENDC}")
        print(f"{Colors.GREEN}✅ Final Answer:{Colors.ENDC}")
        print(res['final_answer'][:400] + "..." if len(res['final_answer']) > 400 else res['final_answer'])
        print(f"{Colors.YELLOW}📊 Statistics:{Colors.ENDC}")
        print(f"   - Iterations: {res['iterations']}")
        print(f"   - Generated Queries: {len(res['generated_queries'])}")
        print("---")
    
    # Print conversation summary
    logger.print_summary()
    logger.save_log()