# -*- coding: utf-8 -*-
"""Multi-agent system for verification"""

from typing import TypedDict, List, Dict, Any
from langchain_core.prompts import PromptTemplate
from models import call_llm
from search import web_search
from utils import get_logger, estimate_tokens, print_agent_output

# State TypedDict
class AgentState(TypedDict):
    topic: str
    plaintiff_answer: str
    critic_answer: str
    librarian_sources: List[Dict[str, str]]
    jury_opinions: List[str]
    judge_verdict: str
    final_answer: str
    history: List[Dict[str, Any]]
    summary_memory: str
    iteration: int
    max_iterations: int
    stop: bool
    metrics: Dict[str, Any]
    rewards: List[float]
    generated_queries: List[str]


# ==================== AGENTS ====================

def plaintiff_agent(state: AgentState) -> AgentState:
    """Generate initial answer - plaintiff agent"""
    prompt_template = PromptTemplate(
        input_variables=["topic", "summary_memory"],
        template="""[ROLE: PLAINTIFF - ИСТЕЦ]

ТЕМА РАССМОТРЕНИЯ:
{topic}

ПАМЯТЬ (предыдущие итерации):
{summary_memory}

Ты — экспертный аналитический агент, специализирующийся на создании обоснованных, проверяемых и логически связных ответов.

ТВОИ ОБЯЗАТЕЛЬСТВА:
1. Дай полный, структурированный ответ по теме.
2. Учитывай ПАМЯТЬ ПРОЦЕССА — ранее выявленные ошибки, замечания критиков и требования судьи.
3. Избегай голословных утверждений — подкрепляй каждый тезис аргументами.
4. Если факт спорный или неоднозначный — явно отметь это и объясни почему.
5. Структурируй ответ: тезис → подробные аргументы → ограничения и оговорки → итоговый вывод.
6. Не повторяй ошибки, выявленные в прошлых итерациях.

ЗАПРЕЩЕНО:
- Выдумывать источники и цитаты
- Использовать расплывчатые или двусмысленные формулировки
- Игнорировать замечания прошлых итераций
- Быть субъективным — придерживайся фактов

Цель: Создать лучшую версию ответа по сравнению с предыдущей итерацией (если они есть).

ВАЖНО: Ты не должен писать исковое заявление или включать в ответ юридическую направленность! Ты должен отвечать как ИИ-эксперт-ассистент

ОТВЕТ:
"""
    )
    
    prompt = prompt_template.format(
        topic=state['topic'],
        summary_memory=state['summary_memory'] or "Это первая итерация."
    )
    # If this is the second iteration or later, include web-search results produced by the librarian
    if state.get('iteration', 0) >= 1 and state.get('librarian_sources'):
        try:
            last_search = state['librarian_sources'][-1]
            search_text = last_search.get('results') if isinstance(last_search, dict) else str(last_search)
            if search_text:
                prompt = prompt + "\n\n[WEB SEARCH RESULTS FROM LIBRARIAN]\n" + str(search_text)
        except Exception:
            pass

    answer = call_llm(prompt)
    state["plaintiff_answer"] = answer
    
    # Log and display
    logger = get_logger()
    logger.log_agent("plaintiff", state['iteration'], prompt, answer)
    print_agent_output("plaintiff", answer, state['iteration'])
    
    return state


def critic_agent(state: AgentState) -> AgentState:
    """Analyze and critique the answer"""
    prompt_template = PromptTemplate(
        input_variables=["plaintiff_answer", "summary_memory"],
        template="""[ROLE: CRITIC - КРИТИК]

АНАЛИЗИРУЕМЫЙ ОТВЕТ ИСТЦА:
{plaintiff_answer}

ИСТОРИЯ ПРОЦЕССА:
{summary_memory}

Ты — аналитический критик, задача которого выявить слабые места, логические разрывы и недостатки.

ТВОЙ АНАЛИЗ ДОЛЖЕН ВКЛЮЧАТЬ:
1. Логические ошибки и внутренние противоречия
2. Недостаточно обоснованные утверждения (где нужны факты)
3. Возможные галлюцинации, предположения, выданные за факты
4. Упущенные важные аспекты темы
5. Повторение ошибок из прошлых итераций (если есть история)

ФОРМАТ ВЫВОДА:
- Начни с общей оценки (1-2 предложения)
- Перечисли 5-7 конкретных проблем (нумерованно)
- Для каждой проблемы: почему это ошибка, чем она может быть опасна
- Закончи 3-4 конкретными рекомендациями по улучшению

ВАЖНО: Не переписывай ответ сам, только выявляй проблемы!

КРИТИЧЕСКИЙ АНАЛИЗ:
"""
    )
    
    prompt = prompt_template.format(
        plaintiff_answer=state['plaintiff_answer'],
        summary_memory=state['summary_memory'] or "Нет истории"
    )
    
    answer = call_llm(prompt)
    state["critic_answer"] = answer
    
    # Log and display
    logger = get_logger()
    logger.log_agent("critic", state['iteration'], prompt, answer)
    print_agent_output("critic", answer, state['iteration'])
    
    return state


def librarian_agent(state: AgentState) -> AgentState:
    """Search for sources and information"""
    prompt_template = PromptTemplate(
        input_variables=["topic"],
        template="""[ROLE: LIBRARIAN - БИБЛИОТЕКАРЬ]

ТЕМА ДЛЯ ПОИСКА:
{topic}

Ты — специалист по поиску надежной информации и источников.

ТВОЯ ЗАДАЧА:
1. Анализируй основной вопрос/тему
2. Сформулируй поисковый запрос (на английском или русском)
3. Запрос должен быть конкретным и ориентирован на проверяемые факты

ТРЕБОВАНИЯ К ЗАПРОСУ:
- Краткость (2-5 слов)
- Специфичность (ищи конкретные факты, исследования, статистику)
- Фокус на академических источниках или официальной информации

ПОИСКОВЫЙ ЗАПРОС:
"""
    )
    
    prompt = prompt_template.format(topic=state['topic'])
    query = call_llm(prompt).strip()
    
    # Perform web search
    search_results = web_search(query)
    state["librarian_sources"] = [{"query": query, "results": search_results}]
    
    # Log and display
    logger = get_logger()
    logger.log_agent("librarian", state['iteration'], prompt, search_results)
    print_agent_output("librarian", search_results, state['iteration'])
    
    return state


def compute_jurors(target_confidence=0.85):
    """Calculate number of jurors based on target confidence"""
    return 3 + int((target_confidence - 0.7) * 10)


def juror_agent(state: AgentState, juror_id: int) -> str:
    """Individual juror opinion"""
    prompt_template = PromptTemplate(
        input_variables=["plaintiff_answer", "critic_answer", "librarian_sources"],
        template="""[ROLE: JUROR #{juror_id} - ПРИСЯЖНЫЙ #{juror_id}]

РАССМАТРИВАЕМЫЙ ОТВЕТ:
{plaintiff_answer}

КРИТИЧЕСКОЕ ЗАМЕЧАНИЕ:
{critic_answer}

ИСТОЧНИКИ И ИССЛЕДОВАНИЯ:
{librarian_sources}

Ты — независимый эксперт-присяжный. Ты имеешь право иметь собственное мнение.

ТВОЙ АНАЛИЗ ДОЛЖЕН СОДЕРЖАТЬ:
1. Независимую оценку корректности ответа (0-10)
2. Согласие или несогласие с критиком (и почему)
3. Соответствие ответа источникам
4. 1-2 конкретных предложения по улучшению
5. Финальное суждение: приемлем ли ответ в текущем виде?

БУДЬ ЧЕСТЕН И НЕЗАВИСИМ. Не просто соглашайся с критиком.

МНЕНИЕ ПРИСЯЖНОГО:
"""
    )
    
    prompt = prompt_template.format(
        juror_id=juror_id,
        plaintiff_answer=state['plaintiff_answer'][:600],
        critic_answer=state['critic_answer'][:400],
        librarian_sources=str(state['librarian_sources'])[:300]
    )
    
    opinion = call_llm(prompt)
    
    # Log and display
    logger = get_logger()
    logger.log_agent(f"juror_{juror_id}", state['iteration'], prompt, opinion)
    print_agent_output(f"juror_{juror_id}", opinion, state['iteration'])
    
    return opinion


def jury_node(state: AgentState) -> AgentState:
    """Collect all jury opinions"""
    num_jurors = compute_jurors()
    print(f"\n👥 Собирается жюри из {num_jurors} присяжных...\n")
    
    opinions = []
    for i in range(num_jurors):
        opinion = juror_agent(state, i + 1)
        opinions.append(opinion)
    
    state["jury_opinions"] = opinions
    state["metrics"][f"jurors_{state['iteration']}"] = num_jurors
    
    return state


def judge_agent(state: AgentState) -> AgentState:
    """Judge makes final decision"""
    prompt_template = PromptTemplate(
        input_variables=["plaintiff_answer", "critic_answer", "jury_opinions", "librarian_sources"],
        template="""[ROLE: JUDGE - СУДЬЯ]

ПОЛНЫЙ КОНТЕКСТ ДЕЛА:

ОТВЕТ ИСТЦА:
{plaintiff_answer}

КРИТИЧЕСКОЕ АНАЛИЗ:
{critic_answer}

МНЕНИЯ ПРИСЯЖНЫХ (жюри):
{jury_opinions}

ИСТОЧНИКИ И ДОКАЗАТЕЛЬСТВА:
{librarian_sources}

Ты — финальный судья в системе верификации. Твоя задача — вынести обоснованное решение.

ТВОЙ ВЕРДИКТ ДОЛЖЕН СОДЕРЖАТЬ:
1. Объективную оценку качества ответа (0-10 баллов)
2. Анализ основных достижений (что сделано хорошо)
3. Выявленные проблемы (что нужно улучшить)
4. Учет мнений присяжных
5. Финальное решение: STOP (ответ готов) или CONTINUE (нужны доработки)

КРИТЕРИИ ОЦЕНКИ:
- 0-3: Неприемлемо (много ошибок, недостоверно)
- 4-6: Частично корректно (есть проблемы, но основа есть)
- 7-8: Хорошо (незначительные замечания)
- 9-10: Отлично (полностью обоснованно и проверяемо)

РЕШЕНИЕ ПРИНИМАЕТСЯ ТАК:
- STOP если SCORE >= 8 (ответ готов к использованию)
- CONTINUE если SCORE < 8 (нужна еще работа)

ФОРМАТ ВЫВОДА (СТРОГО):
VERDICT: [твой полный вердикт]
SCORE: [число 0-10]
DECISION: STOP|CONTINUE

СУДЕБНЫЙ ВЕРДИКТ:
"""
    )
    
    prompt = prompt_template.format(
        plaintiff_answer=state['plaintiff_answer'][:800],
        critic_answer=state['critic_answer'][:500],
        jury_opinions="\n---\n".join(state['jury_opinions'][:2]),
        librarian_sources=str(state['librarian_sources'])[:300]
    )
    
    verdict = call_llm(prompt)
    state["judge_verdict"] = verdict
    
    # Log and display
    logger = get_logger()
    logger.log_agent("judge", state['iteration'], prompt, verdict)
    print_agent_output("judge", verdict, state['iteration'])
    
    # Parse decision
    lines = verdict.splitlines()
    decision_line = next((line for line in lines if "DECISION:" in line), "DECISION: CONTINUE")
    score_line = next((line for line in lines if "SCORE:" in line), "SCORE: 0")
    
    try:
        score = float(score_line.split("SCORE:")[-1].strip())
    except:
        score = 0.0
    
    state["stop"] = "STOP" in decision_line.upper()
    
    # RL reward
    reward = score / 10.0 - 0.5
    state["rewards"].append(reward)
    
    state["iteration"] += 1
    return state


def final_agent(state: AgentState) -> AgentState:
    """Compose final answer based on the full chain of reasoning"""
    prompt_template = PromptTemplate(
        input_variables=[
            "topic", "plaintiff_answer", "critic_answer", "librarian_sources",
            "jury_opinions", "judge_verdict", "summary_memory", "history"
        ],
        template="""[ROLE: FINALIZER - ИТОГОВЫЙ АГЕНТ]

ТЕМА: {topic}

ИМЕЮТСЯ СЛЕДУЮЩИЕ ДАННЫЕ И РАССУЖДЕНИЯ (СВЯЖИ ИХ В ЦЕЛОЕ):

ОТВЕТ ИСТЦА:
{plaintiff_answer}

КРИТИКА:
{critic_answer}

НАЙДЕННЫЕ ИСТОЧНИКИ (ЛИБРАРИАН):
{librarian_sources}

МНЕНИЯ ЖЮРИ:
{jury_opinions}

ВЕРДИКТ СУДЬИ:
{judge_verdict}

ОБЩАЯ ПАМЯТЬ И ИСТОРИЯ ПРОЦЕССА:
{summary_memory}

ИНСТРУКЦИЯ:
1) Синтезируй связный, окончательный ответ по теме в 6-12 абзацев.
2) Начни с краткого резюме (2-3 предложения) итогового вывода.
3) Ясно укажи, какие утверждения подтверждены источниками, а какие остаются спорными.
4) Ссылки и упоминания источников — только если они есть в предоставленных данных.
5) Заключи практическим советом/рекомендацией (1-2 предложения).

ИТОГОВЫЙ ОТВЕТ:
"""
    )

    prompt = prompt_template.format(
        topic=state.get('topic', ''),
        plaintiff_answer=state.get('plaintiff_answer', ''),
        critic_answer=state.get('critic_answer', ''),
        librarian_sources=str(state.get('librarian_sources', '')),
        jury_opinions="\n---\n".join(state.get('jury_opinions', [])),
        judge_verdict=state.get('judge_verdict', ''),
        summary_memory=state.get('summary_memory', ''),
        history=str(state.get('history', []))[:2000]
    )

    answer = call_llm(prompt)
    state["final_answer"] = answer

    # Log and display
    logger = get_logger()
    logger.log_agent("final", state.get('iteration', 0), prompt, answer)
    print_agent_output("final", answer, state.get('iteration', 0))

    return state


def update_memory(state: AgentState) -> AgentState:
    """Update history and memory"""
    # Store compressed history
    state["history"].append({
        "plaintiff": state["plaintiff_answer"][:200],
        "critic": state["critic_answer"][:150],
        "verdict": state["judge_verdict"][:100],
        "iteration": state["iteration"] - 1
    })
    
    # Create compact memory summary
    compact_history = "\n".join([
        f"Итерация {h.get('iteration', i)}: {h['plaintiff'][:80]}... | Вердикт: {h['verdict'][:60]}..."
        for i, h in enumerate(state["history"][-3:])
    ])
    
    prompt_template = PromptTemplate(
        input_variables=["history"],
        template="""Суммируй прогресс верификации в 5-7 кратких пунктов.
Сосредоточься на: ошибках → исправлениях → оценке качества.

ИСТОРИЯ:
{history}

ИТОГОВАЯ СЖАТАЯ СВОДКА:
"""
    )
    
    prompt = prompt_template.format(history=compact_history)
    state["summary_memory"] = call_llm(prompt)
    
    # Log
    logger = get_logger()
    logger.log_agent("memory", state['iteration'] - 1, prompt, state["summary_memory"])
    
    return state
