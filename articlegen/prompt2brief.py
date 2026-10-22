"""Expand a single natural-language prompt into a fully-specified ``Brief``.

The autonomous ``auto`` flow takes one sentence from the user and must still satisfy
the decision gate (which rejects incomplete input), so this builds a complete,
gate-valid ``Brief``. When the planner model is unavailable - offline mode, or a
provider error such as a rate limit - it falls back to a deterministic heuristic so
the pipeline can still run end to end without anyone hand-editing JSON.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .llm import LLMClient, ProviderError, parse_json
from .models import ArticleType, Brief, CitationStyle, TASK_ID_RE
from .prompts import PromptError, load_template

#: Keyword heuristics used by the deterministic fallback to pick an article type.
_RESEARCH_KEYWORDS = {
    "computational": r"\b(code|program|implement|algorithm|script|function|library|api|software|python|runtime)\b",
    "empirical": r"\b(experiment|study|dataset|empirical|measure|clinical|user study|benchmark)\b",
    "systematic_review": r"\b(systematic review|literature review|meta-analysis|survey of the literature)\b",
    "narrative_review": r"\b(review|survey|overview)\b",
    "educational": r"\b(explainer|tutorial|guide|introduction to|how to| Primer)\b",
}


def _system_prompt(prompt_dir: Optional[str] = None) -> str:
    try:
        return load_template("brief_writer", prompt_dir)
    except PromptError:
        return (
            "You convert a single user request into a JSON article brief. "
            "Return only JSON with title, objective, audience, mode, article_type, "
            "citation_style, requires_citations, requires_experiments, target_words "
            "and a sections array of {id,title,purpose,word_target} objects."
        )


def _parse_brief(text: str) -> Optional[Dict[str, Any]]:
    try:
        value = parse_json(text)
    except Exception:  # noqa: BLE001 - any parse failure means "use the fallback"
        return None
    return value if isinstance(value, dict) else None


def _infer_article_type(prompt: str) -> str:
    p = (prompt or "").lower()
    for article_type, pattern in _RESEARCH_KEYWORDS.items():
        if re.search(pattern, p):
            return article_type
    return "educational"


def _default_sections(target: int) -> List[Dict[str, Any]]:
    target = max(200, min(12000, int(target)))
    specs = [
        ("introduction", "Introduction",
         "Frame the problem and state what the reader will gain.", 0.15),
        ("overview", "Overview",
         "Explain the core idea precisely enough to apply it.", 0.35),
        ("worked-example", "Worked Example",
         "Show the concept applied end to end with real inputs and outputs.", 0.30),
        ("tradeoffs", "Trade-offs and Limits",
         "Name what this approach costs and when it is the wrong choice.", 0.20),
    ]
    sections: List[Dict[str, Any]] = []
    for index, (sid, title, purpose, fraction) in enumerate(specs):
        word_target = max(150, int(round(target * fraction / 50.0)) * 50)
        sections.append({
            "id": sid,
            "title": title,
            "purpose": purpose,
            "word_target": word_target,
        })
    return sections


def _fallback_dict(prompt: str, *, target_words: int, max_sources: int) -> Dict[str, Any]:
    slug = re.sub(r"[^a-z0-9]+", "-", (prompt or "article").lower()).strip("-")[:40]
    task_id = f"auto-{slug}" if slug else "auto-article"
    if len(task_id) > 49:
        task_id = task_id[:49]
    article_type = ArticleType.parse(_infer_article_type(prompt))
    title = (prompt or "Untitled article").strip().replace("\n", " ")
    if len(title) > 120:
        title = title[:117].rstrip() + "..."
    objective = (
        f"Produce a complete, self-contained {article_type.value} article about: "
        f"{prompt}. The reader should finish with a clear, evidence-based "
        f"understanding and, where applicable, a reproducible example."
    )
    audience = ("practitioners and interested readers who want a precise, working "
                "treatment of the topic.")
    return {
        "task_id": task_id,
        "title": title,
        "objective": objective,
        "audience": audience,
        "mode": "article",
        "tone": "neutral, precise, professional",
        "target_words": target_words,
        "sections": _default_sections(target_words),
        "constraints": ["avoid unqualified performance claims",
                        "state assumptions explicitly"],
        "requires_citations": bool(article_type.is_research),
        "language": "en",
        "article_type": article_type.value,
        "citation_style": "apa",
        "requires_experiments": bool(article_type.wants_experiments),
        "execute_experiments": True,
        "source_retrieval": "auto",
        "max_sources": max_sources,
        "review_iterations": 1,
        "experiment_timeout": 30,
        "novelty_claim": False,
        "sources": [],
    }


def _clean_sections(raw: Any, target_words: int) -> List[Dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        return []
    cleaned: List[Dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        sid = str(item.get("id") or item.get("section_id") or f"section-{index + 1}").strip().lower()
        sid = re.sub(r"[^a-z0-9]+", "-", sid).strip("-")
        if not sid or not TASK_ID_RE.match(sid) or len(sid) > 41:
            sid = f"section-{index + 1}"
        title = str(item.get("title") or sid).strip()[:120] or sid
        purpose = str(item.get("purpose") or item.get("description") or title).strip()
        if len(purpose) < 15:
            purpose = (f"{purpose} - explain it clearly for the reader."
                       if purpose else f"Explain {title} clearly for the reader.")
        try:
            word_target = int(item.get("word_target") or 0)
        except (TypeError, ValueError):
            word_target = 0
        if word_target <= 0:
            word_target = 200
        cleaned.append({
            "id": sid,
            "title": title,
            "purpose": purpose,
            "word_target": word_target,
        })
    return cleaned


def _merge(base: Dict[str, Any], model: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    if isinstance(model.get("title"), str) and model["title"].strip():
        out["title"] = model["title"].strip()[:120]
    if isinstance(model.get("objective"), str) and len(model["objective"].strip()) >= 40:
        out["objective"] = model["objective"].strip()
    if isinstance(model.get("audience"), str) and model["audience"].strip():
        out["audience"] = model["audience"].strip()
    if isinstance(model.get("mode"), str) and model["mode"].strip().lower() in ("article", "code"):
        out["mode"] = model["mode"].strip().lower()
    if isinstance(model.get("article_type"), str):
        out["article_type"] = ArticleType.parse(model["article_type"]).value
    if isinstance(model.get("citation_style"), str):
        out["citation_style"] = CitationStyle.parse(model["citation_style"]).value
    if isinstance(model.get("requires_citations"), bool):
        out["requires_citations"] = model["requires_citations"]
    if isinstance(model.get("requires_experiments"), bool):
        out["requires_experiments"] = model["requires_experiments"]
    if isinstance(model.get("target_words"), (int, float)) and 200 <= int(model["target_words"]) <= 12000:
        out["target_words"] = int(model["target_words"])
    sections = _clean_sections(model.get("sections"), out["target_words"])
    if len(sections) >= 3:
        out["sections"] = sections
    return out


def brief_from_prompt(prompt: str, client: Optional[LLMClient] = None, *,
                      target_words: int = 1400, max_sources: int = 20,
                      prompt_dir: Optional[str] = None) -> Brief:
    """Build a gate-valid ``Brief`` from a single natural-language ``prompt``.

    Tries the planner model first; on any unavailability it falls back to a
    deterministic heuristic so the autonomous flow never blocks on manual input.
    """
    system = _system_prompt(prompt_dir)
    user = f"USER REQUEST:\n{prompt}\n\nReturn the JSON brief now."

    model_dict: Optional[Dict[str, Any]] = None
    if client is not None and getattr(client, "available", False):
        try:
            response = client.complete(system, user, max_output_tokens=2048)
            model_dict = _parse_brief(response.text)
        except (ProviderError, ValueError):
            model_dict = None

    base = _fallback_dict(prompt, target_words=target_words, max_sources=max_sources)
    if model_dict:
        base = _merge(base, model_dict)
    return Brief.from_dict(base)
