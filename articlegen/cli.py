"""Command-line interface for the ArticleGen pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from . import __version__
from .changelog import ChangeLog, read_jsonl
from .gate import verify_brief
from .llm import resolve_resilient_client
from .models import ArticleType, Brief, CitationStyle
from .orchestrator import Orchestrator, OrchestratorConfig

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "config", "default.json")

TEMPLATE_BRIEF: Dict[str, Any] = {
    "task_id": "my-article",
    "title": "TODO: replace with a specific article title",
    "objective": (
        "TODO: describe the specific outcome the reader should achieve, why it matters, "
        "and what the article will cover."
    ),
    "audience": "TODO: name the intended readers",
    "mode": "article",
    "tone": "neutral, precise, professional",
    "target_words": 1200,
    "requires_citations": False,
    "constraints": ["include at least one worked example", "avoid unqualified performance claims"],
    "sections": [
        {"id": "introduction", "title": "Introduction",
         "purpose": "Frame the problem and state what the reader will gain.", "word_target": 200},
        {"id": "core-concept", "title": "The Core Concept",
         "purpose": "Explain the mechanism precisely enough to implement it.", "word_target": 450},
        {"id": "worked-example", "title": "Worked Example",
         "purpose": "Show the concept applied end to end with real inputs and outputs.",
         "word_target": 350},
        {"id": "tradeoffs", "title": "Trade-offs and Limits",
         "purpose": "Name what this approach costs and when it is the wrong choice.",
         "word_target": 200},
    ],
}


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object at the top level")
    return value


def _load_config(path: Optional[str]) -> Dict[str, Any]:
    target = path or (DEFAULT_CONFIG if os.path.exists(DEFAULT_CONFIG) else None)
    if not target or not os.path.exists(target):
        return {}
    return _load_json(target)


def _print_issues(issues: List[Any]) -> None:
    if not issues:
        print("  no issues")
        return
    for issue in sorted(issues, key=lambda f: (-f.severity.weight, f.category)):
        marker = {"blocker": "BLOCK", "major": "MAJOR", "minor": "minor", "note": "note"}[
            issue.severity.value
        ]
        print(f"  [{marker}] {issue.category} @{issue.section or '-'}: {issue.claim}")
        print(f"           -> {issue.action}")


def cmd_init(args: argparse.Namespace) -> int:
    if os.path.exists(args.path) and not args.force:
        print(f"refusing to overwrite {args.path} (use --force)", file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(args.path)) or ".", exist_ok=True)
    with open(args.path, "w", encoding="utf-8") as handle:
        json.dump(TEMPLATE_BRIEF, handle, indent=2)
        handle.write("\n")
    print(f"wrote brief template to {args.path}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    brief = Brief.from_dict(_load_json(args.brief))
    result = verify_brief(brief)
    print(f"brief: {brief.task_id} ({brief.mode.value}, {len(brief.sections)} sections)")
    print(f"checks run: {', '.join(result.checks_run)}")
    print(f"decision: {'PASS' if result.ok else 'ROUTE_BACK'}")
    _print_issues(result.issues)
    return 0 if result.ok else 1


def cmd_run(args: argparse.Namespace) -> int:
    config_data = _load_config(args.config)
    config = OrchestratorConfig.from_dict(config_data)
    # These two overrides rewrite the brief, not the orchestration config.
    article_type_override: Optional[ArticleType] = None
    citation_style_override: Optional[CitationStyle] = None

    if args.iterations is not None:
        config.max_iterations = args.iterations
    if args.target is not None:
        config.score_target = args.target
    if args.provider:
        config.provider = args.provider
    if getattr(args, "model", None):
        config.model = args.model
    if args.workspace:
        config.workspace = args.workspace
    if args.changelog:
        config.changelog_path = args.changelog
    if getattr(args, "no_research", False):
        config.research = False
    if getattr(args, "review_iterations", None) is not None:
        config.review_iterations = args.review_iterations
    if getattr(args, "source_retrieval", None):
        config.source_retrieval = args.source_retrieval
    if getattr(args, "max_sources", None) is not None:
        config.max_sources = args.max_sources
    if getattr(args, "experiment_timeout", None) is not None:
        config.experiment_timeout = args.experiment_timeout
    if getattr(args, "no_experiments", False):
        config.execute_experiments = False
    if getattr(args, "article_type", None):
        article_type_override = ArticleType.parse(args.article_type)
    if getattr(args, "citation_style", None):
        citation_style_override = CitationStyle.parse(args.citation_style)

    brief = Brief.from_dict(_load_json(args.brief))
    if article_type_override is not None:
        brief.article_type = article_type_override
    if citation_style_override is not None:
        brief.citation_style = citation_style_override
    orchestrator = Orchestrator(config, changelog=ChangeLog(config.changelog_path))
    result = orchestrator.run(brief)

    print(f"\ntask      : {result.task_id}")
    print(f"status    : {result.status}")
    print(f"stop      : {result.stop_reason}")
    print(f"score     : {result.final_score:.4f} (target {config.score_target})")
    print(f"iterations: {len(result.iterations)}")
    print(f"tokens    : {result.tokens_total}")
    print(f"workspace : {result.workspace or '(not created; run stopped before generation)'}")
    if result.quality_status:
        print(f"research  : {result.quality_status}")
        gate = (result.research or {}).get("quality_gate") or {}
        for reason in (gate.get("reasons") or [])[:8]:
            print(f"  - {reason}")
        report_path = os.path.join(result.workspace, "research", "report.md")
        if os.path.exists(report_path):
            print(f"report    : {report_path}")
    if result.final_draft is not None:
        print(f"draft     : {os.path.join(result.workspace, 'master_draft.md')}")
    else:
        print(f"draft     : not created ({result.stop_reason})")
    print(f"log       : {config.changelog_path}")

    for record in result.iterations:
        previous = "-" if record.previous_score is None else f"{record.previous_score:.4f}"
        print(f"  iter {record.iteration}: {previous} -> {record.score:.4f} "
              f"({record.delta:+.4f}) findings={record.finding_count} "
              f"applied={record.applied} deferred={record.deferred} "
              f"tokens={record.tokens_total} [{record.decision}]")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(result.to_dict(), handle, indent=2)
        print(f"result json: {args.json}")

    if result.gate is not None and result.gate.issues:
        print("\nfinal draft verification:")
        _print_issues(result.gate.issues)

    return 0 if result.status == "PASS" else 1


def cmd_auto(args: argparse.Namespace) -> int:
    """One prompt in, a full article out - build the brief and run autonomously."""
    prompt = args.prompt
    if getattr(args, "prompt_file", None):
        with open(args.prompt_file, "r", encoding="utf-8") as handle:
            prompt = handle.read()
    if not prompt or not prompt.strip():
        print("error: no prompt supplied (pass a prompt argument or --prompt-file)",
              file=sys.stderr)
        return 2

    config_data = _load_config(args.config)
    config = OrchestratorConfig.from_dict(config_data)
    # Autonomous mode: build the brief from the prompt, self-heal the client, and
    # loop the pipeline to a ready quality gate.
    config.auto = True
    if args.iterations is not None:
        config.max_iterations = args.iterations
    if args.target is not None:
        config.score_target = args.target
    if args.provider:
        config.provider = args.provider
    if getattr(args, "model", None):
        config.model = args.model
    if args.workspace:
        config.workspace = args.workspace
    if args.changelog:
        config.changelog_path = args.changelog
    if getattr(args, "auto_rounds", None) is not None:
        config.max_autonomous_rounds = args.auto_rounds
    if getattr(args, "no_research", False):
        config.research = False
    if getattr(args, "review_iterations", None) is not None:
        config.review_iterations = args.review_iterations
    if getattr(args, "source_retrieval", None):
        config.source_retrieval = args.source_retrieval
    if getattr(args, "max_sources", None) is not None:
        config.max_sources = args.max_sources
    if getattr(args, "experiment_timeout", None) is not None:
        config.experiment_timeout = args.experiment_timeout
    if getattr(args, "no_experiments", False):
        config.execute_experiments = False
    if getattr(args, "fallback_models", None):
        config.fallback_models = [m.strip() for m in args.fallback_models.split(",") if m.strip()]

    client = resolve_resilient_client(
        provider=config.provider, model=config.model,
        fallback_models=config.fallback_models,
    )
    orchestrator = Orchestrator(config, client=client, changelog=ChangeLog(config.changelog_path))
    result = orchestrator.run_autonomous(prompt)

    # Mirror cmd_run's reporting, plus always write the article to disk.
    print(f"\ntask      : {result.task_id}")
    print(f"status    : {result.status}")
    print(f"stop      : {result.stop_reason}")
    print(f"score     : {result.final_score:.4f} (target {config.score_target})")
    print(f"iterations: {len(result.iterations)}")
    print(f"tokens    : {result.tokens_total}")
    print(f"workspace : {result.workspace or '(not created; run stopped before generation)'}")
    if result.quality_status:
        print(f"research  : {result.quality_status}")
        gate = (result.research or {}).get("quality_gate") or {}
        for reason in (gate.get("reasons") or [])[:8]:
            print(f"  - {reason}")
        report_path = os.path.join(result.workspace, "research", "report.md")
        if os.path.exists(report_path):
            print(f"report    : {report_path}")
    if result.final_draft is not None:
        print(f"draft     : {os.path.join(result.workspace, 'master_draft.md')}")
    else:
        print(f"draft     : not created ({result.stop_reason})")
    print(f"log       : {config.changelog_path}")

    for record in result.iterations:
        previous = "-" if record.previous_score is None else f"{record.previous_score:.4f}"
        print(f"  iter {record.iteration}: {previous} -> {record.score:.4f} "
              f"({record.delta:+.4f}) findings={record.finding_count} "
              f"applied={record.applied} deferred={record.deferred} "
              f"tokens={record.tokens_total} [{record.decision}]")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(result.to_dict(), handle, indent=2)
        print(f"result json: {args.json}")

    out_path = args.out or os.path.join(result.workspace or ".", "article.md")
    if result.final_draft is not None and any(
            s.body.strip() for s in result.final_draft.sections):
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(result.final_draft.to_markdown())
        print(f"article   : {out_path}")
    else:
        print("article   : not written (no usable draft produced)")

    # Exit 0 when the article reached the validated gate; otherwise report the
    # block so callers (and CI) can tell a finished article from a proposal.
    return 0 if result.quality_status == "ready_for_final_review" else 1


def cmd_log(args: argparse.Namespace) -> int:
    jsonl = args.path or "change_log.jsonl"
    entries = read_jsonl(jsonl)
    if args.task:
        entries = [e for e in entries if e.get("task_id") == args.task]
    if not entries:
        print(f"no log entries found in {jsonl}")
        return 0
    for entry in entries[-args.limit:]:
        event = entry.get("event", "?")
        if event == "iteration":
            print(f"[{entry.get('timestamp')}] {entry.get('task_id')} iter "
                  f"{entry.get('iteration')}: score={entry.get('score')} "
                  f"applied={entry.get('applied')} deferred={entry.get('deferred')} "
                  f"tokens={entry.get('tokens_total')} -> {entry.get('decision')}")
        else:
            print(f"[{entry.get('timestamp')}] {entry.get('task_id')} {event}: "
                  f"{entry.get('status') or entry.get('reason') or ''}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    example = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "examples", "brief_example.json")
    if not os.path.exists(example):
        print(f"example brief not found at {example}", file=sys.stderr)
        return 1
    args.brief = example
    args.iterations = args.iterations if args.iterations is not None else 3
    args.json = args.json or os.path.join(args.workspace or "runs", "demo_result.json")
    return cmd_run(args)


def cmd_ui(args: argparse.Namespace) -> int:
    """Launch the Streamlit dashboard.

    Streamlit is an optional extra - the pipeline itself has no runtime dependencies -
    so a missing install gets an actionable message instead of a traceback.
    """
    import shutil
    import subprocess

    try:
        import streamlit  # noqa: F401
    except ImportError:
        print(
            "error: streamlit is not installed.\n"
            "  install it with:  pip install -e '.[ui]'   (or: pip install streamlit)\n"
            "  then re-run:      articlegen ui",
            file=sys.stderr,
        )
        return 2

    # Resolve from the package, not the repo root: the script ships as package data,
    # so this works from site-packages as well as from a source checkout.
    from .ui import DASHBOARD_PATH

    candidates = [args.app] if args.app else []
    candidates += [DASHBOARD_PATH, os.path.join(os.getcwd(), "app.py")]
    app_path = next((c for c in candidates if c and os.path.exists(c)), None)
    if app_path is None:
        print(
            "error: could not locate the dashboard script.\n"
            f"  looked for: {', '.join(c for c in candidates if c)}\n"
            "  pass --app /path/to/dashboard.py to override",
            file=sys.stderr,
        )
        return 2

    streamlit_bin = shutil.which("streamlit")
    command: List[str]
    if streamlit_bin:
        command = [streamlit_bin, "run", app_path]
    else:
        command = [sys.executable, "-m", "streamlit", "run", app_path]

    command += [
        "--server.port", str(args.port),
        "--server.address", args.address,
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    if args.no_cors:
        # Required when the app is served behind a reverse proxy that rewrites the
        # Origin header; unsafe on an untrusted network.
        command += ["--server.enableCORS", "false", "--server.enableXsrfProtection", "false"]

    print("launching: " + " ".join(command))

    if os.name == "posix":
        # Replace this process rather than forking a child. A launcher that forks
        # leaves the real server orphaned (reparented to PID 1) the moment the
        # wrapper is killed - which is how a stray dashboard outlives a CI job and
        # keeps holding its port. exec also means SIGTERM/SIGINT reach Streamlit
        # directly instead of being absorbed by an idle wrapper.
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            os.execvp(command[0], command)
        except OSError as exc:
            print(f"error: could not exec '{command[0]}': {exc}", file=sys.stderr)
            return 127
        return 127  # unreachable: execvp does not return on success

    try:
        return subprocess.call(command)
    except KeyboardInterrupt:  # pragma: no cover
        return 130


def _add_research_options(parser: argparse.ArgumentParser) -> None:
    """Research-stage switches shared by ``run`` and ``demo``."""
    parser.add_argument("--article-type", default=None,
                        choices=[t.value for t in ArticleType],
                        help="override the brief's article type (sets required sections "
                             "and which checks apply)")
    parser.add_argument("--citation-style", default=None,
                        choices=[s.value for s in CitationStyle])
    parser.add_argument("--no-research", action="store_true",
                        help="disable research stages (plan, sources, experiments, "
                             "review loop, quality gate)")
    parser.add_argument("--review-iterations", type=int, default=None,
                        help="review -> targeted revision -> re-validate iterations")
    parser.add_argument("--source-retrieval", default=None,
                        choices=["auto", "crossref", "user", "static", "none"])
    parser.add_argument("--max-sources", type=int, default=None)
    parser.add_argument("--no-experiments", action="store_true",
                        help="plan experiments but do not execute them (the run will be "
                             "reported as incomplete)")
    parser.add_argument("--experiment-timeout", type=int, default=None,
                        help="per-experiment timeout in seconds")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="articlegen",
        description="Iterative multi-agent article/code pipeline.",
    )
    parser.add_argument("--version", action="version", version=f"articlegen {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="write a starter brief")
    init.add_argument("path", nargs="?", default="brief.json")
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=cmd_init)

    verify = sub.add_parser("verify", help="run the input decision gate only (no tokens spent)")
    verify.add_argument("brief")
    verify.set_defaults(func=cmd_verify)

    run = sub.add_parser("run", help="run the full orchestrated loop")
    run.add_argument("brief")
    run.add_argument("--iterations", type=int, default=None)
    run.add_argument("--target", type=float, default=None, help="quality score target")
    run.add_argument("--provider", default=None,
                     choices=["auto", "anthropic", "openai", "dahl"])
    run.add_argument("--model", default=None,
                     help="model id, e.g. zai-org/GLM-5.3-Flash for the dahl provider")
    run.add_argument("--workspace", default=None)
    run.add_argument("--changelog", default=None)
    run.add_argument("--config", default=None)
    run.add_argument("--json", default=None, help="write the full result JSON here")
    _add_research_options(run)
    run.set_defaults(func=cmd_run)

    log = sub.add_parser("log", help="summarise the machine-readable change log")
    log.add_argument("--path", default=None)
    log.add_argument("--task", default=None)
    log.add_argument("--limit", type=int, default=20)
    log.set_defaults(func=cmd_log)

    auto = sub.add_parser(
        "auto", help="build the brief from ONE prompt and run the pipeline autonomously "
                     "to a full, validated article")
    auto.add_argument("prompt", nargs="?", default=None,
                      help="one natural-language prompt describing the article")
    auto.add_argument("--prompt-file", default=None,
                      help="read the prompt from a file instead of the argument")
    auto.add_argument("--out", default=None,
                      help="write the final article markdown here")
    auto.add_argument("--auto-rounds", type=int, default=4,
                      help="max autonomous re-run rounds (stops early on no progress)")
    auto.add_argument("--iterations", type=int, default=None)
    auto.add_argument("--target", type=float, default=None, help="quality score target")
    auto.add_argument("--provider", default=None,
                      choices=["auto", "anthropic", "openai", "dahl"])
    auto.add_argument("--model", default=None,
                      help="model id, e.g. zai-org/GLM-5.3-Flash for the dahl provider")
    auto.add_argument("--fallback-models", default=None,
                      help="comma-separated fallback models, e.g. "
                           "deepseek-ai/DeepSeek-V4-Flash-0731,MiniMaxAI/MiniMax-M2.7")
    auto.add_argument("--workspace", default=None)
    auto.add_argument("--changelog", default=None)
    auto.add_argument("--config", default=None)
    auto.add_argument("--json", default=None, help="write the full result JSON here")
    auto.add_argument("--no-research", action="store_true",
                      help="disable research stages")
    auto.add_argument("--review-iterations", type=int, default=None)
    auto.add_argument("--source-retrieval", default=None,
                      choices=["auto", "crossref", "user", "static", "none"])
    auto.add_argument("--max-sources", type=int, default=None)
    auto.add_argument("--no-experiments", action="store_true")
    auto.add_argument("--experiment-timeout", type=int, default=None)
    auto.set_defaults(func=cmd_auto)

    ui = sub.add_parser("ui", help="launch the Streamlit web dashboard")
    ui.add_argument("--port", type=int, default=8501)
    ui.add_argument("--address", default="0.0.0.0",
                    help="bind address; 0.0.0.0 so reverse-proxied previews work")
    ui.add_argument("--app", default=None, help="path to app.py (auto-detected by default)")
    ui.add_argument("--no-cors", action="store_true",
                    help="disable CORS/XSRF checks for reverse-proxy hosting")
    ui.set_defaults(func=cmd_ui)

    demo = sub.add_parser("demo", help="run the bundled example brief end to end")
    demo.add_argument("--iterations", type=int, default=None)
    demo.add_argument("--workspace", default=None)
    demo.add_argument("--changelog", default=None)
    demo.add_argument("--provider", default=None,
                      choices=["auto", "anthropic", "openai", "dahl"])
    demo.add_argument("--model", default=None,
                      help="model id, e.g. zai-org/GLM-5.3-Flash for the dahl provider")
    demo.add_argument("--config", default=None)
    demo.add_argument("--target", type=float, default=None)
    demo.add_argument("--json", default=None)
    _add_research_options(demo)
    demo.set_defaults(func=cmd_demo)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"error: malformed JSON: {exc}", file=sys.stderr)
        return 2
    except (TypeError, ValueError) as exc:
        print(f"error: invalid input: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
