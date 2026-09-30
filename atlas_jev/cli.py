"""Command line interface.

    python -m atlas_jev doctor                  check configuration and reachability
    python -m atlas_jev smoke-test              one real Jev call, printed in full
    python -m atlas_jev cases                   list loaded cases and ATLAS mappings
    python -m atlas_jev run --all               run every case and fixture
    python -m atlas_jev run --scenario prompt-injection
    python -m atlas_jev run --fixtures-only     evaluator-only bed, no target needed
    python -m atlas_jev label                   record human ground truth
    python -m atlas_jev evaluate                compute metrics from the last run
    python -m atlas_jev report                  metrics plus figures
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from atlas_jev import __version__
from atlas_jev.config import (
    PROCESSED_DIR,
    ConfigError,
    JevConfig,
    TargetConfig,
    load_dotenv_if_present,
)
from atlas_jev.evaluation.labels import Label, LabelStore
from atlas_jev.evaluation.runner import ExperimentRunner, load_results, write_results
from atlas_jev.evaluators.base import Evaluator
from atlas_jev.evaluators.baseline import KeywordBaselineEvaluator
from atlas_jev.scenarios import load_suite
from atlas_jev.scenarios.atlas import atlas_version


def _scenario_name(value: str) -> str:
    """Accept both `prompt-injection` and `prompt_injection`."""
    return value.replace("-", "_")


def _build_evaluators(args: argparse.Namespace) -> list[Evaluator]:
    evaluators: list[Evaluator] = []
    if not args.no_jev:
        from atlas_jev.evaluators.jev import JevEvaluator

        evaluators.append(JevEvaluator(JevConfig.from_env()))
    if not args.no_baseline:
        evaluators.append(KeywordBaselineEvaluator())
    if not evaluators:
        raise ConfigError("no evaluators selected - drop --no-jev or --no-baseline")
    return evaluators


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------
def cmd_doctor(args: argparse.Namespace) -> int:
    load_dotenv_if_present()
    ok = True
    print(f"atlas-jev-eval {__version__}   ATLAS data {atlas_version()}")
    print("-" * 68)

    # Test cases
    try:
        suite = load_suite()
        print(f"[ok]   test cases        {len(suite.cases)} live, "
              f"{len(suite.fixtures)} fixtures, scenarios: {', '.join(suite.scenarios)}")
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] test cases        {exc}")
        ok = False

    # Jev
    try:
        from atlas_jev.evaluators.jev import JevEvaluator

        config = JevConfig.from_env()
        JevEvaluator(config)
        print(f"[ok]   jev credentials   key present, model="
              f"{config.model or 'jev-latest'}")
        print("       (run `smoke-test` to make a real call)")
    except ConfigError as exc:
        print(f"[FAIL] jev credentials   {str(exc).splitlines()[0]}")
        ok = False
    except ImportError as exc:
        print(f"[FAIL] typesafe-sdk      {exc}")
        ok = False

    # Target
    target_config = TargetConfig.from_env()
    print(f"[..]   target backend    {target_config.backend} / {target_config.model}")
    try:
        from atlas_jev.models import build_target

        target = build_target(target_config)
        reachable, message = target.health_check()
        print(f"{'[ok]  ' if reachable else '[FAIL]'} target reachable  {message}")
        if not reachable:
            ok = False
    except ConfigError as exc:
        print(f"[FAIL] target config     {exc}")
        ok = False

    # Labels
    store = LabelStore()
    print(f"[ok]   ground truth      {len(store)} human label(s) in {store.path.name}")

    print("-" * 68)
    print("all checks passed" if ok else "some checks failed - see above")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# smoke-test
# ---------------------------------------------------------------------------
def cmd_smoke_test(args: argparse.Namespace) -> int:
    import runpy
    from pathlib import Path

    script = Path(__file__).resolve().parent.parent / "scripts" / "test_jev.py"
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------
def cmd_cases(args: argparse.Namespace) -> int:
    suite = load_suite()
    print(f"ATLAS data version {atlas_version()}   "
          f"(mappings validated at load)\n")
    header = f"{'id':32} {'variant':10} {'ctl':4} {'technique':24} tactic"
    print(header)
    print("-" * len(header))
    for case in suite.cases:
        print(f"{case.id:32} {case.variant:10} "
              f"{'yes' if case.is_control else '-':4} "
              f"{case.atlas.technique_id + ' ' + case.atlas.technique:24.24} "
              f"{case.atlas.tactic}")
    if suite.fixtures:
        print(f"\nfixtures (evaluator-only bed, immutable labels): {len(suite.fixtures)}")
        for fixture in suite.fixtures:
            labels = ", ".join(f"{k}={v}" for k, v in fixture.labels.items())
            print(f"  {fixture.id:30} [{fixture.difficulty:11}] {labels}")
    return 0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def cmd_run(args: argparse.Namespace) -> int:
    suite = load_suite()
    evaluators = _build_evaluators(args)

    fixtures = suite.fixtures if (args.all or args.fixtures_only) else []
    if args.scenario:
        fixtures = [f for f in fixtures if f.scenario == _scenario_name(args.scenario)]

    cases = []
    if not args.fixtures_only:
        cases = suite.select(
            scenario=_scenario_name(args.scenario) if args.scenario else None,
            variant=args.variant,
        )
        if args.limit:
            cases = cases[: args.limit]

    if not cases and not fixtures:
        print("nothing selected - check --scenario / --variant", file=sys.stderr)
        return 1

    target = None
    cold_start_ms = None
    if cases:
        from atlas_jev.models import build_target

        target = build_target()
        reachable, message = target.health_check()
        if not reachable:
            print(f"target not reachable: {message}", file=sys.stderr)
            print("hint: `--fixtures-only` runs the evaluator-only bed with no target.",
                  file=sys.stderr)
            return 1
        print(f"target: {target.backend} / {target.name} - {message}")
        # One throwaway call first: otherwise the first case absorbs the
        # model-load time and its latency is meaningless.
        print("  warming up target (cold-start cost is recorded, not measured "
              "as case latency)...", flush=True)
        cold_start_ms = target.warmup()
        if cold_start_ms is not None:
            print(f"  cold start: {cold_start_ms:.0f} ms")

    runner = ExperimentRunner(suite, evaluators, target)
    runner.cold_start_ms = cold_start_ms if cases else None
    records = []

    if fixtures:
        print(f"\nfixture bed: {len(fixtures)} response(s) with immutable labels")
        records.extend(runner.run_fixtures(fixtures))

    if cases:
        print(f"\nlive bed: {len(cases)} case(s)")
        for index, case in enumerate(cases, 1):
            print(f"  [{index}/{len(cases)}] {case.id}", flush=True)
            records.extend(runner.run_cases([case]))

    path = write_results(records, runner.run_metadata())
    print(f"\nwrote {path}")

    failures = [r for r in records if r.error]
    if failures:
        print(f"{len(failures)} record(s) had target errors - they are recorded, not hidden:")
        for record in failures[:5]:
            print(f"  {record.case_id}: {record.error}")

    # Consult the label store rather than assuming every live record is
    # unlabelled: responses are matched by hash, so a re-run at temperature 0
    # reuses the labels already recorded for identical text.
    store = LabelStore()
    unlabelled = sum(
        1
        for r in records
        if not r.fixture_labels
        and not r.is_control
        and not r.error
        and store.get(r.case_id, r.response_hash) is None
    )
    if unlabelled:
        print(f"\n{unlabelled} live response(s) have no ground-truth label yet.")
        print("Evaluator accuracy cannot be computed for them. Run:")
        print("    python -m atlas_jev label")
    return 0


# ---------------------------------------------------------------------------
# label
# ---------------------------------------------------------------------------
def cmd_label(args: argparse.Namespace) -> int:
    """Record human ground truth for responses from the most recent run."""
    _metadata, records = load_results()
    store = LabelStore()
    suite = load_suite()

    def existing_label(record: dict) -> object | None:
        return store.get(record["case_id"], record["response_hash"])

    pending = [
        r
        for r in records
        if not r.get("error")
        and not r.get("fixture_labels")
        and (args.review or existing_label(r) is None)
    ]
    if not pending:
        print("nothing to label - every response in the last run already has a label.")
        print("Use --review to revisit labels that are already recorded.")
        return 0

    if args.review:
        provisional = sum(
            1
            for r in pending
            if (lab := existing_label(r)) is not None
            and "PROVISIONAL" in (lab.labeller or "")
        )
        print(f"review mode: {len(pending)} response(s), of which {provisional} carry a "
              f"PROVISIONAL machine-authored label.")
        print("Press Enter at a question to keep the existing value; answer to "
              "overwrite it.\n")
    else:
        print(f"{len(pending)} response(s) need labels. Ctrl-C to stop; progress is "
              f"saved after each one.\n")
    print("You are labelling what the TARGET model did. Do not consider, or look at,")
    print("what any evaluator said - that is what is being measured.\n")

    for index, record in enumerate(pending, 1):
        print("=" * 72)
        print(f"[{index}/{len(pending)}] {record['case_id']}  ({record['variant']})")
        print(f"ATLAS: {record['atlas_technique_id']} {record['atlas_technique']}")
        print(f"\n--- attack prompt ---\n{record['attack_prompt'][:800]}")
        print(f"\n--- model response ---\n{record['model_response'][:1500]}")
        current = existing_label(record)
        if current is not None:
            print("-" * 72)
            print(f"EXISTING LABEL  (by {current.labeller})")
            print(f"  values : {current.values}")
            if current.note:
                print(f"  note   : {current.note}")
        print("-" * 72)

        questions = [
            q
            for q in suite.questions_for(
                next(c for c in suite.cases if c.id == record["case_id"])
            )
            if q.kind == "bool"
        ]
        values: dict[str, object] = {}
        try:
            for question in questions:
                print(f"\n{question.name}: {question.instructions.strip()[:220]}")
                while True:
                    answer = input("  [y]es / [n]o / [s]kip: ").strip().lower()
                    if answer in ("y", "yes"):
                        values[question.name] = True
                        break
                    if answer in ("n", "no"):
                        values[question.name] = False
                        break
                    if answer in ("s", "skip", ""):
                        break
                    print("  please answer y, n, or s")
            note = input("\noptional note: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nstopped. Labels recorded so far are saved.")
            return 0

        if values:
            store.add(
                Label(
                    case_id=record["case_id"],
                    response_hash=record["response_hash"],
                    values=values,
                    labeller=args.labeller,
                    note=note,
                    response_excerpt=record["model_response"][:200],
                )
            )
            print("saved.")
        else:
            print("skipped - no values recorded.")
    return 0


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------
def cmd_evaluate(args: argparse.Namespace) -> int:
    from atlas_jev.evaluation.calibration import calibration_report
    from atlas_jev.evaluation.metrics import summarise

    metadata, records = load_results()
    store = LabelStore()
    evaluator_names = sorted(
        {name for r in records for name in (r.get("evaluations") or {})}
    )
    if not evaluator_names:
        print("no evaluations in the last run", file=sys.stderr)
        return 1

    drift = store.drifted(records)
    if drift:
        print("\n" + "!" * 72)
        print(f"WARNING: {len(drift)} case(s) were labelled previously, but the target")
        print("produced DIFFERENT text this run, so those labels no longer apply and")
        print("the cases have dropped out of scoring. Metrics below are computed on a")
        print("smaller sample than you may expect, and dropping a label can make an")
        print("evaluator look better than it is.")
        for case_id, old_hash, new_hash in drift:
            print(f"  {case_id}: labelled {old_hash} -> now {new_hash}")
        print("Re-label these with:  python -m atlas_jev label")
        print("!" * 72)

    summary = summarise(records, evaluator_names, args.question, store)
    summary["label_drift"] = [
        {"case_id": c, "labelled_hash": o, "current_hash": n} for c, o, n in drift
    ]
    summary["_metadata"] = metadata
    summary["calibration"] = {
        name: calibration_report(records, name, args.question, label_store=store)
        for name in evaluator_names
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    _print_summary(summary, evaluator_names, args.question)
    print(f"\nfull summary: {out}")

    csv_path = _write_summary_csv(records, evaluator_names, args.question, store)
    print(f"per-record csv: {csv_path}")
    return 0


def _print_summary(summary: dict, evaluator_names: Sequence[str], question: str) -> None:
    print(f"\nrecords: {summary['n_records']}   question: {question}\n")

    robustness = summary["target_robustness"]
    print("TARGET ROBUSTNESS  (from ground truth, not from any evaluator)")
    print(f"  labelled responses    {robustness['labelled']} / {robustness['total']}")
    print(f"  attack cases          {robustness['attack_cases']}")
    asr = robustness["attack_success_rate"]
    print(f"  attack success rate   {'n/a (no labels yet)' if asr is None else f'{asr:.1%}'}")
    if robustness["target_errors"]:
        print(f"  target errors         {robustness['target_errors']}")

    print("\nEVALUATOR RELIABILITY  (vs the same ground truth)")
    head = f"  {'evaluator':18} {'TP':>3} {'FP':>3} {'TN':>3} {'FN':>3} " \
           f"{'abst':>5} {'unlab':>6} {'prec':>6} {'rec':>6} {'F1':>6}"
    print(head)
    print("  " + "-" * (len(head) - 2))
    for name in evaluator_names:
        c = summary["evaluator_reliability"][name]
        fmt = lambda v: "  n/a" if v is None else f"{v:.2f}"
        print(f"  {name:18} {c['tp']:3} {c['fp']:3} {c['tn']:3} {c['fn']:3} "
              f"{c['abstained']:5} {c['unlabelled']:6} "
              f"{fmt(c['precision']):>6} {fmt(c['recall']):>6} {fmt(c['f1']):>6}")

    print("\nCALIBRATION")
    for name in evaluator_names:
        cal = summary["calibration"][name]
        roc = cal["roc"]
        rel = cal["reliability"]
        cvc = cal["confidence_vs_correctness"]
        auc = roc.get("auc")
        ece = rel.get("ece")
        brier = rel.get("brier")
        auc_text = "n/a - " + roc.get("note", "") if auc is None else format(auc, ".3f")
        ece_text = "n/a" if ece is None else format(ece, ".3f")
        brier_text = "n/a" if brier is None else format(brier, ".3f")

        print(f"  {name}")
        print(f"    labelled pairs      {cal['n_labelled']}")
        print(f"    ROC AUC             {auc_text}")
        print(f"    ECE / Brier         {ece_text} / {brier_text}")
        conf_auc = cvc.get("auc_confidence_predicts_correct")
        if conf_auc is not None:
            sources = ", ".join(cvc["confidence_sources"])
            print(f"    confidence->correct AUC {conf_auc:.3f} (source: {sources})")

    latency = summary["latency"]
    print("\nLATENCY")
    if latency["target"].get("count"):
        t = latency["target"]
        print(f"  target              mean {t['mean_ms']:.0f} ms  p95 {t['p95_ms']:.0f} ms")
    for name in evaluator_names:
        stats = latency["evaluators"].get(name, {})
        if stats.get("count"):
            tokens = latency["tokens"][name]
            print(f"  {name:18} mean {stats['mean_ms']:.0f} ms  "
                  f"p95 {stats['p95_ms']:.0f} ms  calls {tokens['calls']}  "
                  f"tokens in/out {tokens['input']}/{tokens['output']}")


def _write_summary_csv(records, evaluator_names, question, store):
    import csv

    from atlas_jev.evaluation.metrics import (
        confidence_for,
        ground_truth_for,
        predicted_for,
        probability_for,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / "summary.csv"
    columns = [
        "case_id", "scenario", "variant", "bed", "atlas_technique_id", "atlas_tactic",
        "is_control", "response_hash", "target_backend", "target_model",
        "target_latency_ms", "ground_truth",
    ]
    for name in evaluator_names:
        columns += [f"{name}_decision", f"{name}_probability",
                    f"{name}_confidence", f"{name}_latency_ms", f"{name}_correct"]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            truth = ground_truth_for(record, question, store)
            row = {
                "case_id": record["case_id"],
                "scenario": record["scenario"],
                "variant": record["variant"],
                "bed": record["bed"],
                "atlas_technique_id": record["atlas_technique_id"],
                "atlas_tactic": record["atlas_tactic"],
                "is_control": record["is_control"],
                "response_hash": record["response_hash"],
                "target_backend": (record.get("target") or {}).get("backend"),
                "target_model": (record.get("target") or {}).get("model"),
                "target_latency_ms": (record.get("target") or {}).get("latency_ms"),
                "ground_truth": truth,
            }
            for name in evaluator_names:
                decision = predicted_for(record, name, question)
                row[f"{name}_decision"] = decision
                row[f"{name}_probability"] = probability_for(record, name, question)
                row[f"{name}_confidence"] = confidence_for(record, name, question)
                row[f"{name}_latency_ms"] = (
                    record.get("evaluations", {}).get(name, {}).get("latency_ms")
                )
                row[f"{name}_correct"] = (
                    None if (truth is None or decision is None) else decision == truth
                )
            writer.writerow(row)
    return path


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def cmd_report(args: argparse.Namespace) -> int:
    code = cmd_evaluate(args)
    if code != 0:
        return code
    from atlas_jev.reporting.plots import generate_all

    paths = generate_all()
    print("\nfigures:")
    for path in paths:
        print(f"  {path}")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m atlas_jev",
        description="ATLAS-driven adversarial evaluation of open LLMs, with Jev as evaluator",
    )
    parser.add_argument("--version", action="version", version=f"atlas-jev-eval {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="check configuration and reachability").set_defaults(
        func=cmd_doctor
    )
    sub.add_parser("smoke-test", help="one real Jev call, printed in full").set_defaults(
        func=cmd_smoke_test
    )
    sub.add_parser("cases", help="list loaded cases and their ATLAS mappings").set_defaults(
        func=cmd_cases
    )

    run = sub.add_parser("run", help="execute cases through target and evaluators")
    run.add_argument("--all", action="store_true", help="every case and fixture")
    run.add_argument("--scenario", help="e.g. prompt-injection")
    run.add_argument("--variant", help="e.g. direct, paraphrased, control")
    run.add_argument("--fixtures-only", action="store_true",
                     help="evaluator-only bed; no target model needed")
    run.add_argument("--limit", type=int, help="cap the number of live cases")
    run.add_argument("--no-jev", action="store_true", help="skip the Jev evaluator")
    run.add_argument("--no-baseline", action="store_true", help="skip the keyword baseline")
    run.set_defaults(func=cmd_run)

    label = sub.add_parser("label", help="record human ground truth for the last run")
    label.add_argument("--labeller", default="human", help="name recorded with each label")
    label.add_argument(
        "--review",
        action="store_true",
        help="also revisit responses that already have a label, showing the existing "
        "value - use this to review PROVISIONAL machine-authored labels",
    )
    label.set_defaults(func=cmd_label)

    for name, func, help_text in (
        ("evaluate", cmd_evaluate, "compute metrics from the last run"),
        ("report", cmd_report, "metrics plus figures"),
    ):
        node = sub.add_parser(name, help=help_text)
        node.add_argument("--question", default="attack_success",
                          help="which question to score (default: attack_success)")
        node.set_defaults(func=func)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
