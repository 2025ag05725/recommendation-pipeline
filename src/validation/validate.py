"""Validation stage: run all checks, quarantine bad records, emit report + logs, enforce quality gate."""
from __future__ import annotations

import pandas as pd

from src.common.io_utils import file_md5, read_json, read_raw_csv, write_json
from src.common.logging_utils import get_logger
from src.ingestion.generate_synthetic import GENRES
from src.validation.checks import IssueLog, validate_interactions, validate_items, validate_users

log = get_logger("validation")


class DataQualityError(RuntimeError):
    """Raised when a table exceeds the configured maximum error rate (the quality gate)."""


def run(cfg) -> dict:
    raw, processed, errors, reports = cfg.path("raw"), cfg.path("processed"), cfg.path("errors"), cfg.path("reports")
    manifest = read_json(raw / "manifest.json")
    for table, meta in manifest.items():  # the data being validated must be what was ingested
        if file_md5(cfg.root / meta["file"]) != meta["md5"]:
            raise RuntimeError(f"{meta['file']} changed after ingestion (md5 mismatch) - re-run ingestion")

    users, items, inter = (read_raw_csv(raw / f"{t}.csv") for t in ("users", "items", "interactions"))
    il = IssueLog()
    engine = cfg["validation"].get("engine", "great_expectations")
    vocab = set(cfg.get("genre_vocabulary", GENRES))
    if engine == "great_expectations":
        from src.validation import ge_checks as engine_mod
        runner = engine_mod.GXRunner(il)
        v_users = engine_mod.validate_users(users, cfg, runner)
        v_items = engine_mod.validate_items(items, cfg, runner, vocab)
        v_inter = engine_mod.validate_interactions(inter, cfg, runner, v_users, v_items)
        import great_expectations as gx
        engine_label = f"great_expectations {gx.__version__}"
    elif engine == "custom":
        v_users = validate_users(users, cfg, il)
        v_items = validate_items(items, cfg, il, vocab)
        v_inter = validate_interactions(inter, cfg, il, v_users, v_items)
        engine_label = "custom (lightweight fallback)"
    else:
        raise ValueError(f"validation.engine must be great_expectations|custom, got {engine!r}")
    log.info("Validation engine: %s", engine_label)

    issues = il.issues_df()
    cfg.write_table(issues, "errors", "validation_issues")
    issues.to_csv(errors / "validation_issues.csv", index=False)  # human-readable copy

    summary: dict[str, dict] = {}
    for name, raw_df, valid_df in (("users", users, v_users), ("items", items, v_items), ("interactions", inter, v_inter)):
        cfg.write_table(valid_df.reset_index(drop=True), "processed", f"valid_{name}")
        rej = raw_df.loc[~raw_df["_row_id"].isin(valid_df["_row_id"])].copy()
        reasons = (issues[(issues["table"] == name) & (issues["severity"] == "error")]
                   .groupby("row_id")["check"].agg(lambda s: ";".join(sorted(set(s)))))
        rej["reject_reasons"] = rej["_row_id"].map(reasons)
        cfg.write_table(rej.reset_index(drop=True), "errors", f"rejected_{name}")
        summary[name] = {"rows_in": len(raw_df), "rows_valid": len(valid_df), "rows_rejected": len(rej),
                         "error_rate": round(len(rej) / max(1, len(raw_df)), 4),
                         "warning_rows": int(issues[(issues["table"] == name) & (issues["severity"] == "warning")]["row_id"].nunique())}
        log.info("%-13s in=%-6d valid=%-6d rejected=%-5d (%.2f%%) warning_rows=%d", name, len(raw_df), len(valid_df),
                 len(rej), 100 * summary[name]["error_rate"], summary[name]["warning_rows"])

    for r in il.results:
        if r["rows_failed"]:
            log.log(30 if r["severity"] == "warning" else 40, "[%s] %-28s %-7s failed=%d  (%s)", r["table"], r["check"],
                    r["severity"].upper(), r["rows_failed"], r["description"])

    gate = cfg["validation"]["max_error_rate"]
    failed_tables = [t for t, s in summary.items() if s["error_rate"] > gate]
    report = {"engine": engine_label, "quality_gate_max_error_rate": gate, "gate_passed": not failed_tables, "tables": summary, "checks": il.results}
    write_json(report, reports / "validation_report.json")
    _write_markdown(report, reports / "validation_report.md")
    if failed_tables:
        raise DataQualityError(f"Quality gate failed (> {gate:.0%} rows rejected) for: {failed_tables}")
    return {"gate_passed": True, **{f"{t}_rejected": s["rows_rejected"] for t, s in summary.items()}}


def _write_markdown(report: dict, path) -> None:
    L = ["# Data Validation Report", "", f"Engine: **{report['engine']}**", "",
         f"Quality gate: reject-rate per table must be <= **{report['quality_gate_max_error_rate']:.0%}** -> "
         f"**{'PASSED' if report['gate_passed'] else 'FAILED'}**", "",
         "Severity: `error` = record quarantined to `data/errors/`; `warning` = record kept and repaired in preparation.", "",
         "## Summary by table", "", "| Table | Rows in | Valid | Rejected | Reject rate | Rows with warnings |", "|---|---:|---:|---:|---:|---:|"]
    for t, s in report["tables"].items():
        L.append(f"| {t} | {s['rows_in']} | {s['rows_valid']} | {s['rows_rejected']} | {s['error_rate']:.2%} | {s['warning_rows']} |")
    L += ["", "## All checks (pass and fail)", "", "| Table | Check | Severity | Description | Failed / checked | Status |", "|---|---|---|---|---:|---|"]
    for r in report["checks"]:
        L.append(f"| {r['table']} | `{r['check']}` | {r['severity']} | {r['description']} | {r['rows_failed']} / {r['rows_checked']} | {r['status']} |")
    L += ["", "Rejected records with reasons: `data/errors/rejected_<table>.csv`; every individual issue: `data/errors/validation_issues.csv`."]
    path.write_text("\n".join(L) + "\n")
