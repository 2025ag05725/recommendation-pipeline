"""Render a self-contained HTML run report (DAG picture + task table + log tail) from a run summary."""
from __future__ import annotations

import html
import shutil
import subprocess
from pathlib import Path

COLORS = {"success": "#2e9d5b", "failed": "#d64545", "upstream_failed": "#e0a030", "skipped": "#9aa5b1", "pending": "#cfd6dd", "running": "#4c8dc9"}


def _dot(summary: dict) -> str:
    L = ['digraph G {rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, fontcolor=white, margin="0.15,0.08"]; edge [color="#6b7785"];']
    for n, t in summary["tasks"].items():
        L.append(f'"{n}" [label="{n}\\n{t["status"].upper()} · {t.get("seconds", 0)}s", fillcolor="{COLORS.get(t["status"], "#999")}"];')
        for d in t["deps"]:
            L.append(f'"{d}" -> "{n}";')
    return "\n".join(L) + "}"


def render_run_report(summary: dict, log_file: Path, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    img_tag = ""
    if shutil.which("dot"):
        png = fig_dir / "pipeline_dag.png"
        subprocess.run(["dot", "-Tpng", "-Gdpi=130", "-o", str(png)], input=_dot(summary).encode(), check=True)
        img_tag = '<img src="figures/pipeline_dag.png" alt="DAG">'
    rows = "".join(
        f"<tr><td>{n}</td><td>{html.escape(t['stage'])}</td><td><span class='b' style='background:{COLORS.get(t['status'], '#999')}'>{t['status']}</span></td>"
        f"<td>{t.get('attempts', '')}</td><td>{t.get('seconds', '')}</td><td>{', '.join(t['deps']) or '—'}</td><td class='r'>{html.escape(str(t.get('result', t.get('error', ''))))[:140]}</td></tr>"
        for n, t in summary["tasks"].items())
    tail = ""
    if Path(log_file).exists():
        lines = [l for l in Path(log_file).read_text().splitlines() if "| INFO" in l and ("START" in l or "SUCCESS" in l or "finished" in l or "FAILED" in l)]
        tail = html.escape("\n".join(l.split(" | ", 3)[0][11:19] + "  " + l.split(" | ", 3)[-1] for l in lines[-24:]))
    verdict = "SUCCESS" if summary["success"] else "FAILED"
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Pipeline run {summary['run_id']}</title><style>
body{{font-family:Helvetica,Arial,sans-serif;margin:28px;color:#1f2933;background:#fff;max-width:1180px}}
h1{{font-size:22px;margin:0 0 4px}} .sub{{color:#52606d;margin-bottom:16px}}
.v{{display:inline-block;padding:3px 12px;border-radius:5px;color:#fff;font-weight:700;background:{COLORS['success'] if summary['success'] else COLORS['failed']}}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:14px 0}} th,td{{border-bottom:1px solid #e4e7eb;padding:6px 8px;text-align:left;vertical-align:top}} th{{background:#f5f7fa}}
.b{{color:#fff;padding:1px 8px;border-radius:4px;font-size:12px}} .r{{font-family:monospace;font-size:11px;color:#52606d}}
pre{{background:#0f1720;color:#d6e2ee;padding:12px;border-radius:6px;font-size:12px;overflow:auto}} img{{max-width:100%}}
</style></head><body>
<h1>Recommendation pipeline — run report</h1>
<div class="sub">Run <b>{summary['run_id']}</b> · started {summary['started']} · wall time {summary['seconds']}s · <span class="v">{verdict}</span><br>engines: {html.escape(str(summary.get('engines', '')))}</div>
{img_tag}<table><tr><th>Task</th><th>Stage</th><th>Status</th><th>Attempts</th><th>Seconds</th><th>Depends on</th><th>Result</th></tr>{rows}</table>
<h3>Execution log (excerpt)</h3><pre>{tail}</pre></body></html>"""
    (out_dir / "run_report.html").write_text(page)
