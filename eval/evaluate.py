"""Evaluation script: build eval set from traces, run scoring, generate report."""

import json
import sqlite3
import sys
import os
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import SQLITE_DB_PATH

EVAL_DB_PATH = str(Path(__file__).resolve().parent / "eval.db")
REPORT_DIR = Path(__file__).resolve().parent

_CREATE_EVAL_TABLE = """
CREATE TABLE IF NOT EXISTS eval_set (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    user_query TEXT NOT NULL,
    intent TEXT NOT NULL,
    actual_answer TEXT NOT NULL,
    actual_sources TEXT NOT NULL,
    expected_answer TEXT DEFAULT '',
    accuracy_score INTEGER DEFAULT 0,
    source_hit INTEGER DEFAULT 0,
    notes TEXT DEFAULT '',
    evaluated INTEGER DEFAULT 0
)
"""


def init_eval_db():
    conn = sqlite3.connect(EVAL_DB_PATH)
    conn.execute(_CREATE_EVAL_TABLE)
    conn.commit()
    conn.close()


def import_traces(limit: int = 30):
    """Import recent traces from trace DB into eval set."""
    if not os.path.exists(SQLITE_DB_PATH):
        print(f"Trace DB not found: {SQLITE_DB_PATH}")
        return 0

    trace_conn = sqlite3.connect(SQLITE_DB_PATH)
    trace_conn.row_factory = sqlite3.Row
    rows = trace_conn.execute(
        "SELECT * FROM trace_logs ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    trace_conn.close()

    if not rows:
        print("No traces found in trace DB.")
        return 0

    eval_conn = sqlite3.connect(EVAL_DB_PATH)
    # Get existing request_ids to avoid duplicates
    existing = {
        r[0]
        for r in eval_conn.execute("SELECT request_id FROM eval_set").fetchall()
    }

    imported = 0
    for row in rows:
        if row["request_id"] in existing:
            continue
        output = json.loads(row["output"])
        eval_conn.execute(
            """INSERT INTO eval_set
               (request_id, user_query, intent, actual_answer, actual_sources)
               VALUES (?, ?, ?, ?, ?)""",
            (
                row["request_id"],
                row["user_query"],
                row["intent"],
                output.get("answer", ""),
                json.dumps(output.get("sources", []), ensure_ascii=False),
            ),
        )
        imported += 1

    eval_conn.commit()
    eval_conn.close()
    print(f"Imported {imported} new traces (skipped {len(rows) - imported} duplicates).")
    return imported


def annotate_interactive():
    """Interactive annotation: score each unevaluated record."""
    conn = sqlite3.connect(EVAL_DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM eval_set WHERE evaluated = 0 ORDER BY id"
    ).fetchall()

    if not rows:
        print("No unevaluated records. Import more traces or all records are scored.")
        conn.close()
        return

    print(f"\n{'='*60}")
    print(f"  Annotation Mode: {len(rows)} records to evaluate")
    print(f"{'='*60}\n")

    for i, row in enumerate(rows):
        print(f"\n--- Record {i+1}/{len(rows)} [id={row['id']}] ---")
        print(f"Query:    {row['user_query']}")
        print(f"Intent:   {row['intent']}")
        print(f"Answer:   {row['actual_answer'][:200]}{'...' if len(row['actual_answer']) > 200 else ''}")
        print(f"Sources:  {row['actual_sources']}")

        # Accuracy score
        while True:
            score = input("\nAccuracy (1-5, or 's' to skip, 'q' to quit): ").strip()
            if score == "q":
                conn.close()
                return
            if score == "s":
                break
            if score.isdigit() and 1 <= int(score) <= 5:
                score = int(score)
                break
            print("Please enter 1-5, 's', or 'q'.")

        if score == "s":
            continue

        # Source hit
        while True:
            hit = input("Source hit? (1=yes, 0=no): ").strip()
            if hit in ("0", "1"):
                hit = int(hit)
                break
            print("Please enter 0 or 1.")

        # Expected answer
        expected = input("Expected answer (enter to skip): ").strip()

        # Notes
        notes = input("Notes (enter to skip): ").strip()

        conn.execute(
            """UPDATE eval_set
               SET accuracy_score=?, source_hit=?, expected_answer=?, notes=?, evaluated=1
               WHERE id=?""",
            (score, hit, expected, notes, row["id"]),
        )
        conn.commit()
        print(f"  -> Saved: score={score}, source_hit={hit}")

    conn.close()
    print("\nAnnotation complete!")


def generate_report() -> str:
    """Generate evaluation report from scored records. Returns the report path."""
    conn = sqlite3.connect(EVAL_DB_PATH)
    conn.row_factory = sqlite3.Row

    total = conn.execute("SELECT COUNT(*) as c FROM eval_set").fetchone()["c"]
    evaluated = conn.execute(
        "SELECT COUNT(*) as c FROM eval_set WHERE evaluated = 1"
    ).fetchone()["c"]

    if evaluated == 0:
        conn.close()
        print("No evaluated records found. Run annotation first.")
        return ""

    # Metrics
    scores = conn.execute(
        "SELECT accuracy_score FROM eval_set WHERE evaluated = 1"
    ).fetchall()
    accuracy_scores = [r["accuracy_score"] for r in scores]
    avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)

    source_hits = conn.execute(
        "SELECT source_hit FROM eval_set WHERE evaluated = 1"
    ).fetchall()
    hit_rate = sum(r["source_hit"] for r in source_hits) / len(source_hits) * 100

    # Fallback rate from trace DB
    fallback_rate = 0.0
    avg_latency = 0.0
    if os.path.exists(SQLITE_DB_PATH):
        trace_conn = sqlite3.connect(SQLITE_DB_PATH)
        trace_conn.row_factory = sqlite3.Row
        trace_total = trace_conn.execute("SELECT COUNT(*) as c FROM trace_logs").fetchone()["c"]
        if trace_total > 0:
            fallback_count = 0
            latency_sum = 0
            for row in trace_conn.execute("SELECT output, latency_ms FROM trace_logs").fetchall():
                output = json.loads(row["output"])
                if output.get("fallback", False):
                    fallback_count += 1
                latency_sum += row["latency_ms"]
            fallback_rate = fallback_count / trace_total * 100
            avg_latency = latency_sum / trace_total
        trace_conn.close()

    # Score distribution
    dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for s in accuracy_scores:
        dist[s] = dist.get(s, 0) + 1

    # Intent breakdown
    intent_rows = conn.execute(
        "SELECT intent, COUNT(*) as c, AVG(accuracy_score) as avg_s "
        "FROM eval_set WHERE evaluated = 1 GROUP BY intent"
    ).fetchall()

    # Detail table
    details = conn.execute(
        "SELECT * FROM eval_set WHERE evaluated = 1 ORDER BY id"
    ).fetchall()

    conn.close()

    # Build report
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report = f"""# Knowledge Base Agent - Evaluation Report

Generated: {now}

## Summary

| Metric | Value |
|--------|-------|
| Total records | {total} |
| Evaluated records | {evaluated} |
| **Avg accuracy score** | **{avg_accuracy:.2f} / 5** |
| **Source hit rate** | **{hit_rate:.1f}%** |
| **Fallback trigger rate** | **{fallback_rate:.1f}%** |
| **Avg latency** | **{avg_latency:.0f} ms** |

## Score Distribution

| Score | Count | Percentage |
|-------|-------|------------|
"""
    for s in range(1, 6):
        pct = dist[s] / evaluated * 100 if evaluated else 0
        bar = "█" * int(pct / 5)
        report += f"| {s} | {dist[s]} | {pct:.1f}% {bar} |\n"

    report += "\n## Intent Breakdown\n\n"
    report += "| Intent | Count | Avg Score |\n"
    report += "|--------|-------|-----------|\n"
    for r in intent_rows:
        report += f"| {r['intent']} | {r['c']} | {r['avg_s']:.2f} |\n"

    report += "\n## Detail Records\n\n"
    for d in details:
        sources = d["actual_sources"]
        report += f"""### Record #{d['id']}

- **Query**: {d['user_query']}
- **Intent**: {d['intent']}
- **Answer**: {d['actual_answer'][:300]}{'...' if len(d['actual_answer']) > 300 else ''}
- **Sources**: {sources}
- **Score**: {d['accuracy_score']}/5 | Source hit: {'Yes' if d['source_hit'] else 'No'}
- **Expected**: {d['expected_answer'] or '(not provided)'}
- **Notes**: {d['notes'] or '(none)'}

"""

    # Save
    report_path = REPORT_DIR / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Report saved to: {report_path}")
    return str(report_path)


def auto_score_from_traces():
    """Auto-populate eval set with basic heuristic scores for quick demo.

    Scores based on: confidence level and fallback status from trace output.
    This is a convenience for bootstrapping the eval set without full manual annotation.
    """
    conn = sqlite3.connect(EVAL_DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM eval_set WHERE evaluated = 0 ORDER BY id"
    ).fetchall()

    if not rows:
        print("No unevaluated records to auto-score.")
        conn.close()
        return 0

    trace_conn = sqlite3.connect(SQLITE_DB_PATH)
    trace_conn.row_factory = sqlite3.Row

    scored = 0
    for row in rows:
        trace = trace_conn.execute(
            "SELECT output FROM trace_logs WHERE request_id = ?",
            (row["request_id"],),
        ).fetchone()

        if trace is None:
            continue

        output = json.loads(trace["output"])
        confidence = output.get("confidence", "low")
        fallback = output.get("fallback", True)

        # Heuristic scoring
        if fallback:
            score = 1
            source_hit = 0
        elif confidence == "high":
            score = 4
            source_hit = 1
        elif confidence == "medium":
            score = 3
            source_hit = 1
        else:
            score = 2
            source_hit = 0

        conn.execute(
            """UPDATE eval_set
               SET accuracy_score=?, source_hit=?, notes='auto-scored', evaluated=1
               WHERE id=?""",
            (score, source_hit, row["id"]),
        )
        scored += 1

    conn.commit()
    conn.close()
    trace_conn.close()
    print(f"Auto-scored {scored} records.")
    return scored


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Knowledge Base Agent Evaluation")
    sub = parser.add_subparsers(dest="command", help="Available commands")

    sub.add_parser("import", help="Import traces from trace DB into eval set")
    sub.add_parser("annotate", help="Interactively annotate eval records")
    sub.add_parser("auto-score", help="Auto-score unevaluated records (heuristic)")
    sub.add_parser("report", help="Generate evaluation report")
    sub.add_parser("status", help="Show eval set status")

    args = parser.parse_args()
    init_eval_db()

    if args.command == "import":
        import_traces()
    elif args.command == "annotate":
        annotate_interactive()
    elif args.command == "auto-score":
        auto_score_from_traces()
    elif args.command == "report":
        generate_report()
    elif args.command == "status":
        conn = sqlite3.connect(EVAL_DB_PATH)
        total = conn.execute("SELECT COUNT(*) FROM eval_set").fetchone()[0]
        evaluated = conn.execute("SELECT COUNT(*) FROM eval_set WHERE evaluated=1").fetchone()[0]
        conn.close()
        print(f"Eval set: {total} total, {evaluated} evaluated, {total - evaluated} pending")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
