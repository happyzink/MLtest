"""
Report service: generate markdown/HTML comparison reports from saved runs.
"""

import json
from pathlib import Path
from datetime import datetime
from app.services.artifact_service import load_metadata, load_predictions, _run_dir, APP_STORAGE_DIR
from app.core.config import APP_STORAGE_DIR as STORAGE


def generate_comparison_report(
    runs: list[dict],
    title: str = "模型比较报告",
    output_dir: Path = None,
) -> dict:
    """Generate a comparison report from multiple runs.

    Each run dict should have: experiment_id, run_id, algorithm, metrics, params.
    """
    now = datetime.now().isoformat()

    lines = [
        f"# {title}",
        f"",
        f"生成时间: {now}",
        f"",
        f"## 模型列表",
        f"",
        f"| 算法 | R² | RMSE | MAE | 训练样本 | 测试样本 |",
        f"|------|-----|------|-----|---------|---------|",
    ]

    for run in runs:
        tm = run.get("test_metrics", {})
        r2 = f"{tm.get('r2', 'N/A'):.4f}" if isinstance(tm.get('r2'), (int, float)) else "N/A"
        rmse = f"{tm.get('rmse', 'N/A'):.4f}" if isinstance(tm.get('rmse'), (int, float)) else "N/A"
        mae = f"{tm.get('mae', 'N/A'):.4f}" if isinstance(tm.get('mae'), (int, float)) else "N/A"
        lines.append(
            f"| {run.get('algorithm', '-')} | {r2} | {rmse} | {mae} | "
            f"{tm.get('n', '-')} | {tm.get('n', '-')} |"
        )

    lines.extend([
        "",
        "## 详细说明",
        "",
        "### 评估协议",
        "- 数据划分: 80/20 随机划分 (random_state=42)",
        "- 指标: R² (决定系数), RMSE (均方根误差), MAE (平均绝对误差)",
        "- R² 可为负值，表示模型比简单均值预测更差",
        "",
        "### 参数详情",
        "",
    ])

    for run in runs:
        lines.append(f"#### {run.get('algorithm', 'Unknown')}")
        lines.append(f"```json")
        lines.append(json.dumps(run.get("effective_params", run.get("params", {})), indent=2, ensure_ascii=False))
        lines.append(f"```")
        lines.append("")

    report_md = "\n".join(lines)

    result = {"markdown": report_md, "runs_count": len(runs)}

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        md_path = output_dir / "comparison_report.md"
        md_path.write_text(report_md, encoding="utf-8")
        result["markdown_path"] = str(md_path)

        # Simple HTML version
        html = _markdown_to_html(report_md, title)
        html_path = output_dir / "comparison_report.html"
        html_path.write_text(html, encoding="utf-8")
        result["html_path"] = str(html_path)

    return result


def _markdown_to_html(md: str, title: str) -> str:
    """Minimal markdown to HTML conversion for reports."""
    import re
    lines = md.split("\n")
    html_lines = ["<!DOCTYPE html>", "<html><head>", f"<title>{title}</title>",
                  "<meta charset='utf-8'>",
                  "<style>",
                  "body { font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; color: #172033; line-height: 1.6; }",
                  "h1 { color: #2563EB; border-bottom: 2px solid #DDE3EB; padding-bottom: 8px; }",
                  "h2 { color: #172033; margin-top: 24px; }",
                  "h4 { color: #566477; }",
                  "table { border-collapse: collapse; width: 100%; margin: 16px 0; }",
                  "th, td { border: 1px solid #DDE3EB; padding: 8px 12px; text-align: left; font-size: 14px; }",
                  "th { background: #F5F7FA; font-weight: 600; }",
                  "pre { background: #F5F7FA; padding: 12px; border-radius: 6px; overflow-x: auto; font-size: 13px; }",
                  "code { font-family: 'Consolas', monospace; }",
                  "</style>",
                  "</head><body>"]

    in_code = False
    for line in lines:
        if line.startswith("```"):
            if in_code:
                html_lines.append("</code></pre>")
                in_code = False
            else:
                html_lines.append("<pre><code>")
                in_code = True
            continue
        if in_code:
            html_lines.append(line)
            continue
        if line.startswith("# "):
            html_lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("### "):
            html_lines.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("#### "):
            html_lines.append(f"<h4>{line[5:]}</h4>")
        elif line.startswith("| "):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            tag = "th" if "算法" in line or "---" not in line else "td"
            row = "".join(f"<{tag}>{c}</{tag}>" for c in cells)
            html_lines.append(f"<tr>{row}</tr>")
        elif line.startswith("- "):
            html_lines.append(f"<li>{line[2:]}</li>")
        elif line.strip() == "":
            html_lines.append("<br>")
        else:
            html_lines.append(f"<p>{line}</p>")

    html_lines.extend(["</body></html>"])
    return "\n".join(html_lines)
