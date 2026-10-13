"""
LLM-based auto-tuning service.
No silent fallback: LLM failures are reported explicitly.
Uses backend .env config for API credentials.
"""

import json
import traceback
from app.core.config import (
    is_llm_configured, QUALITY_THRESHOLD, MAX_TUNING_ITERATIONS,
    TUNING_MAX_TRIALS, TUNING_PATIENCE,
)
from app.services.ml_engine import train_model, get_primary_score, ALL_ALGOS
from app.services.llm_client import call_llm


def _build_dataset_context(df, features, target, task_type, algorithm) -> str:
    lines = []
    lines.append(f"Task type: {task_type}")
    lines.append(f"Algorithm: {algorithm}")
    lines.append(f"Total samples: {len(df)}")
    lines.append(f"Features ({len(features)}): {', '.join(features)}")
    lines.append(f"Target: {target}")
    lines.append("Feature statistics:")
    for f in features:
        if df[f].dtype in ["int64", "float64"]:
            lines.append(f"  {f}: mean={df[f].mean():.4f}, std={df[f].std():.4f}, "
                         f"min={df[f].min():.4f}, max={df[f].max():.4f}")
        else:
            lines.append(f"  {f}: {df[f].nunique()} unique values")
    return "\n".join(lines)


def _build_mixer_prompt(dataset_ctx: str, algorithm: str, task_type: str,
                        history: list[dict], algo_entry: dict) -> str:
    default_params = algo_entry["defaults"]
    history_text = ""
    if history:
        history_text = "\nPrevious tuning attempts:\n"
        for h in history:
            score_str = f"{h['score']:.4f}" if h.get('score') is not None else "failed"
            history_text += f"  Round {h['round']}: params={json.dumps(h['params'])}, "
            history_text += f"score={score_str}, status={h.get('status', 'unknown')}"
            if h.get('rationale'):
                history_text += f", rationale={h['rationale']}"
            history_text += "\n"

    prompt = f"""You are an expert ML engineer performing hyperparameter tuning.

Dataset Context:
{dataset_ctx}

Algorithm: {algorithm}
Default parameters: {json.dumps(default_params)}
{history_text}

Your task: Generate a NEW set of hyperparameters that could improve model performance.
- If this is the first round, suggest reasonable starting parameters.
- If previous rounds exist, analyze what worked/didn't and adjust accordingly.
- For regression, optimize for R² score (target >= {QUALITY_THRESHOLD}).
- For classification, optimize for F1 score.

Respond with a JSON object containing:
{{
  "params": {{"n_estimators": 200, "max_depth": 8, "learning_rate": 0.05}},
  "rationale": "Brief explanation of why these parameters"
}}

Only respond with valid JSON, no other text."""
    return prompt


def _parse_llm_response(response: str) -> tuple[dict, str]:
    """Extract params and rationale from LLM response.
    Returns (params_dict, rationale_string).
    Raises ValueError on parse failure.
    """
    text = response.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    if text.startswith("json"):
        text = text[4:].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{[^{}]+\}', text)
        if match:
            data = json.loads(match.group())
        else:
            raise ValueError(f"Could not parse LLM response as JSON: {response[:200]}")

    if isinstance(data, dict) and "params" in data:
        return data["params"], data.get("rationale", "")
    elif isinstance(data, dict):
        return data, ""
    else:
        raise ValueError(f"Unexpected LLM response structure: {type(data)}")


async def run_tuning_loop(
    df, features: list[str], target: str, task_type: str, algorithm: str,
) -> dict:
    """Execute the tuning loop with proper error handling.

    No silent fallback - LLM failures are recorded explicitly.
    """
    algo_entry = ALL_ALGOS[task_type][algorithm]
    dataset_ctx = _build_dataset_context(df, features, target, task_type, algorithm)

    iterations = []
    best_score = -float("inf")
    best_params = {}
    best_metrics = {}
    history = []
    llm_calls = 0
    llm_failures = 0
    no_improve_count = 0
    stop_reason = None

    max_trials = TUNING_MAX_TRIALS
    max_llm_calls = max(MAX_TUNING_ITERATIONS, 10)
    patience = TUNING_PATIENCE

    for round_num in range(1, max_trials + 1):
        if llm_calls >= max_llm_calls:
            stop_reason = "max_llm_calls"
            break

        # === Call LLM for parameters ===
        mixer_prompt = _build_mixer_prompt(dataset_ctx, algorithm, task_type, history, algo_entry)
        new_params = None
        rationale = ""
        llm_error = None

        if is_llm_configured():
            try:
                llm_response, llm_meta = await call_llm(mixer_prompt)
                llm_calls += 1
                new_params, rationale = _parse_llm_response(llm_response)
            except Exception as e:
                llm_calls += 1
                llm_failures += 1
                llm_error = str(e)
        else:
            # No LLM configured - use defaults with slight variation
            import random
            defaults = algo_entry["defaults"].copy()
            new_params = {}
            for k, v in defaults.items():
                if isinstance(v, (int, float)):
                    new_params[k] = type(v)(v * random.uniform(0.8, 1.2))
                else:
                    new_params[k] = v
            rationale = "LLM 未配置，使用默认参数微调"

        # === Train model ===
        iteration = {
            "round": round_num,
            "params": new_params or {},
            "metrics": {},
            "score": None,
            "rationale": rationale,
            "status": "succeeded",
            "error": None,
        }

        if llm_error:
            iteration["status"] = "llm_failed"
            iteration["error"] = llm_error
            iteration["score"] = None
        elif new_params is not None:
            try:
                result = train_model(df, features, target, task_type, algorithm, new_params)
                metrics = result["metrics"]
                score = get_primary_score(metrics, task_type)

                iteration["metrics"] = metrics
                iteration["score"] = score
                iteration["params"] = result["params"]  # Use actual effective params

                # Update best
                if score is not None and score > best_score:
                    best_score = score
                    best_params = result["params"]
                    best_metrics = metrics
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                # Check threshold
                if score is not None and score >= QUALITY_THRESHOLD:
                    stop_reason = "target_reached"
                    iterations.append(iteration)
                    history.append(iteration)
                    break

            except Exception as e:
                iteration["status"] = "train_failed"
                iteration["error"] = str(e)
                iteration["score"] = None

        iterations.append(iteration)
        history.append(iteration)

        # Check patience
        if no_improve_count >= patience:
            stop_reason = "patience_exhausted"
            break

    reached = best_score >= QUALITY_THRESHOLD if best_score > -float("inf") else False

    # Build report
    report_lines = [
        f"=== ML-TEST 调参报告 ===",
        f"算法: {algorithm} ({task_type})",
        f"特征: {', '.join(features)}",
        f"目标: {target}",
        f"迭代次数: {len(iterations)}",
        f"LLM 调用: {llm_calls} 次 (失败: {llm_failures} 次)",
        f"最佳分数: {best_score:.4f}" if best_score > -float("inf") else "最佳分数: 无有效结果",
        f"目标阈值 ({QUALITY_THRESHOLD}): {'已达到' if reached else '未达到'}",
        f"停止原因: {stop_reason or '达到最大轮数'}",
        f"最佳参数: {json.dumps(best_params, indent=2, ensure_ascii=False)}",
        "",
        "迭代历史:",
    ]
    for it in iterations:
        score_str = f"{it['score']:.4f}" if it['score'] is not None else "N/A"
        report_lines.append(f"  第 {it['round']} 轮: 分数={score_str} | 状态={it['status']} | {json.dumps(it['params'], ensure_ascii=False)}")

    return {
        "iterations": iterations,
        "best_params": best_params,
        "best_score": round(best_score, 4) if best_score > -float("inf") else None,
        "reached_threshold": reached,
        "final_report": "\n".join(report_lines),
        "stop_reason": stop_reason,
        "llm_calls": llm_calls,
    }
