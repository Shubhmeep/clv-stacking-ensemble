"""
LangGraph Orchestrator for CLV Modeling (CKPT2/3/4/5).

Behavior:
  - Runs full pipeline under a LangGraph StateGraph
  - Gemini decides next steps and final model selection (validation only)
  - Test metrics computed only after selection
  - Output is minimal: step completion + iteration progress
"""

from __future__ import annotations

import os
import json
import io
import contextlib
from pathlib import Path
from typing import Dict, List, Literal, TypedDict

import numpy as np
import pandas as pd

from dotenv import load_dotenv

from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

from langchain_core.messages import SystemMessage, HumanMessage

from langchain_core.messages import SystemMessage, HumanMessage

from src.data import load_online_retail_ii, clean_data
from src.features import create_temporal_splits_multi, create_temporal_splits_multi_extended
from src.baselines import (
    train_elasticnet,
    train_random_forest,
    train_xgboost,
    train_extra_trees,
    train_hist_gb,
    train_poisson,
    train_knn,
    train_svr,
    train_mlp,
    evaluate_model,
)
from src.stacking import StackedEnsemble


# =============================================================================
# State Definition
# =============================================================================

class OrchestratorState(TypedDict, total=False):
    next_step: Literal[
        "temporal_splits_baseline",
        "train_baselines",
        "stacking_search_ckpt3",
        "decide_extended_features",
        "temporal_splits_extended",
        "train_extended_baselines",
        "stacking_search_ckpt4",
        "decide_final_model",
        "final_report",
        "end",
    ]
    df_raw: pd.DataFrame
    df_clean: pd.DataFrame
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    train_df_ext: pd.DataFrame
    val_df_ext: pd.DataFrame
    test_df_ext: pd.DataFrame
    feature_cols: List[str]
    feature_cols_ext: List[str]
    baseline_results_val: Dict[str, Dict[str, float]]
    stacking_results_val: Dict[str, Dict[str, float]]
    ext_baseline_results_val: Dict[str, Dict[str, float]]
    ext_stacking_results_val: Dict[str, Dict[str, float]]
    baseline_models: Dict[str, object]
    stack_models: Dict[str, object]
    ext_baseline_models: Dict[str, object]
    ext_stack_models: Dict[str, object]
    decision: Dict[str, str]
    report_tables: Dict[str, pd.DataFrame]
    report_summary: str
    file_2009_2010: str
    file_2010_2011: str


# =============================================================================
# Helpers
# =============================================================================

def _get_llm():
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        load_dotenv()
        if not os.getenv("GEMINI_API_KEY"):
            return None
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    except Exception:
        return None


SYSTEM_PROMPT = (
    "You are the CLV model orchestration agent. "
    "Use ONLY the provided validation metrics for decisions. "
    "Return STRICT JSON with the exact keys requested. "
    "No extra text."
)


def _llm_json(llm, user_prompt: str) -> Dict:
    try:
        resp = llm.invoke([SystemMessage(SYSTEM_PROMPT), HumanMessage(user_prompt)])
        return json.loads(resp.content.strip())
    except Exception:
        return {}


SYSTEM_PROMPT = (
    "You are the CLV model orchestration agent. "
    "You must ONLY use the provided validation metrics to make decisions. "
    "Return STRICT JSON with the exact keys requested. "
    "Do not include any extra text."
)


def _llm_json(llm, user_prompt: str) -> Dict:
    """Invoke LLM with a strict system prompt and parse JSON."""
    try:
        resp = llm.invoke([SystemMessage(SYSTEM_PROMPT), HumanMessage(user_prompt)])
        return json.loads(resp.content.strip())
    except Exception:
        return {}


def _build_features(df: pd.DataFrame) -> List[str]:
    exclude = {"customer_id", "CustomerID", "target", "cutoff_date", "horizon_start", "horizon_end"}
    return [c for c in df.columns if c not in exclude]


def _safe_mae_dict(results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    return {k: float(v.get("MAE", np.inf)) for k, v in results.items()}


@contextlib.contextmanager
def _silent():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        yield


def _log_step(step: str):
    print(f"[Step complete] {step}")


@contextlib.contextmanager
def _silent():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        yield


def _log_step(step: str):
    print(f"[Step complete] {step}")


def _stacking_configs(has_xgb: bool) -> List[Dict[str, object]]:
    configs = [
        {"name": "EN_Config_A", "meta": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42), "use_features": False},
        {"name": "EN_Config_B", "meta": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42), "use_features": True},
        {"name": "RF_Config_B", "meta": RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1), "use_features": True},
        {"name": "HGB_Config_B", "meta": HistGradientBoostingRegressor(max_depth=6, learning_rate=0.1, random_state=42), "use_features": True},
    ]
    if has_xgb:
        try:
            import xgboost as xgb
            configs.append({
                "name": "XGB_Config_B",
                "meta": xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=-1),
                "use_features": True
            })
        except Exception:
            pass
    return configs[:5]


def _select_config_names(llm, candidates: List[str], max_k: int, context: str) -> List[str]:
    if llm is None:
        return candidates[:max_k]
    prompt = (
        "Select the most promising stacking configs to try. "
        "Choose up to max_k items from the candidate list. "
        "Return JSON: {\"chosen\": [..]}.\n\n"
        f"max_k: {max_k}\n"
        f"candidates: {candidates}\n"
        f"context: {context}\n"
    )
    parsed = _llm_json(llm, prompt)
    chosen = parsed.get("chosen", [])
    if isinstance(chosen, list):
        chosen = [c for c in chosen if c in candidates]
    if not chosen:
        return candidates[:max_k]
    return chosen[:max_k]


def _select_config_names(llm, candidates: List[str], max_k: int, context: str) -> List[str]:
    """Let LLM choose up to max_k configs; fallback to first max_k."""
    if llm is None:
        return candidates[:max_k]

    prompt = (
        "Select the most promising stacking configs to try. "
        "Choose up to max_k items from the candidate list. "
        "Return JSON: {\"chosen\": [..]}.\n\n"
        f"max_k: {max_k}\n"
        f"candidates: {candidates}\n"
        f"context: {context}\n"
    )
    parsed = _llm_json(llm, prompt)
    chosen = parsed.get("chosen", [])
    if isinstance(chosen, list):
        chosen = [c for c in chosen if c in candidates]
    if not chosen:
        return candidates[:max_k]
    return chosen[:max_k]


# =============================================================================
# Nodes
# =============================================================================

def load_and_clean_data(state: OrchestratorState) -> OrchestratorState:
    repo_root = Path(__file__).resolve().parents[1]
    default_2009 = repo_root / "data" / "Year 2009-2010.csv"
    default_2010 = repo_root / "data" / "Year 2010-2011.csv"

    file_2009_2010 = state.get("file_2009_2010", str(default_2009))
    file_2010_2011 = state.get("file_2010_2011", str(default_2010))

    if not Path(file_2009_2010).exists() or not Path(file_2010_2011).exists():
        raise FileNotFoundError(
            "Dataset files not found. Expected:\n"
            f"  {file_2009_2010}\n"
            f"  {file_2010_2011}\n"
            "Set state['file_2009_2010'] and state['file_2010_2011'] to override."
        )

    with _silent():
        df_raw = load_online_retail_ii(file_2009_2010, file_2010_2011)
        df_clean = clean_data(df_raw)

    state.update({"df_raw": df_raw, "df_clean": df_clean, "next_step": "temporal_splits_baseline"})
    _log_step("load_and_clean_data")
    return state


def temporal_splits_baseline(state: OrchestratorState) -> OrchestratorState:
    train_cutoffs = ["2010-06-01", "2010-09-01", "2010-12-01", "2011-03-01"]
    val_cutoff = "2011-06-01"
    test_cutoff = "2011-09-01"

    with _silent():
        train_df, val_df, test_df = create_temporal_splits_multi(
            state["df_clean"],
            train_cutoffs=train_cutoffs,
            val_cutoff=val_cutoff,
            test_cutoff=test_cutoff,
            obs_months=6,
            horizon_months=3,
        )

    state.update(
        {
            "train_df": train_df,
            "val_df": val_df,
            "test_df": test_df,
            "feature_cols": _build_features(train_df),
            "next_step": "train_baselines",
        }
    )
    _log_step("temporal_splits_baseline")
    return state


def train_baselines(state: OrchestratorState) -> OrchestratorState:
    train_df = state["train_df"]
    val_df = state["val_df"]
    feature_cols = state["feature_cols"]

    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_val = val_df[feature_cols]
    y_val = val_df["target"]

    results_val = {}
    baseline_models = {}

    model_en = train_elasticnet(X_train, y_train)
    baseline_models["ElasticNet"] = model_en
    results_val["ElasticNet"] = evaluate_model(y_val, model_en.predict(X_val), "ElasticNet")

    model_rf = train_random_forest(X_train, y_train)
    baseline_models["RandomForest"] = model_rf
    results_val["RandomForest"] = evaluate_model(y_val, model_rf.predict(X_val), "RandomForest")

    model_xgb = train_xgboost(X_train, y_train)
    if model_xgb is not None:
        baseline_models["XGBoost"] = model_xgb
        pred_xgb_val = model_xgb.predict(X_val)
        results_val["XGBoost"] = evaluate_model(y_val, pred_xgb_val, "XGBoost")
        pred_avg_val = (model_en.predict(X_val) + model_rf.predict(X_val) + pred_xgb_val) / 3
        results_val["SimpleAvg"] = evaluate_model(y_val, pred_avg_val, "SimpleAvg")

    model_et = train_extra_trees(X_train, y_train)
    baseline_models["ExtraTrees"] = model_et
    results_val["ExtraTrees"] = evaluate_model(y_val, model_et.predict(X_val), "ExtraTrees")

    model_hgb = train_hist_gb(X_train, y_train)
    baseline_models["HistGB"] = model_hgb
    results_val["HistGB"] = evaluate_model(y_val, model_hgb.predict(X_val), "HistGB")

    model_pr = train_poisson(X_train, y_train)
    baseline_models["Poisson"] = model_pr
    results_val["Poisson"] = evaluate_model(y_val, model_pr.predict(X_val), "Poisson")

    model_knn = train_knn(X_train, y_train)
    baseline_models["KNN"] = model_knn
    results_val["KNN"] = evaluate_model(y_val, model_knn.predict(X_val), "KNN")

    model_svr = train_svr(X_train, y_train)
    baseline_models["SVR"] = model_svr
    results_val["SVR"] = evaluate_model(y_val, model_svr.predict(X_val), "SVR")

    model_mlp = train_mlp(X_train, y_train, use_log_target=False)
    baseline_models["MLP"] = model_mlp
    results_val["MLP"] = evaluate_model(y_val, model_mlp.predict(X_val), "MLP")

    state.update(
        {
            "baseline_results_val": results_val,
            "baseline_models": baseline_models,
            "next_step": "stacking_search_ckpt3",
        }
    )
    _log_step("train_baselines")
    return state


def stacking_search_ckpt3(state: OrchestratorState) -> OrchestratorState:
    train_df = state["train_df"].sort_values("cutoff_date").reset_index(drop=True)
    val_df = state["val_df"]
    feature_cols = state["feature_cols"]

    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_val = val_df[feature_cols]
    y_val = val_df["target"]

    base_models = {
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1),
    }
    has_xgb = False
    try:
        import xgboost as xgb
        base_models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1
        )
        has_xgb = True
    except Exception:
        pass

    configs = _stacking_configs(has_xgb)
    llm = _get_llm()
    candidate_names = [c["name"] for c in configs]
    chosen_names = _select_config_names(
        llm,
        candidate_names,
        max_k=min(5, len(candidate_names)),
        context="CKPT3 stacking search (baseline features)",
    )
    name_to_cfg = {c["name"]: c for c in configs}
    configs = [name_to_cfg[n] for n in chosen_names if n in name_to_cfg]
    results_val = {}
    stack_models = {}

    for i, cfg in enumerate(configs, 1):
        print(f"[Iteration {i}/{len(configs)}] Trying {cfg['name']} ...")
        stack = StackedEnsemble(meta_learner=cfg["meta"], use_features=cfg["use_features"], n_folds=5)
        with _silent():
            oof = stack.generate_oof_predictions(X_train, y_train, base_models)
            stack.train(X_train, y_train, oof, save_dir=None, save_models=False)
            stack.train_base_models_final(X_train, y_train, base_models)
        pred_val, _ = stack.predict(X_val)
        res = evaluate_model(y_val, pred_val, cfg["name"])
        results_val[cfg["name"]] = res
        stack_models[cfg["name"]] = stack
        print(f"[Iteration {i}/{len(configs)}] {cfg['name']} complete (Val MAE={res['MAE']:.4f})")

    state.update(
        {
            "stacking_results_val": results_val,
            "stack_models": stack_models,
            "next_step": "decide_extended_features",
        }
    )
    _log_step("stacking_search_ckpt3")
    return state


def decide_extended_features(state: OrchestratorState) -> OrchestratorState:
    baseline_mae = _safe_mae_dict(state["baseline_results_val"])
    stack_mae = _safe_mae_dict(state["stacking_results_val"])

    decision = {
        "best_baseline": min(baseline_mae, key=baseline_mae.get),
        "best_stack": min(stack_mae, key=stack_mae.get),
        "next_step": "temporal_splits_extended",
        "reason": "Default: proceed to extended features",
    }

    llm = _get_llm()
    if llm is not None:
        prompt = (
            "Decide if we should proceed to extended features based ONLY on validation MAE. "
            "Return JSON with keys: next_step, reason. "
            "Valid next_step: temporal_splits_extended, final_report.\n\n"
            f"Baseline MAE (val): {json.dumps(baseline_mae)}\n"
            f"Stacking MAE (val): {json.dumps(stack_mae)}\n"
        )
        parsed = _llm_json(llm, prompt)
        if parsed.get("next_step") in {"temporal_splits_extended", "final_report"}:
            decision["next_step"] = parsed["next_step"]
            decision["reason"] = parsed.get("reason", "")

    state.update({"decision": decision, "next_step": decision["next_step"]})
    _log_step("decide_extended_features")
    return state


def temporal_splits_extended(state: OrchestratorState) -> OrchestratorState:
    train_cutoffs = ["2010-06-01", "2010-09-01", "2010-12-01", "2011-03-01"]
    val_cutoff = "2011-06-01"
    test_cutoff = "2011-09-01"

    with _silent():
        train_df_ext, val_df_ext, test_df_ext = create_temporal_splits_multi_extended(
            state["df_clean"],
            train_cutoffs=train_cutoffs,
            val_cutoff=val_cutoff,
            test_cutoff=test_cutoff,
            obs_months=6,
            horizon_months=3,
        )

    state.update(
        {
            "train_df_ext": train_df_ext,
            "val_df_ext": val_df_ext,
            "test_df_ext": test_df_ext,
            "feature_cols_ext": _build_features(train_df_ext),
            "next_step": "train_extended_baselines",
        }
    )
    _log_step("temporal_splits_extended")
    return state


def train_extended_baselines(state: OrchestratorState) -> OrchestratorState:
    train_df = state["train_df_ext"]
    val_df = state["val_df_ext"]
    feature_cols = state["feature_cols_ext"]

    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_val = val_df[feature_cols]
    y_val = val_df["target"]

    results_val = {}
    ext_models = {}

    model_en = train_elasticnet(X_train, y_train)
    ext_models["ElasticNet_ext"] = model_en
    results_val["ElasticNet_ext"] = evaluate_model(y_val, model_en.predict(X_val), "ElasticNet_ext")

    model_rf = train_random_forest(X_train, y_train)
    ext_models["RandomForest_ext"] = model_rf
    results_val["RandomForest_ext"] = evaluate_model(y_val, model_rf.predict(X_val), "RandomForest_ext")

    model_xgb = train_xgboost(X_train, y_train)
    if model_xgb is not None:
        ext_models["XGBoost_ext"] = model_xgb
        pred_xgb_val = model_xgb.predict(X_val)
        results_val["XGBoost_ext"] = evaluate_model(y_val, pred_xgb_val, "XGBoost_ext")
        pred_avg_val = (model_en.predict(X_val) + model_rf.predict(X_val) + pred_xgb_val) / 3
        results_val["SimpleAvg_ext"] = evaluate_model(y_val, pred_avg_val, "SimpleAvg_ext")

    model_et = train_extra_trees(X_train, y_train)
    ext_models["ExtraTrees_ext"] = model_et
    results_val["ExtraTrees_ext"] = evaluate_model(y_val, model_et.predict(X_val), "ExtraTrees_ext")

    model_hgb = train_hist_gb(X_train, y_train)
    ext_models["HistGB_ext"] = model_hgb
    results_val["HistGB_ext"] = evaluate_model(y_val, model_hgb.predict(X_val), "HistGB_ext")

    model_pr = train_poisson(X_train, y_train)
    ext_models["Poisson_ext"] = model_pr
    results_val["Poisson_ext"] = evaluate_model(y_val, model_pr.predict(X_val), "Poisson_ext")

    model_knn = train_knn(X_train, y_train)
    ext_models["KNN_ext"] = model_knn
    results_val["KNN_ext"] = evaluate_model(y_val, model_knn.predict(X_val), "KNN_ext")

    model_svr = train_svr(X_train, y_train)
    ext_models["SVR_ext"] = model_svr
    results_val["SVR_ext"] = evaluate_model(y_val, model_svr.predict(X_val), "SVR_ext")

    model_mlp = train_mlp(X_train, y_train, use_log_target=False)
    ext_models["MLP_ext"] = model_mlp
    results_val["MLP_ext"] = evaluate_model(y_val, model_mlp.predict(X_val), "MLP_ext")

    state.update(
        {
            "ext_baseline_results_val": results_val,
            "ext_baseline_models": ext_models,
            "next_step": "stacking_search_ckpt4",
        }
    )
    _log_step("train_extended_baselines")
    return state


def stacking_search_ckpt4(state: OrchestratorState) -> OrchestratorState:
    train_df = state["train_df_ext"].sort_values("cutoff_date").reset_index(drop=True)
    val_df = state["val_df_ext"]
    feature_cols = state["feature_cols_ext"]

    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_val = val_df[feature_cols]
    y_val = val_df["target"]

    base_models = {
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1),
    }
    has_xgb = False
    try:
        import xgboost as xgb
        base_models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1
        )
        has_xgb = True
    except Exception:
        pass

    configs = _stacking_configs(has_xgb)
    llm = _get_llm()
    candidate_names = [c["name"] for c in configs]
    chosen_names = _select_config_names(
        llm,
        candidate_names,
        max_k=min(5, len(candidate_names)),
        context="CKPT4 stacking search (extended features)",
    )
    name_to_cfg = {c["name"]: c for c in configs}
    configs = [name_to_cfg[n] for n in chosen_names if n in name_to_cfg]
    results_val = {}
    ext_stack_models = {}

    for i, cfg in enumerate(configs, 1):
        print(f"[Iteration {i}/{len(configs)}] Trying {cfg['name']} (extended) ...")
        stack = StackedEnsemble(meta_learner=cfg["meta"], use_features=cfg["use_features"], n_folds=5)
        with _silent():
            oof = stack.generate_oof_predictions(X_train, y_train, base_models)
            stack.train(X_train, y_train, oof, save_dir=None, save_models=False)
            stack.train_base_models_final(X_train, y_train, base_models)
        pred_val, _ = stack.predict(X_val)
        res = evaluate_model(y_val, pred_val, f"{cfg['name']}_ext")
        results_val[f"{cfg['name']}_ext"] = res
        ext_stack_models[f"{cfg['name']}_ext"] = stack
        print(f"[Iteration {i}/{len(configs)}] {cfg['name']}_ext complete (Val MAE={res['MAE']:.4f})")

    state.update(
        {
            "ext_stacking_results_val": results_val,
            "ext_stack_models": ext_stack_models,
            "next_step": "decide_final_model",
        }
    )
    _log_step("stacking_search_ckpt4")
    return state


def decide_final_model(state: OrchestratorState) -> OrchestratorState:
    val_metrics = {}
    val_metrics.update({f"Baseline:{k}": v["MAE"] for k, v in state["baseline_results_val"].items()})
    val_metrics.update({f"Stack:{k}": v["MAE"] for k, v in state["stacking_results_val"].items()})
    if state.get("ext_baseline_results_val"):
        val_metrics.update({f"ExtBaseline:{k}": v["MAE"] for k, v in state["ext_baseline_results_val"].items()})
    if state.get("ext_stacking_results_val"):
        val_metrics.update({f"ExtStack:{k}": v["MAE"] for k, v in state["ext_stacking_results_val"].items()})

    best = min(val_metrics, key=val_metrics.get)
    decision = state.get("decision", {})
    decision["final_model"] = best
    decision["final_reason"] = "Default: lowest validation MAE"

    llm = _get_llm()
    if llm is not None:
        prompt = (
            "Select the best model based ONLY on validation MAE. "
            "Return JSON with keys: final_model, final_reason. "
            "Choose one key from the provided metrics.\n\n"
            f"Validation MAE: {json.dumps(val_metrics)}\n"
        )
        parsed = _llm_json(llm, prompt)
        if parsed.get("final_model") in val_metrics:
            decision["final_model"] = parsed["final_model"]
            decision["final_reason"] = parsed.get("final_reason", "")

    state.update({"decision": decision, "next_step": "final_report"})
    _log_step("decide_final_model")
    return state


def final_report(state: OrchestratorState) -> OrchestratorState:
    tables = {}
    decision = state.get("decision", {})

    # CKPT2 test metrics (after selection)
    test_results_baseline = {}
    test_df = state["test_df"]
    feature_cols = state["feature_cols"]
    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    for name, model in state["baseline_models"].items():
        preds = model.predict(X_test)
        test_results_baseline[name] = evaluate_model(y_test, preds, name)

    if {"ElasticNet", "RandomForest", "XGBoost"}.issubset(state["baseline_models"].keys()):
        en = state["baseline_models"]["ElasticNet"]
        rf = state["baseline_models"]["RandomForest"]
        xgb = state["baseline_models"]["XGBoost"]
        pred_avg = (en.predict(X_test) + rf.predict(X_test) + xgb.predict(X_test)) / 3
        test_results_baseline["SimpleAvg"] = evaluate_model(y_test, pred_avg, "SimpleAvg")

    tables["ckpt2_baselines"] = pd.DataFrame(
        [{"Model": k, "MAE": v["MAE"], "RMSE": v["RMSE"]} for k, v in test_results_baseline.items()]
    ).sort_values("MAE")

    # CKPT3 test metrics for all stacking iterations
    ckpt3_rows = []
    for name, stack in state["stack_models"].items():
        preds, _ = stack.predict(X_test)
        res = evaluate_model(y_test, preds, name)
        ckpt3_rows.append({"Model": name, "MAE": res["MAE"], "RMSE": res["RMSE"]})
    tables["ckpt3_stacking"] = pd.DataFrame(ckpt3_rows).sort_values("MAE")

    # CKPT4 test metrics if extended was run
    if state.get("ext_baseline_models"):
        test_results_ext = {}
        test_df_ext = state["test_df_ext"]
        feature_cols_ext = state["feature_cols_ext"]
        X_test_ext = test_df_ext[feature_cols_ext]
        y_test_ext = test_df_ext["target"]

        for name, model in state["ext_baseline_models"].items():
            preds = model.predict(X_test_ext)
            test_results_ext[name] = evaluate_model(y_test_ext, preds, name)

        if {"ElasticNet_ext", "RandomForest_ext", "XGBoost_ext"}.issubset(state["ext_baseline_models"].keys()):
            en = state["ext_baseline_models"]["ElasticNet_ext"]
            rf = state["ext_baseline_models"]["RandomForest_ext"]
            xgb = state["ext_baseline_models"]["XGBoost_ext"]
            pred_avg = (en.predict(X_test_ext) + rf.predict(X_test_ext) + xgb.predict(X_test_ext)) / 3
            test_results_ext["SimpleAvg_ext"] = evaluate_model(y_test_ext, pred_avg, "SimpleAvg_ext")

        tables["ckpt4_baselines"] = pd.DataFrame(
            [{"Model": k, "MAE": v["MAE"], "RMSE": v["RMSE"]} for k, v in test_results_ext.items()]
        ).sort_values("MAE")

        if state.get("ext_stack_models"):
            rows = []
            for name, model in state["ext_stack_models"].items():
                pred_test, _ = model.predict(X_test_ext)
                res = evaluate_model(y_test_ext, pred_test, name)
                rows.append({"Model": name, "MAE": res["MAE"], "RMSE": res["RMSE"]})
            tables["ckpt4_stacking"] = pd.DataFrame(rows).sort_values("MAE")

    best_ckpt2 = tables["ckpt2_baselines"].iloc[0]
    summary = f"Best CKPT2 baseline: {best_ckpt2['Model']} MAE={best_ckpt2['MAE']:.6f}"
    if "ckpt4_baselines" in tables:
        best_ckpt4 = tables["ckpt4_baselines"].iloc[0]
        delta = best_ckpt4["MAE"] - best_ckpt2["MAE"]
        sign = "+" if delta >= 0 else ""
        summary += f" | Best CKPT4 baseline: {best_ckpt4['Model']} MAE={best_ckpt4['MAE']:.6f} (Delta {sign}{delta:.6f})"
    if decision.get("final_model"):
        summary += f" | Final selection (val): {decision.get('final_model')} ({decision.get('final_reason','')})"

    state.update({"report_tables": tables, "report_summary": summary, "next_step": "end"})
    _log_step("final_report_ready")
    return state


# =============================================================================
# Graph Construction
# =============================================================================

def _route_next(state: OrchestratorState):
    return state["next_step"]


class LangGraphCLVOrchestrator:
    """End-to-end orchestrator using LangGraph StateGraph."""

    def __init__(self):
        from langgraph.graph import StateGraph, END

        graph = StateGraph(OrchestratorState)
        graph.add_node("load_and_clean_data", load_and_clean_data)
        graph.add_node("temporal_splits_baseline", temporal_splits_baseline)
        graph.add_node("train_baselines", train_baselines)
        graph.add_node("stacking_search_ckpt3", stacking_search_ckpt3)
        graph.add_node("decide_extended_features", decide_extended_features)
        graph.add_node("temporal_splits_extended", temporal_splits_extended)
        graph.add_node("train_extended_baselines", train_extended_baselines)
        graph.add_node("stacking_search_ckpt4", stacking_search_ckpt4)
        graph.add_node("decide_final_model", decide_final_model)
        graph.add_node("final_report", final_report)

        graph.set_entry_point("load_and_clean_data")

        graph.add_edge("load_and_clean_data", "temporal_splits_baseline")
        graph.add_edge("temporal_splits_baseline", "train_baselines")
        graph.add_edge("train_baselines", "stacking_search_ckpt3")
        graph.add_edge("stacking_search_ckpt3", "decide_extended_features")

        graph.add_conditional_edges(
            "decide_extended_features",
            _route_next,
            {"temporal_splits_extended": "temporal_splits_extended", "final_report": "decide_final_model"},
        )

        graph.add_edge("temporal_splits_extended", "train_extended_baselines")
        graph.add_edge("train_extended_baselines", "stacking_search_ckpt4")
        graph.add_edge("stacking_search_ckpt4", "decide_final_model")
        graph.add_edge("decide_final_model", "final_report")

        graph.add_conditional_edges("final_report", _route_next, {"end": END})

        self.graph = graph.compile()

    def run(self) -> OrchestratorState:
        initial_state: OrchestratorState = {"next_step": "temporal_splits_baseline"}
        return self.graph.invoke(initial_state)

    def get_graph(self):
        return self.graph


__all__ = ["LangGraphCLVOrchestrator", "OrchestratorState"]
