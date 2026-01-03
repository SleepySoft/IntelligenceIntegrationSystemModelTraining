import streamlit as st
import json
import os
import re
import pandas as pd
from collections import Counter

# ================= 配置 =================
# DATA_FILE = "Data/v1/result_ckpt100.jsonl"
DATA_FILE = "evaluation-20260103/result_ckpt360.jsonl"
REVIEWED_FILE = "eval_reviewed.jsonl"

st.set_page_config(layout="wide", page_title="Model Evaluation Tool - Advanced")


# --- 1. 核心解析与评估逻辑 (Core Logic) ---

def safe_parse_json(text):
    """
    尝试解析 JSON，如果失败返回 None。
    """
    if text is None:
        return None
    if isinstance(text, dict):
        return text

    text = str(text).strip()
    try:
        return json.loads(text)
    except:
        pass

    # 尝试提取 markdown json
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass

    # 尝试提取第一个 {}
    match = re.search(r'(\{.*\})', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass

    return None


def is_negative_sample(data_dict):
    """
    判断是否为负例（无价值情报）。
    逻辑：如果是 None，视为错误（不是负例）。
    如果 Key 只有 UUID，或者没有 RATE/EVENT_TEXT 字段，视为负例。
    """
    if not data_dict or not isinstance(data_dict, dict):
        return False

    # 逻辑：只有 UUID 或 显式为空
    keys = set(data_dict.keys())
    if keys == {'UUID'}:
        return True

    # 或者没有核心内容字段
    if 'RATE' not in data_dict and 'EVENT_TEXT' not in data_dict:
        return True

    return False


def extract_scores(rate_data):
    """
    解析 RATE 字典。
    返回:
    1. independent_scores: { '内容准确率': val, '规模及影响': val, '潜力及传承': val }
    2. primary_category: (Name, Score) - 除去上述三个key后的最高分项
    """
    if not isinstance(rate_data, dict):
        return {}, ("N/A", 0)

    independent_keys = {"内容准确率", "规模及影响", "潜力及传承"}

    # 提取独立分数
    independent_scores = {k: rate_data.get(k, 0) for k in independent_keys}

    # 提取主要维度
    candidates = {k: v for k, v in rate_data.items() if k not in independent_keys}

    if not candidates:
        return independent_scores, ("无有效领域", 0)

    best_category = max(candidates, key=candidates.get)
    best_score = candidates[best_category]

    return independent_scores, (best_category, best_score)


def evaluate_single_sample(gt_raw, pred_raw):
    """
    对单条数据进行自动化评估，返回评估结果对象
    """
    gt = safe_parse_json(gt_raw)
    pred = safe_parse_json(pred_raw)

    result = {
        "format_error": False,
        "uuid_missing": False,
        "classification": "Unknown",  # TP, TN, FP, FN
        "dim_match": None,  # 主要维度是否一致
        "score_deltas": {},  # 独立评分偏差
        "details": ""
    }

    # 1. 检查格式错误
    if pred is None:
        result["format_error"] = True
        return result

    if "UUID" not in pred:
        result["uuid_missing"] = True
        result["format_error"] = True  # 视作格式错误
        return result

    # 2. 检查正负例
    gt_is_neg = is_negative_sample(gt)
    pred_is_neg = is_negative_sample(pred)

    if not gt_is_neg and not pred_is_neg:
        result["classification"] = "TP"  # 都有内容
    elif gt_is_neg and pred_is_neg:
        result["classification"] = "TN"  # 都认为没内容
    elif gt_is_neg and not pred_is_neg:
        result["classification"] = "FP"  # GT无内容，模型编造了内容
    elif not gt_is_neg and pred_is_neg:
        result["classification"] = "FN"  # GT有内容，模型忽略了

    # 3. 如果是 TP (两者都有内容)，深入对比维度和分数
    if result["classification"] == "TP":
        gt_indep, (gt_cat, _) = extract_scores(gt.get("RATE", {}))
        pred_indep, (pred_cat, _) = extract_scores(pred.get("RATE", {}))

        # 维度对比
        result["dim_match"] = (gt_cat == pred_cat)
        result["gt_primary"] = gt_cat
        result["pred_primary"] = pred_cat

        # 分数对比 (Pred - GT)
        for k in gt_indep:
            result["score_deltas"][k] = pred_indep.get(k, 0) - gt_indep[k]

    return result


def calculate_global_metrics(data_list):
    """
    遍历所有数据，计算全局指标
    """
    stats = {
        "total": len(data_list),
        "format_errors": 0,
        "TP": 0, "TN": 0, "FP": 0, "FN": 0,
        "dim_match_count": 0,
        "tp_count_for_dim": 0,
        "score_mae": {"内容准确率": [], "规模及影响": [], "潜力及传承": []}
    }

    for item in data_list:
        eval_res = evaluate_single_sample(item.get('ground_truth'), item.get('model_output'))

        if eval_res["format_error"]:
            stats["format_errors"] += 1
            continue  # 格式错误不参与后续逻辑混淆矩阵计算

        cls = eval_res["classification"]
        stats[cls] += 1

        if cls == "TP":
            stats["tp_count_for_dim"] += 1
            if eval_res["dim_match"]:
                stats["dim_match_count"] += 1

            for k, delta in eval_res["score_deltas"].items():
                if k in stats["score_mae"]:
                    stats["score_mae"][k].append(abs(delta))

    return stats


# --- 2. Helper Functions (File IO) ---
def load_data():
    data = []
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data


def save_progress(index, label, comment, current_data):
    current_data[index]['human_label'] = label
    current_data[index]['comments'] = comment
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        for entry in current_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# --- 3. UI 渲染函数 ---

def render_metrics_sidebar(data):
    st.sidebar.title("📊 Auto Evaluation Stats")

    if not data:
        st.sidebar.warning("No Data Loaded")
        return

    stats = calculate_global_metrics(data)
    total_valid = stats["TP"] + stats["TN"] + stats["FP"] + stats["FN"]

    # 1. 格式错误率
    err_rate = (stats["format_errors"] / stats["total"]) * 100 if stats["total"] > 0 else 0
    st.sidebar.metric("JSON Format Error Rate", f"{err_rate:.1f}%", help="无法解析JSON或缺少UUID的比例")

    st.sidebar.divider()

    # 2. 混淆矩阵指标
    # Precision = TP / (TP + FP)
    precision = stats["TP"] / (stats["TP"] + stats["FP"]) if (stats["TP"] + stats["FP"]) > 0 else 0
    # Recall = TP / (TP + FN)
    recall = stats["TP"] / (stats["TP"] + stats["FN"]) if (stats["TP"] + stats["FN"]) > 0 else 0
    # Accuracy = (TP + TN) / Total Valid
    acc = (stats["TP"] + stats["TN"]) / total_valid if total_valid > 0 else 0

    c1, c2 = st.sidebar.columns(2)
    c1.metric("Precision", f"{precision:.2%}")
    c2.metric("Recall", f"{recall:.2%}")
    st.sidebar.metric("Classification Acc", f"{acc:.2%}", help="正确判断 '有价值' vs '无价值' 的准确率")

    st.sidebar.text(f"TP:{stats['TP']} | TN:{stats['TN']} | FP:{stats['FP']} | FN:{stats['FN']}")

    st.sidebar.divider()

    # 3. 维度与评分
    dim_acc = stats["dim_match_count"] / stats["tp_count_for_dim"] if stats["tp_count_for_dim"] > 0 else 0
    st.sidebar.metric("Primary Dimension Match", f"{dim_acc:.1f}%", help="在双方都认为有价值时，主要分类维度的一致性")

    st.sidebar.write("Score MAE (平均绝对误差):")
    for k, v_list in stats["score_mae"].items():
        avg_mae = sum(v_list) / len(v_list) if v_list else 0
        st.sidebar.caption(f"{k}: {avg_mae:.2f}")


def render_content_card(column, title, raw_data, style="default", compare_eval=None, is_gt=False):
    """
    Enhanced render function based on evaluation results.
    """
    data_dict = safe_parse_json(raw_data)

    with column:
        # 标题行
        header_cols = st.columns([3, 1])
        header_cols[0].markdown(f"### {title}")

        # 如果是模型输出且有错误，显示在这里
        if not is_gt and compare_eval:
            if compare_eval["format_error"]:
                header_cols[1].error("FORMAT ERR")
            elif compare_eval["classification"] == "FN":
                header_cols[1].error("MISSED (FN)")
            elif compare_eval["classification"] == "FP":
                header_cols[1].warning("NOISE (FP)")
            elif compare_eval["classification"] == "TN":
                header_cols[1].info("IGNORE (TN)")

        if data_dict is None:
            st.error("⚠️ JSON Parse Error")
            st.code(str(raw_data), language="text")
            return

        # 判断是否为负例 (仅 UUID)
        is_neg = is_negative_sample(data_dict)

        if is_neg:
            st.info(f"🚫 Negative Sample (No Value)\nUUID: {data_dict.get('UUID', 'Unknown')}")
        else:
            # 正常内容展示
            indep_scores, (prim_cat, prim_score) = extract_scores(data_dict.get("RATE", {}))

            # 颜色逻辑：如果维度不匹配，且当前是模型输出，且不是GT，显示醒目颜色
            cat_delta = None
            if not is_gt and compare_eval and compare_eval["classification"] == "TP":
                if not compare_eval["dim_match"]:
                    cat_delta = "MISMATCH"

            # 指标展示
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(label="主要维度", value=prim_cat, delta=cat_delta, delta_color="inverse")
            m2.metric(label="主分", value=prim_score)
            m3.metric(label="规模影响", value=indep_scores.get("规模及影响", 0))
            m4.metric(label="内容准确", value=indep_scores.get("内容准确率", 0))

            st.divider()

            display_text = data_dict.get("EVENT_TEXT", str(raw_data))
            if style == "success":
                st.success(display_text)
            elif style == "warning":
                st.warning(display_text)
            else:
                st.info(display_text)

        with st.expander("查看原始 JSON"):
            st.json(data_dict)


# --- Main App Logic ---
def main():
    st.title("🤖 LLM Evaluation: Auto-Metrics & Human Review")

    # 1. 初始化
    if 'data' not in st.session_state:
        st.session_state.data = load_data()

    if 'current_index' not in st.session_state:
        unreviewed_indices = [i for i, d in enumerate(st.session_state.data) if d.get('human_label') is None]
        st.session_state.current_index = unreviewed_indices[0] if unreviewed_indices else 0

    data = st.session_state.data

    # --- 渲染侧边栏统计 ---
    render_metrics_sidebar(data)

    # --- 主界面 ---
    idx = st.session_state.current_index
    total_count = len(data)

    # 顶部进度
    reviewed_count = sum(1 for d in data if d.get('human_label') is not None)
    st.progress(reviewed_count / total_count if total_count > 0 else 0)

    if idx < total_count:
        item = data[idx]

        # 实时计算当前条目的自动评估结果
        eval_result = evaluate_single_sample(item.get('ground_truth'), item.get('model_output'))

        st.subheader(f"Sample #{idx + 1} | Auto-Eval: {eval_result['classification']}")

        # 对比区
        col1, col2 = st.columns(2)

        # Ground Truth
        render_content_card(
            column=col1,
            title="✅ Ground Truth",
            raw_data=item.get('ground_truth', '{}'),
            style="success",
            is_gt=True
        )

        # Model Output
        render_content_card(
            column=col2,
            title="🤖 Model Output",
            raw_data=item.get('model_output', '{}'),
            style="warning",
            compare_eval=eval_result,  # 传入评估结果用于高亮差异
            is_gt=False
        )

        # --- 详细对比信息 (如果出错或不一致) ---
        if eval_result["format_error"]:
            st.error(f"❌ Critical Error: Model output format is invalid or missing UUID.")
        elif eval_result["classification"] == "TP" and not eval_result["dim_match"]:
            st.warning(
                f"⚠️ Dimension Mismatch: GT implies '{eval_result['gt_primary']}' but Model predicts '{eval_result['pred_primary']}'")
        elif eval_result["classification"] == "FN":
            st.error("⚠️ Recall Failure: Ground truth has valid info, model returned Negative.")
        elif eval_result["classification"] == "FP":
            st.warning("⚠️ Precision Failure: Ground truth is Negative, model hallucinated info.")

        # --- 操作区 ---
        st.divider()
        c1, c2, c3 = st.columns([1, 1, 4])

        with c1:
            if st.button("👍 Pass / Good", use_container_width=True, type="primary"):
                save_progress(idx, "pass", "", data)
                st.session_state.current_index += 1
                st.rerun()

        with c2:
            if st.button("👎 Fail / Bad", use_container_width=True):
                save_progress(idx, "fail", "", data)
                st.session_state.current_index += 1
                st.rerun()

        with c3:
            comment = st.text_input("Comments", key="comment_input", placeholder="e.g. Logic error, Wrong score...")
            if st.button("Submit Comment"):
                save_progress(idx, "commented", comment, data)
                st.session_state.current_index += 1
                st.rerun()

        # 导航
        st.divider()
        prev, center, next_btn = st.columns([1, 8, 1])
        if prev.button("Previous"):
            st.session_state.current_index = max(0, idx - 1)
            st.rerun()
        if next_btn.button("Next"):
            st.session_state.current_index = min(len(data) - 1, idx + 1)
            st.rerun()

        with st.expander("Show Instruction & Input"):
            st.info(f"**Instruction:** {item.get('instruction', '')}")
            st.text(f"**Input:** {item.get('input', '')}")

    else:
        st.balloons()
        st.success("🎉 All samples reviewed!")

        # 最终下载
        st.download_button(
            label="Download Reviewed Data",
            data=json.dumps(data, indent=2, ensure_ascii=False),
            file_name="reviewed_final.json",
            mime="application/json"
        )


if __name__ == "__main__":
    main()
