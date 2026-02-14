# -*- coding: utf-8 -*-
import warnings
import os
import logging

# 1. Suppress noisy library warnings for a cleaner UI
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Specific suppression for the Streamlit/Plotly collision warning until Streamlit resolves the 'width' kwarg conflict
warnings.filterwarnings("ignore", message=".*use_container_width.*")
try:
    from urllib3.exceptions import NotOpenSSLWarning

    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except ImportError:
    pass
os.environ["WANDB_SILENT"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MPLBACKEND"] = "Agg"

# Initialize global dashboard logging
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename=os.path.join(log_dir, "dashboard.log"),
    filemode="a",
)
logger = logging.getLogger("fishy.dashboard")

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import re
from pathlib import Path
import sys
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_fscore_support,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap
except ImportError:
    umap = None
try:
    import paramiko
except ImportError:
    paramiko = None

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fishy._core.config import TrainingConfig
from fishy._core.config_loader import load_config
from fishy.data.module import create_data_module
from fishy import get_data_path
from fishy.cli.main import detect_method
from fishy.experiments.deep_training import ModelTrainer
from fishy.experiments.classic_training import SklearnTrainer
from fishy.analysis.xai import GradCAM, ModelWrapper
from fishy._core.utils import get_device
from lime.lime_tabular import LimeTabularExplainer
from fishy.analysis.statistical import summarize_results

st.set_page_config(
    page_title="Fishy Business | Spectral Analysis", page_icon="🐟", layout="wide"
)
st.markdown(
    """<style>
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    [data-testid="stMetric"] { background-color: #ffffff; padding: 10px; border-radius: 10px; }
    [data-testid="stMetricValue"] > div, [data-testid="stMetricLabel"] p, [data-testid="stMetricLabel"] div, [data-testid="stMetricLabel"] span { color: #1f2937 !important; }
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_data
def get_metadata():
    models = load_config("models")
    datasets = load_config("datasets")
    all_model_names = []
    for section in [
        "deep_models",
        "classic_models",
        "evolutionary_models",
        "probabilistic_models",
    ]:
        all_model_names.extend(list(models.get(section, {}).keys()))
    return sorted(list(set(all_model_names))), sorted(list(datasets.keys()))


def crawl_local_results():
    results = []
    output_root = Path("outputs")
    if not output_root.exists():
        return pd.DataFrame()
    for metric_file in output_root.glob("**/results/metrics.json"):
        try:
            with open(metric_file, "r") as f:
                data = json.load(f)
            # Find statistical or aggregated results
            summary_file = metric_file.parent.parent / "statistical_analysis.csv"
            if summary_file.exists():
                sdf = pd.read_csv(summary_file)
                results.append(sdf)
            else:
                # Add single run fallback
                parts = metric_file.parts
                # Find outputs index
                try:
                    out_idx = parts.index("outputs")
                    dataset = parts[out_idx+1]
                    method = parts[out_idx+3].split("_")[0]
                except:
                    dataset = "unknown"
                    method = "unknown"
                    
                results.append(
                    pd.DataFrame(
                        [
                            {
                                "Dataset": dataset,
                                "Method": method,
                                "Test": data.get("val_balanced_accuracy", 0),
                                "Train": data.get("train_balanced_accuracy", 0),
                            }
                        ]
                    )
                )
        except:
            continue
    return pd.concat(results).drop_duplicates() if results else pd.DataFrame()


def fetch_remote_data(
    host, port, username, password, remote_path, jump_host=None, jump_user=None, otp=None
):
    import io
    import tarfile
    import json

    results_map = {}
    aggregated_summaries = []
    
    try:
        target_client = paramiko.SSHClient()
        target_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        if jump_host:
            # Multi-hop via ProxyJump
            jump_client = paramiko.SSHClient()
            jump_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            # ECS MFA: Password + OTP. Further increased timeout for 1,920+ folders.
            jump_client.connect(
                jump_host, 
                username=jump_user, 
                password=f"{password}{otp}", 
                timeout=300, 
                banner_timeout=60
            )
            transport = jump_client.get_transport()
            transport.set_keepalive(30)
            dest_addr = (host, port)
            local_addr = ("127.0.0.1", 0)
            channel = transport.open_channel("direct-tcpip", dest_addr, local_addr)
            target_client.connect(
                host, 
                username=username, 
                password=password, 
                sock=channel, 
                timeout=300,
                banner_timeout=60
            )
        else:
            target_client.connect(
                host, 
                port=port, 
                username=username, 
                password=password, 
                timeout=300, 
                banner_timeout=60
            )

        # Ensure the transport doesn't time out during the long find/tar operation
        # 300s (5 mins) should be plenty for 1,920 folders.
        target_client.get_transport().set_timeout(300)

        # Optimization: Use a single tar pipe to fetch all result files at once.
        # This avoids the O(N) overhead of SFTP open/read calls for 1000s of files.
        # We search for summary.json (batch) and metrics.json (single runs).
        find_cmd = f"find {remote_path} \( -name 'summary.json' -o -name 'metrics.json' \) -print0 | tar -czf - --null -T -"
        stdin, stdout, stderr = target_client.exec_command(find_cmd)
        
        # Read the entire tarball into memory
        tar_data = stdout.read()
        err_data = stderr.read().decode()
        
        if err_data:
            logger.warning(f"Remote command stderr: {err_data}")
        
        if tar_data:
            with tarfile.open(fileobj=io.BytesIO(tar_data), mode="r:gz") as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    
                    f = tar.extractfile(member)
                    if not f:
                        continue
                        
                    try:
                        data = json.load(f)
                        if "summary.json" in member.name:
                            # Already summarized batch results
                            aggregated_summaries.append(pd.DataFrame(data))
                        elif "metrics.json" in member.name:
                            # Raw metrics from a single run - needs grouping for summarize_results
                            # Path pattern: .../outputs/{dataset}/{method}_{timestamp}/results/metrics.json
                            parts = member.name.split("/")
                            try:
                                # Look for 'outputs' and extract metadata
                                out_idx = -1
                                for i, p in enumerate(parts):
                                    if p == "outputs":
                                        out_idx = i
                                        break
                                
                                if out_idx != -1 and len(parts) > out_idx + 2:
                                    dataset = parts[out_idx + 1]
                                    method = parts[out_idx + 2].split("_")[0]
                                    key = f"{dataset}|||{method}"
                                    if key not in results_map:
                                        results_map[key] = []
                                    results_map[key].append(data)
                            except:
                                continue
                    except Exception as e:
                        logger.warning(f"Failed to parse {member.name}: {e}")
                        continue

        target_client.close()
        if jump_host:
            jump_client.close()
            
    except Exception as e:
        st.error(f"Network Error: {e}")
        logger.error(f"Remote fetch failed: {e}", exc_info=True)

    # Combine re-summarized raw results with existing batch summaries
    final_df = summarize_results(results_map) if results_map else pd.DataFrame()
    if aggregated_summaries:
        batch_df = pd.concat(aggregated_summaries, ignore_index=True)
        final_df = pd.concat([final_df, batch_df], ignore_index=True).drop_duplicates()
        
    return final_df


def fetch_wandb_data(entity, project, api_key=None):
    """Fetches results directly from W&B API and performs statistical analysis."""
    import wandb
    import os
    
    try:
        if api_key:
            os.environ["WANDB_API_KEY"] = api_key
            
        api = wandb.Api()
        # Optimization: Filter for finished runs on the server side to reduce data transfer
        # and GraphQL overhead.
        runs = api.runs(f"{entity}/{project}", filters={"state": "finished"})
        
        results_map = {}
        raw_rows = []
        found_oil = False
        
        # We need a count for the progress bar, but getting exact count might be slow
        # so we'll just show a "Processing..." message with a counter.
        progress_text = "Fetching and processing finished runs from W&B..."
        bar = st.progress(0, text=progress_text)
        
        # Use a list to avoid iterator issues if needed
        all_runs = list(runs)
        total_runs = len(all_runs)
        
        for i, run in enumerate(all_runs):
            if i % 10 == 0:
                bar.progress((i + 1) / total_runs, text=f"{progress_text} ({i+1}/{total_runs})")
                
            # Extract configuration
            ds = run.config.get("dataset")
            model = run.config.get("model")
            
            if not ds or not model:
                continue
            
            if ds == "oil":
                found_oil = True
                
            # Extract metrics
            train_acc = run.summary.get("train_balanced_accuracy", run.summary.get("train_accuracy", 0))
            val_acc = run.summary.get("val_balanced_accuracy", run.summary.get("accuracy", 0))
            f1 = run.summary.get("val_f1", run.summary.get("f1", 0))
            runtime = run.summary.get("_runtime", run.summary.get("total_training_time_s", 0))
                
            key = f"{ds}|||{model}"
            if key not in results_map:
                results_map[key] = []
            
            res_entry = {
                "train_balanced_accuracy": train_acc,
                "val_balanced_accuracy": val_acc,
                "f1": f1,
                "runtime": runtime
            }
            results_map[key].append(res_entry)
            raw_rows.append({
                "Dataset": ds,
                "Method": model,
                "Test Accuracy": val_acc,
                "Train Accuracy": train_acc,
                "F1 Score": f1,
                "Runtime (s)": runtime
            })
            
        bar.empty()
        
        if not results_map:
            st.warning(f"No valid finished runs found in {entity}/{project}")
            return pd.DataFrame()
            
        if not found_oil:
            st.info("Note: No runs with dataset='oil' found in W&B project summary.")
        else:
            st.success(f"Successfully found 'oil' results ({len([v for k,v in results_map.items() if 'oil' in k])} runs) in W&B!")

        st.session_state["raw_results_df"] = pd.DataFrame(raw_rows)
        summary_df = summarize_results(results_map, baseline_model="opls-da")
        
        # Display pretty Rich table in logs
        from fishy.analysis.statistical import display_statistical_summary
        display_statistical_summary(summary_df)
        
        return summary_df
    except Exception as e:
        st.error(f"W&B API Error: {e}")
        logger.error(f"W&B Fetch failed: {e}", exc_info=True)
        return pd.DataFrame()


def process_wandb_csv(file_path):
    """Processes W&B export CSV and performs statistical significance tests."""
    try:
        df = pd.read_csv(file_path)
        
        col_map = {
            "val_balanced_accuracy": ["val_balanced_accuracy", "val_acc", "accuracy", "Test Acc"],
            "train_balanced_accuracy": ["train_balanced_accuracy", "train_acc", "Train Acc"],
            "dataset": ["dataset", "Dataset"],
            "model": ["model", "Method", "model_name"],
            "f1": ["f1", "val_f1", "F1"],
            "runtime": ["runtime", "Runtime", "time", "_runtime"]
        }
        
        def normalize(curr_df):
            for standard, alternates in col_map.items():
                if standard not in curr_df.columns:
                    for alt in alternates:
                        if alt in curr_df.columns:
                            curr_df[standard] = curr_df[alt]; break
            return curr_df

        df = normalize(df)

        # Only flatten if essential columns are missing
        if not all(c in df.columns for c in ["dataset", "model", "val_balanced_accuracy"]):
            if "summary" in df.columns and "config" in df.columns:
                import ast
                def safe_parse(val):
                    if pd.isna(val): return {}
                    if isinstance(val, dict): return val
                    try: return ast.literal_eval(val)
                    except: return {}
                
                flattened_data = []
                for _, row in df.iterrows():
                    s, c = safe_parse(row["summary"]), safe_parse(row["config"])
                    flattened_data.append({
                        "dataset": c.get("dataset"),
                        "model": c.get("model"),
                        "train_balanced_accuracy": s.get("train_balanced_accuracy", s.get("train_accuracy", 0)),
                        "val_balanced_accuracy": s.get("val_balanced_accuracy", s.get("accuracy", 0)),
                        "f1": s.get("val_f1", s.get("f1", 0)),
                        "runtime": s.get("_runtime", s.get("total_training_time_s", 0))
                    })
                df = pd.DataFrame(flattened_data)
                df = normalize(df)

        for col in ["train_balanced_accuracy", "val_balanced_accuracy", "f1", "runtime"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        results_map, raw_rows = {}, []
        for (ds, model), group in df.groupby(["dataset", "model"]):
            if pd.isna(ds) or pd.isna(model) or ds == "dataset": continue
            key = f"{ds}|||{model}"
            runs = []
            for _, row in group.iterrows():
                run_data = {
                    "train_balanced_accuracy": row["train_balanced_accuracy"],
                    "val_balanced_accuracy": row["val_balanced_accuracy"],
                    "f1": row.get("f1", 0),
                    "runtime": row.get("runtime", 0)
                }
                runs.append(run_data)
                raw_rows.append({
                    "Dataset": ds, "Method": model, 
                    "Test Accuracy": row["val_balanced_accuracy"], 
                    "Train Accuracy": row["train_balanced_accuracy"],
                    "F1 Score": row.get("f1", 0), "Runtime (s)": row.get("runtime", 0)
                })
            results_map[key] = runs

        st.session_state["raw_results_df"] = pd.DataFrame(raw_rows)
        summary_df = summarize_results(results_map, baseline_model="opls-da")
        from fishy.analysis.statistical import display_statistical_summary
        display_statistical_summary(summary_df)
        return summary_df
    except Exception as e:
        st.error(f"Error: {e}"); return pd.DataFrame()


def get_color_map(methods):
    """Generates a consistent color map for a list of methods."""
    colors = px.colors.qualitative.Plotly + px.colors.qualitative.Bold
    unique_methods = sorted(list(set(methods)))
    return {m: colors[i % len(colors)] for i, m in enumerate(unique_methods)}


def render_advanced_benchmarks(df_summary, df_raw, color_map):
    st.write("### 🚀 Advanced Benchmarking Insights")
    
    tab_box, tab_heat, tab_radar = st.tabs(["🛡️ Stability", "📈 Heatmap & Efficiency", "🏆 Top 3 Profiles"])
    
    with tab_box:
        st.write("#### Performance Distribution (Stability)")
        ds_choice = st.selectbox("Select Dataset for Distribution", df_raw["Dataset"].unique(), key="box_ds")
        fig_box = px.box(
            df_raw[df_raw["Dataset"] == ds_choice], 
            x="Method", y="Test Accuracy", color="Method",
            color_discrete_map=color_map, template="plotly_white", points="all",
            title=f"Stability across all runs: {ds_choice.upper()}"
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
    with tab_heat:
        c_h1, c_h2 = st.columns(2)
        with c_h1:
            st.write("#### Global Performance Heatmap")
            pivot_df = df_summary.pivot(index="Method", columns="Dataset", values="Test")
            fig_heat = px.imshow(
                pivot_df, text_auto=".3f", aspect="auto",
                color_continuous_scale="Viridis", title="Method Performance across Datasets",
                template="plotly_white"
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        with c_h2:
            st.write("#### Efficiency: Runtime vs. Accuracy")
            # Determine best Y axis for efficiency
            y_axis = "runtime" if "runtime" in df_summary.columns else ("Runtime (s)" if "Runtime (s)" in df_summary.columns else "Train")
            fig_eff = px.scatter(
                df_summary, x="Test", y=y_axis, size="Test Std", color="Method",
                hover_name="Method", facet_col="Dataset", 
                color_discrete_map=color_map, template="plotly_white",
                labels={"Test": "Test Accuracy", y_axis: y_axis.capitalize()},
                title=f"Accuracy vs {y_axis.capitalize()}"
            )
            st.plotly_chart(fig_eff, use_container_width=True)
    
    with tab_radar:
        st.write("#### Top 3 Methods Radar Comparison")
        ds_radar = st.selectbox("Select Dataset for Radar", df_summary["Dataset"].unique(), key="radar_ds")
        top_3 = df_summary[df_summary["Dataset"] == ds_radar].sort_values("Test", ascending=False).head(3)
        
        fig_radar = go.Figure()
        for _, row in top_3.iterrows():
            # Gather metrics for radar
            # We normalize to 0-1 range for the radar plot
            train_val = row.get("Train", 0)
            test_val = row.get("Test", 0)
            f1_val = row.get("f1", row.get("F1 Score", test_val)) # fallback to test acc
            stability = 1.0 - row.get("Test Std", 0) # higher is better
            
            fig_radar.add_trace(go.Scatterpolar(
                r=[train_val, test_val, f1_val, stability],
                theta=["Train Acc", "Test Acc", "F1 Score", "Stability (1-Std)"],
                fill="toself", name=row["Method"],
                line=dict(color=color_map.get(row["Method"]))
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True, title=f"Top 3 Comparison: {ds_radar.upper()}", template="plotly_white"
        )
        st.plotly_chart(fig_radar, use_container_width=True)


st.sidebar.title("🛠️ Configuration")
model_names, dataset_names = get_metadata()
data_path = Path(get_data_path())
selected_model = st.sidebar.selectbox(
    "Select Model",
    model_names,
    index=model_names.index("transformer") if "transformer" in model_names else 0,
)
selected_dataset = st.sidebar.selectbox(
    "Select Dataset",
    dataset_names,
    index=dataset_names.index("species") if "species" in dataset_names else 0,
)

dm = create_data_module(selected_dataset, str(data_path))

with st.sidebar.expander("🚀 Hyperparameters", expanded=True):
    epochs = st.slider("Epochs", 1, 100, 10)
    batch_size = st.select_slider("Batch Size", options=[8, 16, 32, 64, 128], value=32)
    lr = st.number_input("Learning Rate", value=1e-4, format="%.1e")
    k_folds = st.slider(
        "Cross-Validation Folds",
        2,
        10,
        3,
        help="Higher folds = better stability estimate",
    )

train_button = st.sidebar.button("🚀 Run Training", width="stretch")

st.title("🐟 Fishy Business")
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Data Exploration",
        "📈 Training & Results",
        "🧠 Interpretability & Biomarkers",
        "🏆 Leaderboard",
    ]
)

def render_results(results):
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Train Bal Acc", f"{results.get('train_balanced_accuracy', 0):.4f}")
    mc2.metric("Val Bal Acc", f"{results.get('val_balanced_accuracy', 0):.4f}")
    mc3.metric("Loss (Val)", f"{results.get('val_loss', 0):.4f}")
    mc4.metric("Time", f"{results.get('total_training_time_s', 0):.2f}s")
    if "epoch_metrics" in results and results["epoch_metrics"]:
        st.write("### Training Progress")
        history = results["epoch_metrics"]
        c_loss, c_acc = st.columns(2)
        c_loss.plotly_chart(
            px.line(
                pd.DataFrame(
                    {
                        "Epoch": range(1, len(history["train_losses"]) + 1),
                        "Train Loss": history["train_losses"],
                        "Val Loss": history["val_losses"],
                    }
                ),
                # Fixed: ensure list columns are correctly assigned
                x="Epoch",
                y=["Train Loss", "Val Loss"],
                title="Loss Curve",
                template="plotly_white",
            ),
            use_container_width=True,
        )
        c_acc.plotly_chart(
            px.line(
                pd.DataFrame(
                    {
                        "Epoch": range(1, len(history["val_metrics"]) + 1),
                        "Val Acc": [
                            m.get("balanced_accuracy", 0)
                            for m in history["val_metrics"]
                        ],
                        "Train Acc": [
                            m.get("balanced_accuracy", 0)
                            for m in history["train_metrics"]
                        ],
                    }
                ),
                x="Epoch",
                y=["Train Acc", "Val Acc"],
                title="Accuracy Curve",
                template="plotly_white",
            ),
            use_container_width=True,
        )
    if "predictions" in results and results["predictions"]:
        preds = results["predictions"]
        y_true, y_pred, y_probs = (
            preds["labels"],
            preds["preds"],
            preds.get("probs"),
        )
        st.write("### Detailed Analysis")
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.plotly_chart(
                px.imshow(
                    confusion_matrix(y_true, y_pred),
                    x=dm.get_class_names(),
                    y=dm.get_class_names(),
                    text_auto=True,
                    color_continuous_scale="Blues",
                    title="Confusion Matrix",
                    template="plotly_white",
                ),
                use_container_width=True,
            )
        with r1c2:
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=range(len(dm.get_class_names()))
            )
            st.plotly_chart(
                px.bar(
                    pd.DataFrame(
                        {
                            "Class": dm.get_class_names(),
                            "Precision": prec,
                            "Recall": rec,
                            "F1": f1,
                        }
                    ),
                    x="Class",
                    y=["Precision", "Recall", "F1"],
                    barmode="group",
                    title="Class Metrics",
                    template="plotly_white",
                ),
                use_container_width=True,
            )
        st.markdown("---")
        st.write("### Error Spotlight & Performance")
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            if y_probs is not None:
                pr_fig = go.Figure()
                for i in range(len(dm.get_class_names())):
                    precision, recall, _ = precision_recall_curve(
                        y_true == i, y_probs[:, i]
                    )
                    avg_prec = average_precision_score(y_true == i, y_probs[:, i])
                    pr_fig.add_trace(
                        go.Scatter(
                            x=recall,
                            y=precision,
                            name=f"{dm.get_class_names()[i]} (AP={avg_prec:.2f})",
                            mode="lines",
                        )
                    )
                pr_fig.update_layout(
                    template="plotly_white",
                    xaxis_title="Recall",
                    yaxis_title="Precision",
                    title="PR Curve",
                )
                st.plotly_chart(pr_fig, use_container_width=True)
        with r2c2:
            if y_probs is not None:
                correct = y_true == y_pred
                err_idx = np.where(~correct)[0]
                if len(err_idx) > 0:
                    top_errs = err_idx[
                        np.argsort(np.max(y_probs[err_idx], axis=1))[-3:][::-1]
                    ]
                    for idx in top_errs:
                        st.error(
                            f"Sample {idx}: Truth={dm.get_class_names()[y_true[idx]]}, Pred={dm.get_class_names()[y_pred[idx]]} (Conf: {np.max(y_probs[idx]):.2f})"
                        )
                else:
                    st.success("Perfect classification!")

        st.markdown("---")
        st.write("### Stability")
        if "folds" in results:
            fold_accs = [
                f.get("val_balanced_accuracy", 0) for f in results.get("folds", [])
            ]
            if fold_accs:
                st.plotly_chart(
                    px.violin(
                        y=fold_accs,
                        box=True,
                        points="all",
                        title=f"Cross-Validation Stability ({k_folds} folds)",
                        template="plotly_white",
                    ),
                    use_container_width=True,
                )
            else:
                st.info("Fold data unavailable.")
        else:
            st.info("Stability data requires multiple folds.")

if data_path.exists():
    dm.setup()
    df_filtered = dm.get_filtered_dataframe()
    class_names = dm.get_class_names()
    X_all, y_all = dm.get_numpy_data(labels_as_indices=True)
    label_col = (
        "Class Name" if "Class Name" in df_filtered.columns else df_filtered.columns[0]
    )
    feature_cols = [c for c in df_filtered.columns if c not in [label_col, "m/z", "Label"]]
    # Fix: Ensure column names are treated as strings before calling .replace()
    mz_axis = np.array([float(c) for c in feature_cols]) if all(str(c).replace('.','',1).isdigit() for c in feature_cols) else np.arange(len(feature_cols))
    
    with tab1:
        st.header("Deep Data Exploration")
        mc1, mc2, mc3 = st.columns([1, 1, 2])
        mc1.metric("Total Samples", len(df_filtered))
        mc2.metric("Feature Count", dm.get_input_dim())
        with mc3:
            dist = df_filtered[label_col].value_counts()
            st.plotly_chart(
                px.bar(
                    x=dist.index,
                    y=dist.values,
                    labels={"x": "Class", "y": "Count"},
                    height=200,
                    template="plotly_white",
                ),
                use_container_width=True,
            )
        st.markdown("---")
        st.write("### Mean Spectral Signature")
        avg_fig = go.Figure()
        for idx, name in enumerate(class_names):
            mask = y_all == idx
            if np.any(mask):
                m, s = X_all[mask].mean(axis=0), X_all[mask].std(axis=0)
                avg_fig.add_trace(go.Scatter(x=mz_axis, y=m, name=name, mode="lines"))
                avg_fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([mz_axis, mz_axis[::-1]]),
                        y=np.concatenate([m + s, (m - s)[::-1]]),
                        fill="toself",
                        fillcolor="rgba(0,100,80,0.1)",
                        line=dict(color="rgba(255,255,255,0)"),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
        avg_fig.update_layout(
            template="plotly_white",
            xaxis_title="m/z",
            yaxis_title="Intensity",
            height=500,
        )
        st.plotly_chart(avg_fig, use_container_width=True)
        st.markdown("---")
        st.write("### Interactive Spectrum Viewer")
        all_classes = sorted(df_filtered[label_col].unique().tolist())
        c_sel, n_sel = st.columns([3, 1])
        sel_classes = c_sel.multiselect(
            "Filter Classes",
            all_classes,
            default=all_classes[: min(2, len(all_classes))],
        )
        n_view = n_sel.slider("Samples to overlay", 1, 50, 10)
        v_df = (
            df_filtered[df_filtered[label_col].isin(sel_classes)]
            if sel_classes
            else df_filtered
        )
        if not v_df.empty:
            s_df = v_df.sample(min(n_view, len(v_df)))
            num_cols = s_df.select_dtypes(include=[np.number]).columns.tolist()
            melted = s_df[[label_col] + num_cols].melt(
                id_vars=label_col, var_name="Feature", value_name="Int"
            )
            st.plotly_chart(
                px.line(
                    melted[melted["Feature"] != "m/z"],
                    x="Feature",
                    y="Int",
                    color=label_col,
                    template="plotly_white",
                    height=500,
                ),
                use_container_width=True,
            )
        st.markdown("---")
        col_pca, col_tsne, col_umap = st.columns(3)
        pca_obj = PCA(n_components=10)
        X_pca = pca_obj.fit_transform(X_all)
        with col_pca:
            st.plotly_chart(
                px.scatter(
                    pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(10)]),
                    x="PC1",
                    y="PC2",
                    color=[class_names[i] for i in y_all],
                    title="PCA",
                    template="plotly_white",
                ),
                use_container_width=True,
            )
        with col_tsne:
            with st.spinner("t-SNE..."):
                X_tsne = TSNE(
                    n_components=2, perplexity=min(30, len(X_all) - 1)
                ).fit_transform(X_all)
            st.plotly_chart(
                px.scatter(
                    pd.DataFrame(X_tsne, columns=["TSNE1", "TSNE2"]),
                    x="TSNE1",
                    y="TSNE2",
                    color=[class_names[i] for i in y_all],
                    title="t-SNE",
                    template="plotly_white",
                ),
                use_container_width=True,
            )
        with col_umap:
            if umap:
                with st.spinner("UMAP..."):
                    X_umap = umap.UMAP().fit_transform(X_all)
                st.plotly_chart(
                    px.scatter(
                        pd.DataFrame(X_umap, columns=["UMAP1", "UMAP2"]),
                        x="UMAP1",
                        y="UMAP2",
                        color=[class_names[i] for i in y_all],
                        title="UMAP",
                        template="plotly_white",
                    ),
                    use_container_width=True,
                )
            else:
                st.info("UMAP not installed")
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.plotly_chart(
                px.area(
                    x=range(1, 11),
                    y=np.cumsum(pca_obj.explained_variance_ratio_),
                    labels={"x": "Comp", "y": "Cum Var"},
                    title="PCA Information Retention",
                    template="plotly_white",
                ),
                use_container_width=True,
            )
        with dcol2:
            avg_int = pd.DataFrame(
                {
                    "Avg Int": X_all.mean(axis=1),
                    "Class": [class_names[i] for i in y_all],
                }
            )
            st.plotly_chart(
                px.violin(
                    avg_int,
                    x="Class",
                    y="Avg Int",
                    color="Class",
                    box=True,
                    points="all",
                    title="Intensity Distribution",
                    template="plotly_white",
                ),
                use_container_width=True,
            )

    with tab2:
        if train_button:
            config = TrainingConfig(
                model=selected_model,
                dataset=selected_dataset,
                file_path=str(data_path),
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=lr,
                k_folds=k_folds,
                wandb_log=False,
            )
            method = detect_method(selected_model)
            config.method = method
            try:
                with st.spinner(f"Training {selected_model}..."):
                    if method == "deep":
                        trainer = ModelTrainer(config)
                        model, results = trainer.train()
                        st.session_state["dm"] = trainer.data_module
                    else:
                        trainer = SklearnTrainer(
                            config,
                            selected_model,
                            selected_dataset,
                            file_path=str(data_path),
                        )
                        model, results = trainer.run()
                        st.session_state["dm"] = dm
                    st.session_state["model"] = model
                    st.session_state["results"] = results
                    st.session_state["method"] = method
                st.success("🎉 Training Complete!")
                render_results(results)
            except Exception as e:
                st.error(f"Failed: {e}")
                st.exception(e)
        elif "results" in st.session_state:
            render_results(st.session_state["results"])

    with tab3:
        if "model" in st.session_state:
            st.header("Interpretability & Biomarkers")
            model = st.session_state["model"]
            current_dm = st.session_state["dm"]
            X_xai, y_xai = current_dm.get_numpy_data(labels_as_indices=True)
            if "rep_indices" not in st.session_state or len(
                st.session_state["rep_indices"]
            ) != len(class_names):
                st.session_state["rep_indices"] = {
                    c: (
                        np.where(y_xai == c)[0][0]
                        if len(np.where(y_xai == c)[0]) > 0
                        else 0
                    )
                    for c in range(len(class_names))
                }
            exp_col1, exp_col2 = st.columns([1, 2])
            with exp_col1:
                sample_idx = st.selectbox("Select instance", range(len(X_xai)))
                method = st.radio(
                    "Method",
                    (
                        ["Grad-CAM", "LIME"]
                        if st.session_state.get("method") == "deep"
                        else ["LIME"]
                    ),
                    horizontal=True,
                )
                st.info(f"Targeting: **{class_names[y_xai[sample_idx]]}**", icon="🎯")
            if method == "Grad-CAM":
                target_layer = getattr(model, "layer_norm2", next(
                    (
                        m
                        for m in reversed(list(model.modules()))
                        if isinstance(m, (nn.Conv1d, nn.Linear))
                    ),
                    None,
                ))
                if target_layer:
                    gc = GradCAM(model, target_layer)
                    cam = (
                        gc.generate_cam(
                            torch.tensor(X_xai[sample_idx])
                            .unsqueeze(0)
                            .to(get_device())
                        )
                        .cpu()
                        .numpy()[0]
                    )
                    gc.remove_hooks()
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=mz_axis, y=X_xai[sample_idx], name="Spectrum", line=dict(color="lightgray", width=1)))
                    fig.add_trace(go.Scatter(x=mz_axis, y=X_xai[sample_idx], mode="markers", marker=dict(color=cam, colorscale="Viridis", size=8, showscale=True), name="Importance"))
                    fig.update_layout(title="Importance Map", template="plotly_white", xaxis_title="m/z", yaxis_title="Intensity")
                    exp_col2.plotly_chart(fig, use_container_width=True)
            else:
                wrapper = ModelWrapper(model, str(get_device()))
                explainer = LimeTabularExplainer(X_xai, feature_names=[f"{m:.4f}" for m in mz_axis], class_names=class_names, discretize_continuous=True)
                with st.spinner("LIME..."):
                    exp = explainer.explain_instance(X_xai[sample_idx], wrapper.predict_proba, num_features=15)
                exp_col2.plotly_chart(px.bar(x=[x[1] for x in exp.as_list()], y=[x[0] for x in exp.as_list()], orientation="h", color=[x[1] for x in exp.as_list()], color_continuous_scale="RdBu", title="LIME Weights"), use_container_width=True)
            
            st.markdown("---")
            if st.button("🔍 Run Class-Specific Biomarker Analysis"):
                with st.spinner("Analyzing stable biomarkers..."):
                    wrapper = ModelWrapper(model, str(get_device()))
                    exp_int = LimeTabularExplainer(X_xai, feature_names=[str(i) for i in range(X_xai.shape[1])], class_names=class_names, discretize_continuous=False)
                    class_biomarkers = {}
                    for c_idx in range(len(class_names)):
                        c_indices = np.where(y_xai == c_idx)[0]
                        if len(c_indices) > 0:
                            sub = np.random.choice(c_indices, min(20, len(c_indices)), replace=False)
                            w_list = []
                            for idx in sub:
                                w_list.append(dict(exp_int.explain_instance(X_xai[idx], wrapper.predict_proba, num_features=X_xai.shape[1], labels=(c_idx,)).as_list(label=c_idx)))
                            avg_w = pd.DataFrame(w_list).mean().sort_values()
                            class_biomarkers[c_idx] = [int(i) for i in avg_w.tail(20).index.tolist()]
                    
                    comp_fig = go.Figure()
                    for c_idx, c_name in enumerate(class_names):
                        rep_idx = st.session_state["rep_indices"].get(c_idx)
                        if rep_idx is not None and c_idx in class_biomarkers:
                            comp_fig.add_trace(go.Scatter(x=mz_axis, y=X_xai[rep_idx], name=f"{c_name}", line=dict(width=1), opacity=0.4))
                            comp_fig.add_trace(go.Scatter(x=mz_axis[class_biomarkers[c_idx]], y=X_xai[rep_idx][class_biomarkers[c_idx]], mode="markers", marker=dict(size=10), name=f"Diagnostic: {c_name}"))
                    st.plotly_chart(comp_fig, use_container_width=True)

    with tab4:
        st.header("🏆 Leaderboard")
        src = st.radio("Leaderboard Data Source", ["Local (outputs/)", "Remote (SSH Proxy Jump)", "W&B Export (CSV)", "W&B API (Live)"], horizontal=True, key="leaderboard_src_selector")
        if "leaderboard_df" not in st.session_state: st.session_state["leaderboard_df"] = pd.DataFrame()
        
        if src == "Local (outputs/)":
            if st.button("🔄 Refresh Local Leaderboard"):
                with st.spinner("Crawling..."): st.session_state["leaderboard_df"] = crawl_local_results()
        elif src == "Remote (SSH Proxy Jump)":
            if not paramiko: st.error("paramiko not installed")
            else:
                with st.expander("🔑 Jump Host Configuration", expanded=True):
                    c1, c2, c_otp = st.columns([2, 2, 2])
                    jh = c1.text_input("Jump Host", value="entry.ecs.vuw.ac.nz")
                    ju = c2.text_input("ECS Username", key="remote_user")
                    otp = c_otp.text_input("OTP Token", type="password", key="remote_otp")
                with st.expander("🎯 Target Server Configuration", expanded=True):
                    c3, c4, c5 = st.columns([2, 1, 2])
                    th = c3.text_input("Target Host", key="remote_target")
                    tp = c4.number_input("Target Port", value=22, key="remote_port")
                    pwd = c5.text_input("ECS Password", type="password", key="remote_pwd")
                    rp = st.text_input("Remote Path", "/home/ecs/username/fishy-business/outputs", key="remote_path")
                if st.button("🚀 Connect & Aggregate Remote"):
                    with st.spinner("Tunnelling..."): st.session_state["leaderboard_df"] = fetch_remote_data(th, tp, ju, pwd, rp, jump_host=jh, jump_user=ju, otp=otp)
        elif src == "W&B Export (CSV)":
            uploaded_file = st.file_uploader("Upload W&B Export CSV", type="csv", key="csv_uploader")
            if uploaded_file:
                if st.button("📊 Run Statistical Analysis on Uploaded CSV"):
                    with open("temp_wandb_export.csv", "wb") as f: f.write(uploaded_file.getbuffer())
                    df_res = process_wandb_csv("temp_wandb_export.csv")
                    if not df_res.empty: st.session_state["leaderboard_df"] = df_res; st.rerun()
        elif src == "W&B API (Live)":
            w_c1, w_c2 = st.columns(2)
            w_entity = w_c1.text_input("W&B Entity", value="victoria-university-of-wellington", key="wb_ent")
            w_project = w_c2.text_input("W&B Project", value="fishy-business", key="wb_proj")
            w_key = st.text_input("W&B API Key (Optional)", type="password", key="wb_key")
            if st.button("🌐 Fetch & Analyze Live W&B Data"):
                with st.spinner("Talking to W&B..."): st.session_state["leaderboard_df"] = fetch_wandb_data(w_entity, w_project, w_key)

        df_summary = st.session_state["leaderboard_df"]
        if not df_summary.empty:
            st.markdown("---")
            st.dataframe(df_summary.sort_values(["Dataset", "Test"], ascending=[True, False]).style.background_gradient(subset=["Test"], cmap="Greens"), use_container_width=True)
            color_map = get_color_map(df_summary["Method"].unique())
            for ds in df_summary["Dataset"].unique():
                ds_df = df_summary[df_summary["Dataset"] == ds].sort_values("Test", ascending=False)
                st.plotly_chart(px.bar(ds_df, x="Method", y="Test", error_y="Test Std", title=f"Leaderboard: {ds.upper()}", color="Method", color_discrete_map=color_map, template="plotly_white"), use_container_width=True)
            
            if "raw_results_df" in st.session_state:
                render_advanced_benchmarks(df_summary, st.session_state["raw_results_df"], color_map)
else:
    st.error("📉 REIMS Dataset Missing")
