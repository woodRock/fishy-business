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
                results.append(
                    pd.DataFrame(
                        [
                            {
                                "Dataset": parts[1],
                                "Method": parts[3].split("_")[0],
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
    host,
    port,
    username,
    password,
    remote_path,
    jump_host=None,
    jump_user=None,
    otp=None,
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
                banner_timeout=60,
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
                banner_timeout=60,
            )
        else:
            target_client.connect(
                host,
                port=port,
                username=username,
                password=password,
                timeout=300,
                banner_timeout=60,
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
                bar.progress(
                    (i + 1) / total_runs, text=f"{progress_text} ({i+1}/{total_runs})"
                )

            # Extract configuration
            ds = run.config.get("dataset")
            model = run.config.get("model")

            if not ds or not model:
                continue

            if ds == "oil":
                found_oil = True

            # Extract final metrics from summary
            train_acc = run.summary.get("train_balanced_accuracy")
            val_acc = run.summary.get("val_balanced_accuracy")

            if train_acc is None or val_acc is None:
                # Fallback check for different naming conventions
                train_acc = run.summary.get("train_accuracy", 0)
                val_acc = run.summary.get(
                    "val_balanced_accuracy", run.summary.get("accuracy", 0)
                )

            key = f"{ds}|||{model}"
            if key not in results_map:
                results_map[key] = []

            results_map[key].append(
                {"train_balanced_accuracy": train_acc, "val_balanced_accuracy": val_acc}
            )

        bar.empty()

        if not results_map:
            st.warning(f"No valid finished runs found in {entity}/{project}")
            return pd.DataFrame()

        if not found_oil:
            st.info("Note: No runs with dataset='oil' found in W&B project summary.")
        else:
            st.success(
                f"Successfully found 'oil' results ({len([v for k,v in results_map.items() if 'oil' in k])} runs) in W&B!"
            )

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
    """Processes W&B export CSV and performs statistical significance tests.
    Supports both flat CSVs and nested W&B API exports.
    """
    try:
        df = pd.read_csv(file_path)

        # Handle "nested" W&B API script format (columns: summary, config, name)
        if "summary" in df.columns and "config" in df.columns:
            import ast

            def safe_parse(val):
                if pd.isna(val):
                    return {}
                if isinstance(val, dict):
                    return val
                try:
                    # API CSVs often use string representations of dicts
                    return ast.literal_eval(val)
                except:
                    return {}

            flattened_data = []
            for _, row in df.iterrows():
                s = safe_parse(row["summary"])
                c = safe_parse(row["config"])
                flattened_data.append(
                    {
                        "dataset": c.get("dataset"),
                        "model": c.get("model"),
                        "train_balanced_accuracy": s.get(
                            "train_balanced_accuracy", s.get("train_accuracy", 0)
                        ),
                        "val_balanced_accuracy": s.get(
                            "val_balanced_accuracy", s.get("accuracy", 0)
                        ),
                    }
                )
            df = pd.DataFrame(flattened_data)

        # Standard column names normalization
        col_map = {
            "val_balanced_accuracy": [
                "val_balanced_accuracy",
                "val_acc",
                "accuracy",
                "Test Acc",
            ],
            "train_balanced_accuracy": [
                "train_balanced_accuracy",
                "train_acc",
                "Train Acc",
            ],
            "dataset": ["dataset", "Dataset"],
            "model": ["model", "Method", "model_name"],
        }

        for standard, alternates in col_map.items():
            if standard not in df.columns:
                for alt in alternates:
                    if alt in df.columns:
                        df[standard] = df[alt]
                        break

        # CRITICAL: Ensure accuracy columns are numeric. W&B exports sometimes contain strings or NaNs.
        for col in ["train_balanced_accuracy", "val_balanced_accuracy"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        required = [
            "dataset",
            "model",
            "train_balanced_accuracy",
            "val_balanced_accuracy",
        ]
        if not all(c in df.columns for c in required):
            st.error(
                f"CSV missing columns. Required: {required}. Found: {list(df.columns)}"
            )
            return pd.DataFrame()

        # Map to internal format for summarize_results
        results_map = {}
        for (ds, model), group in df.groupby(["dataset", "model"]):
            if pd.isna(ds) or pd.isna(model):
                continue
            key = f"{ds}|||{model}"
            runs = []
            for _, row in group.iterrows():
                runs.append(
                    {
                        "train_balanced_accuracy": row["train_balanced_accuracy"],
                        "val_balanced_accuracy": row["val_balanced_accuracy"],
                    }
                )
            results_map[key] = runs

        # Perform significance testing using OPLS-DA as baseline
        summary_df = summarize_results(results_map, baseline_model="opls-da")

        # Display pretty Rich table in logs/console
        from fishy.analysis.statistical import display_statistical_summary

        display_statistical_summary(summary_df)

        return summary_df
    except Exception as e:
        st.error(f"Error processing CSV: {e}")
        return pd.DataFrame()


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
    feature_cols = [c for c in df_filtered.columns if c not in [label_col, "m/z"]]
    try:
        mz_axis = np.array([float(c) for c in feature_cols])
    except:
        mz_axis = np.arange(len(feature_cols))

    # ... rest of the existing code for tab1, tab2, etc.

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
            st.info("Showing last results.")
            render_results(st.session_state["results"])
        else:
            st.info("Run training to see results.")

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
                # Prioritize layer_norm2 for Transformers to get full spectral resolution
                # Fallback to the last Linear/Conv layer for other models
                target_layer = getattr(
                    model,
                    "layer_norm2",
                    next(
                        (
                            m
                            for m in reversed(list(model.modules()))
                            if isinstance(m, (nn.Conv1d, nn.Linear))
                        ),
                        None,
                    ),
                )
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
                    fig.add_trace(
                        go.Scatter(
                            x=mz_axis,
                            y=X_xai[sample_idx],
                            name="Spectrum",
                            line=dict(color="lightgray", width=1),
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=mz_axis,
                            y=X_xai[sample_idx],
                            mode="markers",
                            marker=dict(
                                color=cam, colorscale="Viridis", size=8, showscale=True
                            ),
                            name="Importance",
                            hovertemplate="m/z: %{x}<br>Int: %{y}<br>Imp: %{marker.color:.4f}",
                        )
                    )
                    fig.update_layout(
                        title="Importance Map",
                        template="plotly_white",
                        xaxis_title="m/z",
                        yaxis_title="Intensity",
                    )
                    exp_col2.plotly_chart(fig, use_container_width=True)
            else:
                wrapper = ModelWrapper(model, str(get_device()))
                explainer = LimeTabularExplainer(
                    X_xai,
                    feature_names=[f"{m:.4f}" for m in mz_axis],
                    class_names=class_names,
                    discretize_continuous=True,
                )
                with st.spinner("LIME..."):
                    exp = explainer.explain_instance(
                        X_xai[sample_idx], wrapper.predict_proba, num_features=15
                    )
                exp_col2.plotly_chart(
                    px.bar(
                        x=[x[1] for x in exp.as_list()],
                        y=[x[0] for x in exp.as_list()],
                        orientation="h",
                        color=[x[1] for x in exp.as_list()],
                        color_continuous_scale="RdBu",
                        title="LIME Weights",
                    ),
                    use_container_width=True,
                )
            st.markdown("---")
            st.write("### 🌐 Distinct Class Biomarker Comparison")
            c_btn1, c_btn2 = st.columns([1, 4])
            if c_btn1.button("🔀 Shuffle Representatives"):
                for c in range(len(class_names)):
                    indices = np.where(y_xai == c)[0]
                    if len(indices) > 0:
                        st.session_state["rep_indices"][c] = np.random.choice(indices)
            if st.button("🔍 Run Class-Specific Analysis"):
                with st.spinner("Analyzing stable biomarkers..."):
                    wrapper = ModelWrapper(model, str(get_device()))
                    exp_int = LimeTabularExplainer(
                        X_xai,
                        feature_names=[str(i) for i in range(X_xai.shape[1])],
                        class_names=class_names,
                        discretize_continuous=False,
                    )
                    class_biomarkers = {}
                    for c_idx in range(len(class_names)):
                        c_indices = np.where(y_xai == c_idx)[0]
                        if len(c_indices) > 0:
                            sub = np.random.choice(
                                c_indices, min(20, len(c_indices)), replace=False
                            )
                            w_list = []
                            for idx in sub:
                                w_list.append(
                                    dict(
                                        exp_int.explain_instance(
                                            X_xai[idx],
                                            wrapper.predict_proba,
                                            num_features=X_xai.shape[1],
                                            labels=(c_idx,),
                                        ).as_list(label=c_idx)
                                    )
                                )
                            avg_w = pd.DataFrame(w_list).mean().sort_values()
                            class_biomarkers[c_idx] = [
                                int(i) for i in avg_w.tail(20).index.tolist()
                            ]
                    comp_fig = go.Figure()
                    styles = [
                        {"c": "gold", "s": "diamond"},
                        {"c": "silver", "s": "circle"},
                        {"c": "cyan", "s": "square"},
                    ]
                    for c_idx, c_name in enumerate(class_names):
                        rep_idx = st.session_state["rep_indices"].get(c_idx)
                        if rep_idx is not None and c_idx in class_biomarkers:
                            spec = X_xai[rep_idx]
                            stl = styles[c_idx % len(styles)]
                            comp_fig.add_trace(
                                go.Scatter(
                                    x=mz_axis,
                                    y=spec,
                                    name=f"{c_name} (Smp {rep_idx})",
                                    line=dict(width=1),
                                    opacity=0.4,
                                )
                            )
                            comp_fig.add_trace(
                                go.Scatter(
                                    x=mz_axis[class_biomarkers[c_idx]],
                                    y=spec[class_biomarkers[c_idx]],
                                    mode="markers",
                                    marker=dict(
                                        color=stl["c"],
                                        size=12,
                                        symbol=stl["s"],
                                        line=dict(width=2, color="black"),
                                    ),
                                    name=f"Top Diagnostic: {c_name}",
                                    hovertemplate="m/z: %{x}<br>Int: %{y}",
                                )
                            )
                    comp_fig.update_layout(
                        title="Distinct Class Representatives & Biomarkers",
                        template="plotly_white",
                        xaxis_title="m/z",
                        yaxis_title="Intensity",
                    )
                    st.plotly_chart(comp_fig, use_container_width=True)
                    st.write("#### Diagnostic Peaks (m/z)")
                    cols = st.columns(len(class_names))
                    for i, c_name in enumerate(class_names):
                        if i in class_biomarkers:
                            top_v = [
                                f"{mz_axis[idx]:.4f}"
                                for idx in class_biomarkers[i][-10:][::-1]
                            ]
                            cols[i].write(f"**{c_name}**")
                            for v in top_v:
                                cols[i].code(v)
            if "class_biomarkers" in locals() and class_biomarkers:
                st.markdown("---")
                st.write("### 🧬 Advanced Biomarker Network")
                net_col1, net_col2 = st.columns(2)
                with net_col1:
                    all_top_features = []
                    for k in class_biomarkers:
                        all_top_features.extend(class_biomarkers[k])
                    feat_counts = (
                        pd.Series(all_top_features)
                        .value_counts()
                        .sort_values(ascending=False)
                        .head(15)
                    )
                    st.plotly_chart(
                        px.bar(
                            x=[f"{mz_axis[i]:.2f}" for i in feat_counts.index],
                            y=feat_counts.values,
                            labels={"x": "m/z Feature", "y": "Freq"},
                            title="Biomarker Stability",
                            template="plotly_white",
                        ),
                        use_container_width=True,
                    )
                with net_col2:
                    top_indices = list(set(all_top_features))
                    if len(top_indices) > 1:
                        subset_data = X_xai[:, top_indices]
                        corr_matrix = np.corrcoef(subset_data.T)
                        G = nx.Graph()
                        for i in range(len(top_indices)):
                            G.add_node(i, label=f"{mz_axis[top_indices[i]]:.2f}")
                        for i in range(len(top_indices)):
                            for j in range(i + 1, len(top_indices)):
                                if abs(corr_matrix[i, j]) > 0.8:
                                    G.add_edge(i, j, weight=abs(corr_matrix[i, j]))
                        pos = nx.spring_layout(G, seed=42)
                        edge_x, edge_y = [], []
                        for edge in G.edges():
                            x0, y0 = pos[edge[0]]
                            x1, y1 = pos[edge[1]]
                            edge_x.extend([x0, x1, None])
                            edge_y.extend([y0, y1, None])
                        edge_trace = go.Scatter(
                            x=edge_x,
                            y=edge_y,
                            line=dict(width=0.5, color="#888"),
                            hoverinfo="none",
                            mode="lines",
                        )
                        node_trace = go.Scatter(
                            x=[pos[node][0] for node in G.nodes()],
                            y=[pos[node][1] for node in G.nodes()],
                            mode="markers+text",
                            text=[G.nodes[node]["label"] for node in G.nodes()],
                            textposition="top center",
                            marker=dict(
                                showscale=True,
                                colorscale="YlGnBu",
                                size=10,
                                color=[len(list(G.neighbors(n))) for n in G.nodes()],
                                line_width=2,
                            ),
                        )
                        st.plotly_chart(
                            go.Figure(
                                data=[edge_trace, node_trace],
                                layout=go.Layout(
                                    title="Biomarker Network (r > 0.8)",
                                    showlegend=False,
                                    hovermode="closest",
                                    xaxis=dict(
                                        showgrid=False,
                                        zeroline=False,
                                        showticklabels=False,
                                    ),
                                    yaxis=dict(
                                        showgrid=False,
                                        zeroline=False,
                                        showticklabels=False,
                                    ),
                                ),
                            ),
                            use_container_width=True,
                        )
                    else:
                        st.info("Not enough biomarkers for network.")

        with tab4:

            st.header("🏆 Leaderboard")

            src = st.radio(
                "Leaderboard Data Source",
                [
                    "Local (outputs/)",
                    "Remote (SSH Proxy Jump)",
                    "W&B Export (CSV)",
                    "W&B API (Live)",
                ],
                horizontal=True,
            )

            if "leaderboard_df" not in st.session_state:

                st.session_state["leaderboard_df"] = pd.DataFrame()

            # ... existing source blocks ...

            if src == "Local (outputs/)":

                if st.button("🔄 Refresh Local Leaderboard"):

                    with st.spinner("Crawling local metrics..."):

                        st.session_state["leaderboard_df"] = crawl_local_results()

            elif src == "Remote (SSH Proxy Jump)":

                # ... (keep existing remote logic) ...

                if not paramiko:

                    st.error("paramiko not installed")

                else:

                    with st.expander(
                        "🔑 Jump Host Configuration (entry.ecs.vuw.ac.nz)",
                        expanded=True,
                    ):

                        c1, c2, c_otp = st.columns([2, 2, 2])

                        jh = c1.text_input("Jump Host", value="entry.ecs.vuw.ac.nz")

                        ju = c2.text_input("ECS Username")

                        otp = c_otp.text_input(
                            "OTP Token (Google Authenticator)",
                            type="password",
                            help="6-digit code",
                        )

                    with st.expander("🎯 Target Server Configuration", expanded=True):

                        c3, c4, c5 = st.columns([2, 1, 2])

                        th = c3.text_input("Target Host (e.g. greta-pt)")

                        tp = c4.number_input("Target Port", value=22)

                        pwd = c5.text_input("ECS Password", type="password")

                        rp = st.text_input(
                            "Remote Path to /outputs",
                            "/home/ecs/username/fishy-business/outputs",
                        )

                    if st.button("🚀 Connect & Aggregate Remote"):

                        with st.spinner(
                            "Tunnelling through entry and crawling metrics..."
                        ):

                            st.session_state["leaderboard_df"] = fetch_remote_data(
                                th, tp, ju, pwd, rp, jump_host=jh, jump_user=ju, otp=otp
                            )

            elif src == "W&B Export (CSV)":

                st.info(
                    "Analyze a W&B export CSV with statistical significance tests (OPLS-DA baseline)."
                )

                uploaded_file = st.file_uploader("Upload W&B Export CSV", type="csv")

                csv_path = "wanb_export_csv.csv"

                if uploaded_file is not None:

                    if st.button("📊 Run Statistical Analysis on Uploaded CSV"):

                        with st.spinner("Calculating significance against OPLS-DA..."):

                            # Save temporarily to process

                            with open("temp_wandb_export.csv", "wb") as f:

                                f.write(uploaded_file.getbuffer())

                            st.session_state["leaderboard_df"] = process_wandb_csv(
                                "temp_wandb_export.csv"
                            )

                elif os.path.exists(csv_path):

                    st.success(f"Found default export: {csv_path}")

                    if st.button("📊 Run Statistical Analysis on Default CSV"):

                        with st.spinner("Calculating significance against OPLS-DA..."):

                            st.session_state["leaderboard_df"] = process_wandb_csv(
                                csv_path
                            )

                else:

                    st.warning(
                        f"Default file '{csv_path}' not found. Please upload one."
                    )

            else:  # W&B API (Live)

                st.info("Fetch live data directly from Weights & Biases API.")

                w_c1, w_c2 = st.columns(2)

                w_entity = w_c1.text_input(
                    "W&B Entity", value="victoria-university-of-wellington"
                )

                w_project = w_c2.text_input("W&B Project", value="fishy-business")

                w_key = st.text_input(
                    "W&B API Key (Optional if logged in)", type="password"
                )

                if st.button("🌐 Fetch & Analyze Live W&B Data"):

                    with st.spinner("Talking to W&B servers..."):

                        st.session_state["leaderboard_df"] = fetch_wandb_data(
                            w_entity, w_project, w_key
                        )

            df_summary = st.session_state["leaderboard_df"]

        if not df_summary.empty:
            preferred_cols = [
                "Dataset",
                "Method",
                "Train",
                "Test",
                "Sig Te",
                "Baseline",
            ]
            actual_cols = [c for c in preferred_cols if c in df_summary.columns]
            st.dataframe(
                df_summary[actual_cols]
                .sort_values(["Dataset", "Test"], ascending=[True, False])
                .style.background_gradient(
                    subset=["Test"] if "Test" in df_summary.columns else [],
                    cmap="Greens",
                ),
                use_container_width=True,
            )
            if st.button("💾 Save Leaderboard Snapshot Locally"):
                os.makedirs("outputs/all", exist_ok=True)
                df_summary.to_csv("outputs/all/leaderboard_snapshot.csv", index=False)
                st.success("Snapshot saved!")
            for ds in df_summary["Dataset"].unique():
                # Sort descending by Test performance for clarity
                ds_df = df_summary[df_summary["Dataset"] == ds].sort_values(
                    "Test", ascending=False
                )
                st.plotly_chart(
                    px.bar(
                        ds_df,
                        x="Method",
                        y="Test",
                        error_y="Test Std",
                        title=f"Leaderboard: {ds.upper()}",
                        color="Method",
                        template="plotly_white",
                    ),
                    use_container_width=True,
                )
        else:
            st.info("No leaderboard data loaded. Click 'Refresh' or 'Connect' above.")
else:
    st.error("📉 REIMS Dataset Missing")
    st.markdown(
        """
        The REIMS dataset is private and must be downloaded before you can use this dashboard.
        
        ### 1. Obtain a Token
        Generate a **Personal Access Token (classic)** with the `repo` scope from 
        [GitHub Developer Settings](https://github.com/settings/tokens).
        
        ### 2. Download Data
        Enter your token below to securely download the dataset to the internal package assets.
        """
    )

    with st.form("download_form"):
        token_input = st.text_input("GitHub Personal Access Token", type="password")
        submit = st.form_submit_button("🚀 Download Dataset")

        if submit:
            if token_input:
                from fishy._core.data_manager import download_dataset

                with st.spinner("Downloading REIMS.xlsx..."):
                    if download_dataset(token=token_input):
                        st.success("✅ Dataset downloaded! Please refresh the page.")
                        st.balloons()
                    else:
                        st.error("❌ Download failed. Check your token and connection.")
            else:
                st.warning("Please enter a valid token.")
