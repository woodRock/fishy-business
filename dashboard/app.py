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
    return sorted(all_model_names), sorted(list(datasets.keys()))


@st.cache_resource
def get_data_module(dataset_name, file_path, version="v12"):
    dm = create_data_module(dataset_name=dataset_name, file_path=file_path)
    dm.setup()
    return dm


def parse_result_path(path_str):
    """Robustly extracts dataset and model from result path using regex."""
    # Pattern: .../<dataset>/<method>/<model>_<timestamp>/results/metrics.json
    # We want the dataset and the first part of the model_timestamp folder
    match = re.search(r"outputs/([^/]+)/([^/]+)/([^/_]+)_", path_str)
    if match:
        return match.group(1), match.group(3)
    return None, None


def crawl_local_results(base_path="outputs"):
    results_map = {}
    p = Path(base_path)
    # Search for all possible result filenames
    for m_path in (
        list(p.glob("**/results/metrics.json"))
        + list(p.glob("**/results/final_metrics.json"))
        + list(p.glob("**/results/aggregated_stats_*.json"))
    ):
        dataset, model = parse_result_path(str(m_path))
        if dataset and model:
            key = f"{dataset}|||{model}"
            if key not in results_map:
                results_map[key] = []
            try:
                with open(m_path, "r") as f:
                    results_map[key].append(json.load(f))
            except Exception as e:
                logger.warning(f"Malformed local JSON {m_path}: {e}")
    return summarize_results(results_map) if results_map else pd.DataFrame()


def fetch_remote_data(
    host, port, user, pwd, remote_path, jump_host=None, jump_user=None, otp=None
):
    results_map = {}
    try:
        ssh_target = paramiko.SSHClient()
        ssh_target.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        if jump_host:

            def handler(title, instructions, prompt_list):
                responses = []
                for prompt, _ in prompt_list:
                    p = prompt.lower()
                    if "password" in p:
                        responses.append(pwd)
                    elif "verification" in p or "token" in p or "code" in p:
                        responses.append(otp if otp else "")
                    else:
                        responses.append("")
                return responses

            transport = paramiko.Transport((jump_host, 22))
            transport.start_client()
            transport.auth_interactive(jump_user or user, handler)
            dest_addr = (host, port)
            local_addr = ("localhost", 0)
            jump_channel = transport.open_channel("direct-tcpip", dest_addr, local_addr)
            ssh_target.connect(
                host,
                port=port,
                username=user,
                password=pwd,
                sock=jump_channel,
                timeout=60,
            )
        else:
            ssh_target.connect(host, port=port, username=user, password=pwd, timeout=60)

        sftp = ssh_target.open_sftp()
        # Find all types of result files
        cmd = f"find {remote_path} -maxdepth 5 \( -name 'metrics.json' -o -name 'final_metrics.json' -o -name 'aggregated_stats_*.json' \)"
        stdin, stdout, stderr = ssh_target.exec_command(cmd)
        paths = stdout.read().decode().splitlines()
        for p in paths:
            dataset, model = parse_result_path(p)
            if dataset and model:
                key = f"{dataset}|||{model}"
                if key not in results_map:
                    results_map[key] = []
                try:
                    with sftp.open(p, "r") as f:
                        results_map[key].append(json.load(f))
                except Exception as e:
                    logger.warning(f"Malformed remote JSON {p}: {e}")
        ssh_target.close()
        if jump_host:
            transport.close()
    except Exception as e:
        st.error(f"Network Error: {e}")
    return summarize_results(results_map) if results_map else pd.DataFrame()


st.sidebar.title("🛠️ Configuration")
model_names, dataset_names = get_metadata()
data_path = get_data_path()
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

dm = get_data_module(selected_dataset, str(data_path))
df_filtered = dm.get_filtered_dataframe()
if "Class Name" in df_filtered.columns:
    st.sidebar.success(f"✅ Loaded {len(df_filtered)} samples")
else:
    st.sidebar.warning("⚠️ Raw labels in use")

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

st.sidebar.markdown("---")
run_all_button = st.sidebar.button("📊 Run Full Benchmark (Quick)", width="stretch")

st.title("🐟 Fishy Business")
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Data Exploration",
        "📈 Training & Results",
        "🧠 Interpretability & Biomarkers",
        "🏆 Leaderboard",
    ]
)

if data_path.exists():
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
        st.write("### Mean Spectral Signature (Full Panel)")
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
        st.write("### Interactive Spectrum Viewer (Full Panel)")
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
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric(
                    "Train Bal Acc", f"{results.get('train_balanced_accuracy', 0):.4f}"
                )
                mc2.metric(
                    "Val Bal Acc", f"{results.get('val_balanced_accuracy', 0):.4f}"
                )
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
                                x=class_names,
                                y=class_names,
                                text_auto=True,
                                color_continuous_scale="Blues",
                                title="Confusion Matrix",
                                template="plotly_white",
                            ),
                            use_container_width=True,
                        )
                    with r1c2:
                        prec, rec, f1, _ = precision_recall_fscore_support(
                            y_true, y_pred, labels=range(len(class_names))
                        )
                        st.plotly_chart(
                            px.bar(
                                pd.DataFrame(
                                    {
                                        "Class": class_names,
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
                            for i in range(len(class_names)):
                                precision, recall, _ = precision_recall_curve(
                                    y_true == i, y_probs[:, i]
                                )
                                avg_prec = average_precision_score(
                                    y_true == i, y_probs[:, i]
                                )
                                pr_fig.add_trace(
                                    go.Scatter(
                                        x=recall,
                                        y=precision,
                                        name=f"{class_names[i]} (AP={avg_prec:.2f})",
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
                                    np.argsort(np.max(y_probs[err_idx], axis=1))[-3:][
                                        ::-1
                                    ]
                                ]
                                for idx in top_errs:
                                    st.error(
                                        f"Sample {idx}: Truth={class_names[y_true[idx]]}, Pred={class_names[y_pred[idx]]} (Conf: {np.max(y_probs[idx]):.2f})"
                                    )
                            else:
                                st.success("Perfect classification!")

                    st.markdown("---")
                    st.write("### Stability")
                    if "folds" in results:
                        fold_accs = [
                            f.get("val_balanced_accuracy", 0)
                            for f in results.get("folds", [])
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
            except Exception as e:
                st.error(f"Failed: {e}")
                st.exception(e)
        elif "results" in st.session_state:
            st.info("Showing last results.")
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
                target_layer = next(
                    (
                        m
                        for m in reversed(list(model.modules()))
                        if isinstance(m, (nn.Conv1d, nn.Linear))
                    ),
                    None,
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
            ["Local (outputs/)", "Remote (SSH Proxy Jump)"],
            horizontal=True,
        )
        if "leaderboard_df" not in st.session_state:
            st.session_state["leaderboard_df"] = pd.DataFrame()
        if src == "Local (outputs/)":
            if st.button("🔄 Refresh Local Leaderboard"):
                with st.spinner("Crawling local metrics..."):
                    st.session_state["leaderboard_df"] = crawl_local_results()
        else:
            if not paramiko:
                st.error("paramiko not installed")
            else:
                with st.expander(
                    "🔑 Jump Host Configuration (entry.ecs.vuw.ac.nz)", expanded=True
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
                    with st.spinner("Tunnelling through entry and crawling metrics..."):
                        st.session_state["leaderboard_df"] = fetch_remote_data(
                            th, tp, ju, pwd, rp, jump_host=jh, jump_user=ju, otp=otp
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
                st.plotly_chart(
                    px.bar(
                        df_summary[df_summary["Dataset"] == ds],
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
    st.warning("Data file not found.")
