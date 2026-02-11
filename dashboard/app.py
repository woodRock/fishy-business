# -*- coding: utf-8 -*-
import warnings; import os
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning)
try: from urllib3.exceptions import NotOpenSSLWarning; warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except ImportError: pass
os.environ['WANDB_SILENT'] = 'true'; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; os.environ['MPLBACKEND'] = 'Agg'

import streamlit as st; import pandas as pd; import numpy as np
import plotly.express as px; import plotly.graph_objects as go
from pathlib import Path; import sys; import torch; import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_fscore_support, precision_recall_curve, average_precision_score
from sklearn.decomposition import PCA; from sklearn.manifold import TSNE
try: import umap
except ImportError: umap = None

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fishy._core.config import TrainingConfig; from fishy._core.config_loader import load_config
from fishy.data.module import create_data_module; from fishy.cli.main import detect_method
from fishy.experiments.deep_training import ModelTrainer; from fishy.experiments.classic_training import SklearnTrainer
from fishy.analysis.xai import GradCAM, ModelWrapper; from fishy._core.utils import get_device
from lime.lime_tabular import LimeTabularExplainer

st.set_page_config(page_title="Fishy Business | Spectral Analysis", page_icon="🐟", layout="wide")
st.markdown("""<style>
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    [data-testid="stMetric"] { background-color: #ffffff; padding: 10px; border-radius: 10px; }
    [data-testid="stMetricValue"] > div, [data-testid="stMetricLabel"] p, [data-testid="stMetricLabel"] div, [data-testid="stMetricLabel"] span { color: #1f2937 !important; }
    </style>""", unsafe_allow_html=True)

@st.cache_data
def get_metadata():
    models = load_config("models"); datasets = load_config("datasets"); all_model_names = []
    for section in ["deep_models", "classic_models", "evolutionary_models", "probabilistic_models"]:
        all_model_names.extend(list(models.get(section, {}).keys()))
    return sorted(all_model_names), sorted(list(datasets.keys()))

@st.cache_resource
def get_data_module(dataset_name, file_path, version="v11"):
    dm = create_data_module(dataset_name=dataset_name, file_path=file_path); dm.setup(); return dm

st.sidebar.title("🛠️ Configuration")
model_names, dataset_names = get_metadata(); data_path = Path("data/REIMS.xlsx")
selected_model = st.sidebar.selectbox("Select Model", model_names, index=model_names.index("transformer") if "transformer" in model_names else 0)
selected_dataset = st.sidebar.selectbox("Select Dataset", dataset_names, index=dataset_names.index("species") if "species" in dataset_names else 0)

dm = get_data_module(selected_dataset, str(data_path)); df_filtered = dm.get_filtered_dataframe()
if "Class Name" in df_filtered.columns: st.sidebar.success(f"✅ Loaded {len(df_filtered)} samples")
else: st.sidebar.warning("⚠️ Raw labels in use")

with st.sidebar.expander("🚀 Hyperparameters", expanded=True):
    epochs = st.slider("Epochs", 1, 100, 10); batch_size = st.select_slider("Batch Size", options=[8, 16, 32, 64, 128], value=32); lr = st.number_input("Learning Rate", value=1e-4, format="%.1e")
train_button = st.sidebar.button("🚀 Run Training", use_container_width=True)

st.title("🐟 Fishy Business")
tab1, tab2, tab3 = st.tabs(["📊 Data Exploration", "📈 Training & Results", "🧠 Interpretability & Biomarkers"])

if data_path.exists():
    class_names = dm.get_class_names(); X_all, y_all = dm.get_numpy_data(labels_as_indices=True)
    label_col = "Class Name" if "Class Name" in df_filtered.columns else df_filtered.columns[0]
    feature_cols = [c for c in df_filtered.columns if c not in [label_col, "m/z"]]
    try: mz_axis = np.array([float(c) for c in feature_cols])
    except: mz_axis = np.arange(len(feature_cols))

    with tab1:
        st.header("Deep Data Exploration")
        mc1, mc2, mc3 = st.columns([1, 1, 2])
        mc1.metric("Total Samples", len(df_filtered)); mc2.metric("Feature Count", dm.get_input_dim())
        with mc3:
            dist = df_filtered[label_col].value_counts()
            st.plotly_chart(px.bar(x=dist.index, y=dist.values, labels={'x': 'Class', 'y': 'Count'}, height=200, template="plotly_white"), use_container_width=True)
        st.markdown("---")
        c1, c2 = st.columns([2, 1])
        with c1:
            st.write("### Mean Spectral Signature")
            avg_fig = go.Figure()
            for idx, name in enumerate(class_names):
                mask = (y_all == idx)
                if np.any(mask):
                    m, s = X_all[mask].mean(axis=0), X_all[mask].std(axis=0)
                    avg_fig.add_trace(go.Scatter(x=mz_axis, y=m, name=name, mode='lines'))
                    avg_fig.add_trace(go.Scatter(x=np.concatenate([mz_axis, mz_axis[::-1]]), y=np.concatenate([m+s, (m-s)[::-1]]), fill='toself', fillcolor='rgba(0,100,80,0.1)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False))
            avg_fig.update_layout(template="plotly_white", xaxis_title="m/z", yaxis_title="Intensity")
            st.plotly_chart(avg_fig, use_container_width=True)
        with c2:
            st.write("### Interactive Viewer")
            all_classes = sorted(df_filtered[label_col].unique().tolist())
            sel_classes = st.multiselect("Filter Classes", all_classes, default=all_classes[:min(2, len(all_classes))])
            n_view = st.slider("Samples", 1, 20, 5)
            v_df = df_filtered[df_filtered[label_col].isin(sel_classes)] if sel_classes else df_filtered
            if not v_df.empty:
                s_df = v_df.sample(min(n_view, len(v_df))); num_cols = s_df.select_dtypes(include=[np.number]).columns.tolist()
                melted = s_df[[label_col]+num_cols].melt(id_vars=label_col, var_name="Feature", value_name="Int")
                st.plotly_chart(px.line(melted[melted["Feature"]!="m/z"], x="Feature", y="Int", color=label_col, template="plotly_white"), use_container_width=True)
        st.markdown("---")
        st.write("### Cluster Analysis")
        col_pca, col_tsne, col_umap = st.columns(3)
        pca_obj = PCA(n_components=10); X_pca = pca_obj.fit_transform(X_all)
        with col_pca: st.plotly_chart(px.scatter(pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(10)]), x="PC1", y="PC2", color=[class_names[i] for i in y_all], title="PCA", template="plotly_white"), use_container_width=True)
        with col_tsne:
            with st.spinner("t-SNE..."): X_tsne = TSNE(n_components=2, perplexity=min(30, len(X_all)-1)).fit_transform(X_all)
            st.plotly_chart(px.scatter(pd.DataFrame(X_tsne, columns=["TSNE1", "TSNE2"]), x="TSNE1", y="TSNE2", color=[class_names[i] for i in y_all], title="t-SNE", template="plotly_white"), use_container_width=True)
        with col_umap:
            if umap:
                with st.spinner("UMAP..."): X_umap = umap.UMAP().fit_transform(X_all)
                st.plotly_chart(px.scatter(pd.DataFrame(X_umap, columns=["UMAP1", "UMAP2"]), x="UMAP1", y="UMAP2", color=[class_names[i] for i in y_all], title="UMAP", template="plotly_white"), use_container_width=True)
            else: st.info("UMAP not installed")
        st.markdown("---")
        st.write("### Statistical Insight")
        dcol1, dcol2 = st.columns(2)
        with dcol1: st.plotly_chart(px.area(x=range(1, 11), y=np.cumsum(pca_obj.explained_variance_ratio_), labels={'x': 'Comp', 'y': 'Cum Var'}, title="Information Retention", template="plotly_white"), use_container_width=True)
        with dcol2:
            avg_int = pd.DataFrame({"Avg Int": X_all.mean(axis=1), "Class": [class_names[i] for i in y_all]})
            st.plotly_chart(px.violin(avg_int, x="Class", y="Avg Int", color="Class", box=True, points="all", title="Sample Intensity Dist", template="plotly_white"), use_container_width=True)

    with tab2:
        if train_button:
            config = TrainingConfig(model=selected_model, dataset=selected_dataset, file_path=str(data_path), epochs=epochs, batch_size=batch_size, learning_rate=lr, wandb_log=False)
            method = detect_method(selected_model); config.method = method
            try:
                with st.spinner(f"Training {selected_model}..."):
                    if method == "deep": trainer = ModelTrainer(config); model, results = trainer.train(); st.session_state['dm'] = trainer.data_module
                    else: trainer = SklearnTrainer(config, selected_model, selected_dataset, file_path=str(data_path)); model, results = trainer.run(); st.session_state['dm'] = dm
                    st.session_state['model'] = model; st.session_state['results'] = results; st.session_state['method'] = method
                st.success("🎉 Training Complete!")
                mc1, mc2, mc3, mc4 = st.columns(4); mc1.metric("Train Bal Acc", f"{results.get('train_balanced_accuracy', 0):.4f}"); mc2.metric("Val Bal Acc", f"{results.get('val_balanced_accuracy', 0):.4f}"); mc3.metric("Loss (Val)", f"{results.get('val_loss', 0):.4f}"); mc4.metric("Time", f"{results.get('total_training_time_s', 0):.2f}s")
                if "epoch_metrics" in results and results["epoch_metrics"]:
                    st.write("### Training Progress"); history = results["epoch_metrics"]; c_loss, c_acc = st.columns(2)
                    c_loss.plotly_chart(px.line(pd.DataFrame({"Epoch": range(1, len(history["train_losses"]) + 1), "Train Loss": history["train_losses"], "Val Loss": history["val_losses"]}), x="Epoch", y=["Train Loss", "Val Loss"], title="Loss Curve", template="plotly_white"), use_container_width=True)
                    c_acc.plotly_chart(px.line(pd.DataFrame({"Epoch": range(1, len(history["val_metrics"]) + 1), "Val Acc": [m.get("balanced_accuracy", 0) for m in history["val_metrics"]], "Train Acc": [m.get("balanced_accuracy", 0) for m in history["train_metrics"]]}), x="Epoch", y=["Train Acc", "Val Acc"], title="Accuracy Curve", template="plotly_white"), use_container_width=True)
                if "predictions" in results and results["predictions"]:
                    preds = results["predictions"]; y_true, y_pred, y_probs = preds["labels"], preds["preds"], preds.get("probs")
                    st.write("### Results Analysis")
                    r1c1, r1c2 = st.columns(2)
                    with r1c1: st.plotly_chart(px.imshow(confusion_matrix(y_true, y_pred), x=class_names, y=class_names, text_auto=True, color_continuous_scale='Blues', title="Confusion Matrix", template="plotly_white"), use_container_width=True)
                    with r1c2:
                        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=range(len(class_names)))
                        st.plotly_chart(px.bar(pd.DataFrame({"Class": class_names, "Precision": prec, "Recall": rec, "F1": f1}), x="Class", y=["Precision", "Recall", "F1"], barmode="group", title="Class Metrics", template="plotly_white"), use_container_width=True)
                    
                    st.markdown("---")
                    st.write("### Advanced Performance Metrics")
                    r2c1, r2c2 = st.columns(2)
                    with r2c1:
                        if y_probs is not None:
                            pr_fig = go.Figure()
                            for i in range(len(class_names)):
                                precision, recall, _ = precision_recall_curve(y_true == i, y_probs[:, i]); avg_prec = average_precision_score(y_true == i, y_probs[:, i])
                                pr_fig.add_trace(go.Scatter(x=recall, y=precision, name=f'{class_names[i]} (AP={avg_prec:.2f})', mode='lines'))
                            pr_fig.update_layout(template="plotly_white", xaxis_title="Recall", yaxis_title="Precision", title="Precision-Recall Curve"); st.plotly_chart(pr_fig, use_container_width=True)
                    with r2c2:
                        if y_probs is not None:
                            # Confident Error Analysis
                            correct = (y_true == y_pred)
                            errors_idx = np.where(~correct)[0]
                            if len(errors_idx) > 0:
                                # Sort errors by confidence in the WRONG class
                                error_probs = np.max(y_probs[errors_idx], axis=1)
                                top_errors = errors_idx[np.argsort(error_probs)[-3:][::-1]]
                                st.write("#### Misclassification Spotlight (Confident Errors)")
                                for idx in top_errors:
                                    true_label, pred_label = class_names[y_true[idx]], class_names[y_pred[idx]]
                                    st.error(f"Sample {idx}: Truth = **{true_label}**, Predicted = **{pred_label}** (Conf: {np.max(y_probs[idx]):.2f})")
                            else: st.success("No misclassifications in this fold!")

            except Exception as e: st.error(f"Failed: {e}"); st.exception(e)
        elif 'results' in st.session_state: st.info("Last results active.")
        else: st.info("Click 'Run Training'.")

    with tab3:
        if 'model' in st.session_state:
            st.header("Interpretability & Biomarkers")
            model = st.session_state['model']; current_dm = st.session_state['dm']
            X_xai, y_xai = current_dm.get_numpy_data(labels_as_indices=True)
            
            # INITIALIZE REP INDICES PROPERLY
            if 'rep_indices' not in st.session_state or len(st.session_state['rep_indices']) != len(class_names):
                st.session_state['rep_indices'] = {}
                for c in range(len(class_names)):
                    idxs = np.where(y_xai == c)[0]
                    st.session_state['rep_indices'][c] = idxs[0] if len(idxs) > 0 else 0

            exp_col1, exp_col2 = st.columns([1, 2])
            with exp_col1:
                sample_idx = st.selectbox("Select instance", range(len(X_xai)))
                method = st.radio("Method", ["Grad-CAM", "LIME"] if st.session_state.get('method') == "deep" else ["LIME"], horizontal=True)
                st.info(f"Targeting: **{class_names[y_xai[sample_idx]]}**", icon="🎯")
            if method == "Grad-CAM":
                target_layer = next((m for m in reversed(list(model.modules())) if isinstance(m, (nn.Conv1d, nn.Linear))), None)
                if target_layer:
                    gc = GradCAM(model, target_layer); cam = gc.generate_cam(torch.tensor(X_xai[sample_idx]).unsqueeze(0).to(get_device())).cpu().numpy()[0]; gc.remove_hooks()
                    fig = go.Figure(); fig.add_trace(go.Scatter(x=mz_axis, y=X_xai[sample_idx], name="Spectrum", line=dict(color='lightgray', width=1))); fig.add_trace(go.Scatter(x=mz_axis, y=X_xai[sample_idx], mode='markers', marker=dict(color=cam, colorscale='Viridis', size=8, showscale=True), name="Importance", hovertemplate="m/z: %{x}<br>Int: %{y}<br>Imp: %{marker.color:.4f}")); fig.update_layout(title="Importance Map", template="plotly_white", xaxis_title="m/z", yaxis_title="Intensity"); exp_col2.plotly_chart(fig, use_container_width=True)
            else:
                wrapper = ModelWrapper(model, str(get_device())); explainer = LimeTabularExplainer(X_xai, feature_names=[f"{m:.4f}" for m in mz_axis], class_names=class_names, discretize_continuous=True)
                with st.spinner("LIME..."): exp = explainer.explain_instance(X_xai[sample_idx], wrapper.predict_proba, num_features=15)
                exp_col2.plotly_chart(px.bar(x=[x[1] for x in exp.as_list()], y=[x[0] for x in exp.as_list()], orientation='h', color=[x[1] for x in exp.as_list()], color_continuous_scale='RdBu', title="LIME Weights"), use_container_width=True)
            
            st.markdown("---")
            st.write("### 🌐 Distinct Class Biomarker Comparison")
            c_btn1, c_btn2 = st.columns([1, 4])
            if c_btn1.button("🔀 Shuffle Classes"):
                for c in range(len(class_names)):
                    idxs = np.where(y_xai == c)[0]
                    if len(idxs) > 0: st.session_state['rep_indices'][c] = np.random.choice(idxs)
            
            if st.button("🔍 Run Class-Specific Analysis", use_container_width=True):
                with st.spinner("Analyzing stable biomarkers..."):
                    wrapper = ModelWrapper(model, str(get_device()))
                    exp_int = LimeTabularExplainer(X_xai, feature_names=[str(i) for i in range(X_xai.shape[1])], class_names=class_names, discretize_continuous=False)
                    class_biomarkers = {}
                    for c_idx in range(len(class_names)):
                        c_idxs = np.where(y_xai == c_idx)[0]
                        if len(c_idxs) > 0:
                            sub = np.random.choice(c_idxs, min(10, len(c_idxs)), replace=False); w_list = []
                            for idx in sub: w_list.append(dict(exp_int.explain_instance(X_xai[idx], wrapper.predict_proba, num_features=X_xai.shape[1], labels=(c_idx,)).as_list(label=c_idx)))
                            avg_w = pd.DataFrame(w_list).mean().sort_values(); class_biomarkers[c_idx] = [int(i) for i in avg_w.tail(20).index.tolist()]
                    
                    comp_fig = go.Figure(); styles = [{'c': 'gold', 's': 'diamond'}, {'c': 'silver', 's': 'circle'}, {'c': 'cyan', 's': 'square'}]
                    for c_idx, c_name in enumerate(class_names):
                        rep_idx = st.session_state['rep_indices'].get(c_idx)
                        if rep_idx is not None and c_idx in class_biomarkers:
                            spec = X_xai[rep_idx]; stl = styles[c_idx % len(styles)]
                            comp_fig.add_trace(go.Scatter(x=mz_axis, y=spec, name=f"{c_name} (Smp {rep_idx})", line=dict(width=1), opacity=0.4))
                            top_idx = class_biomarkers[c_idx]
                            comp_fig.add_trace(go.Scatter(x=mz_axis[top_idx], y=spec[top_idx], mode='markers', marker=dict(color=stl['c'], size=12, symbol=stl['s'], line=dict(width=2, color='black')), name=f"Diagnostic: {c_name}", hovertemplate="m/z: %{x}<br>Int: %{y}"))
                    comp_fig.update_layout(title="Gold vs Silver: Distinct Class Representatives & Biomarkers", template="plotly_white", xaxis_title="m/z", yaxis_title="Intensity")
                    st.plotly_chart(comp_fig, use_container_width=True)
                    st.write("#### Top Diagnostic Peaks")
                    cols = st.columns(len(class_names))
                    for i, c_name in enumerate(class_names):
                        if i in class_biomarkers:
                            top_v = [f"{mz_axis[idx]:.4f}" for idx in class_biomarkers[i][-10:][::-1]]
                            cols[i].write(f"**{c_name}**"); 
                            for v in top_v: cols[i].code(v)
        else: st.info("Run training first.")
else: st.warning("Data file not found.")
