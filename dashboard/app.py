# -*- coding: utf-8 -*-
import warnings; import os
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning)
try: from urllib3.exceptions import NotOpenSSLWarning; warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except ImportError: pass
os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'; os.environ['MPLBACKEND'] = 'Agg'

import streamlit as st; import pandas as pd; import numpy as np
import plotly.express as px; import plotly.graph_objects as go
from pathlib import Path; import sys; import torch; import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_curve, auc; from sklearn.decomposition import PCA

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
def get_data_module(dataset_name, file_path):
    dm = create_data_module(dataset_name=dataset_name, file_path=file_path)
    dm.setup()
    return dm

st.sidebar.title("🛠️ Configuration")
model_names, dataset_names = get_metadata()
selected_model = st.sidebar.selectbox("Select Model", model_names, index=model_names.index("transformer") if "transformer" in model_names else 0)
selected_dataset = st.sidebar.selectbox("Select Dataset", dataset_names, index=dataset_names.index("species") if "species" in dataset_names else 0)
with st.sidebar.expander("🚀 Hyperparameters", expanded=True):
    epochs = st.slider("Epochs", 1, 100, 10); batch_size = st.select_slider("Batch Size", options=[8, 16, 32, 64, 128], value=32); lr = st.number_input("Learning Rate", value=1e-4, format="%.1e")
train_button = st.sidebar.button("🚀 Run Training", use_container_width=True)

st.title("🐟 Fishy Business")
tab1, tab2, tab3 = st.tabs(["📊 Data Exploration", "📈 Training & Results", "🧠 Model Interpretability"])
data_path = Path("data/REIMS.xlsx")

if data_path.exists():
    dm = get_data_module(selected_dataset, str(data_path))
    # FIX: Use filtered dataframe for all visualizations
    df_filtered = dm.get_filtered_dataframe(); class_names = dm.get_class_names()
    X_all, y_all = dm.get_numpy_data(labels_as_indices=True)

    with tab1:
        st.header(f"Exploration: {selected_dataset.upper()}")
        
        # Row 1: Key Metrics & Distribution
        mc1, mc2, mc3 = st.columns([1, 1, 2])
        mc1.metric("Total Samples", len(df_filtered))
        mc2.metric("Feature Count", dm.get_input_dim())
        with mc3:
            dist = df_filtered.iloc[:, 0].value_counts()
            fig_dist = px.bar(x=dist.index, y=dist.values, labels={'x': 'Class', 'y': 'Count'}, height=200, template="plotly_white")
            fig_dist.update_layout(margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown("---")

        # Row 2: Filtered Viewer
        c1, c2 = st.columns([3, 1])
        with c1:
            st.write("### Interactive Spectrum Viewer")
            all_classes = sorted(df_filtered.iloc[:, 0].unique().tolist())
            selected_classes = st.multiselect("Filter Viewer by Class", all_classes, default=all_classes[:min(2, len(all_classes))])
            
            n_samples = st.slider("Max samples to overlay", 1, 50, 10)
            
            if selected_classes:
                view_df = df_filtered[df_filtered.iloc[:, 0].isin(selected_classes)]
            else:
                view_df = df_filtered
                
            if not view_df.empty:
                sample_df = view_df.sample(min(n_samples, len(view_df)))
                # Col 0 is targets, Col 1 is m/z identifier, rest are intensities
                # Need to check column structure for melting correctly
                melted = sample_df.melt(id_vars=view_df.columns[0], var_name="Feature", value_name="Intensity")
                # Filter out the 'm/z' string identifier from plot if it exists
                melted = melted[melted["Feature"] != "m/z"]
                fig = px.line(melted, x="Feature", y="Intensity", color=view_df.columns[0], template="plotly_white", title=f"Overlaid Spectra ({len(sample_df)} samples)")
                fig.update_layout(xaxis_title="m/z index", yaxis_title="Intensity")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Select at least one class to visualize.")
            
        with c2:
            st.write("### Feature Variance")
            variances = np.var(X_all, axis=0); top_idx = np.argsort(variances)[-30:][::-1]
            st.plotly_chart(px.bar(x=[f"mz_{i}" for i in top_idx], y=variances[top_idx], labels={'x': 'Feature', 'y': 'Var'}, height=450, template="plotly_white"), use_container_width=True)

        st.markdown("---")

        # Row 3: Averages and PCA
        avg_col, pca_col = st.columns(2)
        with avg_col:
            st.write("### Mean Spectral Signature")
            avg_fig = go.Figure()
            for idx, name in enumerate(class_names):
                class_mask = (y_all == idx)
                if np.any(class_mask):
                    mean_spec = X_all[class_mask].mean(axis=0); std_spec = X_all[class_mask].std(axis=0)
                    x_vals = np.arange(len(mean_spec))
                    avg_fig.add_trace(go.Scatter(x=x_vals, y=mean_spec, name=name, mode='lines'))
                    avg_fig.add_trace(go.Scatter(x=np.concatenate([x_vals, x_vals[::-1]]), y=np.concatenate([mean_spec + std_spec, (mean_spec - std_spec)[::-1]]), fill='toself', fillcolor='rgba(0,100,80,0.1)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False))
            avg_fig.update_layout(template="plotly_white", margin=dict(t=30))
            st.plotly_chart(avg_fig, use_container_width=True)
            
        with pca_col:
            st.write("### Global Data Topology (PCA)")
            pca = PCA(n_components=2); X_proj = pca.fit_transform(X_all)
            proj_df = pd.DataFrame(X_proj, columns=["PC1", "PC2"]); proj_df["Class"] = [class_names[i] for i in y_all]
            fig_pca = px.scatter(proj_df, x="PC1", y="PC2", color="Class", template="plotly_white")
            fig_pca.update_layout(margin=dict(t=30))
            st.plotly_chart(fig_pca, use_container_width=True)

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
                mc1, mc2, mc3, mc4 = st.columns(4); mc1.metric("Train Acc", f"{results.get('train_balanced_accuracy', 0):.4f}"); mc2.metric("Val Acc", f"{results.get('val_balanced_accuracy', 0):.4f}"); mc3.metric("F1 (Val)", f"{results.get('val_f1', 0):.4f}"); mc4.metric("Loss (Val)", f"{results.get('val_loss', 0):.4f}")
                
                if "epoch_metrics" in results:
                    st.write("### Training Progress")
                    history = results["epoch_metrics"]; c_loss, c_acc = st.columns(2)
                    loss_df = pd.DataFrame({"Epoch": range(1, len(history["train_losses"]) + 1), "Train Loss": history["train_losses"], "Val Loss": history["val_losses"]})
                    c_loss.plotly_chart(px.line(loss_df, x="Epoch", y=["Train Loss", "Val Loss"], title="Loss Curve", template="plotly_white"), use_container_width=True)
                    acc_df = pd.DataFrame({"Epoch": range(1, len(history["val_metrics"]) + 1), "Val Acc": [m.get("balanced_accuracy", 0) for m in history["val_metrics"]], "Train Acc": [m.get("balanced_accuracy", 0) for m in history["train_metrics"]]})
                    c_acc.plotly_chart(px.line(acc_df, x="Epoch", y=["Train Acc", "Val Acc"], title="Accuracy Curve", template="plotly_white"), use_container_width=True)

                if "predictions" in results and results["predictions"]:
                    preds = results["predictions"]; y_true, y_pred, y_probs = preds["labels"], preds["preds"], preds.get("probs")
                    st.write("### Error Analysis")
                    ec1, ec2 = st.columns(2)
                    with ec1:
                        st.write("#### Confusion Matrix")
                        st.plotly_chart(px.imshow(confusion_matrix(y_true, y_pred), x=class_names, y=class_names, text_auto=True, color_continuous_scale='Blues', aspect="auto"), use_container_width=True)
                    with ec2:
                        if y_probs is not None:
                            st.write("#### ROC Curve")
                            roc_fig = go.Figure()
                            for i in range(len(class_names)):
                                fpr, tpr, _ = roc_curve(y_true == i, y_probs[:, i]); roc_auc = auc(fpr, tpr)
                                roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'{class_names[i]} (AUC={roc_auc:.2f})', mode='lines'))
                            roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash='dash'), name='Random'))
                            roc_fig.update_layout(template="plotly_white", xaxis_title="FPR", yaxis_title="TPR")
                            st.plotly_chart(roc_fig, use_container_width=True)
                        else: st.info("Probabilities not available for this model type.")
                    
                    if y_probs is not None:
                        st.write("#### Prediction Confidence Distribution")
                        confidences = np.max(y_probs, axis=1); correct = (y_true == y_pred)
                        conf_df = pd.DataFrame({"Confidence": confidences, "Outcome": ["Correct" if c else "Incorrect" for c in correct]})
                        st.plotly_chart(px.histogram(conf_df, x="Confidence", color="Outcome", barmode="overlay", template="plotly_white"), use_container_width=True)

            except Exception as e: st.error(f"Failed: {e}"); st.exception(e)
        elif 'results' in st.session_state: st.info("Showing results from last session.")
        else: st.info("Configure and run training to see results.")

    with tab3:
        if 'model' in st.session_state:
            st.header("Explainable AI Analysis")
            model = st.session_state['model']; current_dm = st.session_state['dm']
            X_xai, y_xai = current_dm.get_numpy_data(labels_as_indices=True)
            sample_idx = st.selectbox("Select sample", range(len(X_xai)))
            available_methods = ["LIME"]; 
            if st.session_state.get('method') == "deep": available_methods.append("Grad-CAM")
            method = st.radio("Method", available_methods, horizontal=True)
            
            if method == "Grad-CAM":
                target_layer = next((m for m in reversed(list(model.modules())) if isinstance(m, (nn.Conv1d, nn.Linear))), None)
                if target_layer:
                    gc = GradCAM(model, target_layer)
                    cam = gc.generate_cam(torch.tensor(X_xai[sample_idx]).unsqueeze(0).to(get_device())).cpu().numpy()[0]
                    gc.remove_hooks(); fig = go.Figure()
                    fig.add_trace(go.Scatter(x=np.arange(len(X_xai[sample_idx])), y=X_xai[sample_idx], name="Spectrum", line=dict(color='lightgray')))
                    fig.add_trace(go.Scatter(x=np.arange(len(cam)), y=X_xai[sample_idx], mode='markers', marker=dict(color=cam, colorscale='Hot', size=6, showscale=True), name="Importance"))
                    fig.update_layout(title=f"Grad-CAM (Actual: {class_names[y_xai[sample_idx]]})", template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                wrapper = ModelWrapper(model, str(get_device()))
                explainer = LimeTabularExplainer(X_xai, feature_names=[f"mz_{i}" for i in range(X_xai.shape[1])], class_names=class_names, discretize_continuous=True)
                with st.spinner("LIME..."): exp = explainer.explain_instance(X_xai[sample_idx], wrapper.predict_proba, num_features=10)
                lime_data = exp.as_list(); features, weights = zip(*lime_data)
                st.plotly_chart(px.bar(x=weights, y=features, orientation='h', color=weights, color_continuous_scale='RdBu', title=f"LIME Features (Actual: {class_names[y_xai[sample_idx]]})"), use_container_width=True)
        else: st.info("Run training first.")
else: st.warning("Data not found at data/REIMS.xlsx")
