import os
import glob
import random
import time
import math

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score


# -------------------------------------------------------------------
# --- 1. Early Stopping Class
# -------------------------------------------------------------------
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self, patience=5, verbose=False, delta=0, path="checkpoint.pt", trace_func=print
    ):
        self.patience, self.verbose, self.counter = patience, verbose, 0
        self.best_score, self.early_stop, self.val_loss_min = None, False, np.Inf
        self.delta, self.path, self.trace_func = delta, path, trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(
                f"Validation MAE decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# -------------------------------------------------------------------
# --- 2. Custom Transformer Model Definition
# -------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.dropout(x + self.pe[: x.size(0), :])


class TransformerOrdinal(nn.Module):
    def __init__(
        self,
        input_features,
        d_model,
        nhead,
        num_encoder_layers,
        dim_feedforward,
        dropout,
        method="regression",
        num_classes=None,
        batch_first=True,
    ):
        super(TransformerOrdinal, self).__init__()
        self.method = method
        self.input_embedding = nn.Linear(input_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=5000)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=batch_first
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers
        )
        self.d_model = d_model

        if method == "regression":
            self.output_layer = nn.Linear(d_model, 1)
        elif method == "classification":
            self.output_layer = nn.Linear(d_model, num_classes)
        elif method == "coral":
            self.output_layer = nn.Linear(d_model, num_classes - 1)
        elif method == "clm":
            self.output_layer = nn.Linear(d_model, 1)
            self.cutpoints_gaps = nn.Parameter(torch.zeros(num_classes - 1))
            self.cutpoints_gaps.data[0] = 0.5
        else:
            raise ValueError(f"Method '{method}' not recognized.")

    def forward(self, src):
        embedded_src = self.input_embedding(src) * math.sqrt(self.d_model)
        pos_encoded_src = self.pos_encoder(embedded_src.permute(1, 0, 2)).permute(
            1, 0, 2
        )
        encoder_output = self.transformer_encoder(pos_encoded_src)
        aggregated_output = encoder_output[:, 0, :]

        if self.method == "clm":
            eta = self.output_layer(aggregated_output)
            cutpoints = torch.cumsum(
                torch.nn.functional.softplus(self.cutpoints_gaps), dim=0
            )
            return cutpoints - eta
        else:
            return self.output_layer(aggregated_output)


# -------------------------------------------------------------------
# --- 3. Custom Loss and Helper Functions
# -------------------------------------------------------------------
class CumulativeLinkLoss(nn.Module):
    def forward(self, cum_logits, y_true):
        cum_probs = torch.sigmoid(cum_logits)
        padded_cum_probs = torch.cat(
            [
                torch.zeros_like(cum_probs[:, :1]),
                cum_probs,
                torch.ones_like(cum_probs[:, :1]),
            ],
            dim=1,
        )
        class_probs = padded_cum_probs[:, 1:] - padded_cum_probs[:, :-1]
        true_class_probs = class_probs.gather(1, y_true.unsqueeze(1)).squeeze(1)
        return -torch.log(true_class_probs + 1e-8).mean()


def label_to_coral_levels(labels, num_classes):
    return torch.stack(
        [torch.where(labels >= i, 1.0, 0.0) for i in range(1, num_classes)], dim=1
    )


def coral_logits_to_prediction(logits):
    return torch.sum(torch.sigmoid(logits) > 0.5, dim=1)


def read_reims_excel_file(fp):
    return pd.read_excel(fp)


def filter_data_for_oil(df):
    return df[df["m/z"].str.contains("MO ", na=False)].copy()


def remove_first_two_characters(df):
    return df["m/z"].str[2:]


def convert_class_labels_to_integers(df):
    df.loc[:, "m/z"] = df["m/z"].astype("category").cat.codes
    return df


def get_dataloader(df, batch_size=32):
    # Ensure not to shuffle here, as splitting is done externally
    for start in range(0, len(df), batch_size):
        yield df.iloc[start : min(start + batch_size, len(df))]


# -------------------------------------------------------------------
# --- 4. Training and Evaluation Functions
# -------------------------------------------------------------------
def train(
    model,
    optimizer,
    criterion,
    train_df,
    val_df,
    epochs,
    device,
    batch_size,
    method,
    num_classes,
    patience,
    checkpoint_path,
    verbose=False,
):
    early_stopper = EarlyStopping(
        patience=patience, verbose=verbose, path=checkpoint_path
    )

    for epoch in range(epochs):
        model.train()
        for batch in get_dataloader(train_df, batch_size):
            x, y = batch.drop(columns=["m/z"]).values, batch["m/z"].values
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(-1).to(device)
            y_true_labels = torch.tensor(y.astype(np.int64), dtype=torch.long).to(
                device
            )

            if method == "regression":
                y_target = y_true_labels.float().view(-1, 1)
            elif method in ["classification", "clm"]:
                y_target = y_true_labels
            elif method == "coral":
                y_target = label_to_coral_levels(y_true_labels, num_classes).to(device)

            optimizer.zero_grad()
            outputs = model(x_tensor)
            loss = criterion(outputs, y_target)
            loss.backward()
            optimizer.step()

        val_mae, val_bca = evaluate(
            model, get_dataloader(val_df, batch_size), device, method
        )
        if verbose:
            print(
                f"Epoch {epoch+1:02d}/{epochs}, Val MAE: {val_mae:.4f}, Val BCA: {val_bca:.4f}"
            )

        early_stopper(val_mae, model)  # Use MAE for early stopping
        if early_stopper.early_stop:
            if verbose:
                print("Early stopping triggered")
            break


def evaluate(model, data_loader, device, method):
    """Calculates MAE and Balanced Classification Accuracy (BCA)."""
    model.eval()
    all_preds_float = []  # For MAE
    all_preds_int = []  # For BCA
    all_true_labels = []  # For both

    with torch.no_grad():
        for batch in data_loader:
            x, y = batch.drop(columns=["m/z"]).values, batch["m/z"].values
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(-1).to(device)
            y_tensor_int = torch.tensor(y.astype(np.int64)).to(
                device
            )  # True labels as int

            all_true_labels.append(y_tensor_int.cpu().numpy())

            outputs = model(x_tensor)

            if method == "regression":
                preds_float = outputs.view(-1)
                preds_int = torch.round(
                    preds_float
                ).long()  # Round to nearest int for class
            elif method == "classification":
                preds_float = torch.argmax(outputs, dim=1).float()
                preds_int = torch.argmax(outputs, dim=1).long()
            elif method in ["coral", "clm"]:
                preds_float = coral_logits_to_prediction(outputs).float()
                preds_int = coral_logits_to_prediction(outputs).long()

            all_preds_float.append(preds_float.cpu().numpy())
            all_preds_int.append(preds_int.cpu().numpy())

    # Concatenate all batches
    all_true_labels = np.concatenate(all_true_labels)
    all_preds_float = np.concatenate(all_preds_float)
    all_preds_int = np.concatenate(all_preds_int)

    mae = np.mean(np.abs(all_preds_float - all_true_labels.astype(float)))

    try:
        bca = balanced_accuracy_score(all_true_labels, all_preds_int)
    except ValueError:
        print(
            f"Warning: Could not compute BCA for method '{method}'. May be missing classes in predictions."
        )
        bca = 0.0

    return mae, bca


# -------------------------------------------------------------------
# --- 5. Main Execution Block
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting the script...")

    if torch.cuda.is_available():
        device = "cuda"
        print("🚀 CUDA is available! Using GPU.")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("🚀 MPS is available! Using Apple Silicon GPU.")
    else:
        device = "cpu"
        print("🐌 No GPU available, using CPU.")

    # --- Hyperparameters ---
    BATCH_SIZE, D_MODEL, N_HEAD = 8, 64, 4
    NUM_LAYERS, DIM_FEEDFORWARD = 2, 256
    EPOCHS, LR, PATIENCE = 50, 1e-4, 5
    K_FOLDS = 5
    SEED = 42

    # --- Set master seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # --- Load and preprocess all data ---
    df = read_reims_excel_file("/Users/woodj/Desktop/fishy-business/data/REIMS.xlsx")
    df = filter_data_for_oil(df)
    df["m/z"] = remove_first_two_characters(df)
    df = convert_class_labels_to_integers(df)

    # Separate features (X) and labels (y) for SKFold
    X = df.drop(columns=["m/z"])
    y = df["m/z"]

    # Create a clean NumPy array of ints for skf.split
    y_skf = y.to_numpy().astype(int)

    NUM_CLASSES = y.nunique()
    print(f"Detected {NUM_CLASSES} unique classes.")

    methods_to_compare = ["regression", "classification", "coral", "clm"]

    # --- Data structure to hold all results from all folds ---
    all_results = {
        method: {"Train MAE": [], "Train BCA": [], "Test MAE": [], "Test BCA": []}
        for method in methods_to_compare
    }

    # --- Initialize Stratified K-Fold ---
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

    # --- K-Fold Cross-Validation Loop ---

    # <-- MODIFIED THIS LINE
    # Use the clean NumPy array 'y_skf' for splitting
    for fold, (train_val_index, test_index) in enumerate(skf.split(X, y_skf)):
        print(f"\n{'='*20} FOLD {fold + 1}/{K_FOLDS} {'='*20}")

        # --- Create Train/Val and Test sets for this fold ---
        # We still use the original pandas X and y for iloc
        X_train_val, y_train_val = X.iloc[train_val_index], y.iloc[train_val_index]
        X_test, y_test = X.iloc[test_index], y.iloc[test_index]

        # --- Split Train/Val into Train and Validation (for early stopping) ---
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=0.15,
            stratify=y_train_val,
            random_state=SEED,
        )

        # --- Re-create DataFrames for the dataloader function ---
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        # --- Method Loop (inside K-Fold loop) ---
        for method in methods_to_compare:
            print(f"--- Training method: {method.upper()} ---")

            checkpoint_path = f"checkpoint_{method}_fold{fold}.pt"

            model = TransformerOrdinal(
                input_features=1,
                d_model=D_MODEL,
                nhead=N_HEAD,
                num_encoder_layers=NUM_LAYERS,
                dim_feedforward=DIM_FEEDFORWARD,
                dropout=0.2,
                method=method,
                num_classes=NUM_CLASSES,
            ).to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

            if method == "regression":
                criterion = nn.MSELoss()
            elif method == "classification":
                criterion = nn.CrossEntropyLoss()
            elif method == "coral":
                criterion = nn.BCEWithLogitsLoss()
            elif method == "clm":
                criterion = CumulativeLinkLoss()

            train(
                model,
                optimizer,
                criterion,
                train_df,
                val_df,
                EPOCHS,
                device,
                BATCH_SIZE,
                method,
                NUM_CLASSES,
                PATIENCE,
                checkpoint_path,
                verbose=False,
            )

            model.load_state_dict(torch.load(checkpoint_path))

            train_mae, train_bca = evaluate(
                model, get_dataloader(train_df, BATCH_SIZE), device, method
            )
            test_mae, test_bca = evaluate(
                model, get_dataloader(test_df, BATCH_SIZE), device, method
            )

            print(f"  Test MAE: {test_mae:.4f} | Test BCA: {test_bca:.4f}")

            all_results[method]["Train MAE"].append(train_mae)
            all_results[method]["Train BCA"].append(train_bca)
            all_results[method]["Test MAE"].append(test_mae)
            all_results[method]["Test BCA"].append(test_bca)

    # --- Final Results Summary (after all folds) ---
    print("\n" + "=" * 50)
    print(f"--- FINAL STATISTICAL RESULTS (MEAN ± STD OVER {K_FOLDS} FOLDS) ---")
    print("=" * 50)

    summary_df = pd.DataFrame()
    metrics_to_report = ["Train MAE", "Train BCA", "Test MAE", "Test BCA"]

    for method in methods_to_compare:
        for metric in metrics_to_report:
            mean = np.mean(all_results[method][metric])
            std = np.std(all_results[method][metric])
            summary_df.loc[method, metric] = f"{mean:.4f} ± {std:.4f}"

    print(summary_df.to_string())

    # --- Clean up checkpoint files ---
    print("\nCleaning up checkpoint files...")
    for fold in range(K_FOLDS):
        for method in methods_to_compare:
            try:
                os.remove(f"checkpoint_{method}_fold{fold}.pt")
            except OSError:
                pass
    print("Done.")
