"""Edge gateway client for ZTrust-FED: local model training, serialization, and cryptographic signing."""

import os
import io
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from crypto_engine import CryptoEngine

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
LABEL_COL     = "label"


class StreamingOnlineModel:
    """Out-of-core online streaming classifier with feature standardizer."""

    def __init__(self, alpha: float = 1e-4, random_state: int = 42):
        self.scaler = StandardScaler()
        self.clf = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=alpha,
            random_state=random_state,
            warm_start=True
        )
        self.classes_ = np.array([0, 1])
        self.is_fitted = False

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.scaler.partial_fit(X)
        X_scaled = self.scaler.transform(X)
        self.clf.partial_fit(X_scaled, y, classes=self.classes_)
        self.is_fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X)
        X_scaled = self.scaler.transform(X_arr)
        return self.clf.predict_proba(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X)
        X_scaled = self.scaler.transform(X_arr)
        return self.clf.predict(X_scaled)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        X_arr = np.asarray(X)
        X_scaled = self.scaler.transform(X_arr)
        return self.clf.score(X_scaled, y)


class GatewayClient:
    """Edge gateway node in the ZTrust-FED framework."""

    def __init__(self,
                 client_id: str,
                 data_path: str,
                 crypto_engine: CryptoEngine,
                 model_type: str = "auto",
                 chunk_size: int = 50_000,
                 max_rf_samples: int = 150_000):
        self.client_id = client_id
        self.data_path = data_path
        self.crypto = crypto_engine
        self.model_type = model_type
        self.chunk_size = chunk_size
        self.max_rf_samples = max_rf_samples
        self.model = None
        self.model_bytes = None
        self.total_samples = 0

        self.private_key, self.public_key = self.crypto.generate_keypair(client_id)

        # Inspect feature column names without loading full CSV into RAM
        preview = pd.read_csv(self.data_path, nrows=5)
        self.feature_columns = [c for c in preview.columns if c != LABEL_COL]

    def train_local_model(self):
        """Train a local classifier on this gateway's partition (streaming or RF)."""
        start = time.time()
        file_size_mb = os.path.getsize(self.data_path) / (1024 * 1024)

        # Determine model type: use streaming SGD for large datasets to protect RAM
        use_streaming = (
            self.model_type == "sgd" or
            (self.model_type == "auto" and file_size_mb > 60.0)
        )

        display_name = self.client_id.replace('_', ' ').title()

        if use_streaming:
            self.model = StreamingOnlineModel(alpha=1e-4, random_state=42)
            total_trained = 0
            last_chunk_X = None
            last_chunk_y = None

            print(f"  • {display_name:<12}: Streaming and training model...", flush=True)
            for chunk in pd.read_csv(self.data_path, chunksize=self.chunk_size):
                X_chunk = chunk[self.feature_columns].values.astype(np.float32)
                y_chunk = chunk[LABEL_COL].values.astype(np.int8)
                self.model.partial_fit(X_chunk, y_chunk)
                total_trained += len(y_chunk)
                last_chunk_X = X_chunk
                last_chunk_y = y_chunk

                if total_trained % 2_000_000 < self.chunk_size:
                    print(f"    [{display_name}] Processed {total_trained:,} samples...", flush=True)

            elapsed = time.time() - start
            train_acc = self.model.score(last_chunk_X, last_chunk_y) if last_chunk_X is not None else 1.0
            self.total_samples = total_trained
            print(f"  • {display_name:<12}: {total_trained:>10,} samples trained ({elapsed:.2f}s, acc={train_acc:.4f}) [Streaming SGD]")
        else:
            # Random Forest on moderate dataset or bounded sample
            if file_size_mb > 50.0:
                df = pd.read_csv(self.data_path, nrows=self.max_rf_samples)
            else:
                df = pd.read_csv(self.data_path)

            X = df[self.feature_columns]
            y = df[LABEL_COL]
            self.total_samples = len(df)

            self.model = RandomForestClassifier(
                n_estimators=50,
                n_jobs=-1,
                random_state=42
            )
            self.model.fit(X, y)
            elapsed = time.time() - start
            train_acc = self.model.score(X, y)
            print(f"  • {display_name:<12}: {len(df):>9,} samples trained ({elapsed:.2f}s, acc={train_acc:.4f}) [Random Forest]")

        return self.model

    def package_contribution(self, round_id: int, nonce: str) -> dict:
        """Serialize and cryptographically sign model update payload."""
        if self.model is None:
            raise ValueError(f"[{self.client_id}] No trained model. Call train_local_model() first.")

        buffer = io.BytesIO()
        joblib.dump(self.model, buffer)
        self.model_bytes = buffer.getvalue()

        model_hash = CryptoEngine.compute_sha256(self.model_bytes)
        signature = CryptoEngine.sign(self.private_key, self.model_bytes)

        return {
            "client_id":   self.client_id,
            "round_id":    round_id,
            "nonce":       nonce,
            "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_bytes": self.model_bytes,
            "model_hash":  model_hash,
            "signature":   signature,
        }

    def get_feature_columns(self) -> list:
        """Return the feature column names used for training."""
        return list(self.feature_columns)

