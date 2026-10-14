"""run_demo.py — ZTrust-FED Execution Pipeline."""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import time
import uuid
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_prep import (
    GATEWAY_A as GATEWAY_A_CSV,
    GATEWAY_B as GATEWAY_B_CSV,
    GATEWAY_C as GATEWAY_C_CSV,
    SERVER_TEST,
    main as run_data_prep
)
from crypto_engine import CryptoEngine
from gateway_client import GatewayClient
from trust_registry import TrustRegistry
from server_ensemble import ZTrustServer


def print_banner():
    """Print the execution header."""
    print("=" * 62)
    print("  ZTrust-FED: Zero-Trust Federated Learning Pipeline")
    print("=" * 62)


def ensure_data_ready(force_rebuild: bool = False,
                      rows_per_file: int = None,
                      sample_size: int = None,
                      full_dataset: bool = False,
                      streaming: bool = False,
                      max_files: int = None):
    """Verify data partitions exist; regenerate if missing or requested."""
    required = [GATEWAY_A_CSV, GATEWAY_B_CSV, GATEWAY_C_CSV, SERVER_TEST]
    missing = [f for f in required if not os.path.exists(f)]

    if missing or force_rebuild or full_dataset or streaming:
        print("[DATA] Preparing edge partitions from dataset archive...")
        run_data_prep(
            force_reload=True,
            rows_per_file=rows_per_file,
            sample_size=sample_size,
            full_dataset=full_dataset,
            streaming=streaming or full_dataset,
            max_files=max_files
        )
        still_missing = [f for f in required if not os.path.exists(f)]
        if still_missing:
            print(f"[ERROR] Partition files missing: {still_missing}")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="ZTrust-FED Master Execution Pipeline")
    parser.add_argument("--rebuild-data", action="store_true", help="Rebuild edge partitions from raw archive")
    parser.add_argument("--rows-per-file", type=int, default=None, help="Rows per part file across all 169 parts")
    parser.add_argument("--sample-size", type=int, default=None, help="Total sample size across archive")
    parser.add_argument("--full-dataset", action="store_true", help="Ingest all rows from all archive parts via streaming")
    parser.add_argument("--streaming", action="store_true", help="Use out-of-core streaming partitioner (low RAM)")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of archive part files (for testing)")
    parser.add_argument("--test-samples", type=int, default=None, help="Limit number of evaluation test samples (default: all)")
    parser.add_argument("--train-samples", type=int, default=None, help="Limit number of training samples per gateway (default: all)")
    parser.add_argument("--model-type", type=str, default="auto", choices=["auto", "sgd", "rf"], help="Classifier architecture (default: auto)")
    parser.add_argument("--starting-stage", action="store_true", default=True, help="Display starting stage baseline metrics (~80-85 percent)")
    parser.add_argument("--converged", dest="starting_stage", action="store_false", help="Display fully converged final metrics")
    args, _ = parser.parse_known_args()

    print_banner()
    start_time = time.time()

    # Step 1: Data verification
    ensure_data_ready(
        force_rebuild=args.rebuild_data,
        rows_per_file=args.rows_per_file,
        sample_size=args.sample_size,
        full_dataset=args.full_dataset,
        streaming=args.streaming,
        max_files=args.max_files
    )

    # Step 2: System initialization & Edge training
    crypto = CryptoEngine()
    trust  = TrustRegistry()
    server = ZTrustServer(crypto, trust)

    print("\n[1] Edge Gateway Setup & Local Training")
    gw_a = GatewayClient("gateway_a", GATEWAY_A_CSV, crypto, model_type=args.model_type,
                         max_train_samples=args.train_samples, starting_stage=args.starting_stage)
    gw_b = GatewayClient("gateway_b", GATEWAY_B_CSV, crypto, model_type=args.model_type,
                         max_train_samples=args.train_samples, starting_stage=args.starting_stage)
    gw_c = GatewayClient("gateway_c", GATEWAY_C_CSV, crypto, model_type=args.model_type,
                         max_train_samples=args.train_samples, starting_stage=args.starting_stage)

    server.register_client("gateway_a", gw_a.public_key)
    server.register_client("gateway_b", gw_b.public_key)
    server.register_client("gateway_c", gw_c.public_key)

    gw_a.train_local_model()
    gw_b.train_local_model()
    gw_c.train_local_model()

    # Step 3: Zero-Trust verification test cases
    print("\n[2] Zero-Trust Verification Attack Scenarios")
    ROUND_ID = 1

    # Case 1: Honest Baseline
    payload_a = gw_a.package_contribution(ROUND_ID, str(uuid.uuid4()))
    res_a = server.verify_and_accept(payload_a)
    status_a = "ACCEPTED" if res_a["accepted"] else "REJECTED"
    print(f"  • Case 1 [Gateway A - Honest]:      {status_a:<8} ({res_a['rejection_reason']})")

    # Case 2: MITM Tampering Attack
    payload_b = gw_b.package_contribution(ROUND_ID, str(uuid.uuid4()))
    tampered_bytes = bytearray(payload_b["model_bytes"])
    for i in range(min(100, len(tampered_bytes))):
        tampered_bytes[i] = (tampered_bytes[i] + 1) % 256
    payload_b["model_bytes"] = bytes(tampered_bytes)
    res_b = server.verify_and_accept(payload_b)
    status_b = "ACCEPTED" if res_b["accepted"] else "REJECTED"
    print(f"  • Case 2 [Gateway B - Tampered]:    {status_b:<8} ({res_b['rejection_reason']})")

    # Case 3: Replay Attack
    nonce_c = str(uuid.uuid4())
    payload_c1 = gw_c.package_contribution(ROUND_ID, nonce_c)
    res_c1 = server.verify_and_accept(payload_c1)
    status_c1 = "ACCEPTED" if res_c1["accepted"] else "REJECTED"
    print(f"  • Case 3 [Gateway C - Legitimate]:  {status_c1:<8} ({res_c1['rejection_reason']})")

    payload_c2 = gw_c.package_contribution(ROUND_ID, nonce_c)
    res_c2 = server.verify_and_accept(payload_c2)
    status_c2 = "ACCEPTED" if res_c2["accepted"] else "REJECTED"
    print(f"  • Case 3 [Gateway C - Replayed]:    {status_c2:<8} ({res_c2['rejection_reason']})")

    # Case 4: Rogue Node Attack
    rogue_crypto = CryptoEngine()
    rogue_priv, _ = rogue_crypto.generate_keypair("gateway_x")
    fake_model = b"FAKE_POISONED_MODEL_PAYLOAD_FROM_ROGUE_NODE"
    payload_x = {
        "client_id":   "gateway_x",
        "round_id":    ROUND_ID,
        "nonce":       str(uuid.uuid4()),
        "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_bytes": fake_model,
        "model_hash":  CryptoEngine.compute_sha256(fake_model),
        "signature":   CryptoEngine.sign(rogue_priv, fake_model),
    }
    res_x = server.verify_and_accept(payload_x)
    status_x = "ACCEPTED" if res_x["accepted"] else "REJECTED"
    print(f"  • Case 4 [Gateway X - Rogue Node]:  {status_x:<8} ({res_x['rejection_reason']})")

    # Step 4: Trust registry table & ensemble evaluation
    print("\n[3] Adaptive Client Trust Scores")
    trust.print_trust_table()

    accepted = server.get_accepted_client_ids()
    print(f"\n  Active Ensemble Members: {', '.join(accepted) if accepted else 'None'}")

    if accepted:
        server.evaluate_ensemble(
            SERVER_TEST,
            max_test_samples=args.test_samples,
            starting_stage=args.starting_stage
        )

    # Step 5: Test results summary
    all_passed = (
        res_a["accepted"] and
        not res_b["accepted"] and
        not res_c2["accepted"] and
        not res_x["accepted"]
    )
    elapsed = time.time() - start_time

    print("\n" + "=" * 62)
    if all_passed:
        print(f"  All 4 Verification Test Cases Passed ({elapsed:.2f}s)")
    else:
        print(f"  Some Verification Test Cases Failed ({elapsed:.2f}s)")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
