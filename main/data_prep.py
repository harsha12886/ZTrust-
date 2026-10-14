"""Dataset preprocessing and edge partitioning pipeline for ZTrust-FED."""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_DIR    = os.path.join(BASE_DIR, "archive")
if not os.path.exists(ARCHIVE_DIR):
    parent_archive = os.path.join(os.path.dirname(BASE_DIR), "archive")
    if os.path.exists(parent_archive):
        ARCHIVE_DIR = parent_archive

DATA_DIR       = os.path.join(BASE_DIR, "data")
PARTITION_DIR  = os.path.join(DATA_DIR, "partitions")
DEFAULT_ROWS_PER_FILE = 2000
SAMPLE_SIZE    = 100_000
RANDOM_SEED    = 42
LABEL_COL      = "label"
BENIGN_LABEL   = "BenignTraffic"

SAMPLE_CSV     = os.path.join(DATA_DIR, "ciciot2023_sample.csv")
SERVER_TEST    = os.path.join(PARTITION_DIR, "server_test.csv")
GATEWAY_A      = os.path.join(PARTITION_DIR, "gateway_a.csv")
GATEWAY_B      = os.path.join(PARTITION_DIR, "gateway_b.csv")
GATEWAY_C      = os.path.join(PARTITION_DIR, "gateway_c.csv")


def stream_partition_all_data(archive_dir: str = ARCHIVE_DIR,
                              max_files: int = None,
                              rows_per_file: int = None) -> None:
    """
    Stream-process all multi-part CSV files one by one with constant, low RAM usage:
      - Reads each part file individually (or in row chunks)
      - Drops NaNs/infs and binarizes labels: Benign -> 0, Attack -> 1
      - Shuffles within file to avoid temporal order bias
      - Streams 20% to server_test.csv and 80%/3 to gateway_a, gateway_b, gateway_c
      - Appends directly to disk with mode='a', freeing memory immediately
    """
    if not os.path.exists(archive_dir):
        print(f"[ERROR] Archive folder '{archive_dir}' not found.")
        sys.exit(1)

    csv_files = sorted([
        os.path.join(archive_dir, f)
        for f in os.listdir(archive_dir)
        if f.endswith(".csv")
    ])

    if not csv_files:
        print(f"[ERROR] No CSV files found in {archive_dir}")
        sys.exit(1)

    if max_files:
        csv_files = csv_files[:max_files]

    os.makedirs(PARTITION_DIR, exist_ok=True)

    # Remove any existing partition files so we start a clean append stream
    for f in [SERVER_TEST, GATEWAY_A, GATEWAY_B, GATEWAY_C, SAMPLE_CSV]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except OSError:
                pass

    total_files = len(csv_files)
    print(f"\n[STREAM] Starting out-of-core streaming partition across {total_files} files...")
    if rows_per_file:
        print(f"[STREAM] Limit per file: {rows_per_file:,} rows/part")
    else:
        print(f"[STREAM] Ingesting 100% of all rows across all {total_files} files (memory-safe)")

    total_rows = 0
    benign_count = 0
    attack_count = 0
    server_rows = 0
    gw_a_rows = 0
    gw_b_rows = 0
    gw_c_rows = 0
    sample_cached = False

    for i, fpath in enumerate(csv_files, 1):
        fname = os.path.basename(fpath)
        try:
            chunk = pd.read_csv(fpath, nrows=rows_per_file)
        except Exception as e:
            print(f"  [WARN] Skipping {fname} due to read error: {e}")
            continue

        if chunk.empty:
            continue

        # Clean infs/NaNs
        chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna()
        if chunk.empty:
            continue

        # Binarize label: 0 for BenignTraffic, 1 for any attack
        if chunk[LABEL_COL].dtype not in [np.int64, np.int32, int]:
            chunk[LABEL_COL] = (chunk[LABEL_COL] != BENIGN_LABEL).astype(np.int8)

        # Shuffle file chunk
        chunk = chunk.sample(frac=1.0, random_state=RANDOM_SEED + i).reset_index(drop=True)

        n_rows = len(chunk)
        b_c = int((chunk[LABEL_COL] == 0).sum())
        a_c = int(n_rows - b_c)

        total_rows += n_rows
        benign_count += b_c
        attack_count += a_c

        # Partition 20% server test, 80% across 3 gateways
        split_idx = int(n_rows * 0.20)
        server_chunk = chunk.iloc[:split_idx]
        train_chunk = chunk.iloc[split_idx:]

        g_size = len(train_chunk) // 3
        gw_a = train_chunk.iloc[:g_size]
        gw_b = train_chunk.iloc[g_size : 2 * g_size]
        gw_c = train_chunk.iloc[2 * g_size:]

        server_rows += len(server_chunk)
        gw_a_rows += len(gw_a)
        gw_b_rows += len(gw_b)
        gw_c_rows += len(gw_c)

        is_first = (i == 1)
        server_chunk.to_csv(SERVER_TEST, mode='a', header=is_first, index=False)
        gw_a.to_csv(GATEWAY_A, mode='a', header=is_first, index=False)
        gw_b.to_csv(GATEWAY_B, mode='a', header=is_first, index=False)
        gw_c.to_csv(GATEWAY_C, mode='a', header=is_first, index=False)

        # Cache a modest sample (~100k rows) in SAMPLE_CSV for quick inspects
        if not sample_cached:
            chunk.iloc[:min(len(chunk), 100_000)].to_csv(SAMPLE_CSV, index=False)
            sample_cached = True

        if i % 10 == 0 or i == total_files:
            pct = (i / total_files) * 100
            print(f"  • [{i:>3}/{total_files} | {pct:>5.1f}%] Processed {fname:<45} | Cumulative: {total_rows:>9,} rows")

        # Explicit free
        del chunk, server_chunk, train_chunk, gw_a, gw_b, gw_c

    print(f"\n  +------------------------------------------------------------------+")
    print(f"  |  STREAMING PARTITION SUMMARY (OUT-OF-CORE COMPLETE)              |")
    print(f"  +------------------------------------------------------------------+")
    print(f"  |  Files Processed: {total_files:>7} archive parts                       |")
    print(f"  |  Total Rows:      {total_rows:>7,} (Benign: {benign_count:,}, Attack: {attack_count:,})")
    print(f"  |  Server Test:     {server_rows:>7,} rows -> server_test.csv")
    print(f"  |  Gateway A:       {gw_a_rows:>7,} rows -> gateway_a.csv")
    print(f"  |  Gateway B:       {gw_b_rows:>7,} rows -> gateway_b.csv")
    print(f"  |  Gateway C:       {gw_c_rows:>7,} rows -> gateway_c.csv")
    print(f"  +------------------------------------------------------------------+")


def load_raw_data(archive_dir: str = ARCHIVE_DIR,
                  sample_size: int = None,
                  rows_per_file: int = None,
                  force_reload: bool = False,
                  full_dataset: bool = False) -> pd.DataFrame:
    """Load and aggregate traffic data from multi-part dataset archive."""
    if not force_reload and os.path.exists(SAMPLE_CSV):
        df = pd.read_csv(SAMPLE_CSV, nrows=sample_size)
        print(f"[DATA] Loaded cached sample: {len(df):,} rows from {SAMPLE_CSV}")
        return df

    if not os.path.exists(archive_dir):
        print(f"[ERROR] Neither {SAMPLE_CSV} nor archive folder '{archive_dir}' found.")
        sys.exit(1)

    csv_files = sorted([
        os.path.join(archive_dir, f)
        for f in os.listdir(archive_dir)
        if f.endswith(".csv")
    ])

    if not csv_files:
        print(f"[ERROR] No CSV files found in {archive_dir}")
        sys.exit(1)

    dfs = []
    if rows_per_file is None:
        rows_per_file = max(10, sample_size // len(csv_files)) if sample_size else DEFAULT_ROWS_PER_FILE

    print(f"[DATA] Ingesting {rows_per_file:,} rows/part across all {len(csv_files)} archive files...")
    for i, f in enumerate(csv_files, 1):
        chunk = pd.read_csv(f, nrows=rows_per_file)
        dfs.append(chunk)
        if i % 40 == 0 or i == len(csv_files):
            print(f"  • Processed {i}/{len(csv_files)} files ({sum(len(d) for d in dfs):,} rows)...")

    df = pd.concat(dfs, ignore_index=True)
    print(f"[DATA] Aggregated {len(df):,} total rows from {len(csv_files)} part files.")
    return df


def clean_and_binarize(df: pd.DataFrame) -> pd.DataFrame:
    """Drop NaN/inf values and map labels: Benign -> 0, Attack -> 1."""
    if df[LABEL_COL].dtype in [np.int64, np.int32, int]:
        if set(df[LABEL_COL].unique()).issubset({0, 1}):
            return df

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df[LABEL_COL] = (df[LABEL_COL] != BENIGN_LABEL).astype(int)

    benign_count = (df[LABEL_COL] == 0).sum()
    attack_count = (df[LABEL_COL] == 1).sum()
    print(f"[DATA] Cleaned: {len(df):,} samples ({benign_count:,} Benign, {attack_count:,} Attacks)")
    return df


def partition_data(df: pd.DataFrame) -> None:
    """
    Splits the cleaned dataset:
      - 20% → Server Test Set (unseen evaluation data)
      - 80% → Split equally across 3 edge gateways (A, B, C)
    """
    print(f"\n[STEP] Partitioning data...")

    # Shuffle the data
    df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    # 20% for server test
    split_idx = int(len(df) * 0.20)
    server_df = df.iloc[:split_idx]
    train_df  = df.iloc[split_idx:]

    # Split remaining 80% into 3 equal gateway partitions
    gateway_size = len(train_df) // 3
    gw_a = train_df.iloc[:gateway_size]
    gw_b = train_df.iloc[gateway_size : 2 * gateway_size]
    gw_c = train_df.iloc[2 * gateway_size:]

    # Save all partitions
    os.makedirs(PARTITION_DIR, exist_ok=True)

    df.to_csv(SAMPLE_CSV, index=False)
    server_df.to_csv(SERVER_TEST, index=False)
    gw_a.to_csv(GATEWAY_A, index=False)
    gw_b.to_csv(GATEWAY_B, index=False)
    gw_c.to_csv(GATEWAY_C, index=False)

    print(f"\n  +--------------------------------------------------+")
    print(f"  |  PARTITION SUMMARY                               |")
    print(f"  +--------------------------------------------------+")
    print(f"  |  Full sample:  {len(df):>7,} rows -> {SAMPLE_CSV}")
    print(f"  |  Server Test:  {len(server_df):>7,} rows -> server_test.csv")
    print(f"  |  Gateway A:    {len(gw_a):>7,} rows -> gateway_a.csv")
    print(f"  |  Gateway B:    {len(gw_b):>7,} rows -> gateway_b.csv")
    print(f"  |  Gateway C:    {len(gw_c):>7,} rows -> gateway_c.csv")
    print(f"  +--------------------------------------------------+")


def main(force_reload: bool = False,
         rows_per_file: int = None,
         sample_size: int = None,
         full_dataset: bool = False,
         streaming: bool = False,
         max_files: int = None):
    import argparse
    parser = argparse.ArgumentParser(description="ZTrust-FED Data Preprocessing Pipeline")
    parser.add_argument("--rebuild", "--force", action="store_true", help="Force rebuilding partitions from raw archive")
    parser.add_argument("--rows-per-file", type=int, default=None, help=f"Rows to sample from each archive part file (default: {DEFAULT_ROWS_PER_FILE})")
    parser.add_argument("--sample-size", type=int, default=None, help="Target total sample size")
    parser.add_argument("--full", action="store_true", help="Ingest all rows from all archive files via streaming")
    parser.add_argument("--streaming", action="store_true", help="Use out-of-core streaming partitioner (low RAM)")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of archive part files (for testing)")
    args, unknown = parser.parse_known_args()

    rebuild_flag = force_reload or args.rebuild
    rpf = rows_per_file if rows_per_file is not None else args.rows_per_file
    ss = sample_size if sample_size is not None else args.sample_size
    is_full = full_dataset or args.full or streaming or args.streaming
    mf = max_files if max_files is not None else args.max_files

    print("=" * 60)
    print("  ZTrust-FED — Stage 1 & 2: Data Collection & Preparation")
    print("=" * 60)

    # When full dataset or streaming is requested, use the memory-safe streaming engine
    if is_full:
        stream_partition_all_data(ARCHIVE_DIR, max_files=mf, rows_per_file=rpf)
    else:
        df = load_raw_data(ARCHIVE_DIR, sample_size=ss, rows_per_file=rpf, force_reload=rebuild_flag)
        df = clean_and_binarize(df)
        partition_data(df)

    print(f"\n[DONE] Data preprocessing complete.\n")


if __name__ == "__main__":
    main()
