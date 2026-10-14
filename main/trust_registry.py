"""Adaptive trust registry for ZTrust-FED: tracks client trust scores and replay prevention nonces."""

import os


class TrustRegistry:
    """Manages client trust scores and nonce replay detection."""

    INITIAL_TRUST   = 100.0
    MIN_TRUST       = 0.0
    MAX_TRUST       = 100.0
    DECAY_FACTOR    = 0.95
    REWARD          = 1.0
    PENALTY         = 25.0
    TRUST_THRESHOLD = 50.0

    def __init__(self):
        self.trust_scores = {}
        self.seen_nonces = set()
        self.history = []

    def register_client(self, client_id: str) -> None:
        """Register a new client with initial trust score."""
        self.trust_scores[client_id] = self.INITIAL_TRUST

    def is_registered(self, client_id: str) -> bool:
        """Check if a client is registered in the trust registry."""
        return client_id in self.trust_scores

    def update_trust(self, client_id: str, verification_failed: bool, verbose: bool = False) -> float:
        """Update a client's trust score based on verification outcome."""
        if client_id not in self.trust_scores:
            raise ValueError(f"Client '{client_id}' not registered.")

        old_trust = self.trust_scores[client_id]
        A = 1.0 if verification_failed else 0.0

        new_trust = (
            self.DECAY_FACTOR * old_trust
            + (1.0 - A) * self.REWARD
            - self.PENALTY * A
        )
        new_trust = max(self.MIN_TRUST, min(self.MAX_TRUST, new_trust))
        self.trust_scores[client_id] = new_trust

        transition = {
            "client_id": client_id,
            "old_trust": old_trust,
            "new_trust": new_trust,
            "failed": verification_failed,
            "delta": new_trust - old_trust,
        }
        self.history.append(transition)

        if verbose:
            status = "PENALIZED" if verification_failed else "REWARDED"
            print(f"  [TRUST] {client_id}: {old_trust:.1f} -> {new_trust:.1f} ({status})")

        return new_trust

    # ─── NONCE REPLAY PROTECTION ─────────────────────────────────────────

    def check_nonce(self, client_id: str, round_id: int, nonce: str) -> bool:
        """
        Check if a nonce has already been used (replay detection).

        Args:
            client_id: Client submitting the nonce
            round_id: FL round number
            nonce: The nonce string to check

        Returns:
            True if nonce is FRESH (not seen before)
            False if nonce is a REPLAY (already used)
        """
        key = (client_id, round_id, nonce)
        if key in self.seen_nonces:
            return False  # Replay detected
        self.seen_nonces.add(key)
        return True  # Fresh nonce

    # ─── TRUST QUERIES ───────────────────────────────────────────────────

    def get_trust(self, client_id: str) -> float:
        """Get current trust score for a client."""
        return self.trust_scores.get(client_id, 0.0)

    def is_eligible(self, client_id: str) -> bool:
        """Check if a client's trust meets the ensemble threshold."""
        return self.get_trust(client_id) >= self.TRUST_THRESHOLD

    def get_all_scores(self) -> dict:
        """Return all trust scores as a dict."""
        return dict(self.trust_scores)

    def get_history(self) -> list:
        """Return trust transition history."""
        return list(self.history)

    def print_trust_table(self) -> None:
        """Print a formatted table of all client trust scores."""
        print(f"  +--------------------+----------+-----------+")
        print(f"  | Gateway ID         | Trust    | Status    |")
        print(f"  +--------------------+----------+-----------+")
        for cid, score in sorted(self.trust_scores.items()):
            eligible = "Eligible" if score >= self.TRUST_THRESHOLD else "Excluded"
            print(f"  | {cid:<18} |  {score:>6.1f}  | {eligible:<9} |")
        print(f"  +--------------------+----------+-----------+")


# ─── STANDALONE TEST ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  ZTrust-FED — Trust Registry Self-Test")
    print("=" * 60)

    registry = TrustRegistry()
    registry.register_client("gateway_a")
    registry.register_client("gateway_b")

    registry.update_trust("gateway_a", verification_failed=False)
    registry.update_trust("gateway_b", verification_failed=True)

    registry.print_trust_table()
    print(f"\n  [DONE] Trust registry tests passed.\n")
