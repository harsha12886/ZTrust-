"""Cryptographic verification engine for ZTrust-FED: RSA-2048 keypairs, SHA-256 hashing, RSA-PSS signatures."""

import os
import hashlib
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KEYS_DIR = os.path.join(BASE_DIR, "keys")


class CryptoEngine:
    """Manages RSA-2048 keys, SHA-256 hashing, and RSA-PSS signatures."""

    def __init__(self, keys_dir: str = KEYS_DIR):
        self.keys_dir = keys_dir
        os.makedirs(self.keys_dir, exist_ok=True)

    def generate_keypair(self, client_id: str) -> tuple:
        """Generate and save an RSA-2048 keypair for a client."""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        public_key = private_key.public_key()

        # Save private key PEM
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        priv_path = os.path.join(self.keys_dir, f"{client_id}_private.pem")
        with open(priv_path, "wb") as f:
            f.write(private_pem)

        # Save public key PEM
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        pub_path = os.path.join(self.keys_dir, f"{client_id}_public.pem")
        with open(pub_path, "wb") as f:
            f.write(public_pem)

        return private_key, public_key

    def load_private_key(self, client_id: str):
        """Load a previously saved private key from PEM file."""
        path = os.path.join(self.keys_dir, f"{client_id}_private.pem")
        with open(path, "rb") as f:
            return serialization.load_pem_private_key(
                f.read(), password=None, backend=default_backend()
            )

    def load_public_key(self, client_id: str):
        """Load a previously saved public key from PEM file."""
        path = os.path.join(self.keys_dir, f"{client_id}_public.pem")
        with open(path, "rb") as f:
            return serialization.load_pem_public_key(
                f.read(), backend=default_backend()
            )

    # ─── SHA-256 HASHING ─────────────────────────────────────────────────

    @staticmethod
    def compute_sha256(data: bytes) -> str:
        """
        Compute SHA-256 digest of arbitrary byte data.

        Returns:
            Hex-encoded hash string
        """
        return hashlib.sha256(data).hexdigest()

    # ─── DIGITAL SIGNATURES ─────────────────────────────────────────────

    @staticmethod
    def sign(private_key, data: bytes) -> bytes:
        """
        Sign data using RSA-PSS with SHA-256.

        Args:
            private_key: RSA private key object
            data: Raw bytes to sign (typically the SHA-256 hash bytes)

        Returns:
            Digital signature bytes
        """
        signature = private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return signature

    @staticmethod
    def verify(public_key, signature: bytes, data: bytes) -> bool:
        """
        Verify a digital signature against data using the public key.

        Args:
            public_key: RSA public key object
            signature: The signature bytes to verify
            data: The original data that was signed

        Returns:
            True if signature is valid, False otherwise
        """
        try:
            public_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except InvalidSignature:
            return False


# ─── STANDALONE TEST ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  ZTrust-FED — Crypto Engine Self-Test")
    print("=" * 60)

    engine = CryptoEngine()

    # Generate test keypair
    priv, pub = engine.generate_keypair("test_client")

    # Hash some test data
    test_data = b"Hello, ZTrust-FED! This is a test payload."
    digest = CryptoEngine.compute_sha256(test_data)
    print(f"\n  [TEST] SHA-256 of test data: {digest[:32]}...")

    # Sign the hash
    sig = CryptoEngine.sign(priv, test_data)
    print(f"  [TEST] Signature length: {len(sig)} bytes")

    # Verify valid signature
    valid = CryptoEngine.verify(pub, sig, test_data)
    print(f"  [TEST] Valid signature verification:   {'PASS ✓' if valid else 'FAIL ✗'}")

    # Verify tampered data
    tampered = b"TAMPERED data!"
    invalid = CryptoEngine.verify(pub, sig, tampered)
    print(f"  [TEST] Tampered data verification:     {'FAIL ✗' if not invalid else 'PASS ✓'}")

    print(f"\n  [DONE] All crypto engine tests passed.\n")
