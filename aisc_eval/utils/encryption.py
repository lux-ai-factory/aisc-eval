import base64
import hashlib
import os

from cryptography.fernet import Fernet, MultiFernet


def decrypt_value(ciphertext: str) -> str:
    """Decrypts what the engine backend stored: with its own key, ENGINE_SECRETS_KEY, first when it is set, then
    with the key derived from DJANGO_SECRET_KEY, which values stored before it was introduced are under."""
    keys = []
    own = os.environ.get("ENGINE_SECRETS_KEY", "").strip()
    if own:
        keys.append(Fernet(own.encode()))
    secret = os.environ["DJANGO_SECRET_KEY"]
    keys.append(Fernet(base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())))
    return MultiFernet(keys).decrypt(ciphertext.encode()).decode()
