import base64
import hashlib
import os

from cryptography.fernet import Fernet


def decrypt_value(ciphertext: str) -> str:
    secret = os.environ["DJANGO_SECRET_KEY"]
    key = base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())
    return Fernet(key).decrypt(ciphertext.encode()).decode()
