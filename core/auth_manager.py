import os
import json
import uuid
import base64
import socket
import logging
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import requests
except ImportError:
    pass

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
except ImportError:
    pass

logger = logging.getLogger("NeuroScalpel.AuthManager")

API_BASE_URL = "http://localhost/nexcore/backend/api/app"
PRODUCT_ID = 1  # Replace with actual NexCore product ID if needed

class AuthException(Exception):
    pass

class AuthManager:
    "Singleton instance to manage authentication, IP binding, and token usage."

    def __init__(self):
        self._session_file = Path(__file__).resolve().parent.parent / "data" / "session.enc"
        self._session_file.parent.mkdir(parents=True, exist_ok=True)
        self._session_data: Dict[str, Any] = {}
        self.ip_address: str = ""
        self.settings: Dict[str, Any] = {}

    def _get_encryption_key(self) -> bytes:
        "Derive a unique encryption key based on the machine's MAC address and a constant salt."
        mac = str(uuid.getnode()).encode()
        salt = b"NeuroScalpel_NVIDIA_Agent_Key_Salt"
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(mac))

    def _get_fernet(self) -> 'Fernet':
        return Fernet(self._get_encryption_key())

    def _fetch_ip(self) -> str:
        if self.ip_address:
            return self.ip_address
        try:
            r = requests.get("https://ip.me", timeout=5)
            if r.status_code == 200:
                self.ip_address = r.text.strip()
                return self.ip_address
        except Exception:
            pass
        # Fallback
        self.ip_address = socket.gethostbyname(socket.gethostname())
        return self.ip_address

    def load_session(self) -> bool:
        "Returns True if a valid session exists and was decrypted successfully."
        if not self._session_file.exists():
            return False
        try:
            encrypted_data = self._session_file.read_bytes()
            fernet = self._get_fernet()
            decrypted_data = fernet.decrypt(encrypted_data)
            self._session_data = json.loads(decrypted_data.decode('utf-8'))
            if "uuid" in self._session_data and "token" in self._session_data:
                return True
        except Exception as e:
            logger.warning(f"Failed to decrypt or load session.enc: {e}")
            self._session_data = {}
        return False

    def save_session(self, uuid_str: str, token_str: str, user_info: dict = None):
        self._session_data = {
            "uuid": uuid_str,
            "token": token_str,
            "user": user_info or {}
        }
        try:
            fernet = self._get_fernet()
            data_bytes = json.dumps(self._session_data).encode("utf-8")
            encrypted_data = fernet.encrypt(data_bytes)
            self._session_file.write_bytes(encrypted_data)
        except Exception as e:
            logger.error(f"Save session failed: {e}")
            raise AuthException("Failed to securely save session data.")

    def delete_session(self):
        self._session_data = {}
        if self._session_file.exists():
            self._session_file.unlink()

    def login(self, email: str, password: str) -> bool:
        "Attempts to log in via NexCore API and saves session if successful."
        payload = {"email": email, "password": password}
        try:
            r = requests.post(f"{API_BASE_URL}/login", json=payload, timeout=15)
            data = r.json()
            if data.get("status") == "success":
                self.save_session(data["uuid"], data["token"], data.get("user"))
                return True
            else:
                raise AuthException(data.get("message", "Login failed."))
        except AuthException:
            raise
        except Exception as e:
            raise AuthException(f"Network error during login: {e}")

    def activate(self) -> dict:
        "Binds hardware IP and validates token quota on startup."
        if not self._session_data.get("uuid"):
            raise AuthException("No session loaded.")
        
        payload = {
            "product_id": PRODUCT_ID,
            "uuid": self._session_data["uuid"],
            "token": self._session_data["token"],
            "ip_address": self._fetch_ip()
        }
        try:
            r = requests.post(f"{API_BASE_URL}/activate", json=payload, timeout=15)
            data = r.json()
            if data.get("status") == "success":
                return data
            else:
                self.delete_session() # Invalid token probably
                raise AuthException(data.get("message", "Activation failed: session invalid."))
        except AuthException:
            raise
        except Exception as e:
            raise AuthException(f"Network error during activation: {e}")

    def fetch_settings(self) -> dict:
        "Fetches dynamic app settings and caches them."
        if not self._session_data.get("uuid"):
            raise AuthException("No session loaded.")
        
        payload = {
            "uuid": self._session_data["uuid"],
            "token": self._session_data["token"]
        }
        try:
            r = requests.post(f"{API_BASE_URL}/settings", json=payload, timeout=15)
            data = r.json()
            if data.get("status") == "success":
                self.settings = data.get("settings", {})
                return self.settings
            else:
                raise AuthException(data.get("message", "Failed to fetch settings."))
        except AuthException:
            raise
        except Exception as e:
            raise AuthException(f"Network error fetching settings: {e}")

    def apply_settings_to_env(self):
        "Overwrites current Process env with fetched settings."
        for key, value in self.settings.items():
            if value is not None:
                os.environ[key] = str(value)

    def use_token(self, action: str = "app_use", cost: int = 1) -> dict:
        "Deducts a token for using a feature like Abliteration or Start Word."
        if not self._session_data.get("uuid"):
            raise AuthException("No session loaded.")
        
        payload = {
            "product_id": PRODUCT_ID,
            "uuid": self._session_data["uuid"],
            "token": self._session_data["token"],
            "action": action,
            "tokens": cost
        }
        try:
            r = requests.post(f"{API_BASE_URL}/use", json=payload, timeout=15)
            if r.status_code == 429:
                data = r.json()
                raise AuthException(data.get("message", "Token quota exhausted."))
            data = r.json()
            if data.get("status") == "success":
                return data
            else:
                raise AuthException(data.get("message", "Failed to consume token."))
        except AuthException:
            raise
        except Exception as e:
            raise AuthException(f"Network error consuming token: {e}")

auth_manager = AuthManager()
