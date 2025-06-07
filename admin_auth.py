"""
Admin Authentication System for MasterTrade Bot
Provides secure authentication and permission management for bot administrators.
"""

import json
import os
import hashlib
import hmac
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from functools import wraps

import config

logger = logging.getLogger(__name__)

class AdminAuthenticator:
    """Handles admin authentication and permission management."""
    
    def __init__(self, auth_file: str = 'admin_sessions.json'):
        self.auth_file = auth_file
        self.active_sessions: Dict[int, Dict] = {}
        self.user_permissions: Dict[int, str] = {}
        self.failed_attempts: Dict[int, List[datetime]] = {}
        
        # Load existing sessions and permissions
        self._load_auth_data()
        self._initialize_owner()
        
        logger.info("Admin authenticator initialized")
    
    def _load_auth_data(self):
        """Load authentication data from file."""
        try:
            if os.path.exists(self.auth_file):
                with open(self.auth_file, 'r') as f:
                    data = json.load(f)
                    
                # Load sessions (convert string keys back to int)
                sessions = data.get('sessions', {})
                for user_id_str, session_data in sessions.items():
                    user_id = int(user_id_str)
                    session_time = datetime.fromisoformat(session_data['login_time'])
                    
                    # Check if session is still valid
                    if self._is_session_valid(session_time):
                        self.active_sessions[user_id] = {
                            'login_time': session_time,
                            'permission_level': session_data['permission_level'],
                            'ip_hash': session_data.get('ip_hash', ''),
                            'last_activity': datetime.fromisoformat(session_data.get('last_activity', session_data['login_time']))
                        }
                
                # Load permissions
                permissions = data.get('permissions', {})
                for user_id_str, level in permissions.items():
                    self.user_permissions[int(user_id_str)] = level
                    
        except Exception as e:
            logger.error(f"Error loading auth data: {str(e)}")
    
    def _save_auth_data(self):
        """Save authentication data to file."""
        try:
            data = {
                'sessions': {},
                'permissions': {}
            }
            
            # Save sessions (convert int keys to string for JSON)
            for user_id, session_data in self.active_sessions.items():
                data['sessions'][str(user_id)] = {
                    'login_time': session_data['login_time'].isoformat(),
                    'permission_level': session_data['permission_level'],
                    'ip_hash': session_data.get('ip_hash', ''),
                    'last_activity': session_data['last_activity'].isoformat()
                }
            
            # Save permissions
            for user_id, level in self.user_permissions.items():
                data['permissions'][str(user_id)] = level
            
            with open(self.auth_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving auth data: {str(e)}")
    
    def _initialize_owner(self):
        """Initialize the bot owner with full permissions."""
        if config.BOT_OWNER_ID and config.BOT_OWNER_ID != 0:
            self.user_permissions[config.BOT_OWNER_ID] = 'OWNER'
            logger.info(f"Bot owner initialized: {config.BOT_OWNER_ID}")
        
        # Initialize configured admins
        for admin_id in config.BOT_ADMINS:
            if admin_id not in self.user_permissions:
                self.user_permissions[admin_id] = 'ADMIN'
                logger.info(f"Admin initialized: {admin_id}")
    
    def _is_session_valid(self, login_time: datetime) -> bool:
        """Check if a session is still valid based on timeout."""
        timeout = timedelta(hours=config.SESSION_TIMEOUT_HOURS)
        return datetime.now() - login_time < timeout
    
    def _hash_password(self, password: str, salt: str = None) -> Tuple[str, str]:
        """Hash password with salt."""
        if salt is None:
            salt = os.urandom(32).hex()
        
        password_hash = hashlib.pbkdf2_hmac('sha256', 
                                          password.encode('utf-8'), 
                                          salt.encode('utf-8'), 
                                          100000)
        return password_hash.hex(), salt
    
    def _verify_password(self, password: str, stored_hash: str, salt: str) -> bool:
        """Verify password against stored hash."""
        password_hash, _ = self._hash_password(password, salt)
        return hmac.compare_digest(password_hash, stored_hash)
    
    def _check_rate_limit(self, user_id: int) -> bool:
        """Check if user has exceeded failed login attempts."""
        now = datetime.now()
        if user_id not in self.failed_attempts:
            return True
        
        # Remove attempts older than 1 hour
        self.failed_attempts[user_id] = [
            attempt for attempt in self.failed_attempts[user_id]
            if now - attempt < timedelta(hours=1)
        ]
        
        # Allow max 5 attempts per hour
        return len(self.failed_attempts[user_id]) < 5
    
    def _record_failed_attempt(self, user_id: int):
        """Record a failed login attempt."""
        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = []
        self.failed_attempts[user_id].append(datetime.now())
    
    def get_user_permission_level(self, user_id: int) -> str:
        """Get user's permission level."""
        return self.user_permissions.get(user_id, 'USER')
    
    def get_permission_value(self, user_id: int) -> int:
        """Get numeric permission value for comparison."""
        level = self.get_user_permission_level(user_id)
        return config.PERMISSION_LEVELS.get(level, 1)
    
    def is_authenticated(self, user_id: int) -> bool:
        """Check if user has an active authenticated session."""
        if user_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[user_id]
        if not self._is_session_valid(session['login_time']):
            # Session expired, remove it
            del self.active_sessions[user_id]
            self._save_auth_data()
            return False
        
        # Update last activity
        session['last_activity'] = datetime.now()
        self._save_auth_data()
        return True
    
    def authenticate_user(self, user_id: int, password: str = None) -> Tuple[bool, str]:
        """Authenticate user with optional password."""
        # Check rate limiting
        if not self._check_rate_limit(user_id):
            return False, "Too many failed attempts. Please try again later."
        
        # Check if user is owner or admin
        permission_level = self.get_user_permission_level(user_id)
        if permission_level == 'USER':
            self._record_failed_attempt(user_id)
            return False, "Access denied. You are not authorized as an administrator."
        
        # If password authentication is required and user is not owner
        if config.REQUIRE_PASSWORD_AUTH and user_id != config.BOT_OWNER_ID:
            if not password:
                return False, "Password required for authentication."
            
            # For simplicity, using configured password. In production, use hashed passwords per user.
            if password != config.ADMIN_PASSWORD:
                self._record_failed_attempt(user_id)
                return False, "Invalid password."
        
        # Create session
        self.active_sessions[user_id] = {
            'login_time': datetime.now(),
            'permission_level': permission_level,
            'ip_hash': '',  # Could be enhanced to track IP
            'last_activity': datetime.now()
        }
        
        self._save_auth_data()
        logger.info(f"User {user_id} authenticated with {permission_level} permissions")
        return True, f"Successfully authenticated as {permission_level}"
    
    def logout_user(self, user_id: int) -> bool:
        """Logout user and invalidate session."""
        if user_id in self.active_sessions:
            del self.active_sessions[user_id]
            self._save_auth_data()
            logger.info(f"User {user_id} logged out")
            return True
        return False
    
    def add_admin(self, user_id: int, permission_level: str = 'ADMIN') -> bool:
        """Add new admin (owner only)."""
        if permission_level in config.PERMISSION_LEVELS:
            self.user_permissions[user_id] = permission_level
            self._save_auth_data()
            logger.info(f"Added {permission_level}: {user_id}")
            return True
        return False
    
    def remove_admin(self, user_id: int) -> bool:
        """Remove admin privileges (owner only)."""
        if user_id in self.user_permissions and user_id != config.BOT_OWNER_ID:
            del self.user_permissions[user_id]
            if user_id in self.active_sessions:
                del self.active_sessions[user_id]
            self._save_auth_data()
            logger.info(f"Removed admin: {user_id}")
            return True
        return False
    
    def get_admin_list(self) -> Dict[int, str]:
        """Get list of all admins and their permission levels."""
        return self.user_permissions.copy()
    
    def get_active_sessions(self) -> Dict[int, Dict]:
        """Get list of active admin sessions."""
        # Clean expired sessions first
        now = datetime.now()
        expired_sessions = []
        
        for user_id, session in self.active_sessions.items():
            if not self._is_session_valid(session['login_time']):
                expired_sessions.append(user_id)
        
        for user_id in expired_sessions:
            del self.active_sessions[user_id]
        
        if expired_sessions:
            self._save_auth_data()
        
        return self.active_sessions.copy()


def require_permission(min_level: str):
    """Decorator to require minimum permission level for a function."""
    def decorator(func):
        @wraps(func)
        async def wrapper(self, update, context, *args, **kwargs):
            user_id = update.effective_user.id
            
            # Check if user has required permission
            if not hasattr(self, 'auth') or not isinstance(self.auth, AdminAuthenticator):
                await update.message.reply_text("❌ Authentication system not available.")
                return
            
            user_permission_value = self.auth.get_permission_value(user_id)
            required_permission_value = config.PERMISSION_LEVELS.get(min_level, 100)
            
            if user_permission_value < required_permission_value:
                await update.message.reply_text(
                    f"❌ Access denied. Required permission: {min_level}\n"
                    f"Your permission: {self.auth.get_user_permission_level(user_id)}"
                )
                return
            
            # Check if user is authenticated (has active session)
            if not self.auth.is_authenticated(user_id):
                await update.message.reply_text(
                    "🔐 Please authenticate first using /admin_login"
                )
                return
            
            return await func(self, update, context, *args, **kwargs)
        return wrapper
    return decorator


def require_authentication(func):
    """Decorator to require authentication for a function."""
    @wraps(func)
    async def wrapper(self, update, context, *args, **kwargs):
        user_id = update.effective_user.id
        
        if not hasattr(self, 'auth') or not isinstance(self.auth, AdminAuthenticator):
            await update.message.reply_text("❌ Authentication system not available.")
            return
        
        if not self.auth.is_authenticated(user_id):
            await update.message.reply_text(
                "🔐 Please authenticate first using /admin_login"
            )
            return
        
        return await func(self, update, context, *args, **kwargs)
    return wrapper
