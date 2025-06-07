"""
User Manager module for TradePulse Signals bot
Handles user data and preferences
"""
import os
import json
import logging
from typing import Dict, Any, Optional
from config import USERS_DATA_FILE, SUPPORTED_ASSETS

logger = logging.getLogger(__name__)

def load_users() -> Dict[str, Dict[str, Any]]:
    """
    Load user data from file
    Returns a dictionary of user IDs and their data
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(USERS_DATA_FILE), exist_ok=True)
        
        # Check if file exists, create if not
        if not os.path.exists(USERS_DATA_FILE):
            with open(USERS_DATA_FILE, 'w') as f:
                json.dump({}, f)
            return {}
        
        # Load data from file
        with open(USERS_DATA_FILE, 'r') as f:
            users = json.load(f)
        
        return users
    except Exception as e:
        logger.error(f"Error loading users from file: {e}")
        return {}

def save_users(users: Dict[str, Dict[str, Any]]) -> bool:
    """
    Save all user data to file
    Returns True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(USERS_DATA_FILE), exist_ok=True)
        
        # Save data to file
        with open(USERS_DATA_FILE, 'w') as f:
            json.dump(users, f, indent=2)
        
        return True
    except Exception as e:
        logger.error(f"Error saving users to file: {e}")
        return False

def save_user(user_data: Dict[str, Any]) -> bool:
    """
    Save or update a single user's data
    Returns True if successful, False otherwise
    """
    try:
        users = load_users()
        user_id = str(user_data["id"])  # Convert to string for JSON key
        
        # Update or add user data
        users[user_id] = user_data
        
        return save_users(users)
    except Exception as e:
        logger.error(f"Error saving user {user_data.get('id', 'unknown')}: {e}")
        return False

def get_user(user_id: int) -> Optional[Dict[str, Any]]:
    """
    Get a user's data by ID
    Returns user data dict or None if not found
    """
    try:
        users = load_users()
        return users.get(str(user_id))
    except Exception as e:
        logger.error(f"Error getting user {user_id}: {e}")
        return None

def get_user_preferences(user_id: int) -> Dict[str, Any]:
    """
    Get a user's preferences
    Returns preferences dict or empty dict if not found
    """
    user = get_user(user_id)
    if user and "preferences" in user:
        return user["preferences"]
    
    # Return default preferences if none found
    return {"assets": SUPPORTED_ASSETS[:2]}  # Default to first two assets

def set_user_preferences(user_id: int, preferences: Dict[str, Any]) -> bool:
    """
    Set a user's preferences
    Returns True if successful, False otherwise
    """
    try:
        user = get_user(user_id)
        if not user:
            logger.warning(f"Attempted to set preferences for unknown user {user_id}")
            return False
        
        user["preferences"] = preferences
        return save_user(user)
    except Exception as e:
        logger.error(f"Error setting preferences for user {user_id}: {e}")
        return False
