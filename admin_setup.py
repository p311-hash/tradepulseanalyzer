#!/usr/bin/env python3
"""
MasterTrade Bot Admin Setup Script
Easy configuration and management of admin authentication.
"""

import os
import json
import getpass
from datetime import datetime
from admin_auth import AdminAuthenticator
import config

def print_banner():
    """Print setup banner."""
    print("=" * 60)
    print("🤖 MasterTrade Bot - Admin Setup")
    print("=" * 60)
    print()

def check_environment():
    """Check if environment is properly configured."""
    print("🔍 Checking environment configuration...")
    
    issues = []
    
    # Check essential environment variables
    if not config.TELEGRAM_TOKEN or config.TELEGRAM_TOKEN == 'your-token-here':
        issues.append("❌ TELEGRAM_TOKEN not configured")
    else:
        print("✅ TELEGRAM_TOKEN configured")
    
    if not config.BOT_OWNER_ID or config.BOT_OWNER_ID == 0:
        issues.append("❌ BOT_OWNER_ID not configured")
    else:
        print(f"✅ BOT_OWNER_ID configured: {config.BOT_OWNER_ID}")
    
    if config.REQUIRE_PASSWORD_AUTH and config.ADMIN_PASSWORD == 'MasterTrade2024!':
        issues.append("⚠️  Using default admin password (consider changing)")
    else:
        print("✅ Admin password configured")
    
    if issues:
        print("\n🚨 Configuration Issues:")
        for issue in issues:
            print(f"  {issue}")
        print("\nPlease update your .env file or environment variables.")
        return False
    
    print("\n✅ Environment configuration looks good!")
    return True

def setup_owner():
    """Setup bot owner."""
    print("\n👑 Setting up bot owner...")
    
    if config.BOT_OWNER_ID and config.BOT_OWNER_ID != 0:
        print(f"Current owner ID: {config.BOT_OWNER_ID}")
        change = input("Do you want to change the owner? (y/N): ").lower().strip()
        if change != 'y':
            return
    
    print("\nTo get your Telegram user ID:")
    print("1. Send a message to @userinfobot on Telegram")
    print("2. Copy the 'Id' number from the response")
    
    while True:
        try:
            user_id = input("\nEnter owner Telegram user ID: ").strip()
            user_id = int(user_id)
            
            confirm = input(f"Confirm owner ID {user_id}? (y/N): ").lower().strip()
            if confirm == 'y':
                # Update .env file
                update_env_file('BOT_OWNER_ID', str(user_id))
                print(f"✅ Owner ID set to: {user_id}")
                break
        except ValueError:
            print("❌ Please enter a valid numeric user ID")

def setup_password():
    """Setup admin password."""
    print("\n🔐 Setting up admin password...")
    
    if not config.REQUIRE_PASSWORD_AUTH:
        print("Password authentication is disabled.")
        enable = input("Do you want to enable password authentication? (y/N): ").lower().strip()
        if enable == 'y':
            update_env_file('REQUIRE_PASSWORD_AUTH', 'true')
        else:
            return
    
    print("\nPassword requirements:")
    print("- At least 8 characters")
    print("- Mix of letters, numbers, and symbols")
    print("- Avoid common passwords")
    
    while True:
        password = getpass.getpass("\nEnter new admin password: ")
        if len(password) < 8:
            print("❌ Password must be at least 8 characters")
            continue
        
        confirm = getpass.getpass("Confirm password: ")
        if password != confirm:
            print("❌ Passwords don't match")
            continue
        
        # Update .env file
        update_env_file('ADMIN_PASSWORD', password)
        print("✅ Admin password updated")
        break

def add_admin():
    """Add a new administrator."""
    print("\n➕ Adding new administrator...")
    
    auth = AdminAuthenticator()
    
    while True:
        try:
            user_id = input("Enter admin Telegram user ID: ").strip()
            user_id = int(user_id)
            
            print("\nPermission levels:")
            print("1. MODERATOR - View stats, moderate content")
            print("2. ADMIN - Full admin access except owner functions")
            
            level_choice = input("Select permission level (1-2): ").strip()
            
            if level_choice == '1':
                permission = 'MODERATOR'
            elif level_choice == '2':
                permission = 'ADMIN'
            else:
                print("❌ Invalid choice")
                continue
            
            # Add admin
            if auth.add_admin(user_id, permission):
                print(f"✅ Added {permission}: {user_id}")
                
                # Update environment variable
                current_admins = os.environ.get('BOT_ADMIN_IDS', '')
                if current_admins:
                    new_admins = f"{current_admins},{user_id}"
                else:
                    new_admins = str(user_id)
                update_env_file('BOT_ADMIN_IDS', new_admins)
                break
            else:
                print("❌ Failed to add admin")
                
        except ValueError:
            print("❌ Please enter a valid numeric user ID")

def list_admins():
    """List all administrators."""
    print("\n👥 Current administrators:")
    
    auth = AdminAuthenticator()
    admin_list = auth.get_admin_list()
    active_sessions = auth.get_active_sessions()
    
    if not admin_list:
        print("No administrators configured.")
        return
    
    print(f"{'User ID':<12} {'Level':<10} {'Status':<8}")
    print("-" * 32)
    
    for user_id, level in admin_list.items():
        status = "🟢 Active" if user_id in active_sessions else "⚪ Offline"
        print(f"{user_id:<12} {level:<10} {status}")

def update_env_file(key, value):
    """Update .env file with new value."""
    env_file = '.env'
    
    # Read existing content
    lines = []
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            lines = f.readlines()
    
    # Update or add the key
    updated = False
    for i, line in enumerate(lines):
        if line.startswith(f"{key}="):
            lines[i] = f"{key}={value}\n"
            updated = True
            break
    
    if not updated:
        lines.append(f"{key}={value}\n")
    
    # Write back to file
    with open(env_file, 'w') as f:
        f.writelines(lines)

def main_menu():
    """Show main setup menu."""
    while True:
        print("\n" + "=" * 40)
        print("🛠️  MasterTrade Bot Admin Setup")
        print("=" * 40)
        print("1. Check environment configuration")
        print("2. Setup bot owner")
        print("3. Setup admin password")
        print("4. Add administrator")
        print("5. List administrators")
        print("6. Exit")
        print()
        
        choice = input("Select option (1-6): ").strip()
        
        if choice == '1':
            check_environment()
        elif choice == '2':
            setup_owner()
        elif choice == '3':
            setup_password()
        elif choice == '4':
            add_admin()
        elif choice == '5':
            list_admins()
        elif choice == '6':
            print("\n👋 Setup complete! Start your bot with:")
            print("python run_telegram_bot.py")
            break
        else:
            print("❌ Invalid choice. Please select 1-6.")

def main():
    """Main setup function."""
    print_banner()
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("📝 Creating .env file from template...")
        if os.path.exists('.env.example'):
            import shutil
            shutil.copy('.env.example', '.env')
            print("✅ Created .env file. Please edit it with your settings.")
        else:
            print("❌ .env.example not found. Please create .env manually.")
            return
    
    # Run initial environment check
    if not check_environment():
        print("\n🔧 Use the setup menu to configure your bot.")
    
    # Show main menu
    main_menu()

if __name__ == '__main__':
    main()
