#!/usr/bin/env python3
"""
Check if the Telegram bot is running and responsive
"""

import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

async def check_bot_status():
    """Check if bot is running by getting updates"""
    try:
        from telegram import Bot
        
        token = os.getenv('TELEGRAM_TOKEN')
        bot = Bot(token=token)
        
        # Get bot info
        me = await bot.get_me()
        print(f"Bot Info: {me.first_name} (@{me.username})")
        
        # Get recent updates to see if bot is receiving messages
        updates = await bot.get_updates(limit=5)
        print(f"Recent updates: {len(updates)} messages")
        
        if updates:
            for update in updates[-3:]:  # Show last 3 updates
                if update.message:
                    user = update.message.from_user
                    text = update.message.text
                    date = update.message.date
                    print(f"  - {date}: {user.first_name}: {text}")
        
        return True
        
    except Exception as e:
        print(f"Error checking bot: {e}")
        return False

if __name__ == '__main__':
    print("Checking bot status...")
    result = asyncio.run(check_bot_status())
    print(f"Bot check result: {'✅ Working' if result else '❌ Failed'}")
