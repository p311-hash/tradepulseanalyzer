#!/usr/bin/env python3
"""
Test script to verify Telegram bot connection
"""

import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_bot_connection():
    """Test if the bot token is valid and can connect to Telegram"""
    try:
        from telegram import Bot
        
        token = os.getenv('TELEGRAM_TOKEN')
        if not token:
            print("❌ No TELEGRAM_TOKEN found in environment")
            return False
        
        print(f"🔍 Testing bot with token: {token[:10]}...")
        
        # Create bot instance
        bot = Bot(token=token)
        
        # Test connection by getting bot info
        bot_info = await bot.get_me()
        
        print("✅ Bot connection successful!")
        print(f"📱 Bot Name: {bot_info.first_name}")
        print(f"🆔 Bot Username: @{bot_info.username}")
        print(f"🔢 Bot ID: {bot_info.id}")
        print(f"🤖 Is Bot: {bot_info.is_bot}")
        
        return True
        
    except Exception as e:
        print(f"❌ Bot connection failed: {e}")
        return False

async def test_send_message():
    """Test sending a message to a chat (if chat_id is provided)"""
    try:
        from telegram import Bot
        
        token = os.getenv('TELEGRAM_TOKEN')
        chat_id = os.getenv('TEST_CHAT_ID')  # Optional: add this to .env for testing
        
        if not chat_id:
            print("ℹ️ No TEST_CHAT_ID provided, skipping message test")
            return True
        
        bot = Bot(token=token)
        
        # Send test message
        message = await bot.send_message(
            chat_id=chat_id,
            text="🤖 Test message from TradePulse Bot!\n\nIf you see this, the bot is working correctly."
        )
        
        print(f"✅ Test message sent successfully! Message ID: {message.message_id}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to send test message: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 Starting Telegram Bot Connection Test...")
    print("=" * 50)
    
    # Test 1: Bot connection
    print("\n📡 Test 1: Bot Connection")
    connection_ok = await test_bot_connection()
    
    if connection_ok:
        # Test 2: Message sending (optional)
        print("\n💬 Test 2: Message Sending")
        await test_send_message()
        
        print("\n" + "=" * 50)
        print("✅ Bot tests completed!")
        print("\n📋 Next steps:")
        print("1. Find your bot on Telegram by searching for its username")
        print("2. Send /start to your bot")
        print("3. The bot should respond if it's running properly")
        
    else:
        print("\n" + "=" * 50)
        print("❌ Bot connection failed!")
        print("\n🔧 Troubleshooting:")
        print("1. Check if TELEGRAM_TOKEN is correct in .env file")
        print("2. Verify the token with @BotFather on Telegram")
        print("3. Make sure the bot is not revoked or disabled")

if __name__ == '__main__':
    asyncio.run(main())
