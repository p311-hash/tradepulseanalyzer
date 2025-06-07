
with open('enhanced_telegram_handler.py', 'r') as f:
    content = f.read()

# Update main menu button
content = content.replace(
    'InlineKeyboardButton(\
🎯
GENERATE
SIGNAL\, callback_data=\generate_signal\)',
    'InlineKeyboardButton(\🎯
SIGNAL
MENU\, callback_data=\signal_menu\)'
)

with open('enhanced_telegram_handler.py', 'w') as f:
    f.write(content)

print('Updated main menu button')

