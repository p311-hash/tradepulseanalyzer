
# Read the file
with open('enhanced_telegram_handler.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the main menu keyboard
old_text = '                [InlineKeyboardButton(\
🎯
GENERATE
SIGNAL\, callback_data=\generate_signal\)],'
new_text = '                [InlineKeyboardButton(\🎯
SIGNAL
MENU\, callback_data=\signal_menu\)],'

content = content.replace(old_text, new_text)

# Remove the asset and time selection buttons from main menu
old_buttons = '''                [
                    InlineKeyboardButton(\💱
SELECT
ASSET\, callback_data=\select_asset\),
                    InlineKeyboardButton(\⏱
SELECT
TIME\, callback_data=\select_time\)
                ],'''

content = content.replace(old_buttons, '')

# Write back to file
with open('enhanced_telegram_handler.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('✅ Main menu updated successfully')

