"""
UI Components for MasterTrade Bot including menu and button handlers
"""

from typing import Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class TradingMode(Enum):
    DEMO = "DEMO"
    REAL = "REAL"

@dataclass
class Button:
    text: str
    callback: str
    row: int
    col: int

class UIManager:
    def __init__(self):
        self.trading_mode = TradingMode.DEMO
        self.callbacks = {}
        
    def register_callback(self, name: str, callback: Callable):
        """Register a callback function for a button"""
        self.callbacks[name] = callback
        
    def get_main_menu_buttons(self) -> Dict[str, Button]:
        """Get the main menu buttons configuration"""
        return {
            'refresh': Button('🔄 Refresh Signal', 'refresh_signal', 0, 0),
            'toggle_mode': Button(
                '🔄 Switch to REAL MODE' if self.trading_mode == TradingMode.DEMO 
                else '🔄 Switch to DEMO MODE',
                'toggle_mode',
                0, 1
            ),
            'menu': Button('📋 Menu', 'show_menu', 1, 0),
            'settings': Button('⚙️ Settings', 'show_settings', 1, 1)
        }
        
    def handle_button_press(self, button_id: str) -> Optional[Dict]:
        """Handle button press events"""
        try:
            if button_id in self.callbacks:
                return self.callbacks[button_id]()
            else:
                logger.warning(f"No callback registered for button {button_id}")
                return None
        except Exception as e:
            logger.error(f"Error handling button press: {str(e)}")
            return None
            
    def toggle_trading_mode(self) -> Dict:
        """Toggle between demo and real trading modes"""
        if self.trading_mode == TradingMode.DEMO:
            self.trading_mode = TradingMode.REAL
            return {'status': 'success', 'message': '⚠️ Switched to REAL trading mode'}
        else:
            self.trading_mode = TradingMode.DEMO
            return {'status': 'success', 'message': 'Switched to DEMO mode'}
            
    def get_settings_menu(self) -> Dict[str, Button]:
        """Get settings menu configuration"""
        return {
            'risk_settings': Button('⚠️ Risk Management', 'show_risk_settings', 0, 0),
            'notifications': Button('🔔 Notifications', 'show_notifications', 0, 1),
            'api_settings': Button('🔑 API Settings', 'show_api_settings', 1, 0),
            'back': Button('⬅️ Back', 'show_main_menu', 1, 1)
        }
        
    def format_button_grid(self, buttons: Dict[str, Button]) -> str:
        """Format buttons into a grid layout for display"""
        # Group buttons by row
        rows = {}
        for button in buttons.values():
            if button.row not in rows:
                rows[button.row] = []
            rows[button.row].append(button)
            
        # Sort buttons in each row by column
        for row in rows.values():
            row.sort(key=lambda x: x.col)
            
        # Format rows
        formatted_rows = []
        for row_num in sorted(rows.keys()):
            row_buttons = rows[row_num]
            formatted_rows.append(" | ".join(button.text for button in row_buttons))
            
        return "\n".join(formatted_rows)
