import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

class ActionLogger:
    """Comprehensive logger for browser actions and user interactions"""
    
    def __init__(self, log_file: str = "browser_actions.log"):
        self.log_file = Path(log_file)
        self.setup_logger()
    
    def setup_logger(self):
        """Setup structured logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_action(self, action: str, target: str, value: Optional[str] = None, 
                   user_id: Optional[str] = None, **kwargs):
        """Log a browser action with structured data"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "target": target,
            "value": value,
            "user_id": user_id,
            "metadata": kwargs
        }
        
        self.logger.info(f"ACTION: {json.dumps(log_data)}")
    
    def log_error(self, error: str, context: Dict[str, Any] = None):
        """Log errors with context"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "context": context or {}
        }
        
        self.logger.error(f"ERROR: {json.dumps(log_data)}")
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics"""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "duration_ms": duration * 1000,
            "metadata": kwargs
        }
        
        self.logger.info(f"PERFORMANCE: {json.dumps(log_data)}")

# Global logger instance
action_logger = ActionLogger()
