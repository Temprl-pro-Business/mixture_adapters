"""
Module for colored logging utilities.
"""

import logging
from typing import Optional
from colorama import Fore, Back, Style, init

# Initialize colorama
init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    """Custom formatter for colored log output."""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Back.WHITE
    }
    
    HIGHLIGHTS = {
        'SELECTED': Back.GREEN + Fore.BLACK,
        'ADAPTER': Fore.CYAN,
        'SCORE': Fore.YELLOW,
        'QUERY': Fore.MAGENTA,
        'LOADING': Fore.BLUE,
        'SUCCESS': Fore.GREEN,
        'ERROR': Fore.RED
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors."""
        # Color the log level
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{Style.RESET_ALL}"
        
        # Process the message for highlights
        if hasattr(record, 'highlight') and record.highlight:
            for key, color in self.HIGHLIGHTS.items():
                record.msg = record.msg.replace(f"<{key}>", color)
                record.msg = record.msg.replace(f"</{key}>", Style.RESET_ALL)
        
        return super().format(record)

class ColoredLogger:
    """Logger with colored output and semantic highlighting."""
    
    def __init__(self, name: str, level: int = logging.INFO):
        """
        Initialize the colored logger.
        
        Args:
            name (str): Logger name
            level (int): Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Add console handler with colored formatter
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(ColoredFormatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(console_handler)
    
    def _log(self, level: int, msg: str, highlight: bool = False) -> None:
        """
        Log a message with optional highlighting.
        
        Args:
            level (int): Logging level
            msg (str): Message to log
            highlight (bool): Whether to process highlight tags
        """
        extra = {'highlight': highlight} if highlight else None
        self.logger.log(level, msg, extra=extra)
    
    def info(self, msg: str, highlight: bool = False) -> None:
        """Log an info message."""
        self._log(logging.INFO, msg, highlight)
    
    def warning(self, msg: str, highlight: bool = False) -> None:
        """Log a warning message."""
        self._log(logging.WARNING, msg, highlight)
    
    def error(self, msg: str, highlight: bool = False) -> None:
        """Log an error message."""
        self._log(logging.ERROR, msg, highlight)
    
    def debug(self, msg: str, highlight: bool = False) -> None:
        """Log a debug message."""
        self._log(logging.DEBUG, msg, highlight)
    
    def success(self, msg: str) -> None:
        """Log a success message."""
        self.info(f"<SUCCESS>{msg}</SUCCESS>", highlight=True) 