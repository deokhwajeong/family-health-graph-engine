"""
Logging configuration for the application.
Provides structured logging with JSON output for production environments.
"""

import logging
import sys
from typing import Optional


class StructuredLogger:
    """Structured logging for API requests and errors."""
    
    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Console handler with formatting
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)-8s %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def info(self, msg: str, **kwargs):
        """Log info with optional context."""
        self.logger.info(f"{msg} | {kwargs}" if kwargs else msg)
    
    def error(self, msg: str, exc: Optional[Exception] = None, **kwargs):
        """Log error with optional exception and context."""
        context = f" | {kwargs}" if kwargs else ""
        if exc:
            self.logger.error(f"{msg}{context}", exc_info=True)
        else:
            self.logger.error(f"{msg}{context}")
    
    def debug(self, msg: str, **kwargs):
        """Log debug with optional context."""
        self.logger.debug(f"{msg} | {kwargs}" if kwargs else msg)
    
    def warning(self, msg: str, **kwargs):
        """Log warning with optional context."""
        self.logger.warning(f"{msg} | {kwargs}" if kwargs else msg)


# Global logger instances
api_logger = StructuredLogger("api")
ml_logger = StructuredLogger("ml")
data_logger = StructuredLogger("data")
