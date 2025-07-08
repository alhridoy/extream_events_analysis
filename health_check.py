"""
Simple health check endpoint for the extreme events analysis API
"""

from datetime import datetime
from typing import Dict, Any

def get_health_status() -> Dict[str, Any]:
    """
    Get the health status of the application
    
    Returns:
        Dict containing health status information
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "extreme_events_analysis",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    status = get_health_status()
    print(f"Health Status: {status}")
