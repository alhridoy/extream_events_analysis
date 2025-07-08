from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
from enum import Enum

class ActionType(str, Enum):
    CLICK = "click"
    TYPE = "type"
    NAVIGATE = "navigate"
    SCROLL = "scroll"

class BrowserAction(BaseModel):
    action: ActionType
    target: str = Field(..., min_length=1, description="CSS selector or element description")
    value: Optional[str] = Field(None, description="Value to type or click")
    timeout: int = Field(default=30, ge=1, le=300, description="Timeout in seconds")
    
    @validator('target')
    def validate_target(cls, v):
        if not v.strip():
            raise ValueError('Target cannot be empty')
        return v.strip()

class InteractionRequest(BaseModel):
    command: str = Field(..., min_length=1, max_length=1000)
    options: Dict[str, Any] = Field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    @validator('command')
    def validate_command(cls, v):
        if not v.strip():
            raise ValueError('Command cannot be empty')
        return v.strip()

class ConfigRequest(BaseModel):
    proxy_url: Optional[str] = None
    headless: bool = True
    timeout: int = Field(default=30, ge=1, le=300)
    user_agent: Optional[str] = None
