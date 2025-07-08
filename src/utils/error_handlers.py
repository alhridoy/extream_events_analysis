from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
import logging

logger = logging.getLogger(__name__)

async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle Pydantic validation errors with detailed messages"""
    logger.error(f"Validation error on {request.url}: {exc}")
    
    errors = []
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error["loc"])
        errors.append({
            "field": field,
            "message": error["msg"],
            "type": error["type"]
        })
    
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation failed",
            "details": errors,
            "message": "Please check your input and try again"
        }
    )

async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions with proper error messages"""
    logger.error(f"Unhandled error on {request.url}: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again."
        }
    )
