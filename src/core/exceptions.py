"""
Custom Exceptions - BOILERPLATE
Application-specific exception classes
"""

from typing import Optional, Any, Dict


class EmailAgentException(Exception):
    """Base exception for Email AI Agent"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ClassificationError(EmailAgentException):
    """Error during email classification"""
    pass


class ConfidenceCalculationError(EmailAgentException):
    """Error during confidence calculation"""
    pass


class EscalationError(EmailAgentException):
    """Error during escalation decision"""
    pass


class ResponseGenerationError(EmailAgentException):
    """Error during response generation"""
    pass


class DatabaseError(EmailAgentException):
    """Database operation error"""
    pass


class QueueError(EmailAgentException):
    """Queue operation error"""
    pass


class EmailServiceError(EmailAgentException):
    """Email service error"""
    pass


class ConfigurationError(EmailAgentException):
    """Configuration error"""
    pass


class ValidationError(EmailAgentException):
    """Data validation error"""
    pass


class RateLimitError(EmailAgentException):
    """Rate limit exceeded error"""
    pass


class ExternalServiceError(EmailAgentException):
    """External service error (OpenAI, etc.)"""
    pass


class ProcessingTimeoutError(EmailAgentException):
    """Processing timeout error"""
    pass