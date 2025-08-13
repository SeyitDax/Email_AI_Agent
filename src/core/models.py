"""
Pydantic Models - BOILERPLATE
Data models for API requests/responses and internal data structures
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field


class EmailCategory(str, Enum):
    """Email classification categories"""
    CUSTOMER_SUPPORT = "customer_support"
    SALES_ORDER = "sales_order"
    MARKETING_PROMOTIONS = "marketing_promotions"
    BILLING_FINANCE = "billing_finance"
    INTERNAL_COMMUNICATION = "internal_communication"
    VENDOR_SUPPLIER = "vendor_supplier"
    LEGAL_COMPLIANCE = "legal_compliance"
    TECHNICAL_IT = "technical_it"
    URGENT_HIGH_PRIORITY = "urgent_high_priority"
    SPAM_JUNK = "spam_junk"
    NEWS_INFORMATION = "news_information"
    SOCIAL_PERSONAL = "social_personal"
    ACTION_REQUIRED = "action_required"
    FYI_INFORMATIONAL = "fyi_informational"
    FOLLOW_UP = "follow_up"


class ProcessingAction(str, Enum):
    """Actions that can be taken on an email"""
    SEND = "send"
    REVIEW = "review"
    ESCALATE = "escalate"


class EmailStatus(str, Enum):
    """Email processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    CLASSIFIED = "classified"
    RESPONSE_GENERATED = "response_generated"
    SENT = "sent"
    ESCALATED = "escalated"
    FAILED = "failed"


class EmailRequest(BaseModel):
    """Request model for email processing"""
    subject: str = Field(..., description="Email subject line")
    body: str = Field(..., description="Email content/body")
    sender: str = Field(..., description="Sender email address")
    recipient: Optional[str] = Field(None, description="Recipient email address")
    thread_id: Optional[str] = Field(None, description="Email thread identifier")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class ClassificationResult(BaseModel):
    """Result of email classification"""
    category: EmailCategory = Field(..., description="Predicted email category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence score")
    category_scores: Dict[str, float] = Field(..., description="Scores for all categories")
    structural_features: Dict[str, bool] = Field(..., description="Detected structural features")
    reasoning: List[str] = Field(default_factory=list, description="Classification reasoning")


class ConfidenceAnalysis(BaseModel):
    """Confidence scoring analysis"""
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_factors: Dict[str, float] = Field(...)
    risk_factors: List[str] = Field(default_factory=list)
    threshold_analysis: Dict[str, float] = Field(...)
    recommended_action: ProcessingAction = Field(...)


class EscalationDecision(BaseModel):
    """Escalation decision result"""
    should_escalate: bool = Field(..., description="Whether to escalate to human")
    escalation_reason: Optional[str] = Field(None, description="Reason for escalation")
    priority_level: int = Field(default=1, ge=1, le=5, description="Priority level (1=low, 5=critical)")
    estimated_complexity: float = Field(..., ge=0.0, le=1.0, description="Email complexity score")


class ResponseGeneration(BaseModel):
    """Generated response details"""
    response_text: str = Field(..., description="Generated response content")
    response_type: str = Field(..., description="Type of response generated")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Response quality confidence")
    prompt_used: str = Field(..., description="Prompt template used")
    tokens_used: int = Field(..., description="Number of tokens consumed")


class EmailProcessingResult(BaseModel):
    """Complete email processing result"""
    email_id: str = Field(..., description="Unique email identifier")
    status: EmailStatus = Field(..., description="Processing status")
    
    # Processing stages
    classification: Optional[ClassificationResult] = None
    confidence_analysis: Optional[ConfidenceAnalysis] = None
    escalation_decision: Optional[EscalationDecision] = None
    response_generation: Optional[ResponseGeneration] = None
    
    # Metadata
    processing_time_ms: int = Field(..., description="Total processing time in milliseconds")
    processed_at: datetime = Field(..., description="When processing completed")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")


class EmailProcessingRequest(BaseModel):
    """Request to process an email"""
    email: EmailRequest = Field(..., description="Email to process")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Processing options")
    callback_url: Optional[str] = Field(None, description="Webhook URL for completion notification")


class BatchProcessingRequest(BaseModel):
    """Request to process multiple emails"""
    emails: List[EmailRequest] = Field(..., description="List of emails to process")
    batch_id: Optional[str] = Field(None, description="Batch identifier")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Processing options")


class BatchProcessingResult(BaseModel):
    """Result of batch email processing"""
    batch_id: str = Field(..., description="Batch identifier")
    total_emails: int = Field(..., description="Total emails in batch")
    successful: int = Field(..., description="Successfully processed emails")
    failed: int = Field(..., description="Failed email processing")
    results: List[EmailProcessingResult] = Field(..., description="Individual email results")
    batch_processing_time_ms: int = Field(..., description="Total batch processing time")


class HealthCheck(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Check timestamp")
    version: str = Field(..., description="Application version")
    dependencies: Dict[str, str] = Field(..., description="Dependency status")


class MetricsResponse(BaseModel):
    """System metrics response"""
    emails_processed_total: int = Field(..., description="Total emails processed")
    emails_processed_today: int = Field(..., description="Emails processed today")
    average_processing_time_ms: float = Field(..., description="Average processing time")
    classification_accuracy: float = Field(..., description="Classification accuracy rate")
    escalation_rate: float = Field(..., description="Escalation rate percentage")
    category_distribution: Dict[str, int] = Field(..., description="Distribution of email categories")
    confidence_distribution: Dict[str, int] = Field(..., description="Distribution of confidence scores")


class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")