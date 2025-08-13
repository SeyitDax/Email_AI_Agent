"""
EmailAgent - Legacy Compatibility Wrapper

This module provides backward compatibility for existing code while leveraging
our sophisticated AI pipeline under the hood. It maintains the same interface
but uses our advanced classification, confidence scoring, and response generation.
"""

import os
import asyncio
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, Literal

# Import our sophisticated AI system
from src.agents.responder import ResponseGenerator

# Load environment variables
load_dotenv()

# Define data models (maintained for backward compatibility)
class EmailClassification(BaseModel):
    category: Literal["order_status", "refund_request", "technical_support", "product_inquiry", "other"]
    confidence: float = Field(ge=0, le=1)
    requires_human: bool
    summary: str

class EmailResponse(BaseModel):
    response_text: str
    confidence: float = Field(ge=0, le=1)
    suggested_action: Literal["send", "review", "escalate"]

class EmailAgent:
    """
    Legacy EmailAgent that uses our sophisticated AI system under the hood
    Maintains backward compatibility while providing advanced functionality
    """
    
    def __init__(self):
        """Initialize EmailAgent with our advanced response generator"""
        self.response_generator = ResponseGenerator()
        
        # Category mapping from our new system to legacy format
        self.category_mapping = {
            'CUSTOMER_SUPPORT': 'other',
            'SALES_ORDER': 'order_status', 
            'TECHNICAL_ISSUE': 'technical_support',
            'BILLING_INQUIRY': 'other',
            'PRODUCT_QUESTION': 'product_inquiry',
            'REFUND_REQUEST': 'refund_request',
            'SHIPPING_INQUIRY': 'order_status',
            'ACCOUNT_ISSUE': 'other',
            'COMPLAINT_NEGATIVE': 'other',
            'COMPLIMENT_POSITIVE': 'other',
            'SPAM_PROMOTIONAL': 'other',
            'PARTNERSHIP_BUSINESS': 'other',
            'PRESS_MEDIA': 'other',
            'LEGAL_COMPLIANCE': 'other',
            'OTHER_UNCATEGORIZED': 'other'
        }
        
        # Action mapping from our new system to legacy format
        self.action_mapping = {
            'approved': 'send',
            'generated': 'review',
            'needs_review': 'review',
            'escalated': 'escalate',
            'failed': 'escalate'
        }
    
    def classify_email(self, email_content: str) -> EmailClassification:
        """
        Classify an incoming email using our sophisticated classification system
        """
        
        try:
            # Use our advanced classifier directly
            from src.agents.classifier import EmailClassifier
            classifier = EmailClassifier()
            
            # Get sophisticated classification
            classification_result = classifier.classify(email_content)
            
            # Map to legacy format
            legacy_category = self.category_mapping.get(
                classification_result['category'], 'other'
            )
            
            # Determine if human intervention required based on confidence and complexity
            requires_human = (
                classification_result['confidence'] < 0.7 or  # Low confidence
                classification_result.get('complexity_score', 0) > 0.8 or  # High complexity
                abs(classification_result.get('sentiment_score', 0)) > 0.8  # High emotion
            )
            
            # Generate summary from reasoning
            reasoning = classification_result.get('reasoning', [])
            summary = '; '.join(reasoning[:2]) if reasoning else f"Classified as {legacy_category}"
            
            return EmailClassification(
                category=legacy_category,
                confidence=classification_result['confidence'],
                requires_human=requires_human,
                summary=summary
            )
            
        except Exception as e:
            # Fallback classification
            return EmailClassification(
                category="other",
                confidence=0.5,
                requires_human=True,
                summary=f"Classification failed: {str(e)}"
            )
    
    def generate_response(self, email_content: str, category: str = None) -> EmailResponse:
        """
        Generate a response using our sophisticated AI pipeline
        """
        
        try:
            # Use our advanced response generation system
            response_generation = asyncio.run(
                self.response_generator.generate_response(email_content)
            )
            
            # Map to legacy format
            legacy_action = self.action_mapping.get(
                response_generation.status.value, 'review'
            )
            
            return EmailResponse(
                response_text=response_generation.response_text,
                confidence=response_generation.quality_score,
                suggested_action=legacy_action
            )
            
        except Exception as e:
            # Fallback response
            return EmailResponse(
                response_text="Thank you for contacting us. We've received your inquiry and will respond shortly.",
                confidence=0.6,
                suggested_action="review"
            )
    
    def process_email(self, email_content: str) -> dict:
        """
        Main method to process an email end-to-end using our sophisticated AI system
        Maintains backward compatibility with legacy interface
        """
        
        try:
            # Use our complete sophisticated pipeline
            response_generation = asyncio.run(
                self.response_generator.generate_response(email_content)
            )
            
            # Extract classification info
            classification_result = response_generation.classification_result
            legacy_category = 'other'
            if classification_result:
                legacy_category = self.category_mapping.get(
                    classification_result['category'], 'other'
                )
            
            # Check if escalated
            if (response_generation.escalation_decision and 
                response_generation.escalation_decision.should_escalate):
                
                escalation_reasons = [r.value for r in response_generation.escalation_decision.escalation_reasons]
                return {
                    "action": "escalate",
                    "reason": f"Escalated due to: {', '.join(escalation_reasons)}",
                    "category": legacy_category,
                    "confidence": classification_result['confidence'] if classification_result else 0.0,
                    "summary": response_generation.escalation_decision.business_impact,
                    "priority": response_generation.escalation_decision.priority_level.value,
                    "assigned_team": response_generation.escalation_decision.assigned_team.value
                }
            
            # Map action to legacy format
            legacy_action = self.action_mapping.get(
                response_generation.status.value, 'review'
            )
            
            # Generate summary from our detailed reasoning
            reasoning = response_generation.reasoning
            summary = reasoning[0] if reasoning else "Email processed successfully"
            
            return {
                "action": legacy_action,
                "category": legacy_category,
                "response": response_generation.response_text,
                "confidence": response_generation.quality_score,
                "summary": summary,
                "quality_grade": response_generation.response_quality.value,
                "processing_time_ms": response_generation.generation_time_ms,
                "tokens_used": response_generation.total_tokens_used
            }
            
        except Exception as e:
            # Fallback processing
            return {
                "action": "escalate",
                "reason": f"Processing failed: {str(e)}",
                "category": "other",
                "confidence": 0.0,
                "summary": "System error - requires human attention",
                "error": str(e)
            }