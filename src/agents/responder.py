"""
Intelligent Response Generation System - CORE INTEGRATION

Advanced response generation that orchestrates all AI components to produce
high-quality, contextually appropriate email responses. Integrates classification,
confidence analysis, escalation decisions, and prompt engineering.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging
import json
import re
from email import policy
from email import message_from_string
from email.header import decode_header

from langchain_openai import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_community.callbacks.manager import get_openai_callback

from .classifier import EmailClassifier
from .confidence_scorer import ConfidenceScorer, ConfidenceAnalysis
from .escalation_engine import EscalationEngine, EscalationDecision
from .spam_filter import SmartSpamFilter, EmailDisposition, FilterResult
from .exchange_handler import SmartExchangeHandler, ExchangeDetectionResult, ExchangeComplexity
from ..utils.prompts import PromptEngineer, PromptConfiguration
from ..core.config import settings

logger = logging.getLogger(__name__)

class ResponseQuality(Enum):
    """Response quality assessment levels"""
    EXCELLENT = "excellent"    # High quality, ready to send
    GOOD = "good"             # Good quality, minor review needed
    ACCEPTABLE = "acceptable"  # Acceptable, human review recommended
    POOR = "poor"             # Poor quality, needs human intervention
    FAILED = "failed"         # Generation failed completely

class ResponseStatus(Enum):
    """Response processing status"""
    GENERATED = "generated"           # Successfully generated
    ESCALATED = "escalated"          # Escalated to human
    FAILED = "failed"                # Generation failed
    NEEDS_REVIEW = "needs_review"    # Requires human review
    APPROVED = "approved"            # Ready to send

@dataclass
class ResponseGeneration:
    """Complete response generation result"""
    # Generated response
    response_text: str
    response_quality: ResponseQuality
    status: ResponseStatus
    
    # Analysis results that led to this response
    classification_result: Dict
    confidence_analysis: ConfidenceAnalysis
    escalation_decision: Optional[EscalationDecision]
    prompt_config: PromptConfiguration
    
    # Quality metrics
    quality_score: float  # 0.0-1.0 overall quality score
    quality_factors: Dict[str, float]  # Individual quality metrics
    
    # Performance metrics
    generation_time_ms: int
    total_tokens_used: int
    prompt_tokens: int
    completion_tokens: int
    
    # Metadata
    llm_model: str
    temperature_used: float
    reasoning: List[str]  # Why this response was generated
    
    # Error handling
    error_message: Optional[str] = None
    fallback_used: bool = False

class ResponseGenerator:
    """
    Advanced response generator that orchestrates all AI components
    to produce high-quality, contextually appropriate responses.
    """
    
    def __init__(self):
        """Initialize response generator with all components"""
        
        # Initialize all AI components
        self.spam_filter = SmartSpamFilter()  # Spam filter runs first
        self.exchange_handler = SmartExchangeHandler()  # NEW: Exchange handler runs second
        self.classifier = EmailClassifier()
        self.confidence_scorer = ConfidenceScorer()
        self.escalation_engine = EscalationEngine()
        self.prompt_engineer = PromptEngineer()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.3,  # Default, will be overridden by prompt config
            openai_api_key=settings.openai_api_key,
            max_retries=3,
            request_timeout=30
        )
        
        # Quality assessment thresholds
        self.quality_thresholds = {
            ResponseQuality.EXCELLENT: 0.9,
            ResponseQuality.GOOD: 0.75,
            ResponseQuality.ACCEPTABLE: 0.6,
            ResponseQuality.POOR: 0.4
        }
        
        # Fallback responses for different categories
        self.fallback_responses = {
            "CUSTOMER_SUPPORT": "Thank you for contacting us. We've received your inquiry and will review it carefully. A member of our customer service team will respond to you within 24 hours with detailed assistance.",
            
            "TECHNICAL_ISSUE": "Thank you for reporting this technical issue. Our technical support team has been notified and will investigate this matter. We'll provide you with an update and resolution steps within 4 hours.",
            
            "BILLING_INQUIRY": "Thank you for your billing inquiry. Our billing specialists will review your account and provide a detailed explanation within 2 business days. If this is urgent, please call our billing department directly.",
            
            "REFUND_REQUEST": "We've received your refund request and understand your concerns. Our refund specialists will review your case according to our refund policy and respond within 48 hours.",
            
            # NEW CATEGORY TEMPLATES - Designed to reduce escalation
            "CUSTOMER_PRAISE": "Thank you so much for your wonderful feedback! We truly appreciate you taking the time to share your positive experience with us. Your kind words mean a lot to our team and motivate us to continue providing excellent service. We're thrilled that we've met your expectations!",
            
            "FEATURE_SUGGESTIONS": "Thank you for your valuable suggestion! We appreciate customers who take the time to share ideas for improving our service. Your feedback has been forwarded to our product development team for consideration. While we can't implement every suggestion, all ideas are carefully reviewed and many have led to great improvements. We'll keep you updated if we move forward with similar enhancements.",
            
            "PARTNERSHIP_BUSINESS": "Thank you for your interest in partnering with us. We've received your business inquiry and it has been forwarded to our partnerships team for review. A representative will reach out to you within 2-3 business days to discuss potential collaboration opportunities.",
            
            "SUBSCRIPTION_MANAGEMENT": "We've received your subscription request and are processing it now. You can expect confirmation within 24 hours. If you need immediate assistance with your account, you can also visit our account management portal using your login credentials.",
            
            "DEFAULT": "Thank you for contacting us. We've received your message and will ensure it receives appropriate attention. You can expect a response from our team within 24 hours."
        }
    
    def _decode_mime_header(self, header_value: str) -> str:
        """Decode RFC 2047 encoded words and return a clean string."""
        try:
            parts = decode_header(header_value)
            decoded_fragments: List[str] = []
            for bytes_part, encoding in parts:
                if isinstance(bytes_part, bytes):
                    try:
                        decoded_fragments.append(bytes_part.decode(encoding or 'utf-8', errors='replace'))
                    except Exception:
                        decoded_fragments.append(bytes_part.decode('utf-8', errors='replace'))
                else:
                    decoded_fragments.append(bytes_part)
            return ''.join(decoded_fragments).strip()
        except Exception:
            return header_value.strip() if header_value else ""

    def _extract_subject(self, raw_email: str) -> str:
        """
        Extract and decode the Subject header from raw email content.
        - Handles RFC-compliant headers, folded headers, and common encodings.
        - Returns an empty string if no subject header is present.
        """
        if not raw_email:
            return ""
        try:
            # Parse email headers to find subject
            lines = raw_email.splitlines()
            subject_value: Optional[str] = None
            i = 0
            while i < len(lines):
                line = lines[i]
                if re.match(r'(?i)^subject:\s*', line):
                    # Capture header value and unfold continuation lines
                    value = re.sub(r'(?i)^subject:\s*', '', line).strip()
                    i += 1
                    while i < len(lines) and (lines[i].startswith(' ') or lines[i].startswith('\t')):
                        value += ' ' + lines[i].strip()
                        i += 1
                    subject_value = value
                    break
                i += 1
            if subject_value is None:
                return ""
            return self._decode_mime_header(subject_value)
        except Exception:
            return ""
    
    async def generate_response(self, email_content: str, sender_email: Optional[str] = None) -> ResponseGeneration:
        """
        Generate a complete response for an email using all AI components
        
        Args:
            email_content: The customer email content
            sender_email: Optional sender email for context
            
        Returns:
            ResponseGeneration with complete analysis and response
        """
        
        start_time = time.time()
        
        try:
            # Step 0: SPAM/PROMOTIONAL FILTER (NEW) - Run before everything else
            logger.info("Running spam/promotional filter")
            subject = self._extract_subject(email_content)
            spam_filter_result = self.spam_filter.filter_email(email_content, subject)
            
            # If spam filter suggests auto-handling, do it now
            if spam_filter_result.disposition != EmailDisposition.PROCESS_NORMALLY:
                return self._create_spam_filtered_response(
                    spam_filter_result, email_content, start_time
                )
            
            # Step 1: Classify the email (only if not filtered out)
            logger.info("Starting email classification")
            classification_result = self.classifier.classify(email_content)
            
            # Step 1.5: CHECK FOR EXCHANGE REQUESTS (NEW) - Run after classification to get complexity
            logger.info("Checking for exchange/return requests")
            complexity_score = classification_result.get('complexity_score', 0.5)
            exchange_result = self.exchange_handler.detect_exchange_request(email_content, complexity_score)
            
            # If exchange handler can auto-handle, do it now
            if exchange_result.is_exchange_request:
                return self._create_exchange_response(
                    exchange_result, classification_result, start_time
                )
            
            # Step 2: Analyze confidence
            logger.info(f"Analyzing confidence for category: {classification_result['category']}")
            confidence_analysis = self.confidence_scorer.analyze_confidence(
                classification_result, email_content
            )
            
            # Step 3: Make escalation decision
            logger.info(f"Making escalation decision (confidence: {confidence_analysis.overall_confidence:.2f})")
            escalation_decision = self.escalation_engine.make_escalation_decision(
                email_content, classification_result, confidence_analysis
            )
            
            # Step 4: Check if we should escalate instead of generating response
            if escalation_decision.should_escalate:
                return self._create_escalation_response(
                    classification_result, confidence_analysis, escalation_decision, start_time
                )
            
            # Step 5: Select optimal prompt
            logger.info("Selecting optimal prompt configuration")
            customer_context = escalation_decision.customer_context
            structural_features = classification_result.get('structural_features', {})
            
            prompt_config = self.prompt_engineer.select_optimal_prompt(
                category=classification_result['category'],
                confidence_analysis=confidence_analysis,
                escalation_decision=None,  # Not escalating
                customer_context=customer_context,
                structural_features=structural_features
            )
            
            # Step 6: Generate response using LLM
            logger.info(f"Generating response using {prompt_config.template_type.value} template")
            response_text, llm_metrics = await self._generate_llm_response(
                prompt_config, email_content
            )
            
            # Step 7: Assess response quality
            quality_score, quality_factors = self._assess_response_quality(
                response_text, classification_result, confidence_analysis, prompt_config
            )
            
            # Step 8: Determine response status
            status, quality_level = self._determine_response_status(
                quality_score, confidence_analysis, escalation_decision
            )
            
            # Step 9: Create reasoning
            reasoning = self._generate_response_reasoning(
                classification_result, confidence_analysis, prompt_config,
                quality_score, status
            )
            
            generation_time_ms = int((time.time() - start_time) * 1000)
            
            return ResponseGeneration(
                response_text=response_text,
                response_quality=quality_level,
                status=status,
                classification_result=classification_result,
                confidence_analysis=confidence_analysis,
                escalation_decision=escalation_decision,
                prompt_config=prompt_config,
                quality_score=quality_score,
                quality_factors=quality_factors,
                generation_time_ms=generation_time_ms,
                total_tokens_used=llm_metrics['total_tokens'],
                prompt_tokens=llm_metrics['prompt_tokens'],
                completion_tokens=llm_metrics['completion_tokens'],
                llm_model=settings.openai_model,
                temperature_used=prompt_config.temperature,
                reasoning=reasoning,
                fallback_used=False
            )
            
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}", exc_info=True)
            return self._create_fallback_response(
                email_content, str(e), start_time
            )
    
    async def _generate_llm_response(self, prompt_config: PromptConfiguration, 
                                   email_content: str) -> Tuple[str, Dict]:
        """Generate response using LLM with the configured prompt"""
        
        try:
            # Format prompt for LLM
            llm_prompt = self.prompt_engineer.format_prompt_for_llm(
                prompt_config, email_content
            )
            
            # Configure LLM with prompt-specific settings
            configured_llm = self.llm.bind(
                max_tokens=prompt_config.max_tokens,
                temperature=prompt_config.temperature
            )
            
            # Create messages
            messages = []
            for msg in llm_prompt['messages']:
                if msg['role'] == 'system':
                    messages.append(SystemMessage(content=msg['content']))
                elif msg['role'] == 'user':
                    messages.append(HumanMessage(content=msg['content']))
                elif msg['role'] == 'assistant':
                    messages.append(AIMessage(content=msg['content']))
            
            # Generate response with token tracking
            with get_openai_callback() as cb:
                response = await configured_llm.ainvoke(messages)
                
                metrics = {
                    'total_tokens': cb.total_tokens,
                    'prompt_tokens': cb.prompt_tokens,
                    'completion_tokens': cb.completion_tokens,
                    'total_cost': cb.total_cost
                }
            
            logger.info(f"LLM response generated: {metrics['total_tokens']} tokens, ${metrics['total_cost']:.4f}")
            
            return response.content, metrics
            
        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            raise
    
    def _assess_response_quality(self, response_text: str, 
                               classification_result: Dict,
                               confidence_analysis: ConfidenceAnalysis,
                               prompt_config: PromptConfiguration) -> Tuple[float, Dict[str, float]]:
        """Assess the quality of the generated response"""
        
        quality_factors = {}
        
        # 1. Length appropriateness (0.0-1.0)
        response_length = len(response_text.split())
        if prompt_config.template_type.value == 'concise':
            # Concise should be 50-150 words
            if 50 <= response_length <= 150:
                quality_factors['length'] = 1.0
            elif response_length < 50:
                quality_factors['length'] = response_length / 50.0
            else:
                quality_factors['length'] = max(0.3, 1.0 - (response_length - 150) / 200)
        else:
            # Standard should be 100-400 words
            if 100 <= response_length <= 400:
                quality_factors['length'] = 1.0
            elif response_length < 100:
                quality_factors['length'] = response_length / 100.0
            else:
                quality_factors['length'] = max(0.3, 1.0 - (response_length - 400) / 300)
        
        # 2. Professional tone (basic keyword analysis)
        professional_indicators = [
            'thank you', 'please', 'i apologize', 'i understand', 'happy to help',
            'let me', 'i\'ll', 'we\'ll', 'please let me know', 'if you have'
        ]
        
        unprofessional_indicators = [
            'whatever', 'obviously', 'clearly you', 'you should know', 'that\'s wrong'
        ]
        
        response_lower = response_text.lower()
        professional_count = sum(1 for phrase in professional_indicators if phrase in response_lower)
        unprofessional_count = sum(1 for phrase in unprofessional_indicators if phrase in response_lower)
        
        quality_factors['tone'] = max(0.3, min(1.0, 0.7 + (professional_count * 0.1) - (unprofessional_count * 0.2)))
        
        # 3. Relevance to category
        category = classification_result['category']
        category_keywords = {
            'TECHNICAL_ISSUE': ['troubleshoot', 'technical', 'issue', 'problem', 'error', 'bug'],
            'BILLING_INQUIRY': ['billing', 'charge', 'payment', 'account', 'invoice'],
            'REFUND_REQUEST': ['refund', 'return', 'policy', 'process'],
            'SHIPPING_INQUIRY': ['shipping', 'delivery', 'tracking', 'order'],
            'CUSTOMER_SUPPORT': ['help', 'assist', 'support', 'service']
        }
        
        relevant_keywords = category_keywords.get(category, [])
        relevance_count = sum(1 for keyword in relevant_keywords if keyword in response_lower)
        
        if relevant_keywords:
            quality_factors['relevance'] = min(1.0, relevance_count / len(relevant_keywords) + 0.5)
        else:
            quality_factors['relevance'] = 0.8  # Default for categories without keywords
        
        # 4. Confidence alignment (higher confidence should produce better responses)
        confidence_factor = confidence_analysis.overall_confidence
        quality_factors['confidence_alignment'] = confidence_factor
        
        # 5. Completeness (basic checks for action items)
        completeness_indicators = [
            'i\'ll', 'i will', 'we\'ll', 'we will', 'next steps', 'please',
            'contact', 'follow up', 'let me know'
        ]
        
        completeness_count = sum(1 for phrase in completeness_indicators if phrase in response_lower)
        quality_factors['completeness'] = min(1.0, completeness_count / 3.0 + 0.4)
        
        # Calculate overall quality score (weighted average)
        weights = {
            'length': 0.15,
            'tone': 0.25,
            'relevance': 0.25,
            'confidence_alignment': 0.20,
            'completeness': 0.15
        }
        
        overall_quality = sum(
            quality_factors[factor] * weights[factor] 
            for factor in weights.keys()
        )
        
        logger.info(f"Response quality assessment: {overall_quality:.2f} - {quality_factors}")
        
        return overall_quality, quality_factors
    
    def _determine_response_status(self, quality_score: float,
                                 confidence_analysis: ConfidenceAnalysis,
                                 escalation_decision: EscalationDecision) -> Tuple[ResponseStatus, ResponseQuality]:
        """Determine the appropriate status for the response"""
        
        # Determine quality level
        if quality_score >= self.quality_thresholds[ResponseQuality.EXCELLENT]:
            quality_level = ResponseQuality.EXCELLENT
        elif quality_score >= self.quality_thresholds[ResponseQuality.GOOD]:
            quality_level = ResponseQuality.GOOD
        elif quality_score >= self.quality_thresholds[ResponseQuality.ACCEPTABLE]:
            quality_level = ResponseQuality.ACCEPTABLE
        else:
            quality_level = ResponseQuality.POOR
        
        # Determine status based on quality and confidence
        if quality_level == ResponseQuality.POOR:
            return ResponseStatus.NEEDS_REVIEW, quality_level
        
        if confidence_analysis.overall_confidence >= 0.85 and quality_level in [ResponseQuality.EXCELLENT, ResponseQuality.GOOD]:
            return ResponseStatus.APPROVED, quality_level
        
        if confidence_analysis.overall_confidence >= 0.70:
            return ResponseStatus.GENERATED, quality_level
        
        return ResponseStatus.NEEDS_REVIEW, quality_level
    
    def _generate_response_reasoning(self, classification_result: Dict,
                                   confidence_analysis: ConfidenceAnalysis,
                                   prompt_config: PromptConfiguration,
                                   quality_score: float,
                                   status: ResponseStatus) -> List[str]:
        """Generate reasoning for the response generation decision"""
        
        reasoning = []
        
        # Classification reasoning
        category = classification_result['category']
        category_confidence = classification_result['confidence']
        reasoning.append(f"Classified as {category} with {category_confidence:.2f} confidence")
        
        # Prompt selection reasoning
        template_type = prompt_config.template_type.value
        reasoning.append(f"Selected {template_type} template based on context analysis")
        
        # Confidence reasoning
        overall_confidence = confidence_analysis.overall_confidence
        if overall_confidence >= 0.85:
            reasoning.append(f"High confidence ({overall_confidence:.2f}) in processing capability")
        elif overall_confidence >= 0.65:
            reasoning.append(f"Moderate confidence ({overall_confidence:.2f}) - review recommended")
        else:
            reasoning.append(f"Low confidence ({overall_confidence:.2f}) - human intervention advised")
        
        # Risk factor reasoning
        if confidence_analysis.risk_factors:
            risk_count = len(confidence_analysis.risk_factors)
            reasoning.append(f"Identified {risk_count} risk factors requiring careful handling")
        
        # Quality reasoning
        reasoning.append(f"Generated response quality score: {quality_score:.2f}")
        
        # Status reasoning
        status_descriptions = {
            ResponseStatus.APPROVED: "Response approved for immediate sending",
            ResponseStatus.GENERATED: "Response generated successfully, ready for review",
            ResponseStatus.NEEDS_REVIEW: "Response requires human review before sending",
            ResponseStatus.FAILED: "Response generation failed"
        }
        
        reasoning.append(status_descriptions.get(status, "Unknown status"))
        
        return reasoning
    
    def _create_escalation_response(self, classification_result: Dict,
                                  confidence_analysis: ConfidenceAnalysis,
                                  escalation_decision: EscalationDecision,
                                  start_time: float) -> ResponseGeneration:
        """Create response for escalated emails"""
        
        generation_time_ms = int((time.time() - start_time) * 1000)
        
        # Generate escalation acknowledgment
        escalation_response = (
            "Thank you for contacting us. Your inquiry has been forwarded to our "
            f"{escalation_decision.assigned_team.value.replace('_', ' ').title()} team for specialized assistance. "
            f"You can expect a response within {escalation_decision.response_sla_hours} hours. "
            "We appreciate your patience and will ensure you receive the attention you deserve."
        )
        
        reasoning = [
            f"Email escalated due to: {', '.join([r.value for r in escalation_decision.escalation_reasons])}",
            f"Assigned to {escalation_decision.assigned_team.value.replace('_', ' ').title()} team",
            f"Priority level: {escalation_decision.priority_level.name}",
            f"Response SLA: {escalation_decision.response_sla_hours} hours"
        ]
        
        return ResponseGeneration(
            response_text=escalation_response,
            response_quality=ResponseQuality.GOOD,
            status=ResponseStatus.ESCALATED,
            classification_result=classification_result,
            confidence_analysis=confidence_analysis,
            escalation_decision=escalation_decision,
            prompt_config=None,  # No prompt used for escalation
            quality_score=0.8,  # Standard quality for escalation responses
            quality_factors={'escalation': 1.0},
            generation_time_ms=generation_time_ms,
            total_tokens_used=0,
            prompt_tokens=0,
            completion_tokens=0,
            llm_model='escalation_system',
            temperature_used=0.0,
            reasoning=reasoning,
            fallback_used=False
        )
    
    def _create_fallback_response(self, email_content: str, error_message: str,
                                start_time: float) -> ResponseGeneration:
        """Create fallback response when generation fails"""
        
        generation_time_ms = int((time.time() - start_time) * 1000)
        
        # Try to classify for fallback selection
        try:
            classification_result = self.classifier.classify(email_content)
            category = classification_result['category']
        except Exception:
            classification_result = {'category': 'OTHER_UNCATEGORIZED', 'confidence': 0.0}
            category = 'DEFAULT'
        
        # Get appropriate fallback response
        fallback_text = self.fallback_responses.get(
            category, self.fallback_responses['DEFAULT']
        )
        
        return ResponseGeneration(
            response_text=fallback_text,
            response_quality=ResponseQuality.ACCEPTABLE,
            status=ResponseStatus.NEEDS_REVIEW,
            classification_result=classification_result,
            confidence_analysis=None,
            escalation_decision=None,
            prompt_config=None,
            quality_score=0.6,  # Acceptable quality for fallback
            quality_factors={'fallback': 1.0},
            generation_time_ms=generation_time_ms,
            total_tokens_used=0,
            prompt_tokens=0,
            completion_tokens=0,
            llm_model='fallback_system',
            temperature_used=0.0,
            reasoning=[f"Fallback response used due to error: {error_message}"],
            error_message=error_message,
            fallback_used=True
        )
    
    def _create_spam_filtered_response(self, filter_result: FilterResult,
                                     email_content: str, start_time: float) -> ResponseGeneration:
        """Create response for emails filtered by spam/promotional filter"""
        
        generation_time_ms = int((time.time() - start_time) * 1000)
        
        # Create a basic classification for the filtered email
        classification_result = {
            'category': 'SPAM_PROMOTIONAL',
            'confidence': 1.0 - filter_result.spam_score,  # Inverse of spam score
            'category_scores': {'SPAM_PROMOTIONAL': filter_result.spam_score},
            'structural_features': {},
            'sentiment_score': 0.0,
            'complexity_score': 0.0,
            'reasoning': filter_result.reasoning
        }
        
        # Determine response based on disposition
        if filter_result.disposition == EmailDisposition.DISCARD:
            response_text = ""  # No response for discarded emails
            status = ResponseStatus.GENERATED
            quality = ResponseQuality.EXCELLENT
            reasoning = ["Email discarded as spam/test content - no response needed"]
            
        elif filter_result.disposition == EmailDisposition.AUTO_RESPOND and filter_result.auto_response_template:
            response_text = filter_result.auto_response_template
            status = ResponseStatus.APPROVED
            quality = ResponseQuality.GOOD
            reasoning = ["Auto-response sent for promotional/informational content"]
            
        elif filter_result.disposition == EmailDisposition.AUTO_ACKNOWLEDGE:
            response_text = """Thank you for your email. We've received your message and will review it accordingly.

For immediate assistance, please contact our customer support team.

Best regards,
Customer Support Team"""
            status = ResponseStatus.APPROVED
            quality = ResponseQuality.GOOD
            reasoning = ["Auto-acknowledgment sent for likely promotional content"]
            
        elif filter_result.disposition == EmailDisposition.UNSUBSCRIBE:
            response_text = """We've received your unsubscribe request and are processing it now.

You will be removed from our mailing list within 24-48 hours.

If you continue to receive emails after this period, please contact our support team.

Best regards,
Customer Support Team"""
            status = ResponseStatus.APPROVED
            quality = ResponseQuality.GOOD
            reasoning = ["Unsubscribe confirmation sent"]
            
        else:
            # Fallback for unknown dispositions
            response_text = "Thank you for contacting us. We've received your message."
            status = ResponseStatus.NEEDS_REVIEW
            quality = ResponseQuality.ACCEPTABLE
            reasoning = ["Unknown disposition - requires review"]
        
        # Add spam filter reasoning to final reasoning
        combined_reasoning = reasoning + [f"Spam Filter: {r}" for r in filter_result.reasoning]
        
        return ResponseGeneration(
            response_text=response_text,
            response_quality=quality,
            status=status,
            classification_result=classification_result,
            confidence_analysis=None,  # Skip confidence analysis for filtered emails
            escalation_decision=None,  # Skip escalation for filtered emails
            prompt_config=None,  # No LLM prompt used
            quality_score=0.85 if filter_result.disposition != EmailDisposition.DISCARD else 1.0,
            quality_factors={'spam_filtered': filter_result.spam_score},
            generation_time_ms=generation_time_ms,
            total_tokens_used=0,  # No LLM tokens used
            prompt_tokens=0,
            completion_tokens=0,
            llm_model='spam_filter_system',
            temperature_used=0.0,
            reasoning=combined_reasoning,
            fallback_used=False
        )
    
    def _create_exchange_response(self, exchange_result: ExchangeDetectionResult,
                                classification_result: Dict, start_time: float) -> ResponseGeneration:
        """Create response for emails handled by exchange handler"""
        
        generation_time_ms = int((time.time() - start_time) * 1000)
        
        # Generate the response using exchange handler template
        if exchange_result.suggested_response_template:
            response_text = self.exchange_handler.format_response(
                exchange_result.suggested_response_template
            )
        else:
            # Fallback response for exchanges without templates
            response_text = """Thank you for contacting us about your return/exchange request.

We've received your message and our customer service team will process your request promptly. 
You can expect a response within 24 hours with detailed next steps.

For immediate assistance, you can also visit our returns portal at [returns.company.com] 
using your order number.

Best regards,
Customer Support Team"""
        
        # Update classification to reflect exchange category
        exchange_classification = {
            'category': 'EXCHANGE_RETURN',
            'confidence': exchange_result.confidence,
            'category_scores': {'EXCHANGE_RETURN': exchange_result.confidence},
            'structural_features': {},
            'sentiment_score': 0.0,
            'complexity_score': exchange_result.classifier_complexity,
            'reasoning': exchange_result.reasoning
        }
        
        # Determine status based on complexity
        if exchange_result.complexity == ExchangeComplexity.SIMPLE:
            status = ResponseStatus.APPROVED
            quality = ResponseQuality.GOOD
        elif exchange_result.complexity == ExchangeComplexity.MODERATE:
            status = ResponseStatus.GENERATED
            quality = ResponseQuality.ACCEPTABLE
        else:
            status = ResponseStatus.NEEDS_REVIEW
            quality = ResponseQuality.ACCEPTABLE
        
        # Create reasoning
        reasoning = [
            "EXCHANGE/RETURN REQUEST - Handled automatically by exchange handler",
            f"Exchange type: {exchange_result.exchange_type.value.replace('_', ' ').title()}",
            f"Complexity: {exchange_result.complexity.value.title()}",
            f"Confidence: {exchange_result.confidence:.2f}"
        ]
        reasoning.extend([f"Exchange Handler: {r}" for r in exchange_result.reasoning[:3]])
        
        return ResponseGeneration(
            response_text=response_text,
            response_quality=quality,
            status=status,
            classification_result=exchange_classification,
            confidence_analysis=None,  # Skip confidence analysis for exchange requests
            escalation_decision=None,  # Skip escalation for exchange requests
            prompt_config=None,  # No LLM prompt used
            quality_score=0.8 if exchange_result.complexity != ExchangeComplexity.COMPLEX else 0.6,
            quality_factors={'exchange_handled': exchange_result.confidence},
            generation_time_ms=generation_time_ms,
            total_tokens_used=0,  # No LLM tokens used
            prompt_tokens=0,
            completion_tokens=0,
            llm_model='exchange_handler_system',
            temperature_used=0.0,
            reasoning=reasoning,
            fallback_used=False
        )
    
    def get_response_summary(self, response_gen: ResponseGeneration) -> Dict[str, any]:
        """Get a summary of the response generation for logging/monitoring"""
        
        return {
            'status': response_gen.status.value,
            'quality': response_gen.response_quality.value,
            'quality_score': response_gen.quality_score,
            'category': response_gen.classification_result['category'],
            'confidence': response_gen.confidence_analysis.overall_confidence if response_gen.confidence_analysis else 0.0,
            'escalated': response_gen.escalation_decision.should_escalate if response_gen.escalation_decision else False,
            'generation_time_ms': response_gen.generation_time_ms,
            'tokens_used': response_gen.total_tokens_used,
            'template_type': response_gen.prompt_config.template_type.value if response_gen.prompt_config else 'none',
            'fallback_used': response_gen.fallback_used,
            'error': response_gen.error_message
        }