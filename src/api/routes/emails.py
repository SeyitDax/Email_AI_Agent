"""
Email Processing Routes - MIXED (Boilerplate + Custom Logic Integration)
API endpoints for email processing operations
"""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
import uuid

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from src.core.config import settings
from src.core.models import (
    EmailProcessingRequest,
    EmailProcessingResult,
    BatchProcessingRequest,
    BatchProcessingResult,
    EmailStatus,
    ErrorResponse,
)
from src.core.exceptions import (
    ClassificationError,
    ConfidenceCalculationError,
    EscalationError,
    ResponseGenerationError,
    ValidationError,
)

# Import our sophisticated AI components
from src.agents.responder import ResponseGenerator
from src.services.database import db_manager
from src.services.email_service import email_service

router = APIRouter()
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer(auto_error=False)


def get_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> str:
    """Validate API key (optional security)"""
    if settings.environment == "production" and not credentials:
        raise HTTPException(status_code=401, detail="API key required")
    
    return credentials.credentials if credentials else "development"


# Initialize our complete AI pipeline
response_generator = ResponseGenerator()

# Initialize services on startup
@router.on_event("startup")
async def startup_event():
    """Initialize services when the API starts"""
    logger.info("Initializing AI Email Agent services...")
    
    # Initialize email service
    await email_service.initialize()
    logger.info("Email service initialized")
    
    # Initialize database
    try:
        db_manager.create_tables()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
    
    logger.info("AI Email Agent services initialized successfully")


async def process_single_email(email_request: EmailProcessingRequest) -> EmailProcessingResult:
    """
    Process a single email through the complete AI pipeline
    INTEGRATED COMPONENT: Uses our sophisticated AI system
    """
    
    start_time = datetime.utcnow()
    email_id = None
    
    try:
        email = email_request.email
        
        # Step 1: Store email in database
        email_data = {
            'subject': email.subject,
            'body': email.body,
            'sender_email': email.sender,
            'sender_name': email.sender_name,
            'language': 'en'  # Could be detected
        }
        
        email_id = await db_manager.store_email_record(email_data)
        logger.info(f"Stored email record: {email_id}")
        
        # Step 2: Analyze email content for security
        content_analysis = email_service.analyze_email_content(email.body)
        if content_analysis['is_likely_spam']:
            logger.warning(f"Potential spam detected: score {content_analysis['spam_score']:.2f}")
        
        # Step 3: Generate complete response using our AI pipeline
        response_generation = await response_generator.generate_response(
            email_content=email.body,
            sender_email=email.sender
        )
        
        # Step 4: Store all analysis results in database
        await _store_analysis_results(email_id, response_generation)
        
        # Step 5: Convert our internal models to API response models
        api_result = await _convert_to_api_response(
            email_id, response_generation, start_time
        )
        
        # Step 6: Update email status in database
        await _update_email_status(email_id, api_result.status)
        
        # Step 7: Handle response sending or escalation
        await _handle_response_action(email, response_generation, email_request.options)
        
        logger.info(f"Email {email_id} processed successfully: {api_result.status}")
        return api_result
        
    except Exception as e:
        processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        logger.error(f"Email processing failed: {str(e)}", exc_info=True)
        
        # Store performance metric for failure
        if email_id:
            try:
                await db_manager.store_performance_metric({
                    'email_id': email_id,
                    'metric_type': 'processing_failure',
                    'metric_name': 'pipeline_error',
                    'value': 1.0,
                    'metadata': {'error': str(e)}
                })
            except Exception as db_e:
                logger.error(f"Failed to store error metric: {db_e}")
        
        return EmailProcessingResult(
            email_id=email_id or str(uuid.uuid4()),
            status=EmailStatus.FAILED,
            processing_time_ms=processing_time,
            processed_at=datetime.utcnow(),
            error_message=str(e),
        )


async def _store_analysis_results(email_id: str, response_generation):
    """Store all analysis results in the database"""
    try:
        # Store classification result
        if response_generation.classification_result:
            await db_manager.store_classification_result(email_id, {
                'category': response_generation.classification_result['category'],
                'confidence': response_generation.classification_result['confidence'],
                'category_scores': response_generation.classification_result.get('category_scores', {}),
                'structural_features': response_generation.classification_result.get('structural_features', {}),
                'sentiment_score': response_generation.classification_result.get('sentiment_score'),
                'complexity_score': response_generation.classification_result.get('complexity_score'),
                'reasoning': response_generation.classification_result.get('reasoning', []),
                'processing_time_ms': response_generation.generation_time_ms,
                'version': '1.0'
            })
        
        # Store confidence analysis
        if response_generation.confidence_analysis:
            await db_manager.store_confidence_analysis(email_id, {
                'overall_confidence': response_generation.confidence_analysis.overall_confidence,
                'classification_confidence': response_generation.confidence_analysis.classification_confidence,
                'response_confidence': response_generation.confidence_analysis.response_confidence,
                'contextual_confidence': response_generation.confidence_analysis.contextual_confidence,
                'score_separation': response_generation.confidence_analysis.score_separation,
                'category_certainty': response_generation.confidence_analysis.category_certainty,
                'risk_factors': response_generation.confidence_analysis.risk_factors,
                'risk_score': response_generation.confidence_analysis.risk_score,
                'recommended_action': response_generation.confidence_analysis.recommended_action,
                'confidence_threshold_met': response_generation.confidence_analysis.confidence_threshold_met,
                'reasoning': response_generation.confidence_analysis.reasoning
            })
        
        # Store escalation decision
        if response_generation.escalation_decision:
            await db_manager.store_escalation_decision(email_id, {
                'should_escalate': response_generation.escalation_decision.should_escalate,
                'escalation_reasons': response_generation.escalation_decision.escalation_reasons,
                'priority_level': response_generation.escalation_decision.priority_level,
                'assigned_team': response_generation.escalation_decision.assigned_team,
                'estimated_complexity': response_generation.escalation_decision.estimated_complexity,
                'response_sla_hours': response_generation.escalation_decision.response_sla_hours,
                'resolution_sla_hours': response_generation.escalation_decision.resolution_sla_hours,
                'customer_context': response_generation.escalation_decision.customer_context,
                'business_impact': response_generation.escalation_decision.business_impact,
                'special_instructions': response_generation.escalation_decision.special_instructions,
                'reasoning': response_generation.escalation_decision.reasoning,
                'escalation_score': response_generation.escalation_decision.escalation_score,
                'decision_confidence': response_generation.escalation_decision.decision_confidence
            })
        
        # Store response record
        await db_manager.store_response(email_id, {
            'response_text': response_generation.response_text,
            'response_quality': response_generation.response_quality,
            'status': response_generation.status,
            'quality_score': response_generation.quality_score,
            'quality_factors': response_generation.quality_factors,
            'generation_time_ms': response_generation.generation_time_ms,
            'total_tokens_used': response_generation.total_tokens_used,
            'prompt_tokens': response_generation.prompt_tokens,
            'completion_tokens': response_generation.completion_tokens,
            'llm_model': response_generation.llm_model,
            'temperature_used': response_generation.temperature_used,
            'prompt_template_type': response_generation.prompt_config.template_type.value if response_generation.prompt_config else None,
            'reasoning': response_generation.reasoning,
            'error_message': response_generation.error_message,
            'fallback_used': response_generation.fallback_used
        })
        
        # Store performance metrics
        await db_manager.store_performance_metric({
            'email_id': email_id,
            'metric_type': 'processing_time',
            'metric_name': 'total_generation_time',
            'value': response_generation.generation_time_ms,
            'unit': 'ms',
            'category': response_generation.classification_result.get('category') if response_generation.classification_result else None
        })
        
        await db_manager.store_performance_metric({
            'email_id': email_id,
            'metric_type': 'quality_score',
            'metric_name': 'response_quality',
            'value': response_generation.quality_score,
            'unit': 'score',
            'category': response_generation.classification_result.get('category') if response_generation.classification_result else None
        })
        
        logger.info(f"Stored all analysis results for email: {email_id}")
        
    except Exception as e:
        logger.error(f"Failed to store analysis results for email {email_id}: {e}")
        # Don't re-raise as this shouldn't fail the main processing


async def _convert_to_api_response(email_id: str, response_generation, start_time) -> EmailProcessingResult:
    """Convert internal response generation to API response format"""
    
    processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
    
    # Convert to API models
    from src.core.models import (
        ClassificationResult, EmailCategory, 
        ConfidenceAnalysis, ProcessingAction,
        EscalationDecision, ResponseGeneration
    )
    
    # Map our internal category names to API enums
    category_mapping = {
        'CUSTOMER_SUPPORT': EmailCategory.CUSTOMER_SUPPORT,
        'SALES_ORDER': EmailCategory.SALES_ORDER,
        'TECHNICAL_ISSUE': EmailCategory.TECHNICAL_ISSUE,
        'BILLING_INQUIRY': EmailCategory.BILLING_INQUIRY,
        'PRODUCT_QUESTION': EmailCategory.PRODUCT_QUESTION,
        'REFUND_REQUEST': EmailCategory.REFUND_REQUEST,
        'SHIPPING_INQUIRY': EmailCategory.SHIPPING_INQUIRY,
        'ACCOUNT_ISSUE': EmailCategory.ACCOUNT_ISSUE,
        'COMPLAINT_NEGATIVE': EmailCategory.COMPLAINT_NEGATIVE,
        'COMPLIMENT_POSITIVE': EmailCategory.COMPLIMENT_POSITIVE,
        'SPAM_PROMOTIONAL': EmailCategory.SPAM_PROMOTIONAL,
        'PARTNERSHIP_BUSINESS': EmailCategory.PARTNERSHIP_BUSINESS,
        'PRESS_MEDIA': EmailCategory.PRESS_MEDIA,
        'LEGAL_COMPLIANCE': EmailCategory.LEGAL_COMPLIANCE,
        'OTHER_UNCATEGORIZED': EmailCategory.OTHER_UNCATEGORIZED
    }
    
    # Build classification result
    classification = None
    if response_generation.classification_result:
        category = category_mapping.get(
            response_generation.classification_result['category'], 
            EmailCategory.OTHER_UNCATEGORIZED
        )
        
        classification = ClassificationResult(
            category=category,
            confidence=response_generation.classification_result['confidence'],
            category_scores=response_generation.classification_result.get('category_scores', {}),
            structural_features=response_generation.classification_result.get('structural_features', {}),
            reasoning=response_generation.classification_result.get('reasoning', [])
        )
    
    # Build confidence analysis
    confidence_analysis = None
    if response_generation.confidence_analysis:
        action_mapping = {
            'send': ProcessingAction.SEND,
            'review': ProcessingAction.REVIEW,
            'escalate': ProcessingAction.ESCALATE
        }
        
        recommended_action = action_mapping.get(
            response_generation.confidence_analysis.recommended_action.value,
            ProcessingAction.REVIEW
        )
        
        confidence_analysis = ConfidenceAnalysis(
            overall_confidence=response_generation.confidence_analysis.overall_confidence,
            confidence_factors={'overall': response_generation.confidence_analysis.overall_confidence},
            risk_factors=[rf.value for rf in response_generation.confidence_analysis.risk_factors],
            threshold_analysis={'send_threshold': 0.85},
            recommended_action=recommended_action
        )
    
    # Build escalation decision
    escalation_decision = None
    if response_generation.escalation_decision:
        escalation_decision = EscalationDecision(
            should_escalate=response_generation.escalation_decision.should_escalate,
            escalation_reason=', '.join([r.value for r in response_generation.escalation_decision.escalation_reasons]) if response_generation.escalation_decision.escalation_reasons else None,
            priority_level=response_generation.escalation_decision.priority_level.value,
            estimated_complexity=response_generation.escalation_decision.estimated_complexity
        )
    
    # Build response generation
    response_gen = None
    if response_generation.response_text:
        response_gen = ResponseGeneration(
            response_text=response_generation.response_text,
            response_type=response_generation.response_quality.value,
            confidence=response_generation.quality_score,
            prompt_used=response_generation.prompt_config.template_type.value if response_generation.prompt_config else 'default',
            tokens_used=response_generation.total_tokens_used
        )
    
    # Determine status
    if response_generation.status.value == 'escalated':
        status = EmailStatus.ESCALATED
    elif response_generation.status.value == 'failed':
        status = EmailStatus.FAILED
    elif response_generation.status.value == 'approved':
        status = EmailStatus.SENT
    else:
        status = EmailStatus.RESPONSE_GENERATED
    
    return EmailProcessingResult(
        email_id=email_id,
        status=status,
        classification=classification,
        confidence_analysis=confidence_analysis,
        escalation_decision=escalation_decision,
        response_generation=response_gen,
        processing_time_ms=processing_time,
        processed_at=datetime.utcnow(),
    )


async def _update_email_status(email_id: str, status: EmailStatus):
    """Update email processing status in database"""
    try:
        # This would require adding an update method to DatabaseManager
        # For now, we'll just log the status update
        logger.info(f"Email {email_id} status updated to: {status}")
    except Exception as e:
        logger.error(f"Failed to update email status: {e}")


async def _handle_response_action(email, response_generation, options):
    """Handle the response action (sending, escalation notification, etc.)"""
    try:
        if response_generation.escalation_decision and response_generation.escalation_decision.should_escalate:
            # Send escalation notification
            logger.info("Sending escalation notification")
            # Could implement escalation notification here
            
        elif response_generation.status.value == 'approved':
            # Auto-send approved responses if configured
            if options and options.get('auto_send', False):
                logger.info("Auto-sending approved response")
                # Could implement auto-sending here
                
        else:
            logger.info(f"Response generated with status: {response_generation.status.value}")
            
    except Exception as e:
        logger.error(f"Failed to handle response action: {e}")
        # Don't re-raise as this shouldn't fail the main processing


@router.post("/process", response_model=EmailProcessingResult)
async def process_email(
    request: EmailProcessingRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key),
):
    """
    Process a single email through the AI pipeline
    """
    
    logger.info(f"Processing email from {request.email.sender}")
    
    try:
        result = await process_single_email(request)
        
        # TODO: Add background tasks for post-processing
        # background_tasks.add_task(log_processing_metrics, result)
        # background_tasks.add_task(update_model_feedback, result)
        
        return result
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.message)
    except (ClassificationError, ConfidenceCalculationError, EscalationError, ResponseGenerationError) as e:
        raise HTTPException(status_code=422, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error processing email: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal processing error")


@router.post("/batch", response_model=BatchProcessingResult)
async def process_batch(
    request: BatchProcessingRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key),
):
    """
    Process multiple emails in a batch
    """
    
    batch_id = request.batch_id or str(uuid.uuid4())
    logger.info(f"Processing batch {batch_id} with {len(request.emails)} emails")
    
    start_time = datetime.utcnow()
    results = []
    successful = 0
    failed = 0
    
    for email in request.emails:
        email_request = EmailProcessingRequest(email=email, options=request.options)
        result = await process_single_email(email_request)
        
        results.append(result)
        
        if result.status == EmailStatus.FAILED:
            failed += 1
        else:
            successful += 1
    
    batch_processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
    
    batch_result = BatchProcessingResult(
        batch_id=batch_id,
        total_emails=len(request.emails),
        successful=successful,
        failed=failed,
        results=results,
        batch_processing_time_ms=batch_processing_time,
    )
    
    # TODO: Add background task for batch analytics
    # background_tasks.add_task(log_batch_metrics, batch_result)
    
    return batch_result


@router.get("/status/{email_id}")
async def get_email_status(
    email_id: str,
    api_key: str = Depends(get_api_key),
):
    """
    Get processing status for a specific email
    """
    
    # TODO: Implement database lookup for email status
    # This is a placeholder implementation
    
    return {
        "email_id": email_id,
        "status": "not_implemented",
        "message": "Status lookup not yet implemented - requires database integration"
    }


@router.get("/categories")
async def get_categories():
    """
    Get all available email categories
    """
    
    from src.core.models import EmailCategory
    
    return {
        "categories": [
            {
                "id": category.value,
                "name": category.value.replace("_", " ").title(),
                "description": f"Email category for {category.value.replace('_', ' ')}"
            }
            for category in EmailCategory
        ]
    }


@router.post("/classify")
async def classify_only(
    request: EmailProcessingRequest,
    api_key: str = Depends(get_api_key),
):
    """
    Classify email without generating response (classification only)
    """
    
    try:
        logger.info("Processing classification-only request")
        
        # Use our classifier directly
        from src.agents.classifier import EmailClassifier
        classifier = EmailClassifier()
        
        # Classify the email
        classification_result = classifier.classify(request.email.body)
        
        # Analyze email content for security
        content_analysis = email_service.analyze_email_content(request.email.body)
        
        # Map to API response format
        from src.core.models import EmailCategory
        category_mapping = {
            'CUSTOMER_SUPPORT': EmailCategory.CUSTOMER_SUPPORT,
            'SALES_ORDER': EmailCategory.SALES_ORDER,
            'TECHNICAL_ISSUE': EmailCategory.TECHNICAL_ISSUE,
            'BILLING_INQUIRY': EmailCategory.BILLING_INQUIRY,
            'PRODUCT_QUESTION': EmailCategory.PRODUCT_QUESTION,
            'REFUND_REQUEST': EmailCategory.REFUND_REQUEST,
            'SHIPPING_INQUIRY': EmailCategory.SHIPPING_INQUIRY,
            'ACCOUNT_ISSUE': EmailCategory.ACCOUNT_ISSUE,
            'COMPLAINT_NEGATIVE': EmailCategory.COMPLAINT_NEGATIVE,
            'COMPLIMENT_POSITIVE': EmailCategory.COMPLIMENT_POSITIVE,
            'SPAM_PROMOTIONAL': EmailCategory.SPAM_PROMOTIONAL,
            'PARTNERSHIP_BUSINESS': EmailCategory.PARTNERSHIP_BUSINESS,
            'PRESS_MEDIA': EmailCategory.PRESS_MEDIA,
            'LEGAL_COMPLIANCE': EmailCategory.LEGAL_COMPLIANCE,
            'OTHER_UNCATEGORIZED': EmailCategory.OTHER_UNCATEGORIZED
        }
        
        api_category = category_mapping.get(
            classification_result['category'], 
            EmailCategory.OTHER_UNCATEGORIZED
        )
        
        return {
            "status": "success",
            "classification": {
                "category": api_category,
                "confidence": classification_result['confidence'],
                "category_scores": classification_result.get('category_scores', {}),
                "structural_features": classification_result.get('structural_features', {}),
                "sentiment_score": classification_result.get('sentiment_score'),
                "complexity_score": classification_result.get('complexity_score'),
                "reasoning": classification_result.get('reasoning', [])
            },
            "content_analysis": content_analysis,
            "processing_time_ms": 0  # We could measure this if needed
        }
        
    except Exception as e:
        logger.error(f"Classification failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@router.get("/performance/summary")
async def get_performance_summary(
    hours: int = 24,
    api_key: str = Depends(get_api_key),
):
    """
    Get system performance summary for monitoring and analytics
    """
    
    try:
        # Validate hours parameter
        if hours < 1 or hours > 168:  # Max 1 week
            raise HTTPException(status_code=400, detail="Hours must be between 1 and 168")
        
        # Get performance summary from database
        summary = await db_manager.get_performance_summary(hours)
        
        # Add additional computed metrics
        summary['performance_grade'] = _calculate_performance_grade(summary)
        summary['recommendations'] = _generate_performance_recommendations(summary)
        
        logger.info(f"Generated performance summary for {hours} hours")
        return {
            "status": "success",
            "summary": summary,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Performance summary generation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Performance summary failed: {str(e)}")


def _calculate_performance_grade(summary: Dict[str, Any]) -> str:
    """Calculate overall performance grade"""
    score = 0.0
    
    # Processing rate (0-30 points)
    processing_rate = summary.get('processing_rate', 0.0)
    score += min(processing_rate * 30, 30)
    
    # Quality score (0-40 points)  
    avg_quality = summary.get('avg_quality_score', 0.0)
    score += avg_quality * 40
    
    # Speed (0-20 points) - lower processing time is better
    avg_time = summary.get('avg_processing_time_ms', 5000)
    speed_score = max(0, 20 - (avg_time / 1000) * 2)  # Penalty for >1s processing
    score += speed_score
    
    # Escalation rate (0-10 points) - lower is better
    escalation_rate = summary.get('escalation_rate', 0.5)
    escalation_score = max(0, 10 - escalation_rate * 20)  # Penalty for high escalation
    score += escalation_score
    
    # Convert to letter grade
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"


def _generate_performance_recommendations(summary: Dict[str, Any]) -> List[str]:
    """Generate performance improvement recommendations"""
    recommendations = []
    
    processing_rate = summary.get('processing_rate', 0.0)
    avg_quality = summary.get('avg_quality_score', 0.0)
    avg_time = summary.get('avg_processing_time_ms', 0)
    escalation_rate = summary.get('escalation_rate', 0.0)
    
    if processing_rate < 0.8:
        recommendations.append("Low processing rate detected. Check for system bottlenecks or processing failures.")
    
    if avg_quality < 0.7:
        recommendations.append("Low average quality score. Review prompt engineering and model fine-tuning.")
    
    if avg_time > 3000:  # 3 seconds
        recommendations.append("High processing time detected. Consider optimizing API calls or adding caching.")
    
    if escalation_rate > 0.3:  # 30% escalation rate
        recommendations.append("High escalation rate. Review confidence thresholds and risk factor detection.")
    
    if not recommendations:
        recommendations.append("System performance is good. Continue monitoring for optimization opportunities.")
    
    return recommendations