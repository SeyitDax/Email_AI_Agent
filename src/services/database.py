"""
Database Operations and Models - SERVICE LAYER

SQLAlchemy models and operations for persisting email processing data,
analysis results, and performance metrics. Supports PostgreSQL with
async operations for high-performance email processing.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, Float, Boolean, 
    DateTime, JSON, ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import relationship, sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID, ENUM
import uuid
import json
import logging

from ..core.config import settings

logger = logging.getLogger(__name__)

# Create database base
Base = declarative_base()

class EmailRecord(Base):
    """Email records with original content and metadata"""
    __tablename__ = "emails"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Email content
    subject = Column(String(500), nullable=True)
    body = Column(Text, nullable=False)
    sender_email = Column(String(255), nullable=True)
    sender_name = Column(String(255), nullable=True)
    
    # Processing metadata
    received_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    processed_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(50), nullable=False, default='pending')  # pending, processing, completed, failed
    
    # Email metadata
    email_length = Column(Integer, nullable=True)
    language_detected = Column(String(10), nullable=True)
    
    # System metadata
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    classification = relationship("ClassificationResult", back_populates="email", uselist=False)
    confidence_analysis = relationship("ConfidenceAnalysis", back_populates="email", uselist=False)  
    escalation_decision = relationship("EscalationDecision", back_populates="email", uselist=False)
    responses = relationship("ResponseRecord", back_populates="email")
    performance_metrics = relationship("PerformanceMetric", back_populates="email")
    
    # Indexes
    __table_args__ = (
        Index('idx_emails_received_at', 'received_at'),
        Index('idx_emails_sender', 'sender_email'),
        Index('idx_emails_status', 'status'),
    )

class ClassificationResult(Base):
    """Email classification results from the classifier"""
    __tablename__ = "classification_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email_id = Column(UUID(as_uuid=True), ForeignKey('emails.id'), nullable=False)
    
    # Classification results
    category = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    category_scores = Column(JSON, nullable=False)  # All category scores
    
    # Analysis details
    structural_features = Column(JSON, nullable=True)
    sentiment_score = Column(Float, nullable=True)
    complexity_score = Column(Float, nullable=True)
    reasoning = Column(JSON, nullable=True)  # List of reasoning strings
    
    # Processing metadata
    processing_time_ms = Column(Integer, nullable=True)
    classifier_version = Column(String(50), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    email = relationship("EmailRecord", back_populates="classification")
    
    # Indexes
    __table_args__ = (
        Index('idx_classification_category', 'category'),
        Index('idx_classification_confidence', 'confidence'),
    )

class ConfidenceAnalysisRecord(Base):
    """Confidence analysis results from confidence scorer"""
    __tablename__ = "confidence_analysis"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email_id = Column(UUID(as_uuid=True), ForeignKey('emails.id'), nullable=False)
    
    # Confidence metrics
    overall_confidence = Column(Float, nullable=False)
    classification_confidence = Column(Float, nullable=False)
    response_confidence = Column(Float, nullable=False)
    contextual_confidence = Column(Float, nullable=False)
    
    # Score analysis
    score_separation = Column(Float, nullable=False)
    category_certainty = Column(Float, nullable=False)
    
    # Risk assessment
    risk_factors = Column(JSON, nullable=False)  # List of risk factor names
    risk_score = Column(Float, nullable=False)
    
    # Decision support
    recommended_action = Column(String(20), nullable=False)  # send, review, escalate
    confidence_threshold_met = Column(JSON, nullable=False)  # Dict of threshold results
    reasoning = Column(JSON, nullable=False)  # List of reasoning strings
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    email = relationship("EmailRecord", back_populates="confidence_analysis")
    
    # Indexes
    __table_args__ = (
        Index('idx_confidence_overall', 'overall_confidence'),
        Index('idx_confidence_action', 'recommended_action'),
    )

class EscalationDecisionRecord(Base):
    """Escalation decisions from escalation engine"""
    __tablename__ = "escalation_decisions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email_id = Column(UUID(as_uuid=True), ForeignKey('emails.id'), nullable=False)
    
    # Core decision
    should_escalate = Column(Boolean, nullable=False)
    escalation_reasons = Column(JSON, nullable=False)  # List of reason names
    
    # Priority and routing
    priority_level = Column(Integer, nullable=False)  # 1-5
    assigned_team = Column(String(50), nullable=False)
    estimated_complexity = Column(Float, nullable=False)
    
    # SLA requirements
    response_sla_hours = Column(Integer, nullable=False)
    resolution_sla_hours = Column(Integer, nullable=False)
    
    # Additional context
    customer_context = Column(JSON, nullable=False)
    business_impact = Column(Text, nullable=False)
    special_instructions = Column(JSON, nullable=False)  # List of instructions
    reasoning = Column(JSON, nullable=False)  # List of reasoning strings
    
    # Metadata
    escalation_score = Column(Float, nullable=False)
    decision_confidence = Column(Float, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    email = relationship("EmailRecord", back_populates="escalation_decision")
    
    # Indexes
    __table_args__ = (
        Index('idx_escalation_should_escalate', 'should_escalate'),
        Index('idx_escalation_priority', 'priority_level'),
        Index('idx_escalation_team', 'assigned_team'),
    )

class ResponseRecord(Base):
    """Generated responses and their metadata"""
    __tablename__ = "responses"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email_id = Column(UUID(as_uuid=True), ForeignKey('emails.id'), nullable=False)
    
    # Response content
    response_text = Column(Text, nullable=False)
    response_quality = Column(String(20), nullable=False)  # excellent, good, acceptable, poor, failed
    status = Column(String(20), nullable=False)  # generated, escalated, failed, needs_review, approved
    
    # Quality metrics
    quality_score = Column(Float, nullable=False)
    quality_factors = Column(JSON, nullable=False)  # Dict of quality factor scores
    
    # Performance metrics
    generation_time_ms = Column(Integer, nullable=False)
    total_tokens_used = Column(Integer, nullable=False)
    prompt_tokens = Column(Integer, nullable=False)
    completion_tokens = Column(Integer, nullable=False)
    
    # Generation metadata
    llm_model = Column(String(50), nullable=False)
    temperature_used = Column(Float, nullable=False)
    prompt_template_type = Column(String(20), nullable=True)
    reasoning = Column(JSON, nullable=False)  # List of reasoning strings
    
    # Error handling
    error_message = Column(Text, nullable=True)
    fallback_used = Column(Boolean, nullable=False, default=False)
    
    # Human review
    human_reviewed = Column(Boolean, nullable=False, default=False)
    human_approved = Column(Boolean, nullable=True)
    human_feedback = Column(Text, nullable=True)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    reviewed_by = Column(String(255), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    email = relationship("EmailRecord", back_populates="responses")
    
    # Indexes
    __table_args__ = (
        Index('idx_response_quality', 'response_quality'),
        Index('idx_response_status', 'status'),
        Index('idx_response_human_reviewed', 'human_reviewed'),
    )

class PerformanceMetric(Base):
    """System performance metrics for monitoring and optimization"""
    __tablename__ = "performance_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email_id = Column(UUID(as_uuid=True), ForeignKey('emails.id'), nullable=True)  # Can be email-specific or system-wide
    
    # Metric identification
    metric_type = Column(String(50), nullable=False)  # classification_time, response_quality, etc.
    metric_name = Column(String(100), nullable=False)
    
    # Metric values
    value = Column(Float, nullable=False)
    unit = Column(String(20), nullable=True)  # ms, tokens, score, etc.
    
    # Context
    category = Column(String(50), nullable=True)  # Email category this metric relates to
    metric_metadata = Column(JSON, nullable=True)  # Additional context data
    
    # Timestamps
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    email = relationship("EmailRecord", back_populates="performance_metrics")
    
    # Indexes
    __table_args__ = (
        Index('idx_metrics_type', 'metric_type'),
        Index('idx_metrics_name', 'metric_name'),
        Index('idx_metrics_timestamp', 'timestamp'),
    )

class SystemConfiguration(Base):
    """System configuration and settings"""
    __tablename__ = "system_configuration"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Configuration
    config_key = Column(String(100), nullable=False, unique=True)
    config_value = Column(JSON, nullable=False)
    description = Column(Text, nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    updated_by = Column(String(255), nullable=True)


class DatabaseManager:
    """Database manager for email processing operations"""
    
    def __init__(self, database_url: str = None):
        """Initialize database manager"""
        self.database_url = database_url or settings.database_url
        
        # Create sync engine for migrations and admin tasks
        self.sync_engine = create_engine(
            self.database_url.replace('postgresql+asyncpg://', 'postgresql://'),
            pool_size=10,
            max_overflow=20,
            pool_timeout=30,
            pool_recycle=1800
        )
        
        # Create async engine for main operations
        self.async_engine = create_async_engine(
            self.database_url,
            pool_size=10,
            max_overflow=20,
            pool_timeout=30,
            pool_recycle=1800
        )
        
        # Create session factories
        self.async_session_factory = sessionmaker(
            bind=self.async_engine, class_=AsyncSession, expire_on_commit=False
        )
        self.sync_session_factory = sessionmaker(bind=self.sync_engine)
    
    def create_tables(self):
        """Create all database tables"""
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=self.sync_engine)
        logger.info("Database tables created successfully")
    
    def drop_tables(self):
        """Drop all database tables (use with caution!)"""
        logger.warning("Dropping all database tables...")
        Base.metadata.drop_all(bind=self.sync_engine)
        logger.info("Database tables dropped")
    
    async def store_email_record(self, email_data: Dict[str, Any]) -> str:
        """Store a new email record and return the ID"""
        async with self.async_session_factory() as session:
            email = EmailRecord(
                subject=email_data.get('subject'),
                body=email_data['body'],
                sender_email=email_data.get('sender_email'),
                sender_name=email_data.get('sender_name'),
                email_length=len(email_data['body'].split()),
                language_detected=email_data.get('language', 'en')
            )
            
            session.add(email)
            await session.commit()
            await session.refresh(email)
            
            logger.info(f"Stored email record: {email.id}")
            return str(email.id)
    
    async def store_classification_result(self, email_id: str, classification_data: Dict[str, Any]):
        """Store classification result for an email"""
        async with self.async_session_factory() as session:
            classification = ClassificationResult(
                email_id=uuid.UUID(email_id),
                category=classification_data['category'],
                confidence=classification_data['confidence'],
                category_scores=classification_data.get('category_scores', {}),
                structural_features=classification_data.get('structural_features', {}),
                sentiment_score=classification_data.get('sentiment_score'),
                complexity_score=classification_data.get('complexity_score'),
                reasoning=classification_data.get('reasoning', []),
                processing_time_ms=classification_data.get('processing_time_ms'),
                classifier_version=classification_data.get('version', '1.0')
            )
            
            session.add(classification)
            await session.commit()
            logger.info(f"Stored classification result for email: {email_id}")
    
    async def store_confidence_analysis(self, email_id: str, confidence_data: Dict[str, Any]):
        """Store confidence analysis for an email"""
        async with self.async_session_factory() as session:
            confidence = ConfidenceAnalysisRecord(
                email_id=uuid.UUID(email_id),
                overall_confidence=confidence_data['overall_confidence'],
                classification_confidence=confidence_data['classification_confidence'],
                response_confidence=confidence_data['response_confidence'],
                contextual_confidence=confidence_data['contextual_confidence'],
                score_separation=confidence_data['score_separation'],
                category_certainty=confidence_data['category_certainty'],
                risk_factors=[rf.value for rf in confidence_data['risk_factors']],
                risk_score=confidence_data['risk_score'],
                recommended_action=confidence_data['recommended_action'].value,
                confidence_threshold_met=confidence_data['confidence_threshold_met'],
                reasoning=confidence_data['reasoning']
            )
            
            session.add(confidence)
            await session.commit()
            logger.info(f"Stored confidence analysis for email: {email_id}")
    
    async def store_escalation_decision(self, email_id: str, escalation_data: Dict[str, Any]):
        """Store escalation decision for an email"""
        async with self.async_session_factory() as session:
            escalation = EscalationDecisionRecord(
                email_id=uuid.UUID(email_id),
                should_escalate=escalation_data['should_escalate'],
                escalation_reasons=[er.value for er in escalation_data['escalation_reasons']],
                priority_level=escalation_data['priority_level'].value,
                assigned_team=escalation_data['assigned_team'].value,
                estimated_complexity=escalation_data['estimated_complexity'],
                response_sla_hours=escalation_data['response_sla_hours'],
                resolution_sla_hours=escalation_data['resolution_sla_hours'],
                customer_context=escalation_data['customer_context'],
                business_impact=escalation_data['business_impact'],
                special_instructions=escalation_data['special_instructions'],
                reasoning=escalation_data['reasoning'],
                escalation_score=escalation_data['escalation_score'],
                decision_confidence=escalation_data['decision_confidence']
            )
            
            session.add(escalation)
            await session.commit()
            logger.info(f"Stored escalation decision for email: {email_id}")
    
    async def store_response(self, email_id: str, response_data: Dict[str, Any]) -> str:
        """Store generated response and return response ID"""
        async with self.async_session_factory() as session:
            response = ResponseRecord(
                email_id=uuid.UUID(email_id),
                response_text=response_data['response_text'],
                response_quality=response_data['response_quality'].value,
                status=response_data['status'].value,
                quality_score=response_data['quality_score'],
                quality_factors=response_data['quality_factors'],
                generation_time_ms=response_data['generation_time_ms'],
                total_tokens_used=response_data['total_tokens_used'],
                prompt_tokens=response_data['prompt_tokens'],
                completion_tokens=response_data['completion_tokens'],
                llm_model=response_data['llm_model'],
                temperature_used=response_data['temperature_used'],
                prompt_template_type=response_data.get('prompt_template_type'),
                reasoning=response_data['reasoning'],
                error_message=response_data.get('error_message'),
                fallback_used=response_data['fallback_used']
            )
            
            session.add(response)
            await session.commit()
            await session.refresh(response)
            
            logger.info(f"Stored response for email: {email_id}")
            return str(response.id)
    
    async def store_performance_metric(self, metric_data: Dict[str, Any]):
        """Store performance metric"""
        async with self.async_session_factory() as session:
            metric = PerformanceMetric(
                email_id=uuid.UUID(metric_data['email_id']) if metric_data.get('email_id') else None,
                metric_type=metric_data['metric_type'],
                metric_name=metric_data['metric_name'],
                value=metric_data['value'],
                unit=metric_data.get('unit'),
                category=metric_data.get('category'),
                metric_metadata=metric_data.get('metadata', {})
            )
            
            session.add(metric)
            await session.commit()
    
    async def get_email_with_analysis(self, email_id: str) -> Optional[Dict[str, Any]]:
        """Get email with all analysis results"""
        async with self.async_session_factory() as session:
            email = await session.get(EmailRecord, uuid.UUID(email_id))
            if not email:
                return None
            
            # Build complete record
            return {
                'email': {
                    'id': str(email.id),
                    'subject': email.subject,
                    'body': email.body,
                    'sender_email': email.sender_email,
                    'received_at': email.received_at.isoformat(),
                    'status': email.status
                },
                'classification': self._serialize_classification(email.classification) if email.classification else None,
                'confidence_analysis': self._serialize_confidence(email.confidence_analysis) if email.confidence_analysis else None,
                'escalation_decision': self._serialize_escalation(email.escalation_decision) if email.escalation_decision else None,
                'responses': [self._serialize_response(r) for r in email.responses]
            }
    
    async def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the last N hours"""
        from datetime import timedelta
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        async with self.async_session_factory() as session:
            # Get basic stats
            from sqlalchemy import func, and_
            
            total_emails = await session.scalar(
                func.count(EmailRecord.id).where(EmailRecord.created_at >= cutoff_time)
            )
            
            processed_emails = await session.scalar(
                func.count(EmailRecord.id).where(
                    and_(EmailRecord.created_at >= cutoff_time, EmailRecord.status == 'completed')
                )
            )
            
            escalated_count = await session.scalar(
                func.count(EscalationDecisionRecord.id).where(
                    and_(EscalationDecisionRecord.created_at >= cutoff_time, 
                         EscalationDecisionRecord.should_escalate == True)
                )
            )
            
            avg_processing_time = await session.scalar(
                func.avg(ResponseRecord.generation_time_ms).where(
                    ResponseRecord.created_at >= cutoff_time
                )
            )
            
            avg_quality_score = await session.scalar(
                func.avg(ResponseRecord.quality_score).where(
                    ResponseRecord.created_at >= cutoff_time
                )
            )
            
            return {
                'period_hours': hours,
                'total_emails': total_emails or 0,
                'processed_emails': processed_emails or 0,
                'escalated_emails': escalated_count or 0,
                'escalation_rate': (escalated_count or 0) / max(total_emails or 1, 1),
                'avg_processing_time_ms': float(avg_processing_time) if avg_processing_time else 0,
                'avg_quality_score': float(avg_quality_score) if avg_quality_score else 0,
                'processing_rate': (processed_emails or 0) / max(total_emails or 1, 1)
            }
    
    def _serialize_classification(self, classification: ClassificationResult) -> Dict[str, Any]:
        """Serialize classification result"""
        return {
            'category': classification.category,
            'confidence': classification.confidence,
            'category_scores': classification.category_scores,
            'structural_features': classification.structural_features,
            'sentiment_score': classification.sentiment_score,
            'complexity_score': classification.complexity_score,
            'reasoning': classification.reasoning,
            'processing_time_ms': classification.processing_time_ms
        }
    
    def _serialize_confidence(self, confidence: ConfidenceAnalysisRecord) -> Dict[str, Any]:
        """Serialize confidence analysis"""
        return {
            'overall_confidence': confidence.overall_confidence,
            'classification_confidence': confidence.classification_confidence,
            'response_confidence': confidence.response_confidence,
            'contextual_confidence': confidence.contextual_confidence,
            'score_separation': confidence.score_separation,
            'category_certainty': confidence.category_certainty,
            'risk_factors': confidence.risk_factors,
            'risk_score': confidence.risk_score,
            'recommended_action': confidence.recommended_action,
            'reasoning': confidence.reasoning
        }
    
    def _serialize_escalation(self, escalation: EscalationDecisionRecord) -> Dict[str, Any]:
        """Serialize escalation decision"""
        return {
            'should_escalate': escalation.should_escalate,
            'escalation_reasons': escalation.escalation_reasons,
            'priority_level': escalation.priority_level,
            'assigned_team': escalation.assigned_team,
            'estimated_complexity': escalation.estimated_complexity,
            'response_sla_hours': escalation.response_sla_hours,
            'resolution_sla_hours': escalation.resolution_sla_hours,
            'business_impact': escalation.business_impact,
            'escalation_score': escalation.escalation_score
        }
    
    def _serialize_response(self, response: ResponseRecord) -> Dict[str, Any]:
        """Serialize response record"""
        return {
            'id': str(response.id),
            'response_text': response.response_text,
            'response_quality': response.response_quality,
            'status': response.status,
            'quality_score': response.quality_score,
            'generation_time_ms': response.generation_time_ms,
            'total_tokens_used': response.total_tokens_used,
            'llm_model': response.llm_model,
            'human_reviewed': response.human_reviewed,
            'human_approved': response.human_approved,
            'created_at': response.created_at.isoformat()
        }
    
    async def close(self):
        """Close database connections"""
        await self.async_engine.dispose()
        self.sync_engine.dispose()


# Global database manager instance
db_manager = DatabaseManager()