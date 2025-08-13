"""
Pytest configuration and shared fixtures for the AI Email Agent test suite.
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import MagicMock, AsyncMock
from typing import Dict, Any

# Test data fixtures
@pytest.fixture
def sample_emails():
    """Sample emails for testing different categories and scenarios"""
    return {
        'customer_support': "Hi, I'm having trouble with my account and can't log in. Can you help me reset my password?",
        
        'order_status': "Where is my order #12345? I ordered it 5 days ago and haven't received any tracking information.",
        
        'technical_issue': "The app keeps crashing every time I try to open it. I've tried restarting my phone but it doesn't help. Error code: 500.",
        
        'billing_inquiry': "I was charged twice for the same subscription. Can you help me understand why and process a refund?",
        
        'refund_request': "I want to return this product. It's not what I expected and doesn't work for my needs. Please send a return label.",
        
        'complaint_negative': "This is terrible customer service! I've been trying to get help for weeks and nobody responds. Your service is unacceptable!",
        
        'compliment_positive': "I just wanted to say thank you for the excellent customer service. The representative was very helpful and solved my problem quickly.",
        
        'spam_promotional': "CONGRATULATIONS! You've won $1,000,000! Click here immediately to claim your prize. Limited time offer!",
        
        'legal_compliance': "I am requesting all personal data you have stored about me under GDPR regulations. Please provide this within 30 days.",
        
        'vip_customer': "This is John Smith, CEO of Fortune 500 Company. I need immediate assistance with our enterprise account issue.",
        
        'high_emotion': "I am absolutely furious! This is the worst experience I've ever had. I'm considering legal action against your company!",
        
        'technical_complex': "Our API integration is failing with SSL certificate errors when connecting to your webhook endpoint. The error occurs during the TLS handshake.",
        
        'multi_channel': "I've already called and emailed about this issue. I spoke to someone yesterday who said they would follow up but I haven't heard anything.",
        
        'urgent_business': "URGENT: Our production system is down and customers are affected. We need immediate technical support to resolve this outage."
    }

@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response for testing"""
    return {
        'choices': [{
            'message': {
                'content': 'Thank you for contacting us. We understand your concern and will help resolve this issue promptly.'
            }
        }],
        'usage': {
            'prompt_tokens': 150,
            'completion_tokens': 50,
            'total_tokens': 200
        }
    }

@pytest.fixture
def mock_database():
    """Mock database for testing without real DB connections"""
    mock_db = MagicMock()
    mock_db.store_email_record = AsyncMock(return_value="test-email-id-123")
    mock_db.store_classification_result = AsyncMock()
    mock_db.store_confidence_analysis = AsyncMock()
    mock_db.store_escalation_decision = AsyncMock()
    mock_db.store_response = AsyncMock(return_value="test-response-id-456")
    mock_db.store_performance_metric = AsyncMock()
    mock_db.get_performance_summary = AsyncMock(return_value={
        'total_emails': 100,
        'processed_emails': 95,
        'escalated_emails': 10,
        'avg_processing_time_ms': 1500,
        'avg_quality_score': 0.82,
        'processing_rate': 0.95,
        'escalation_rate': 0.10
    })
    return mock_db

@pytest.fixture
def mock_email_service():
    """Mock email service for testing without external APIs"""
    mock_service = MagicMock()
    mock_service.analyze_email_content = MagicMock(return_value={
        'is_suspicious': False,
        'suspicious_patterns': [],
        'is_likely_spam': False,
        'spam_score': 0.1,
        'content_length': 100,
        'word_count': 20
    })
    mock_service.validate_email_address = MagicMock(return_value=True)
    mock_service.send_response = AsyncMock(return_value=True)
    return mock_service

@pytest.fixture
def test_config():
    """Test configuration with safe defaults"""
    return {
        'openai_api_key': 'test-key',
        'openai_model': 'gpt-3.5-turbo',
        'database_url': 'sqlite:///:memory:',
        'use_gmail_api': False,
        'use_smtp': False,
        'environment': 'test'
    }

@pytest.fixture
def temp_directory():
    """Create a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def classification_results():
    """Sample classification results for testing"""
    return {
        'category': 'CUSTOMER_SUPPORT',
        'confidence': 0.85,
        'category_scores': {
            'CUSTOMER_SUPPORT': 0.85,
            'TECHNICAL_ISSUE': 0.12,
            'OTHER_UNCATEGORIZED': 0.03
        },
        'structural_features': {
            'has_question_mark': True,
            'has_greeting': True,
            'has_phone_number': False,
            'has_email': False,
            'has_order_number': False,
            'word_count': 20,
            'sentence_count': 2
        },
        'sentiment_score': -0.2,
        'complexity_score': 0.3,
        'reasoning': [
            'Contains customer service keywords',
            'Has question mark indicating inquiry',
            'Polite tone detected'
        ]
    }

@pytest.fixture
def confidence_analysis_result():
    """Sample confidence analysis result for testing"""
    from src.agents.confidence_scorer import ConfidenceAnalysis, RiskFactor, ProcessingAction
    
    return ConfidenceAnalysis(
        overall_confidence=0.82,
        classification_confidence=0.85,
        response_confidence=0.80,
        contextual_confidence=0.78,
        score_separation=0.73,
        category_certainty=0.85,
        risk_factors=[],
        risk_score=0.1,
        recommended_action=ProcessingAction.SEND,
        confidence_threshold_met={'send': True, 'review': True, 'escalate': False},
        reasoning=['High classification confidence', 'Clear category distinction', 'Low risk factors']
    )

@pytest.fixture
def escalation_decision_result():
    """Sample escalation decision result for testing"""
    from src.agents.escalation_engine import (
        EscalationDecision, EscalationReason, PriorityLevel, EscalationTeam
    )
    
    return EscalationDecision(
        should_escalate=False,
        escalation_reasons=[],
        priority_level=PriorityLevel.ROUTINE,
        assigned_team=EscalationTeam.SENIOR_SUPPORT,
        estimated_complexity=0.3,
        response_sla_hours=24,
        resolution_sla_hours=72,
        customer_context={'is_vip': False, 'customer_type': 'standard'},
        business_impact='LOW - Standard customer inquiry',
        special_instructions=[],
        reasoning=['No escalation required - automated processing recommended'],
        escalation_score=0.2,
        decision_confidence=0.82
    )

# Performance benchmarks
@pytest.fixture
def performance_benchmarks():
    """Expected performance benchmarks for testing"""
    return {
        'max_processing_time_ms': 5000,  # 5 seconds max
        'min_classification_accuracy': 0.80,  # 80% accuracy minimum
        'min_confidence_correlation': 0.70,  # Confidence should correlate with success
        'max_escalation_rate': 0.30,  # Max 30% escalation rate
        'min_quality_score': 0.65,  # Minimum response quality
        'max_tokens_per_response': 600  # Token usage limit
    }

# Test categories for parameterized tests
@pytest.fixture
def test_categories():
    """List of all email categories for testing"""
    return [
        'CUSTOMER_SUPPORT', 'SALES_ORDER', 'TECHNICAL_ISSUE', 'BILLING_INQUIRY',
        'PRODUCT_QUESTION', 'REFUND_REQUEST', 'SHIPPING_INQUIRY', 'ACCOUNT_ISSUE',
        'COMPLAINT_NEGATIVE', 'COMPLIMENT_POSITIVE', 'SPAM_PROMOTIONAL',
        'PARTNERSHIP_BUSINESS', 'PRESS_MEDIA', 'LEGAL_COMPLIANCE', 'OTHER_UNCATEGORIZED'
    ]

@pytest.fixture
def risk_factors():
    """List of all risk factors for testing"""
    from src.agents.confidence_scorer import RiskFactor
    return list(RiskFactor)

@pytest.fixture
def prompt_templates():
    """List of all prompt template types for testing"""
    from src.utils.prompts import PromptTemplate
    return list(PromptTemplate)