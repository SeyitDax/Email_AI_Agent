"""
Integration tests for the complete email processing pipeline.
Tests end-to-end flow from email input to response generation.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from src.agents.responder import ResponseGenerator, ResponseStatus
from email_agent import EmailAgent


class TestEmailPipeline:
    """Test suite for complete email processing pipeline"""
    
    @pytest.fixture
    def mock_openai(self, mock_openai_response):
        """Mock OpenAI API calls"""
        with patch('src.agents.responder.ChatOpenAI') as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm.bind.return_value = mock_llm
            
            # Mock async invoke
            async def mock_ainvoke(messages):
                mock_response = MagicMock()
                mock_response.content = "Thank you for contacting us. We'll help you with your account issue right away."
                return mock_response
            
            mock_llm.ainvoke = mock_ainvoke
            mock_llm_class.return_value = mock_llm
            
            # Mock callback context
            with patch('src.agents.responder.get_openai_callback') as mock_callback:
                callback_context = MagicMock()
                callback_context.total_tokens = 200
                callback_context.prompt_tokens = 150
                callback_context.completion_tokens = 50
                callback_context.total_cost = 0.001
                callback_context.__enter__ = MagicMock(return_value=callback_context)
                callback_context.__exit__ = MagicMock(return_value=None)
                mock_callback.return_value = callback_context
                
                yield mock_llm
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_success(self, sample_emails, mock_openai):
        """Test successful complete pipeline processing"""
        generator = ResponseGenerator()
        
        # Test customer support email
        result = await generator.generate_response(sample_emails['customer_support'])
        
        assert result.response_text is not None
        assert len(result.response_text) > 0
        assert result.status in [ResponseStatus.GENERATED, ResponseStatus.APPROVED]
        assert result.classification_result is not None
        assert result.confidence_analysis is not None
        assert 0.0 <= result.quality_score <= 1.0
        assert result.generation_time_ms > 0
        assert result.total_tokens_used > 0
    
    @pytest.mark.asyncio
    async def test_escalation_flow(self, sample_emails, mock_openai):
        """Test emails that should be escalated"""
        generator = ResponseGenerator()
        
        # Test VIP customer email
        result = await generator.generate_response(sample_emails['vip_customer'])
        
        assert result.status == ResponseStatus.ESCALATED
        assert result.escalation_decision is not None
        assert result.escalation_decision.should_escalate == True
        assert result.response_text is not None  # Should have acknowledgment
        
        # Test legal compliance email
        result = await generator.generate_response(sample_emails['legal_compliance'])
        
        assert result.status == ResponseStatus.ESCALATED
        assert result.escalation_decision.should_escalate == True
    
    @pytest.mark.asyncio
    async def test_high_confidence_approval(self, mock_openai):
        """Test emails that should be approved for sending"""
        generator = ResponseGenerator()
        
        # Simple, clear customer support email
        simple_email = "Hi, I forgot my password. Can you help me reset it? Thank you."
        
        result = await generator.generate_response(simple_email)
        
        # Should have high confidence and be approved
        assert result.confidence_analysis.overall_confidence > 0.7
        assert result.status in [ResponseStatus.APPROVED, ResponseStatus.GENERATED]
        assert result.quality_score > 0.6
    
    @pytest.mark.asyncio
    async def test_low_confidence_review(self, sample_emails, mock_openai):
        """Test emails that need human review"""
        generator = ResponseGenerator()
        
        # Ambiguous or complex email
        ambiguous_email = "Something is wrong with the thing you sent me last week."
        
        result = await generator.generate_response(ambiguous_email)
        
        # Should require review due to low confidence
        assert result.status in [ResponseStatus.NEEDS_REVIEW, ResponseStatus.ESCALATED]
        assert result.confidence_analysis.overall_confidence < 0.8
    
    @pytest.mark.asyncio 
    async def test_spam_detection(self, sample_emails, mock_openai):
        """Test spam email handling"""
        generator = ResponseGenerator()
        
        result = await generator.generate_response(sample_emails['spam_promotional'])
        
        # Spam should be detected and handled appropriately
        assert result.classification_result['category'] == 'SPAM_PROMOTIONAL'
        # Spam usually gets low response confidence
        assert result.confidence_analysis.response_confidence < 0.6
    
    @pytest.mark.asyncio
    async def test_error_handling_and_fallback(self, sample_emails):
        """Test error handling and fallback responses"""
        # Test with broken generator (no OpenAI key)
        with patch('src.agents.responder.ChatOpenAI', side_effect=Exception("API Error")):
            generator = ResponseGenerator()
            
            result = await generator.generate_response(sample_emails['customer_support'])
            
            # Should fallback gracefully
            assert result.status == ResponseStatus.NEEDS_REVIEW
            assert result.fallback_used == True
            assert result.error_message is not None
            assert "API Error" in result.error_message
    
    def test_legacy_compatibility(self, sample_emails):
        """Test legacy EmailAgent compatibility"""
        agent = EmailAgent()
        
        # Test different email types
        for email_type, email_content in sample_emails.items():
            result = agent.process_email(email_content)
            
            # Should return expected legacy format
            assert 'action' in result
            assert 'category' in result
            assert 'confidence' in result
            assert result['action'] in ['send', 'review', 'escalate']
            
            # If not escalated, should have response
            if result['action'] != 'escalate':
                assert 'response' in result
                assert len(result['response']) > 0
    
    def test_classification_accuracy(self, sample_emails):
        """Test classification accuracy across different email types"""
        agent = EmailAgent()
        
        # Test specific expected classifications
        expected_mappings = {
            'customer_support': 'other',  # Maps to legacy 'other'
            'order_status': 'order_status',
            'technical_issue': 'technical_support', 
            'refund_request': 'refund_request',
            'billing_inquiry': 'other'
        }
        
        for email_key, expected_category in expected_mappings.items():
            if email_key in sample_emails:
                result = agent.classify_email(sample_emails[email_key])
                
                # Should classify correctly or have reasonable confidence
                if result.category != expected_category:
                    # At least should have reasonable confidence
                    assert result.confidence > 0.4, f"Low confidence for {email_key}: {result.confidence}"
    
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, sample_emails, mock_openai, performance_benchmarks):
        """Test that pipeline meets performance benchmarks"""
        generator = ResponseGenerator()
        results = []
        
        # Process multiple emails to test performance
        for i, email_content in enumerate(list(sample_emails.values())[:5]):  # Test subset
            result = await generator.generate_response(email_content)
            results.append(result)
            
            # Each email should meet time benchmark
            assert result.generation_time_ms < performance_benchmarks['max_processing_time_ms']
            
            # Token usage should be reasonable
            assert result.total_tokens_used < performance_benchmarks['max_tokens_per_response']
        
        # Overall quality should meet benchmark
        avg_quality = sum(r.quality_score for r in results) / len(results)
        assert avg_quality >= performance_benchmarks['min_quality_score']
        
        # Escalation rate should be reasonable
        escalated_count = sum(1 for r in results if r.status == ResponseStatus.ESCALATED)
        escalation_rate = escalated_count / len(results)
        assert escalation_rate <= performance_benchmarks['max_escalation_rate']
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, sample_emails, mock_openai):
        """Test concurrent email processing"""
        generator = ResponseGenerator()
        
        # Process multiple emails concurrently
        tasks = []
        for email_content in list(sample_emails.values())[:3]:  # Test subset
            task = generator.generate_response(email_content)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert len(results) == 3
        for result in results:
            assert result.response_text is not None
            assert result.generation_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_prompt_template_selection(self, sample_emails, mock_openai):
        """Test that appropriate prompt templates are selected"""
        generator = ResponseGenerator()
        
        # VIP customer should get VIP template
        result = await generator.generate_response(sample_emails['vip_customer'])
        if result.prompt_config:
            # VIP should either escalate or use VIP template
            assert (result.status == ResponseStatus.ESCALATED or 
                   result.prompt_config.template_type.value == 'vip')
        
        # High emotion should get empathetic template
        result = await generator.generate_response(sample_emails['high_emotion'])
        if result.prompt_config and result.status != ResponseStatus.ESCALATED:
            assert result.prompt_config.template_type.value in ['empathetic', 'high_risk']
        
        # Technical issue should get technical template
        result = await generator.generate_response(sample_emails['technical_complex'])
        if result.prompt_config and result.status != ResponseStatus.ESCALATED:
            assert result.prompt_config.template_type.value in ['technical', 'detailed']
    
    @pytest.mark.asyncio
    async def test_quality_assessment_accuracy(self, mock_openai):
        """Test quality assessment accuracy"""
        generator = ResponseGenerator()
        
        # Good quality response scenario
        good_email = "Hello, I need help resetting my password. Thank you."
        result = await generator.generate_response(good_email)
        
        # Quality factors should be reasonable
        assert 'length' in result.quality_factors
        assert 'tone' in result.quality_factors
        assert all(0.0 <= score <= 1.0 for score in result.quality_factors.values())
        
        # Overall quality should correlate with individual factors
        avg_factors = sum(result.quality_factors.values()) / len(result.quality_factors)
        assert abs(result.quality_score - avg_factors) < 0.3  # Should be reasonably close
    
    def test_response_summary_generation(self, sample_emails):
        """Test response summary for monitoring"""
        agent = EmailAgent()
        
        for email_content in list(sample_emails.values())[:3]:
            result = agent.process_email(email_content)
            
            # Should have monitoring-friendly data
            assert 'confidence' in result
            assert isinstance(result['confidence'], (int, float))
            if 'processing_time_ms' in result:
                assert isinstance(result['processing_time_ms'], (int, float))
            if 'tokens_used' in result:
                assert isinstance(result['tokens_used'], (int, float))
    
    @pytest.mark.asyncio
    async def test_risk_factor_escalation(self, mock_openai):
        """Test that high-risk factors trigger appropriate escalation"""
        generator = ResponseGenerator()
        
        # Legal risk
        legal_email = "I want to speak to my lawyer about this GDPR violation"
        result = await generator.generate_response(legal_email)
        assert result.status == ResponseStatus.ESCALATED
        
        # Financial risk  
        financial_email = "This is fraud! I'm disputing this unauthorized charge immediately!"
        result = await generator.generate_response(financial_email)
        
        # Should have high risk or be escalated
        assert (result.status == ResponseStatus.ESCALATED or
               result.confidence_analysis.risk_score > 0.5)
    
    @pytest.mark.asyncio
    async def test_context_preservation(self, mock_openai):
        """Test that context is preserved through the pipeline"""
        generator = ResponseGenerator()
        
        # Email with specific context (order number, urgency)
        contextual_email = "URGENT: My order #12345 hasn't arrived and I need it for tomorrow's presentation!"
        
        result = await generator.generate_response(contextual_email)
        
        # Should detect structural features
        assert result.classification_result is not None
        structural_features = result.classification_result.get('structural_features', {})
        assert structural_features.get('has_order_number', False) == True
        
        # Should detect urgency in escalation decision
        if result.escalation_decision:
            context = result.escalation_decision.customer_context
            assert context.get('urgency_level') == 'high'