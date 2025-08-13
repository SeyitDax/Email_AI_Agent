"""
Unit tests for the ConfidenceScorer component.
Tests multi-dimensional confidence analysis and risk factor detection.
"""

import pytest
from unittest.mock import MagicMock
from src.agents.confidence_scorer import ConfidenceScorer, ConfidenceAnalysis, RiskFactor, ProcessingAction


class TestConfidenceScorer:
    """Test suite for ConfidenceScorer"""
    
    def test_scorer_initialization(self):
        """Test confidence scorer initializes with correct thresholds"""
        scorer = ConfidenceScorer()
        
        assert 'send' in scorer.thresholds
        assert 'review' in scorer.thresholds
        assert 'escalate' in scorer.thresholds
        assert scorer.thresholds['send'] > scorer.thresholds['review']
        assert scorer.thresholds['review'] > scorer.thresholds['escalate']
    
    def test_confidence_analysis(self, classification_results):
        """Test comprehensive confidence analysis"""
        scorer = ConfidenceScorer()
        
        result = scorer.analyze_confidence(
            classification_results, 
            "Hi, I need help with my account login issue."
        )
        
        assert isinstance(result, ConfidenceAnalysis)
        assert 0.0 <= result.overall_confidence <= 1.0
        assert 0.0 <= result.classification_confidence <= 1.0
        assert 0.0 <= result.response_confidence <= 1.0
        assert 0.0 <= result.contextual_confidence <= 1.0
        assert isinstance(result.risk_factors, list)
        assert isinstance(result.reasoning, list)
    
    def test_score_separation_calculation(self):
        """Test score separation calculation"""
        scorer = ConfidenceScorer()
        
        # Clear separation
        clear_scores = {'CUSTOMER_SUPPORT': 0.8, 'TECHNICAL_ISSUE': 0.2}
        separation = scorer._calculate_score_separation(clear_scores)
        assert separation == 0.6
        
        # Close scores
        close_scores = {'CUSTOMER_SUPPORT': 0.5, 'TECHNICAL_ISSUE': 0.45}
        separation = scorer._calculate_score_separation(close_scores)
        assert separation == 0.05
        
        # Single category
        single_score = {'CUSTOMER_SUPPORT': 1.0}
        separation = scorer._calculate_score_separation(single_score)
        assert separation == 1.0
    
    def test_risk_factor_detection(self):
        """Test risk factor identification"""
        scorer = ConfidenceScorer()
        
        # Legal terms
        legal_email = "I want to speak to my lawyer about this breach of contract"
        risk_factors = scorer._identify_risk_factors(
            legal_email, 0.0, 0.5, {}
        )
        assert RiskFactor.LEGAL_TERMS in risk_factors
        
        # High emotion
        emotional_email = "This is absolutely terrible and unacceptable!"
        risk_factors = scorer._identify_risk_factors(
            emotional_email, -0.9, 0.5, {}
        )
        assert RiskFactor.HIGH_EMOTION in risk_factors
        
        # Financial impact
        financial_email = "I need an immediate refund for this unauthorized charge"
        risk_factors = scorer._identify_risk_factors(
            financial_email, 0.0, 0.5, {}
        )
        assert RiskFactor.FINANCIAL_IMPACT in risk_factors
    
    def test_classification_confidence_calculation(self):
        """Test classification confidence calculation"""
        scorer = ConfidenceScorer()
        
        # High confidence scenario
        high_scores = {'CUSTOMER_SUPPORT': 0.9, 'OTHER': 0.1}
        high_separation = 0.8
        low_complexity = 0.1
        
        confidence = scorer._calculate_classification_confidence(
            high_scores, high_separation, low_complexity
        )
        assert confidence > 0.8
        
        # Low confidence scenario
        low_scores = {'CUSTOMER_SUPPORT': 0.6, 'TECHNICAL_ISSUE': 0.4}
        low_separation = 0.2
        high_complexity = 0.8
        
        confidence = scorer._calculate_classification_confidence(
            low_scores, low_separation, high_complexity
        )
        assert confidence < 0.7
    
    def test_response_confidence_by_category(self):
        """Test response confidence varies by category"""
        scorer = ConfidenceScorer()
        
        # Easy categories should have high response confidence
        easy_confidence = scorer._calculate_response_confidence(
            'COMPLIMENT_POSITIVE', {}, {}
        )
        assert easy_confidence > 0.8
        
        # Hard categories should have lower response confidence
        hard_confidence = scorer._calculate_response_confidence(
            'LEGAL_COMPLIANCE', {}, {}
        )
        assert hard_confidence < 0.5
        
        # Structural features should boost confidence
        with_features = scorer._calculate_response_confidence(
            'SALES_ORDER', {}, {'has_order_number': True, 'has_phone_number': True}
        )
        without_features = scorer._calculate_response_confidence(
            'SALES_ORDER', {}, {}
        )
        assert with_features > without_features
    
    def test_contextual_confidence_calculation(self):
        """Test contextual confidence based on email properties"""
        scorer = ConfidenceScorer()
        
        # Well-structured email
        structured_email = "Dear Support, I have a question about my account. Thank you, John"
        structured_features = {'has_greeting': True, 'has_closing': True, 'has_question_mark': True}
        
        structured_confidence = scorer._calculate_contextual_confidence(
            structured_email, structured_features, 0.0
        )
        
        # Poorly structured email
        poor_email = "help me now!!!"
        poor_features = {}
        
        poor_confidence = scorer._calculate_contextual_confidence(
            poor_email, poor_features, -0.8
        )
        
        assert structured_confidence > poor_confidence
    
    def test_risk_score_calculation(self):
        """Test overall risk score calculation"""
        scorer = ConfidenceScorer()
        
        # No risk factors
        no_risk = scorer._calculate_risk_score([], 0.0)
        assert no_risk == 0.0
        
        # Multiple risk factors
        high_risk_factors = [
            RiskFactor.LEGAL_TERMS,
            RiskFactor.HIGH_EMOTION, 
            RiskFactor.FINANCIAL_IMPACT
        ]
        high_risk = scorer._calculate_risk_score(high_risk_factors, -0.9)
        assert high_risk > 0.5
        
        # Risk score should be clamped at 1.0
        extreme_risk = scorer._calculate_risk_score(list(RiskFactor), -1.0)
        assert extreme_risk <= 1.0
    
    def test_overall_confidence_calculation(self):
        """Test overall confidence combines all factors correctly"""
        scorer = ConfidenceScorer()
        
        # High confidence scenario
        high_overall = scorer._calculate_overall_confidence(0.9, 0.9, 0.8, 0.1)
        assert high_overall > 0.8
        
        # Low confidence scenario
        low_overall = scorer._calculate_overall_confidence(0.5, 0.4, 0.6, 0.8)
        assert low_overall < 0.5
        
        # Should respect weights
        weighted = scorer._calculate_overall_confidence(0.9, 0.1, 0.1, 0.0)
        assert 0.3 <= weighted <= 0.4  # Should be around classification weight * 0.9
    
    def test_action_determination(self):
        """Test recommended action determination"""
        scorer = ConfidenceScorer()
        
        # High confidence, low risk -> SEND
        action = scorer._determine_action(0.9, [], 0.1)
        assert action == ProcessingAction.SEND
        
        # Medium confidence -> REVIEW
        action = scorer._determine_action(0.7, [], 0.2)
        assert action == ProcessingAction.REVIEW
        
        # Low confidence -> ESCALATE
        action = scorer._determine_action(0.4, [], 0.1)
        assert action == ProcessingAction.ESCALATE
        
        # High risk always escalates
        action = scorer._determine_action(0.9, [RiskFactor.LEGAL_TERMS], 0.1)
        assert action == ProcessingAction.ESCALATE
        
        action = scorer._determine_action(0.9, [RiskFactor.VIP_CUSTOMER], 0.1)
        assert action == ProcessingAction.ESCALATE
    
    def test_reasoning_generation(self):
        """Test human-readable reasoning generation"""
        scorer = ConfidenceScorer()
        
        reasoning = scorer._generate_reasoning(
            0.85, 0.8, 0.9, 0.7, 
            [RiskFactor.HIGH_EMOTION], 0.3, 
            ProcessingAction.REVIEW
        )
        
        assert isinstance(reasoning, list)
        assert len(reasoning) > 0
        
        # Should contain confidence assessment
        confidence_mentioned = any('confidence' in r.lower() for r in reasoning)
        assert confidence_mentioned
        
        # Should mention risk factors
        risk_mentioned = any('risk' in r.lower() or 'emotion' in r.lower() for r in reasoning)
        assert risk_mentioned
        
        # Should mention final action
        action_mentioned = any('review' in r.lower() for r in reasoning)
        assert action_mentioned
    
    @pytest.mark.parametrize("confidence_level,expected_action", [
        (0.95, ProcessingAction.SEND),
        (0.75, ProcessingAction.REVIEW), 
        (0.45, ProcessingAction.ESCALATE),
        (0.25, ProcessingAction.ESCALATE)
    ])
    def test_confidence_thresholds(self, confidence_level, expected_action):
        """Test confidence threshold boundaries"""
        scorer = ConfidenceScorer()
        
        action = scorer._determine_action(confidence_level, [], 0.1)
        assert action == expected_action
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        scorer = ConfidenceScorer()
        
        # Empty classification result
        result = scorer.analyze_confidence({}, "")
        assert isinstance(result, ConfidenceAnalysis)
        assert result.overall_confidence >= 0.0
        
        # Extreme values
        extreme_result = scorer.analyze_confidence({
            'category': 'OTHER_UNCATEGORIZED',
            'confidence': 0.0,
            'category_scores': {},
            'structural_features': {},
            'sentiment_score': -1.0,
            'complexity_score': 1.0
        }, "")
        assert isinstance(extreme_result, ConfidenceAnalysis)
    
    def test_threshold_met_calculation(self):
        """Test threshold met calculation"""
        scorer = ConfidenceScorer()
        
        # High confidence meets all thresholds
        result = scorer.analyze_confidence({
            'category': 'CUSTOMER_SUPPORT',
            'confidence': 0.9,
            'category_scores': {'CUSTOMER_SUPPORT': 0.9},
            'structural_features': {},
            'sentiment_score': 0.0,
            'complexity_score': 0.2
        }, "Hello, I need help")
        
        assert result.confidence_threshold_met['send'] == True
        assert result.confidence_threshold_met['review'] == True
        assert result.confidence_threshold_met['escalate'] == False
        
        # Low confidence meets only escalate
        low_result = scorer.analyze_confidence({
            'category': 'OTHER_UNCATEGORIZED',
            'confidence': 0.3,
            'category_scores': {'OTHER_UNCATEGORIZED': 0.3},
            'structural_features': {},
            'sentiment_score': 0.0,
            'complexity_score': 0.8
        }, "unclear message")
        
        assert low_result.confidence_threshold_met['send'] == False
        assert low_result.confidence_threshold_met['review'] == False
        assert low_result.confidence_threshold_met['escalate'] == True