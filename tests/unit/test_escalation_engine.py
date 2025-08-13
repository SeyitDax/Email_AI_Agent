"""
Unit tests for the EscalationEngine component.
Tests complex escalation decision logic and team assignment.
"""

import pytest
from unittest.mock import MagicMock
from src.agents.escalation_engine import (
    EscalationEngine, EscalationDecision, EscalationReason, 
    PriorityLevel, EscalationTeam
)
from src.agents.confidence_scorer import ConfidenceAnalysis, RiskFactor, ProcessingAction


class TestEscalationEngine:
    """Test suite for EscalationEngine"""
    
    def test_engine_initialization(self):
        """Test escalation engine initializes with correct configuration"""
        engine = EscalationEngine()
        
        assert 'confidence_threshold' in engine.escalation_thresholds
        assert len(engine.vip_indicators['titles']) > 0
        assert len(engine.legal_keywords) > 0
        assert len(engine.financial_keywords) > 0
    
    def test_customer_context_analysis(self):
        """Test customer context analysis and VIP detection"""
        engine = EscalationEngine()
        
        # VIP by title
        vip_email = "This is John Smith, CEO of our company. I need immediate help."
        context = engine._analyze_customer_context(vip_email, {})
        assert context['is_vip'] == True
        assert 'ceo' in str(context['vip_indicators']).lower()
        
        # VIP by company
        company_email = "I'm from Microsoft and we're having issues with the integration"
        context = engine._analyze_customer_context(company_email, {})
        assert context['is_vip'] == True
        
        # Regular customer
        regular_email = "I have a question about my account"
        context = engine._analyze_customer_context(regular_email, {})
        assert context['is_vip'] == False
        
        # Urgency detection
        urgent_email = "This is urgent! I need help immediately!"
        context = engine._analyze_customer_context(urgent_email, {})
        assert context['urgency_level'] == 'high'
    
    def test_escalation_reasons_determination(self, confidence_analysis_result):
        """Test determination of escalation reasons"""
        engine = EscalationEngine()
        
        # Low confidence escalation
        low_confidence = ConfidenceAnalysis(
            overall_confidence=0.5, classification_confidence=0.5,
            response_confidence=0.5, contextual_confidence=0.5,
            score_separation=0.1, category_certainty=0.5,
            risk_factors=[], risk_score=0.1,
            recommended_action=ProcessingAction.ESCALATE,
            confidence_threshold_met={}, reasoning=[]
        )
        
        reasons = engine._determine_escalation_reasons(
            "I need help", {}, low_confidence, {'is_vip': False}
        )
        assert EscalationReason.LOW_CONFIDENCE in reasons
        
        # VIP customer escalation
        reasons = engine._determine_escalation_reasons(
            "I need help", {}, confidence_analysis_result, {'is_vip': True}
        )
        assert EscalationReason.VIP_CUSTOMER in reasons
        
        # Legal compliance escalation
        legal_email = "I need all my data under GDPR regulations"
        reasons = engine._determine_escalation_reasons(
            legal_email, {}, confidence_analysis_result, {'is_vip': False}
        )
        assert EscalationReason.LEGAL_COMPLIANCE in reasons
        
        # Financial impact escalation
        financial_email = "I want a refund for this unauthorized charge"
        reasons = engine._determine_escalation_reasons(
            financial_email, {}, confidence_analysis_result, {'is_vip': False}
        )
        assert EscalationReason.FINANCIAL_IMPACT in reasons
    
    def test_escalation_score_calculation(self, confidence_analysis_result):
        """Test escalation score calculation"""
        engine = EscalationEngine()
        
        # Low risk scenario
        low_score = engine._calculate_escalation_score(
            confidence_analysis_result, 0.0, 0.2, [], {'is_vip': False}
        )
        assert low_score < 0.5
        
        # High risk scenario
        high_risk_confidence = ConfidenceAnalysis(
            overall_confidence=0.3, classification_confidence=0.3,
            response_confidence=0.3, contextual_confidence=0.3,
            score_separation=0.05, category_certainty=0.3,
            risk_factors=[RiskFactor.LEGAL_TERMS, RiskFactor.HIGH_EMOTION],
            risk_score=0.8, recommended_action=ProcessingAction.ESCALATE,
            confidence_threshold_met={}, reasoning=[]
        )
        
        high_score = engine._calculate_escalation_score(
            high_risk_confidence, -0.9, 0.8, 
            [EscalationReason.LEGAL_COMPLIANCE, EscalationReason.EMOTIONAL_INTENSITY],
            {'is_vip': True}
        )
        assert high_score > 0.7
    
    def test_escalation_decision_logic(self, confidence_analysis_result):
        """Test final escalation decision logic"""
        engine = EscalationEngine()
        
        # Should not escalate - high confidence, low risk
        should_escalate = engine._should_escalate(
            confidence_analysis_result, [], 0.3
        )
        assert should_escalate == False
        
        # Should escalate - critical reasons
        should_escalate = engine._should_escalate(
            confidence_analysis_result, [EscalationReason.LEGAL_COMPLIANCE], 0.3
        )
        assert should_escalate == True
        
        should_escalate = engine._should_escalate(
            confidence_analysis_result, [EscalationReason.BUSINESS_CRITICAL], 0.3
        )
        assert should_escalate == True
        
        # Should escalate - high escalation score
        should_escalate = engine._should_escalate(
            confidence_analysis_result, [EscalationReason.FINANCIAL_IMPACT], 0.8
        )
        assert should_escalate == True
    
    def test_priority_level_determination(self):
        """Test priority level determination"""
        engine = EscalationEngine()
        
        # Critical priority
        priority = engine._determine_priority_level(
            [EscalationReason.BUSINESS_CRITICAL], 0.0, {}, 0.9
        )
        assert priority == PriorityLevel.CRITICAL
        
        priority = engine._determine_priority_level(
            [EscalationReason.LEGAL_COMPLIANCE], 0.0, {}, 0.5
        )
        assert priority == PriorityLevel.CRITICAL
        
        # High priority for VIP
        priority = engine._determine_priority_level(
            [], 0.0, {'is_vip': True}, 0.6
        )
        assert priority == PriorityLevel.HIGH
        
        # High priority for extreme emotion
        priority = engine._determine_priority_level(
            [EscalationReason.EMOTIONAL_INTENSITY], -0.95, {}, 0.7
        )
        assert priority == PriorityLevel.HIGH
        
        # Medium priority
        priority = engine._determine_priority_level(
            [EscalationReason.TECHNICAL_COMPLEXITY], 0.0, {}, 0.65
        )
        assert priority == PriorityLevel.MEDIUM
    
    def test_team_assignment(self):
        """Test team assignment logic"""
        engine = EscalationEngine()
        
        # VIP customers go to concierge
        team = engine._assign_team('CUSTOMER_SUPPORT', [], {'is_vip': True})
        assert team == EscalationTeam.VIP_CONCIERGE
        
        # Legal issues go to legal
        team = engine._assign_team('LEGAL_COMPLIANCE', [EscalationReason.LEGAL_COMPLIANCE], {})
        assert team == EscalationTeam.LEGAL_COMPLIANCE
        
        # Technical issues go to technical team
        team = engine._assign_team('TECHNICAL_ISSUE', [EscalationReason.TECHNICAL_COMPLEXITY], {})
        assert team == EscalationTeam.TECHNICAL_TEAM
        
        # Billing issues go to specialists
        team = engine._assign_team('BILLING_INQUIRY', [EscalationReason.FINANCIAL_IMPACT], {})
        assert team == EscalationTeam.BILLING_SPECIALISTS
        
        # Business critical goes to management
        team = engine._assign_team('CUSTOMER_SUPPORT', [EscalationReason.BUSINESS_CRITICAL], {})
        assert team == EscalationTeam.MANAGEMENT
        
        # Press media goes to PR
        team = engine._assign_team('PRESS_MEDIA', [], {})
        assert team == EscalationTeam.PR_COMMUNICATIONS
        
        # Default to senior support
        team = engine._assign_team('CUSTOMER_SUPPORT', [], {})
        assert team == EscalationTeam.SENIOR_SUPPORT
    
    def test_complexity_estimation(self):
        """Test complexity estimation"""
        engine = EscalationEngine()
        
        # High complexity category
        complexity = engine._estimate_complexity(
            'LEGAL_COMPLIANCE', [EscalationReason.LEGAL_COMPLIANCE], 0.5, {}
        )
        assert complexity > 0.8
        
        # Low complexity category
        complexity = engine._estimate_complexity(
            'CUSTOMER_SUPPORT', [], 0.2, {}
        )
        assert complexity < 0.5
        
        # VIP adds complexity
        vip_complexity = engine._estimate_complexity(
            'CUSTOMER_SUPPORT', [], 0.2, {'is_vip': True}
        )
        regular_complexity = engine._estimate_complexity(
            'CUSTOMER_SUPPORT', [], 0.2, {'is_vip': False}
        )
        assert vip_complexity > regular_complexity
    
    def test_sla_requirements(self):
        """Test SLA requirement determination"""
        engine = EscalationEngine()
        
        # Critical priority gets tight SLA
        response_sla, resolution_sla = engine._determine_sla_requirements(
            PriorityLevel.CRITICAL, [], {}
        )
        assert response_sla == 1  # 1 hour
        assert resolution_sla == 4  # 4 hours
        
        # VIP gets enhanced SLA
        vip_response, vip_resolution = engine._determine_sla_requirements(
            PriorityLevel.MEDIUM, [], {'is_vip': True}
        )
        regular_response, regular_resolution = engine._determine_sla_requirements(
            PriorityLevel.MEDIUM, [], {'is_vip': False}
        )
        assert vip_response <= regular_response
        assert vip_resolution <= regular_resolution
        
        # Business critical gets 1 hour response
        bc_response, bc_resolution = engine._determine_sla_requirements(
            PriorityLevel.LOW, [EscalationReason.BUSINESS_CRITICAL], {}
        )
        assert bc_response == 1
    
    def test_business_impact_assessment(self):
        """Test business impact assessment"""
        engine = EscalationEngine()
        
        # High impact scenarios
        impact = engine._assess_business_impact(
            [EscalationReason.BUSINESS_CRITICAL], {}, 0.0
        )
        assert impact.startswith('HIGH')
        
        impact = engine._assess_business_impact(
            [], {'is_vip': True}, 0.0
        )
        assert impact.startswith('HIGH')
        
        impact = engine._assess_business_impact(
            [EscalationReason.LEGAL_COMPLIANCE], {}, 0.0
        )
        assert impact.startswith('HIGH')
        
        # Medium impact
        impact = engine._assess_business_impact(
            [EscalationReason.FINANCIAL_IMPACT], {}, -0.8
        )
        assert impact.startswith('MEDIUM')
        
        # Low impact
        impact = engine._assess_business_impact(
            [], {}, 0.0
        )
        assert impact.startswith('LOW')
    
    def test_special_instructions_generation(self):
        """Test special handling instructions generation"""
        engine = EscalationEngine()
        
        # VIP instructions
        instructions = engine._generate_special_instructions(
            [], {'is_vip': True}, 'CUSTOMER_SUPPORT'
        )
        assert any('VIP' in instruction for instruction in instructions)
        
        # Legal instructions
        instructions = engine._generate_special_instructions(
            [EscalationReason.LEGAL_COMPLIANCE], {}, 'LEGAL_COMPLIANCE'
        )
        assert any('LEGAL' in instruction for instruction in instructions)
        
        # Technical instructions
        instructions = engine._generate_special_instructions(
            [EscalationReason.TECHNICAL_COMPLEXITY], {}, 'TECHNICAL_ISSUE'
        )
        assert any('TECHNICAL' in instruction for instruction in instructions)
    
    def test_complete_escalation_decision(self, confidence_analysis_result):
        """Test complete escalation decision process"""
        engine = EscalationEngine()
        
        # Regular email - should not escalate
        regular_decision = engine.make_escalation_decision(
            "I have a question about my account",
            {'category': 'CUSTOMER_SUPPORT', 'sentiment_score': 0.0, 'complexity_score': 0.2},
            confidence_analysis_result
        )
        
        assert isinstance(regular_decision, EscalationDecision)
        assert regular_decision.should_escalate == False
        assert regular_decision.priority_level in list(PriorityLevel)
        assert regular_decision.assigned_team in list(EscalationTeam)
        assert isinstance(regular_decision.reasoning, list)
        
        # VIP email - should escalate
        vip_decision = engine.make_escalation_decision(
            "This is CEO John Smith. I need immediate help with our enterprise account.",
            {'category': 'CUSTOMER_SUPPORT', 'sentiment_score': 0.0, 'complexity_score': 0.2},
            confidence_analysis_result
        )
        
        assert vip_decision.should_escalate == True
        assert EscalationReason.VIP_CUSTOMER in vip_decision.escalation_reasons
        assert vip_decision.assigned_team == EscalationTeam.VIP_CONCIERGE
    
    def test_reasoning_generation(self, confidence_analysis_result):
        """Test escalation reasoning generation"""
        engine = EscalationEngine()
        
        # Escalated case
        reasoning = engine._generate_escalation_reasoning(
            True, [EscalationReason.VIP_CUSTOMER, EscalationReason.HIGH_RISK_FACTORS],
            0.8, confidence_analysis_result, PriorityLevel.HIGH, EscalationTeam.VIP_CONCIERGE
        )
        
        assert isinstance(reasoning, list)
        assert len(reasoning) > 0
        assert any('ESCALATION REQUIRED' in r for r in reasoning)
        assert any('VIP' in r for r in reasoning)
        assert any('HIGH' in r for r in reasoning)
        
        # Non-escalated case
        reasoning = engine._generate_escalation_reasoning(
            False, [], 0.3, confidence_analysis_result, 
            PriorityLevel.ROUTINE, EscalationTeam.SENIOR_SUPPORT
        )
        
        assert any('NO ESCALATION' in r for r in reasoning)
    
    @pytest.mark.parametrize("escalation_reason,expected_priority", [
        (EscalationReason.BUSINESS_CRITICAL, PriorityLevel.CRITICAL),
        (EscalationReason.LEGAL_COMPLIANCE, PriorityLevel.CRITICAL),
        (EscalationReason.VIP_CUSTOMER, PriorityLevel.HIGH),
        (EscalationReason.TECHNICAL_COMPLEXITY, PriorityLevel.MEDIUM),
    ])
    def test_priority_mapping(self, escalation_reason, expected_priority):
        """Test escalation reason to priority mapping"""
        engine = EscalationEngine()
        
        priority = engine._determine_priority_level(
            [escalation_reason], 0.0, {}, 0.6
        )
        
        # Should be at least the expected priority or higher
        assert priority.value <= expected_priority.value