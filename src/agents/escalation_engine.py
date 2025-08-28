"""
Intelligent Escalation Decision Engine - CORE LOGIC

Advanced escalation decision system that combines classification results, confidence analysis,
and complex business rules to determine when human intervention is required. Goes beyond
simple confidence thresholds to implement sophisticated escalation logic.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from enum import Enum
import re
from datetime import datetime, timedelta

from .confidence_scorer import ConfidenceAnalysis, RiskFactor, ProcessingAction

class EscalationReason(Enum):
    """Reasons why an email might be escalated"""
    LOW_CONFIDENCE = "low_confidence"
    HIGH_RISK_FACTORS = "high_risk_factors"
    VIP_CUSTOMER = "vip_customer"
    LEGAL_COMPLIANCE = "legal_compliance"
    FINANCIAL_IMPACT = "financial_impact"
    TECHNICAL_COMPLEXITY = "technical_complexity"
    EMOTIONAL_INTENSITY = "emotional_intensity"
    BUSINESS_CRITICAL = "business_critical"
    POLICY_EXCEPTION = "policy_exception"
    MULTI_CHANNEL = "multi_channel"

class PriorityLevel(Enum):
    """Priority levels for escalated emails"""
    CRITICAL = 1    # Immediate response required (< 1 hour)
    HIGH = 2        # Same-day response required (< 4 hours)  
    MEDIUM = 3      # Next business day (< 24 hours)
    LOW = 4         # Standard SLA (< 48 hours)
    ROUTINE = 5     # Normal queue processing (< 72 hours)

class EscalationTeam(Enum):
    """Teams that can handle escalated emails"""
    SENIOR_SUPPORT = "senior_support"
    TECHNICAL_TEAM = "technical_team"
    BILLING_SPECIALISTS = "billing_specialists"
    VIP_CONCIERGE = "vip_concierge"
    LEGAL_COMPLIANCE = "legal_compliance"
    MANAGEMENT = "management"
    PR_COMMUNICATIONS = "pr_communications"
    PARTNERSHIPS = "partnerships"

@dataclass
class EscalationDecision:
    """Comprehensive escalation decision result"""
    # Core decision
    should_escalate: bool
    escalation_reasons: List[EscalationReason]
    
    # Priority and routing
    priority_level: PriorityLevel
    assigned_team: EscalationTeam
    estimated_complexity: float  # 0.0 = simple, 1.0 = very complex
    
    # SLA requirements
    response_sla_hours: int
    resolution_sla_hours: int
    
    # Additional context
    customer_context: Dict[str, any]
    business_impact: str
    special_instructions: List[str]
    reasoning: List[str]
    
    # Metadata
    escalation_score: float  # Overall escalation likelihood
    decision_confidence: float  # Confidence in this escalation decision

class EscalationEngine:
    """
    Intelligent escalation decision engine that determines when and how
    to escalate emails based on multiple sophisticated factors.
    """
    
    def __init__(self):
        """Initialize escalation engine with business rules and thresholds"""
        
        # Base escalation thresholds
        self.escalation_thresholds = {
            'confidence_threshold': 0.65,  # Below this, consider escalation
            'risk_score_threshold': 0.4,   # Above this, consider escalation
            'sentiment_extreme_threshold': 0.8,  # Very positive/negative
            'complexity_threshold': 0.7,   # High complexity emails
        }
        
        # Priority scoring weights
        self.priority_weights = {
            'confidence_impact': 0.25,     # How confidence affects priority
            'risk_factor_impact': 0.30,    # How risk factors affect priority
            'sentiment_impact': 0.20,      # How sentiment affects priority
            'business_impact': 0.25        # How business impact affects priority
        }
        
        # VIP customer indicators (enhanced detection)
        self.vip_indicators = {
            'email_domains': [
                '@fortune500company.com', '@enterprise.com', '@corporation.com',
                '@government.gov', '@university.edu', '@hospital.org'
            ],
            'titles': [
                'ceo', 'president', 'vice president', 'vp', 'director',
                'manager', 'head of', 'chief', 'senior', 'lead'
            ],
            'companies': [
                'microsoft', 'google', 'apple', 'amazon', 'facebook',
                'tesla', 'netflix', 'spotify', 'uber', 'airbnb'
            ],
            'keywords': [
                'enterprise', 'corporate', 'business', 'organization',
                'department', 'team lead', 'decision maker'
            ]
        }
        
        # Legal/compliance keywords (enhanced)
        self.legal_keywords = [
            'gdpr', 'ccpa', 'privacy', 'data protection', 'compliance',
            'audit', 'regulation', 'attorney', 'lawyer', 'legal counsel',
            'subpoena', 'court order', 'litigation', 'lawsuit', 'damages',
            'breach', 'violation', 'regulatory', 'investigation'
        ]
        
        # Financial impact keywords
        self.financial_keywords = [
            'refund', 'chargeback', 'dispute', 'fraud', 'unauthorized',
            'billing error', 'overcharged', 'money back', 'compensation',
            'financial loss', 'revenue impact', 'budget', 'invoice',
            # NEW: Additional billing/payment specific patterns
            'payment failed', 'charge failed', 'money deducted', 'charged twice',
            'double charged', 'duplicate charge', 'payment went through',
            'billing name', 'receipt shows', 'billing mistake', 'wrong amount',
            'payment attempt', 'account charged', 'payment error'
        ]
        
        # Technical complexity indicators
        self.technical_keywords = [
            'api', 'integration', 'webhook', 'ssl', 'certificate',
            'database', 'server', 'configuration', 'deployment',
            'bug', 'error code', 'exception', 'malfunction'
        ]
        
        # Business critical indicators  
        self.business_critical_keywords = [
            'outage', 'down', 'not working', 'broken', 'critical issue',
            'urgent', 'emergency', 'production', 'live system',
            'customers affected', 'revenue loss', 'business impact'
        ]
    
    def make_escalation_decision(self, 
                               email_content: str,
                               classification_result: Dict,
                               confidence_analysis: ConfidenceAnalysis) -> EscalationDecision:
        """
        Make comprehensive escalation decision based on all available information
        
        Args:
            email_content: Original email text
            classification_result: Output from EmailClassifier
            confidence_analysis: Output from ConfidenceScorer
            
        Returns:
            EscalationDecision with complete escalation analysis
        """
        
        # 1. Gather all relevant data
        category = classification_result.get('category', 'OTHER_UNCATEGORIZED')
        sentiment_score = classification_result.get('sentiment_score', 0.0)
        complexity_score = classification_result.get('complexity_score', 0.0)
        structural_features = classification_result.get('structural_features', {})
        
        # 2. Analyze customer context
        customer_context = self._analyze_customer_context(email_content, structural_features)
        
        # 3. Determine escalation reasons
        escalation_reasons = self._determine_escalation_reasons(
            email_content, classification_result, confidence_analysis, customer_context
        )
        
        # 4. Calculate escalation score
        escalation_score = self._calculate_escalation_score(
            confidence_analysis, sentiment_score, complexity_score, 
            escalation_reasons, customer_context
        )
        
        # 5. Make final escalation decision
        should_escalate = self._should_escalate(
            confidence_analysis, escalation_reasons, escalation_score
        )
        
        # 6. Determine priority level
        priority_level = self._determine_priority_level(
            escalation_reasons, sentiment_score, customer_context, escalation_score
        )
        
        # 7. Assign appropriate team (with smart content analysis)
        assigned_team = self._assign_team(
            category, escalation_reasons, customer_context, email_content
        )
        
        # 8. Calculate complexity estimate
        estimated_complexity = self._estimate_complexity(
            category, escalation_reasons, complexity_score, customer_context
        )
        
        # 9. Set SLA requirements
        response_sla_hours, resolution_sla_hours = self._determine_sla_requirements(
            priority_level, escalation_reasons, customer_context
        )
        
        # 10. Generate business impact assessment
        business_impact = self._assess_business_impact(
            escalation_reasons, customer_context, sentiment_score
        )
        
        # 11. Create special instructions
        special_instructions = self._generate_special_instructions(
            escalation_reasons, customer_context, category
        )
        
        # 12. Generate reasoning
        reasoning = self._generate_escalation_reasoning(
            should_escalate, escalation_reasons, escalation_score,
            confidence_analysis, priority_level, assigned_team
        )
        
        return EscalationDecision(
            should_escalate=should_escalate,
            escalation_reasons=escalation_reasons,
            priority_level=priority_level,
            assigned_team=assigned_team,
            estimated_complexity=estimated_complexity,
            response_sla_hours=response_sla_hours,
            resolution_sla_hours=resolution_sla_hours,
            customer_context=customer_context,
            business_impact=business_impact,
            special_instructions=special_instructions,
            reasoning=reasoning,
            escalation_score=escalation_score,
            decision_confidence=confidence_analysis.overall_confidence
        )
    
    def _analyze_customer_context(self, email_content: str, structural_features: Dict) -> Dict[str, any]:
        """Analyze customer context for VIP detection and special handling"""
        
        email_lower = email_content.lower()
        context = {
            'is_vip': False,
            'vip_indicators': [],
            'customer_type': 'standard',
            'communication_history': 'unknown',
            'account_status': 'unknown'
        }
        
        # VIP detection - email domain
        email_match = re.search(r'[\w\.-]+@([\w\.-]+\.\w+)', email_content)
        if email_match:
            domain = email_match.group(1).lower()
            if any(vip_domain[1:] in domain for vip_domain in self.vip_indicators['email_domains']):
                context['is_vip'] = True
                context['vip_indicators'].append(f'VIP email domain: {domain}')
                context['customer_type'] = 'enterprise'
        
        # VIP detection - titles
        for title in self.vip_indicators['titles']:
            if title in email_lower:
                context['is_vip'] = True
                context['vip_indicators'].append(f'Executive title: {title}')
                context['customer_type'] = 'executive'
        
        # VIP detection - companies
        for company in self.vip_indicators['companies']:
            if company in email_lower:
                context['is_vip'] = True
                context['vip_indicators'].append(f'Major company: {company}')
                context['customer_type'] = 'enterprise'
        
        # Communication urgency indicators
        urgency_indicators = ['urgent', 'asap', 'immediately', 'emergency', 'critical']
        if any(indicator in email_lower for indicator in urgency_indicators):
            context['urgency_level'] = 'high'
        else:
            context['urgency_level'] = 'normal'
        
        # Multi-channel indicators  
        channel_indicators = ['called', 'phoned', 'spoke to', 'chatted', 'messaged']
        if any(indicator in email_lower for indicator in channel_indicators):
            context['multi_channel'] = True
        else:
            context['multi_channel'] = False
        
        return context
    
    def _determine_escalation_reasons(self, 
                                    email_content: str,
                                    classification_result: Dict,
                                    confidence_analysis: ConfidenceAnalysis,
                                    customer_context: Dict) -> List[EscalationReason]:
        """Determine all reasons why this email should be escalated"""
        
        reasons = []
        email_lower = email_content.lower()
        
        # 1. Low confidence escalation
        if confidence_analysis.overall_confidence < self.escalation_thresholds['confidence_threshold']:
            reasons.append(EscalationReason.LOW_CONFIDENCE)
        
        # 2. High-risk factors
        high_risk_factors = [RiskFactor.LEGAL_TERMS, RiskFactor.VIP_CUSTOMER]
        if any(factor in confidence_analysis.risk_factors for factor in high_risk_factors):
            reasons.append(EscalationReason.HIGH_RISK_FACTORS)
        
        # 3. VIP customer
        if customer_context['is_vip']:
            reasons.append(EscalationReason.VIP_CUSTOMER)
        
        # 4. Legal/compliance issues
        if any(keyword in email_lower for keyword in self.legal_keywords):
            reasons.append(EscalationReason.LEGAL_COMPLIANCE)
        
        # 5. Financial impact
        if any(keyword in email_lower for keyword in self.financial_keywords):
            reasons.append(EscalationReason.FINANCIAL_IMPACT)
        
        # 6. Technical complexity
        if any(keyword in email_lower for keyword in self.technical_keywords):
            reasons.append(EscalationReason.TECHNICAL_COMPLEXITY)
        
        # 7. Emotional intensity
        sentiment_score = classification_result.get('sentiment_score', 0.0)
        if abs(sentiment_score) > self.escalation_thresholds['sentiment_extreme_threshold']:
            reasons.append(EscalationReason.EMOTIONAL_INTENSITY)
        
        # 8. Business critical
        if any(keyword in email_lower for keyword in self.business_critical_keywords):
            reasons.append(EscalationReason.BUSINESS_CRITICAL)
        
        # 9. Multi-channel communication
        if customer_context.get('multi_channel', False):
            reasons.append(EscalationReason.MULTI_CHANNEL)
        
        # 10. Policy exception needs
        policy_exception_indicators = [
            'exception', 'special case', 'unusual situation', 'one-time',
            'policy override', 'bend the rules', 'make an exception'
        ]
        if any(indicator in email_lower for indicator in policy_exception_indicators):
            reasons.append(EscalationReason.POLICY_EXCEPTION)
        
        return reasons
    
    def _calculate_escalation_score(self, 
                                  confidence_analysis: ConfidenceAnalysis,
                                  sentiment_score: float,
                                  complexity_score: float,
                                  escalation_reasons: List[EscalationReason],
                                  customer_context: Dict) -> float:
        """Calculate overall escalation likelihood score"""
        
        score = 0.0
        
        # Base score from confidence (inverted - low confidence = high escalation score)
        confidence_factor = (1.0 - confidence_analysis.overall_confidence) * 0.4
        score += confidence_factor
        
        # Risk factor contribution
        risk_factor_count = len(confidence_analysis.risk_factors)
        risk_contribution = min(risk_factor_count * 0.15, 0.45)  # Max 45% from risk
        score += risk_contribution
        
        # Escalation reason contribution
        reason_weights = {
            EscalationReason.LEGAL_COMPLIANCE: 0.25,
            EscalationReason.VIP_CUSTOMER: 0.20,
            EscalationReason.BUSINESS_CRITICAL: 0.20,
            EscalationReason.HIGH_RISK_FACTORS: 0.15,
            EscalationReason.FINANCIAL_IMPACT: 0.15,
            EscalationReason.EMOTIONAL_INTENSITY: 0.10,
            EscalationReason.TECHNICAL_COMPLEXITY: 0.10,
            EscalationReason.MULTI_CHANNEL: 0.10,
            EscalationReason.POLICY_EXCEPTION: 0.08,
            EscalationReason.LOW_CONFIDENCE: 0.05
        }
        
        reason_contribution = sum(reason_weights.get(reason, 0.05) for reason in escalation_reasons)
        score += min(reason_contribution, 0.5)  # Max 50% from reasons
        
        # Sentiment extremes contribute
        sentiment_contribution = abs(sentiment_score) * 0.1
        score += sentiment_contribution
        
        # Complexity contribution
        complexity_contribution = complexity_score * 0.15
        score += complexity_contribution
        
        # VIP boost
        if customer_context.get('is_vip', False):
            score += 0.2
        
        return min(1.0, score)  # Cap at 1.0
    
    def _should_escalate(self, 
                        confidence_analysis: ConfidenceAnalysis,
                        escalation_reasons: List[EscalationReason],
                        escalation_score: float) -> bool:
        """Make final binary escalation decision"""
        
        # Always escalate for certain critical reasons
        critical_reasons = [
            EscalationReason.LEGAL_COMPLIANCE,
            EscalationReason.BUSINESS_CRITICAL,
            EscalationReason.VIP_CUSTOMER,
            EscalationReason.FINANCIAL_IMPACT  # NEW: Force escalation for billing/financial issues
        ]
        
        if any(reason in escalation_reasons for reason in critical_reasons):
            return True
        
        # Escalate if confidence recommends it
        if confidence_analysis.recommended_action == ProcessingAction.ESCALATE:
            return True
        
        # Escalate if escalation score is high enough
        if escalation_score > 0.6:
            return True
        
        # Multiple moderate reasons can trigger escalation
        if len(escalation_reasons) >= 3 and escalation_score > 0.4:
            return True
        
        return False
    
    def _determine_priority_level(self, 
                                escalation_reasons: List[EscalationReason],
                                sentiment_score: float,
                                customer_context: Dict,
                                escalation_score: float) -> PriorityLevel:
        """Determine the priority level for escalated emails"""
        
        # Critical priority triggers
        if EscalationReason.BUSINESS_CRITICAL in escalation_reasons:
            return PriorityLevel.CRITICAL
        
        if EscalationReason.LEGAL_COMPLIANCE in escalation_reasons:
            return PriorityLevel.CRITICAL
        
        # High priority triggers
        if customer_context.get('is_vip', False):
            return PriorityLevel.HIGH
        
        if abs(sentiment_score) > 0.9 and EscalationReason.EMOTIONAL_INTENSITY in escalation_reasons:
            return PriorityLevel.HIGH
        
        if EscalationReason.FINANCIAL_IMPACT in escalation_reasons and escalation_score > 0.7:
            return PriorityLevel.HIGH
        
        # Medium priority triggers
        if escalation_score > 0.6:
            return PriorityLevel.MEDIUM
        
        if len(escalation_reasons) >= 3:
            return PriorityLevel.MEDIUM
        
        # Low priority
        if escalation_score > 0.4:
            return PriorityLevel.LOW
        
        # Default routine
        return PriorityLevel.ROUTINE
    
    def _assign_team(self, 
                    category: str,
                    escalation_reasons: List[EscalationReason],
                    customer_context: Dict,
                    email_content: str = "") -> EscalationTeam:
        """Assign the most appropriate team for handling the escalation with smart content analysis"""
        
        # VIP customers go to concierge (highest priority)
        if customer_context.get('is_vip', False):
            return EscalationTeam.VIP_CONCIERGE
        
        # Legal/compliance issues (critical priority)
        if EscalationReason.LEGAL_COMPLIANCE in escalation_reasons:
            return EscalationTeam.LEGAL_COMPLIANCE
        
        # ENHANCED: Smart content-based team assignment
        if email_content:
            smart_team = self._smart_team_detection(email_content, category, escalation_reasons)
            if smart_team:
                return smart_team
        
        # Business critical or management escalation (high priority)
        if EscalationReason.BUSINESS_CRITICAL in escalation_reasons:
            return EscalationTeam.MANAGEMENT
        
        # Technical issues (standard category-based assignment)
        if (EscalationReason.TECHNICAL_COMPLEXITY in escalation_reasons or 
            category == 'TECHNICAL_ISSUE'):
            return EscalationTeam.TECHNICAL_TEAM
        
        # Billing and financial (standard category-based assignment)
        if (EscalationReason.FINANCIAL_IMPACT in escalation_reasons or
            category in ['BILLING_INQUIRY', 'REFUND_REQUEST']):
            return EscalationTeam.BILLING_SPECIALISTS
        
        # Press and media
        if category == 'PRESS_MEDIA':
            return EscalationTeam.PR_COMMUNICATIONS
        
        # Partnership inquiries
        if category == 'PARTNERSHIP_BUSINESS':
            return EscalationTeam.PARTNERSHIPS
        
        # Default to senior support
        return EscalationTeam.SENIOR_SUPPORT
    
    def _smart_team_detection(self, email_content: str, category: str, 
                            escalation_reasons: List[EscalationReason]) -> Optional[EscalationTeam]:
        """
        Smart content-based team detection with corrected team assignments:
        - Senior Team: Most serious issues (security, system failures, critical software problems)
        - Technical Team: General technical support (login, tracking, passwords, how-to)  
        - Management: Business-wide operational impact
        - Billing Specialists: Financial/payment issues
        """
        
        email_lower = email_content.lower().strip()
        
        # SENIOR TEAM - Most Serious Issues (Highest Priority) 
        senior_team_patterns = {
            'security_issues': [
                'hacking', 'hacked', 'security breach', 'unauthorized access', 'data breach',
                'account compromised', 'suspicious activity', 'security concern', 'fraud alert',
                'identity theft', 'unauthorized login', 'security vulnerability', 
                'suspicious account activity', 'unknown locations', 'investigate immediately'
            ],
            'critical_system_failures': [
                'system failure', 'database down', 'critical error', 'system crash',
                'complete system down', 'major system issue', 'infrastructure failure',
                'server crash', 'database corruption', 'system malfunction'
            ],
            'major_software_problems': [
                'data loss', 'data corruption', 'software crash', 'application failure',
                'major bug', 'critical software error', 'system not responding',
                'software malfunction', 'critical functionality broken', 'system freeze'
            ],
            'high_severity_issues': [
                'critical issue', 'urgent system problem', 'major malfunction',
                'system emergency', 'critical failure', 'serious software issue'
            ]
        }
        
        senior_score = 0.0
        for pattern_type, patterns in senior_team_patterns.items():
            matches = sum(1 for pattern in patterns if pattern in email_lower)
            if matches > 0:
                senior_score += matches * 0.6  # Very high weight for senior team issues
                
        if senior_score > 0.4:  # Lower threshold for senior team
            return EscalationTeam.SENIOR_SUPPORT  # Using SENIOR_SUPPORT as Senior Team
        
        # MANAGEMENT - Business-Wide Operational Impact
        management_patterns = {
            'business_outages': [
                'website down all morning', 'site offline', 'service unavailable for hours',
                'trying all day', 'can\'t place order for hours', 'system down all day',
                'website has been down', 'service disruption', 'widespread outage'
            ],
            'revenue_impact': [
                'business critical', 'revenue impact', 'affecting business', 'lost sales',
                'business operations affected', 'operational impact', 'business disruption'
            ],
            'escalation_requests': [
                'escalate to management', 'speak to manager', 'need supervisor',
                'management attention', 'executive level', 'senior management'
            ]
        }
        
        management_score = 0.0
        for pattern_type, patterns in management_patterns.items():
            matches = sum(1 for pattern in patterns if pattern in email_lower)
            if matches > 0:
                management_score += matches * 0.4
                
        if management_score > 0.6:
            return EscalationTeam.MANAGEMENT
        
        # BILLING SPECIALISTS - Financial/Payment Issues
        billing_patterns = {
            'payment_failures': [
                'payment failed', 'charge failed', 'payment error', 'billing error',
                'payment attempt showed error', 'payment went through but', 'money deducted'
            ],
            'billing_discrepancies': [
                'charged twice', 'double charged', 'duplicate charge', 'wrong amount',
                'billing name wrong', 'receipt shows wrong', 'billing mistake',
                'wrong name on receipt', 'billing address error'
            ],
            'refund_requests': [
                'refund request', 'want refund', 'need refund', 'money back',
                'return money', 'charge back', 'dispute charge', 'cancel and refund'
            ],
            'subscription_issues': [
                'subscription renewed without', 'auto renewal', 'subscription charge',
                'cancel subscription', 'billing cycle', 'recurring charge'
            ]
        }
        
        billing_score = 0.0
        for pattern_type, patterns in billing_patterns.items():
            matches = sum(1 for pattern in patterns if pattern in email_lower)
            if matches > 0:
                billing_score += matches * 0.4
                
        if billing_score > 0.5:
            return EscalationTeam.BILLING_SPECIALISTS
        
        # TECHNICAL TEAM - General Technical Support (Routine Issues)
        technical_patterns = {
            'login_issues': [
                'login issue', 'can\'t log in', 'login problem', 'account access',
                'password reset', 'forgot password', 'password problem', 'login error',
                'account locked', 'login not working', 'sign in problem'
            ],
            'order_tracking': [
                'order status', 'tracking link', 'where is my order', 'order tracking',
                'shipment status', 'delivery status', 'package status', 'order inquiry',
                'tracking not working', 'track my order', 'order delivery'
            ],
            'account_management': [
                'update profile', 'change account settings', 'account information',
                'profile update', 'account settings', 'personal information',
                'contact information', 'account details'
            ],
            'order_management': [
                'cancel order', 'order cancellation', 'modify order', 'change order',
                'order changes', 'cancel my order', 'stop order', 'order modification',
                'cancel it before it ships', 'request for order cancellation', 
                'would like to cancel', 'want to cancel order'
            ],
            'warranty_and_replacements': [
                'replacement part', 'warranty replacement', 'under warranty',
                'replacement part request', 'lid broke', 'part broke', 'send me a replacement',
                'warranty claim', 'defective part', 'broken part'
            ],
            'product_support': [
                'how to use', 'product question', 'feature question', 'how do i',
                'product support', 'using the product', 'product help', 'instructions'
            ],
            'general_technical': [
                'technical question', 'how does this work', 'feature not working',
                'minor issue', 'small problem', 'quick question', 'general inquiry'
            ]
        }
        
        technical_score = 0.0
        for pattern_type, patterns in technical_patterns.items():
            matches = sum(1 for pattern in patterns if pattern in email_lower)
            if matches > 0:
                technical_score += matches * 0.3
                
        if technical_score > 0.4:
            return EscalationTeam.TECHNICAL_TEAM
        
        # Priority Resolution: If multiple scores are close, use hierarchy
        scores = [
            (senior_score, EscalationTeam.SENIOR_SUPPORT, "Senior Team"),
            (management_score, EscalationTeam.MANAGEMENT, "Management"), 
            (billing_score, EscalationTeam.BILLING_SPECIALISTS, "Billing"),
            (technical_score, EscalationTeam.TECHNICAL_TEAM, "Technical")
        ]
        
        # Sort by score descending (sort by first element of tuple only)
        scores.sort(key=lambda x: x[0], reverse=True)
        top_score, top_team, team_name = scores[0]
        
        # Return highest scoring team if confidence threshold met
        if top_score > 0.3:
            return top_team
            
        return None  # Use default assignment logic
    
    def _estimate_complexity(self, 
                           category: str,
                           escalation_reasons: List[EscalationReason],
                           base_complexity_score: float,
                           customer_context: Dict) -> float:
        """Estimate the complexity of handling this escalated email"""
        
        complexity = base_complexity_score
        
        # Category-based complexity
        category_complexity = {
            'TECHNICAL_ISSUE': 0.8,
            'LEGAL_COMPLIANCE': 0.9,
            'PARTNERSHIP_BUSINESS': 0.7,
            'PRESS_MEDIA': 0.8,
            'ACCOUNT_ISSUE': 0.6,
            'BILLING_INQUIRY': 0.4,
            'CUSTOMER_SUPPORT': 0.3
        }
        
        category_factor = category_complexity.get(category, 0.5)
        complexity = max(complexity, category_factor)
        
        # Escalation reason complexity boosts
        reason_complexity_boosts = {
            EscalationReason.LEGAL_COMPLIANCE: 0.3,
            EscalationReason.TECHNICAL_COMPLEXITY: 0.2,
            EscalationReason.POLICY_EXCEPTION: 0.2,
            EscalationReason.BUSINESS_CRITICAL: 0.25,
            EscalationReason.MULTI_CHANNEL: 0.15,
        }
        
        for reason in escalation_reasons:
            boost = reason_complexity_boosts.get(reason, 0.0)
            complexity += boost
        
        # VIP customers add complexity due to special handling needs
        if customer_context.get('is_vip', False):
            complexity += 0.15
        
        return min(1.0, complexity)
    
    def _determine_sla_requirements(self, 
                                  priority_level: PriorityLevel,
                                  escalation_reasons: List[EscalationReason],
                                  customer_context: Dict) -> tuple:
        """Determine SLA requirements for response and resolution"""
        
        # Base SLA by priority
        base_sla = {
            PriorityLevel.CRITICAL: (1, 4),     # 1hr response, 4hr resolution
            PriorityLevel.HIGH: (4, 24),        # 4hr response, 24hr resolution
            PriorityLevel.MEDIUM: (24, 72),     # 24hr response, 72hr resolution  
            PriorityLevel.LOW: (48, 120),       # 48hr response, 120hr resolution
            PriorityLevel.ROUTINE: (72, 168)    # 72hr response, 168hr resolution
        }
        
        response_sla, resolution_sla = base_sla[priority_level]
        
        # VIP customers get enhanced SLA
        if customer_context.get('is_vip', False):
            response_sla = max(1, response_sla // 2)  # Halve response time, min 1hr
            resolution_sla = max(4, int(resolution_sla * 0.75))  # 25% faster resolution
        
        # Business critical gets enhanced SLA
        if EscalationReason.BUSINESS_CRITICAL in escalation_reasons:
            response_sla = 1  # Always 1hr for business critical
            resolution_sla = max(4, resolution_sla // 2)  # Halve resolution time
        
        return response_sla, resolution_sla
    
    def _assess_business_impact(self, 
                              escalation_reasons: List[EscalationReason],
                              customer_context: Dict,
                              sentiment_score: float) -> str:
        """Assess the business impact of this escalation"""
        
        if EscalationReason.BUSINESS_CRITICAL in escalation_reasons:
            return "HIGH - Business critical issue affecting operations or revenue"
        
        if customer_context.get('is_vip', False):
            return "HIGH - VIP customer requiring special attention and care"
        
        if EscalationReason.LEGAL_COMPLIANCE in escalation_reasons:
            return "HIGH - Legal/compliance issue with potential regulatory impact"
        
        if (EscalationReason.FINANCIAL_IMPACT in escalation_reasons and 
            abs(sentiment_score) > 0.7):
            return "MEDIUM - Financial dispute with potential chargeback risk"
        
        if EscalationReason.EMOTIONAL_INTENSITY in escalation_reasons and sentiment_score < -0.8:
            return "MEDIUM - Highly dissatisfied customer with reputation risk"
        
        if len(escalation_reasons) >= 3:
            return "MEDIUM - Complex multi-factor case requiring expert handling"
        
        return "LOW - Standard escalation for quality assurance"
    
    def _generate_special_instructions(self, 
                                     escalation_reasons: List[EscalationReason],
                                     customer_context: Dict,
                                     category: str) -> List[str]:
        """Generate special handling instructions for the escalated case"""
        
        instructions = []
        
        # VIP handling
        if customer_context.get('is_vip', False):
            instructions.append("VIP CUSTOMER: Handle with extra care and attention")
            instructions.append("Consider offering premium support options")
        
        # Legal compliance
        if EscalationReason.LEGAL_COMPLIANCE in escalation_reasons:
            instructions.append("LEGAL REVIEW: Consult legal team before responding")
            instructions.append("Document all interactions for compliance purposes")
        
        # Business critical
        if EscalationReason.BUSINESS_CRITICAL in escalation_reasons:
            instructions.append("BUSINESS CRITICAL: Escalate to management immediately")
            instructions.append("Provide regular status updates every 2 hours")
        
        # Financial impact
        if EscalationReason.FINANCIAL_IMPACT in escalation_reasons:
            instructions.append("FINANCIAL RISK: Review refund/credit policies carefully")
            instructions.append("Consider proactive resolution to prevent chargeback")
        
        # Technical complexity
        if EscalationReason.TECHNICAL_COMPLEXITY in escalation_reasons:
            instructions.append("TECHNICAL ISSUE: Engage technical experts immediately")
            instructions.append("Document technical details and resolution steps")
        
        # High emotion
        if EscalationReason.EMOTIONAL_INTENSITY in escalation_reasons:
            instructions.append("HIGH EMOTION: Use empathetic communication approach")
            instructions.append("Consider phone call instead of email response")
        
        # Multi-channel
        if EscalationReason.MULTI_CHANNEL in escalation_reasons:
            instructions.append("MULTI-CHANNEL: Check all communication channels for context")
            instructions.append("Consolidate response to primary channel only")
        
        return instructions
    
    def _generate_escalation_reasoning(self, 
                                     should_escalate: bool,
                                     escalation_reasons: List[EscalationReason],
                                     escalation_score: float,
                                     confidence_analysis: ConfidenceAnalysis,
                                     priority_level: PriorityLevel,
                                     assigned_team: EscalationTeam) -> List[str]:
        """Generate human-readable reasoning for the escalation decision"""
        
        reasoning = []
        
        if should_escalate:
            reasoning.append("ESCALATION REQUIRED - Human intervention needed")
            
            # Primary escalation reasons
            reason_descriptions = {
                EscalationReason.LOW_CONFIDENCE: "Classification confidence below threshold",
                EscalationReason.HIGH_RISK_FACTORS: "Multiple high-risk factors detected",
                EscalationReason.VIP_CUSTOMER: "VIP customer requiring special attention",
                EscalationReason.LEGAL_COMPLIANCE: "Legal/compliance issues detected",
                EscalationReason.FINANCIAL_IMPACT: "Financial impact or dispute identified",
                EscalationReason.TECHNICAL_COMPLEXITY: "Complex technical issue requiring expertise",
                EscalationReason.EMOTIONAL_INTENSITY: "High emotional intensity requiring careful handling",
                EscalationReason.BUSINESS_CRITICAL: "Business critical issue affecting operations",
                EscalationReason.POLICY_EXCEPTION: "Policy exception or special case handling needed",
                EscalationReason.MULTI_CHANNEL: "Multi-channel communication requiring coordination"
            }
            
            for reason in escalation_reasons:
                if reason in reason_descriptions:
                    reasoning.append(f"• {reason_descriptions[reason]}")
            
            # Escalation score context
            if escalation_score > 0.8:
                reasoning.append(f"Very high escalation score: {escalation_score:.2f}")
            elif escalation_score > 0.6:
                reasoning.append(f"High escalation score: {escalation_score:.2f}")
            else:
                reasoning.append(f"Moderate escalation score: {escalation_score:.2f}")
            
            # Priority and team assignment
            reasoning.append(f"Assigned priority: {priority_level.name} ({priority_level.value})")
            reasoning.append(f"Recommended team: {assigned_team.value.replace('_', ' ').title()}")
            
        else:
            reasoning.append("NO ESCALATION - Automated processing recommended")
            reasoning.append(f"Escalation score {escalation_score:.2f} below threshold")
            reasoning.append(f"Confidence level: {confidence_analysis.overall_confidence:.2f}")
            
            if confidence_analysis.recommended_action == ProcessingAction.SEND:
                reasoning.append("Recommended for immediate automated response")
            elif confidence_analysis.recommended_action == ProcessingAction.REVIEW:
                reasoning.append("Recommended for human review before sending")
        
        return reasoning