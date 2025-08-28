"""
Sophisticated Confidence Scoring System - CORE LOGIC

Advanced multi-dimensional confidence analysis that goes beyond simple classification scores.
Combines classification confidence, contextual factors, and risk assessment to make intelligent
routing decisions (send/review/escalate).
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum

class ProcessingAction(Enum):
    """Actions the system can take based on confidence analysis"""
    SEND = "send"           # High confidence - send immediately
    REVIEW = "review"       # Medium confidence - human review recommended  
    ESCALATE = "escalate"   # Low confidence or high risk - human required

class RiskFactor(Enum):
    """Types of risk factors that affect confidence"""
    HIGH_EMOTION = "high_emotion"
    LEGAL_TERMS = "legal_terms"
    COMPLAINT_LANGUAGE = "complaint_language"
    AMBIGUOUS_INTENT = "ambiguous_intent"
    VIP_CUSTOMER = "vip_customer"
    TECHNICAL_COMPLEXITY = "technical_complexity"
    FINANCIAL_IMPACT = "financial_impact"
    LOW_SCORE_SEPARATION = "low_score_separation"

@dataclass
class ConfidenceAnalysis:
    """Comprehensive confidence analysis result"""
    # Core confidence metrics
    overall_confidence: float
    classification_confidence: float
    response_confidence: float
    contextual_confidence: float
    
    # Score analysis
    score_separation: float  # Gap between top and second category
    category_certainty: float  # How definitive the classification is
    
    # Risk assessment
    risk_factors: List[RiskFactor]
    risk_score: float  # Overall risk (0.0 = no risk, 1.0 = maximum risk)
    
    # Decision support
    recommended_action: ProcessingAction
    confidence_threshold_met: Dict[str, bool]  # Which thresholds were met
    reasoning: List[str]  # Human-readable confidence reasoning

class ConfidenceScorer:
    """
    Advanced confidence scoring system that analyzes classification results
    and determines optimal processing actions.
    """
    
    def __init__(self):
        """Initialize confidence scorer with thresholds and weights"""
        
        # Base confidence thresholds (will be adjusted based on context)
        self.base_thresholds = {
            'send': 0.85,      # Must be very confident to auto-send
            'review': 0.65,    # Medium confidence for human review
            'escalate': 0.45   # Below this always escalate
        }
        
        # Context-specific threshold adjustments
        self.threshold_adjustments = {
            # Financial/billing issues - lower thresholds (easier to escalate)
            'financial_issues': {
                'send': 0.95,      # Nearly perfect confidence required
                'review': 0.75,    # Higher review threshold
                'escalate': 0.60   # Much higher escalation threshold (easier to escalate)
            },
            # Security issues - much lower thresholds
            'security_issues': {
                'send': 0.98,      # Almost never auto-send security issues
                'review': 0.80,    # High review threshold
                'escalate': 0.70   # Very high escalation threshold
            },
            # Simple inquiries - higher thresholds (harder to escalate)
            'simple_inquiries': {
                'send': 0.75,      # Lower send threshold (easier to auto-send)
                'review': 0.55,    # Lower review threshold
                'escalate': 0.30   # Lower escalation threshold (harder to escalate)
            },
            # Exchange/returns - higher thresholds (let AI handle more)
            'exchange_returns': {
                'send': 0.70,      # Much lower send threshold
                'review': 0.50,    # Lower review threshold
                'escalate': 0.25   # Much lower escalation threshold
            }
        }
        
        # Weights for different confidence factors
        self.weights = {
            'classification': 0.40,  # Classification accuracy confidence
            'response': 0.30,        # Response generation confidence
            'contextual': 0.20,      # Context understanding confidence
            'risk_adjustment': 0.10  # Risk factor penalty
        }
        
        # Score separation thresholds
        self.separation_thresholds = {
            'very_clear': 0.30,    # Top score 30%+ higher than second
            'clear': 0.20,         # Top score 20%+ higher than second  
            'moderate': 0.10,      # Top score 10%+ higher than second
            'unclear': 0.05        # Top score <5% higher than second
        }
        
        # Risk factor keywords for detection
        self.risk_keywords = {
            RiskFactor.LEGAL_TERMS: [
                'lawyer', 'attorney', 'legal', 'lawsuit', 'sue', 'court',
                'litigation', 'damages', 'breach', 'contract', 'violation'
            ],
            RiskFactor.COMPLAINT_LANGUAGE: [
                'terrible', 'horrible', 'worst', 'disgusted', 'furious',
                'outraged', 'unacceptable', 'ridiculous', 'pathetic'
            ],
            RiskFactor.HIGH_EMOTION: [
                'urgent', 'emergency', 'immediately', 'asap', 'critical',
                'angry', 'frustrated', 'disappointed', 'upset'
            ],
            RiskFactor.FINANCIAL_IMPACT: [
                'refund', 'money back', 'charge back', 'dispute',
                'fraud', 'unauthorized', 'billing error', 'overcharged'
            ]
        }
    
    def analyze_confidence(self, classification_result: Dict, email_content: str) -> ConfidenceAnalysis:
        """
        Perform comprehensive confidence analysis on classification results
        
        Args:
            classification_result: Output from EmailClassifier.classify()
            email_content: Original email text for additional analysis
            
        Returns:
            ConfidenceAnalysis with detailed confidence metrics and recommendations
        """
        
        # Extract key data from classification result
        category_scores = classification_result.get('category_scores', {})
        structural_features = classification_result.get('structural_features', {})
        sentiment_score = classification_result.get('sentiment_score', 0.0)
        complexity_score = classification_result.get('complexity_score', 0.0)
        
        # 1. Calculate score separation (how clear the classification is)
        score_separation = self._calculate_score_separation(category_scores)
        category_certainty = self._calculate_category_certainty(category_scores, score_separation)
        
        # 2. Calculate individual confidence components
        classification_confidence = self._calculate_classification_confidence(
            category_scores, score_separation, complexity_score
        )
        
        response_confidence = self._calculate_response_confidence(
            classification_result['category'], category_scores, structural_features
        )
        
        contextual_confidence = self._calculate_contextual_confidence(
            email_content, structural_features, sentiment_score
        )
        
        # 3. Identify risk factors
        risk_factors = self._identify_risk_factors(
            email_content, sentiment_score, score_separation, structural_features
        )
        
        risk_score = self._calculate_risk_score(risk_factors, sentiment_score)
        
        # 4. Calculate overall confidence with risk adjustment
        overall_confidence = self._calculate_overall_confidence(
            classification_confidence, response_confidence, 
            contextual_confidence, risk_score
        )
        
        # 5. Get context-adjusted thresholds
        context_type = self._detect_context_type(classification_result, email_content, risk_factors)
        adjusted_thresholds = self._get_adjusted_thresholds(context_type)
        
        # 6. Determine recommended action (with adjusted thresholds)
        recommended_action = self._determine_action(overall_confidence, risk_factors, risk_score, adjusted_thresholds)
        
        # 7. Check which thresholds were met (using adjusted thresholds)
        confidence_threshold_met = {
            'send': overall_confidence >= adjusted_thresholds['send'],
            'review': overall_confidence >= adjusted_thresholds['review'],
            'escalate': overall_confidence < adjusted_thresholds['escalate'],
            'context_type': context_type  # Include context info
        }
        
        # 8. Generate human-readable reasoning
        reasoning = self._generate_reasoning(
            overall_confidence, classification_confidence, response_confidence,
            contextual_confidence, risk_factors, score_separation, recommended_action,
            context_type, adjusted_thresholds
        )
        
        return ConfidenceAnalysis(
            overall_confidence=overall_confidence,
            classification_confidence=classification_confidence,
            response_confidence=response_confidence,
            contextual_confidence=contextual_confidence,
            score_separation=score_separation,
            category_certainty=category_certainty,
            risk_factors=risk_factors,
            risk_score=risk_score,
            recommended_action=recommended_action,
            confidence_threshold_met=confidence_threshold_met,
            reasoning=reasoning
        )
    
    def _calculate_score_separation(self, category_scores: Dict[str, float]) -> float:
        """Calculate the gap between the top two category scores"""
        if len(category_scores) < 2:
            return 1.0  # Perfect separation if only one category
        
        sorted_scores = sorted(category_scores.values(), reverse=True)
        return sorted_scores[0] - sorted_scores[1]
    
    def _calculate_category_certainty(self, category_scores: Dict[str, float], separation: float) -> float:
        """Calculate how certain we are about the category choice"""
        if separation >= self.separation_thresholds['very_clear']:
            return 0.95
        elif separation >= self.separation_thresholds['clear']:
            return 0.85
        elif separation >= self.separation_thresholds['moderate']:
            return 0.70
        else:
            return 0.50  # Low certainty for unclear classifications
    
    def _calculate_classification_confidence(self, category_scores: Dict[str, float], 
                                          score_separation: float, complexity_score: float) -> float:
        """Calculate confidence in the classification decision"""
        
        # Base confidence from top score
        top_score = max(category_scores.values()) if category_scores else 0.0
        
        # Boost confidence based on score separation
        separation_boost = min(score_separation * 2, 0.2)  # Max 20% boost
        
        # Penalty for high complexity
        complexity_penalty = complexity_score * 0.15  # Max 15% penalty
        
        # Calculate final classification confidence
        confidence = top_score + separation_boost - complexity_penalty
        
        return max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
    
    def _calculate_response_confidence(self, category: str, category_scores: Dict[str, float],
                                     structural_features: Dict[str, any]) -> float:
        """Calculate confidence in our ability to generate a good response"""
        
        # Start with category-specific base confidence
        category_base_confidence = {
            'CUSTOMER_SUPPORT': 0.85,    # High - we handle these well
            'SALES_ORDER': 0.90,         # Very high - straightforward 
            'TECHNICAL_ISSUE': 0.70,     # Lower - more complex responses needed
            'BILLING_INQUIRY': 0.80,     # Good - standard procedures
            'PRODUCT_QUESTION': 0.85,    # High - FAQ-style responses
            'REFUND_REQUEST': 0.75,      # Moderate - policy dependent
            'SHIPPING_INQUIRY': 0.88,    # High - tracking info available
            'ACCOUNT_ISSUE': 0.65,       # Lower - may need account access
            'COMPLAINT_NEGATIVE': 0.60,  # Low - requires careful handling
            'COMPLIMENT_POSITIVE': 0.95, # Very high - easy to respond
            'SPAM_PROMOTIONAL': 0.40,    # Low - shouldn't auto-respond
            'PARTNERSHIP_BUSINESS': 0.50, # Low - needs business context
            'PRESS_MEDIA': 0.30,         # Very low - needs PR review
            'LEGAL_COMPLIANCE': 0.20,    # Very low - legal expertise needed
            'OTHER_UNCATEGORIZED': 0.50  # Medium - depends on content
        }
        
        base_confidence = category_base_confidence.get(category, 0.50)
        
        # Boost confidence if we have helpful structural features
        feature_boost = 0.0
        
        # Order-related features boost confidence for order categories
        if category in ['SALES_ORDER', 'SHIPPING_INQUIRY', 'REFUND_REQUEST']:
            if structural_features.get('has_order_number'):
                feature_boost += 0.10
            if structural_features.get('has_tracking_number'):
                feature_boost += 0.10
        
        # Contact info helps with all categories
        if structural_features.get('has_phone_number') or structural_features.get('has_email'):
            feature_boost += 0.05
        
        # Account numbers help with account-related issues
        if category in ['ACCOUNT_ISSUE', 'BILLING_INQUIRY'] and structural_features.get('has_account_number'):
            feature_boost += 0.15
        
        return min(1.0, base_confidence + feature_boost)
    
    def _calculate_contextual_confidence(self, email_content: str, structural_features: Dict[str, any],
                                       sentiment_score: float) -> float:
        """Calculate confidence based on contextual understanding"""
        
        confidence = 0.70  # Base contextual confidence
        
        # Length analysis - very short or very long emails are harder
        content_length = len(email_content.split())
        if 10 <= content_length <= 200:  # Sweet spot for email length
            confidence += 0.15
        elif content_length < 5:  # Too short
            confidence -= 0.20
        elif content_length > 500:  # Too long
            confidence -= 0.10
        
        # Sentiment extremes are riskier
        sentiment_abs = abs(sentiment_score)
        if sentiment_abs > 0.7:  # Very positive or very negative
            confidence -= 0.10
        
        # Clear structure helps understanding
        if structural_features.get('has_greeting') and structural_features.get('has_closing'):
            confidence += 0.10
        
        # Questions are generally easier to understand
        if structural_features.get('has_question_mark'):
            confidence += 0.05
        
        return max(0.0, min(1.0, confidence))
    
    def _identify_risk_factors(self, email_content: str, sentiment_score: float,
                             score_separation: float, structural_features: Dict[str, any]) -> List[RiskFactor]:
        """Identify risk factors that might affect response quality"""
        
        risk_factors = []
        email_lower = email_content.lower()
        
        # Check for risk keywords (using word boundaries to avoid false positives)
        import re
        for risk_type, keywords in self.risk_keywords.items():
            for keyword in keywords:
                # Use word boundaries to match whole words only
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, email_lower):
                    risk_factors.append(risk_type)
                    break  # Only add each risk type once
        
        # Emotional intensity risk
        if abs(sentiment_score) > 0.8:
            risk_factors.append(RiskFactor.HIGH_EMOTION)
        
        # Low score separation means unclear intent
        if score_separation < self.separation_thresholds['unclear']:
            risk_factors.append(RiskFactor.AMBIGUOUS_INTENT)
        
        # Technical complexity indicators
        technical_terms = ['api', 'ssl', 'database', 'server', 'integration', 'configuration']
        if any(term in email_lower for term in technical_terms):
            risk_factors.append(RiskFactor.TECHNICAL_COMPLEXITY)
        
        # VIP indicators (could be enhanced with actual customer data)
        vip_indicators = ['ceo', 'president', 'director', 'manager', 'enterprise', 'corporate']
        if any(indicator in email_lower for indicator in vip_indicators):
            risk_factors.append(RiskFactor.VIP_CUSTOMER)
        
        return risk_factors
    
    def _calculate_risk_score(self, risk_factors: List[RiskFactor], sentiment_score: float) -> float:
        """Calculate overall risk score from individual risk factors"""
        
        if not risk_factors:
            return 0.0
        
        # Risk weights for different factors
        risk_weights = {
            RiskFactor.LEGAL_TERMS: 0.30,
            RiskFactor.COMPLAINT_LANGUAGE: 0.25,
            RiskFactor.HIGH_EMOTION: 0.20,
            RiskFactor.VIP_CUSTOMER: 0.25,
            RiskFactor.FINANCIAL_IMPACT: 0.20,
            RiskFactor.TECHNICAL_COMPLEXITY: 0.15,
            RiskFactor.AMBIGUOUS_INTENT: 0.15,
            RiskFactor.LOW_SCORE_SEPARATION: 0.10
        }
        
        # Calculate weighted risk score
        total_risk = sum(risk_weights.get(factor, 0.10) for factor in risk_factors)
        
        # Add sentiment-based risk
        sentiment_risk = abs(sentiment_score) * 0.15  # Max 15% additional risk
        
        return min(1.0, total_risk + sentiment_risk)
    
    def _calculate_overall_confidence(self, classification_confidence: float, response_confidence: float,
                                    contextual_confidence: float, risk_score: float) -> float:
        """Calculate the final overall confidence score"""
        
        # Weighted combination of confidence factors
        base_confidence = (
            classification_confidence * self.weights['classification'] +
            response_confidence * self.weights['response'] +
            contextual_confidence * self.weights['contextual']
        )
        
        # Apply risk penalty
        risk_penalty = risk_score * self.weights['risk_adjustment']
        overall_confidence = base_confidence - risk_penalty
        
        return max(0.0, min(1.0, overall_confidence))
    
    def _detect_context_type(self, classification_result: Dict, email_content: str, 
                           risk_factors: List[RiskFactor]) -> str:
        """Detect the context type to determine appropriate confidence thresholds"""
        
        email_lower = email_content.lower()
        category = classification_result.get('category', '').upper()
        
        # Check for financial/billing context
        financial_indicators = [
            'billing', 'payment', 'refund', 'charge', 'money', 'invoice', 
            'receipt', 'account charged', 'payment failed', 'double charged'
        ]
        if (RiskFactor.FINANCIAL_IMPACT in risk_factors or 
            category in ['BILLING_INQUIRY', 'REFUND_REQUEST'] or
            any(indicator in email_lower for indicator in financial_indicators)):
            return 'financial_issues'
        
        # Check for security context
        security_indicators = [
            'security', 'hack', 'breach', 'unauthorized', 'suspicious activity',
            'fraud', 'identity theft', 'account compromised', 'login attempt'
        ]
        if any(indicator in email_lower for indicator in security_indicators):
            return 'security_issues'
        
        # Check for exchange/return context
        exchange_indicators = [
            'exchange', 'return', 'wrong size', 'wrong color', 'different item',
            'size too', 'doesn\'t fit', 'not as described', 'want to exchange'
        ]
        if any(indicator in email_lower for indicator in exchange_indicators):
            return 'exchange_returns'
        
        # Check for simple inquiries
        simple_indicators = [
            'how to', 'question about', 'wondering', 'can you help', 'quick question',
            'information about', 'clarification', 'simple question'
        ]
        if (any(indicator in email_lower for indicator in simple_indicators) or
            len(email_content.split()) < 30):  # Short emails are often simple
            return 'simple_inquiries'
        
        # Default to base thresholds
        return 'default'
    
    def _get_adjusted_thresholds(self, context_type: str) -> Dict[str, float]:
        """Get confidence thresholds adjusted for the specific context"""
        
        if context_type in self.threshold_adjustments:
            return self.threshold_adjustments[context_type].copy()
        else:
            return self.base_thresholds.copy()
    
    def _determine_action(self, overall_confidence: float, risk_factors: List[RiskFactor],
                         risk_score: float, thresholds: Dict[str, float] = None) -> ProcessingAction:
        """Determine the recommended processing action with context-aware thresholds"""
        
        # Use provided thresholds or fall back to base thresholds
        if thresholds is None:
            thresholds = self.base_thresholds
        
        # High-risk factors always escalate regardless of confidence
        high_risk_factors = [RiskFactor.LEGAL_TERMS, RiskFactor.VIP_CUSTOMER]
        if any(factor in risk_factors for factor in high_risk_factors):
            return ProcessingAction.ESCALATE
        
        # Context-aware confidence-based decision
        if overall_confidence >= thresholds['send'] and risk_score < 0.3:
            return ProcessingAction.SEND
        elif overall_confidence >= thresholds['review']:
            return ProcessingAction.REVIEW
        else:
            return ProcessingAction.ESCALATE
    
    def _generate_reasoning(self, overall_confidence: float, classification_confidence: float,
                          response_confidence: float, contextual_confidence: float,
                          risk_factors: List[RiskFactor], score_separation: float,
                          action: ProcessingAction, context_type: str = 'default',
                          adjusted_thresholds: Dict[str, float] = None) -> List[str]:
        """Generate human-readable reasoning for the confidence assessment"""
        
        reasoning = []
        
        # Overall confidence assessment
        if overall_confidence >= 0.85:
            reasoning.append("Very high overall confidence in processing this email")
        elif overall_confidence >= 0.65:
            reasoning.append("Moderate confidence - human review recommended")
        else:
            reasoning.append("Low confidence - human intervention required")
        
        # Classification reasoning
        if classification_confidence >= 0.80:
            reasoning.append("Strong classification confidence")
        elif score_separation >= 0.20:
            reasoning.append("Clear category distinction")
        else:
            reasoning.append("Ambiguous classification - multiple categories possible")
        
        # Response capability reasoning
        if response_confidence >= 0.85:
            reasoning.append("High confidence in response generation capability")
        elif response_confidence < 0.60:
            reasoning.append("Limited response generation capability for this category")
        
        # Risk factor reasoning
        if risk_factors:
            risk_descriptions = {
                RiskFactor.LEGAL_TERMS: "Contains legal terminology",
                RiskFactor.COMPLAINT_LANGUAGE: "Contains complaint language",
                RiskFactor.HIGH_EMOTION: "High emotional intensity detected",
                RiskFactor.VIP_CUSTOMER: "Potential VIP customer identified",
                RiskFactor.FINANCIAL_IMPACT: "Financial impact detected",
                RiskFactor.TECHNICAL_COMPLEXITY: "Technical complexity identified",
                RiskFactor.AMBIGUOUS_INTENT: "Customer intent unclear",
                RiskFactor.LOW_SCORE_SEPARATION: "Classification uncertainty"
            }
            
            for factor in risk_factors:
                if factor in risk_descriptions:
                    reasoning.append(f"Risk factor: {risk_descriptions[factor]}")
        
        # Action reasoning
        action_reasoning = {
            ProcessingAction.SEND: "Recommended for immediate automated response",
            ProcessingAction.REVIEW: "Recommended for human review before sending",
            ProcessingAction.ESCALATE: "Requires human agent intervention"
        }
        
        reasoning.append(action_reasoning[action])
        
        return reasoning