"""
Smart Exchange/Return Handler - Integrated Version

This module leverages existing complexity calculation from EmailClassifier
and focuses specifically on detecting exchange/return requests that can be 
handled automatically without escalation.

Key Innovation: Integrates with existing classifier complexity logic instead
of duplicating it, focusing purely on exchange-specific detection patterns.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import re


class ExchangeType(Enum):
    """Types of exchange requests detected"""
    SIZE_EXCHANGE = "size_exchange"           # Wrong size received/ordered
    COLOR_STYLE = "color_style"               # Wrong color or style
    PRODUCT_MISMATCH = "product_mismatch"     # Different product than expected
    GENERAL_RETURN = "general_return"         # General return request
    NOT_EXCHANGE = "not_exchange"             # Not an exchange request


class ExchangeComplexity(Enum):
    """Complexity levels for exchange requests"""
    SIMPLE = "simple"           # AI can handle automatically
    MODERATE = "moderate"       # AI can handle with caution
    COMPLEX = "complex"         # Requires human escalation


@dataclass
class ExchangeDetectionResult:
    """Result of exchange detection analysis"""
    is_exchange_request: bool
    exchange_type: ExchangeType
    complexity: ExchangeComplexity
    confidence: float                        # 0.0-1.0 confidence in detection
    detected_issues: List[str]              # Specific issues found
    keywords_matched: List[str]             # Keywords that triggered detection
    reasoning: List[str]                    # Why this decision was made
    suggested_response_template: Optional[str]  # Which template to use
    classifier_complexity: float            # Complexity from EmailClassifier


class SmartExchangeHandler:
    """
    Intelligent handler for exchange and return requests that leverages
    existing EmailClassifier complexity logic and focuses on exchange detection.
    """
    
    def __init__(self):
        """Initialize exchange handler with detection patterns and templates"""
        
        # Size-related exchange patterns
        self.size_patterns = {
            'explicit_size_mentions': [
                r'wrong size', r'size.*wrong', r'too (big|small|large|tight)',
                r'size [A-Z]{1,3}.*received.*size [A-Z]{1,3}', 
                r'ordered.*size.*got.*size', r'fits (too|doesn\'t)',
                r'(small|medium|large|xl|xs).*instead.*of.*(small|medium|large|xl|xs)'
            ],
            'size_keywords': [
                'wrong size', 'too big', 'too small', 'too tight', 'too loose',
                'doesn\'t fit', 'size up', 'size down', 'exchange size'
            ]
        }
        
        # Color and style exchange patterns  
        self.color_style_patterns = {
            'color_mentions': [
                r'wrong color', r'different color', r'color.*wrong',
                r'(red|blue|green|black|white|yellow|pink|purple).*instead.*of.*(red|blue|green|black|white|yellow|pink|purple)',
                r'not.*color.*shown', r'color.*website', r'not.*expected'
            ],
            'style_mentions': [
                r'wrong style', r'different style', r'not.*style.*ordered',
                r'style.*different', r'not.*shown.*website', r'looks different'
            ],
            'appearance_keywords': [
                'wrong color', 'different color', 'wrong style', 'different style',
                'not as shown', 'not what expected', 'looks different', 'not matching'
            ]
        }
        
        # Product mismatch patterns
        self.product_mismatch_patterns = {
            'mismatch_phrases': [
                r'wrong.*item', r'different.*product', r'not.*what.*ordered',
                r'received.*instead', r'not.*matching.*description',
                r'different.*than.*website', r'not.*as.*advertised'
            ],
            'mismatch_keywords': [
                'wrong item', 'wrong product', 'different item', 'not what ordered',
                'not matching', 'not as described', 'received wrong'
            ]
        }
        
        # Exchange/return intent patterns
        self.exchange_intent_patterns = {
            'exchange_requests': [
                r'exchange', r'swap', r'change.*for', r'return.*exchange',
                r'exchange.*process', r'how.*exchange', r'guide.*exchange'
            ],
            'return_requests': [
                r'return', r'send.*back', r'give.*back', 
                r'return.*process', r'how.*return', r'return.*policy'
            ],
            'process_questions': [
                r'how.*do.*i.*exchange', r'what.*exchange.*process', r'guide.*through.*exchange',
                r'steps.*to.*return', r'procedure.*for.*return', r'process.*for.*exchange'
            ]
        }
        
        # Simple issue indicators (these suggest AI can handle)
        self.simple_issue_indicators = [
            'just need to exchange', 'simple exchange', 'wrong size only',
            'just the wrong color', 'easy exchange', 'straightforward return'
        ]
        
        # Response templates for different exchange types
        self.response_templates = {
            'size_exchange': """Dear Valued Customer,

Thank you for contacting us about the sizing issue with your recent order.

I sincerely apologize that the item doesn't fit as expected. We want to make this right for you immediately.

**Easy Size Exchange Process:**

1. **Quick Exchange**: Visit our returns portal at [returns.company.com] with your order number
2. **Free Return Shipping**: We'll provide a prepaid return label - absolutely no cost to you  
3. **Select Correct Size**: Choose your preferred size during the return process
4. **Fast Processing**: Your new item ships within 24 hours of receiving your return

**Size Guide Help**: If you're unsure about the correct size, our detailed size guide at [company.com/sizing] includes measurements and fit recommendations.

**Express Option**: For faster service, you can order the correct size now and return the original item at your convenience - no waiting required.

**Our Promise**: We guarantee the correct size ships this time, and if there are any issues, you get priority handling.

Is there anything specific about the sizing I can help you with? I'm here to ensure you get the perfect fit.

Best regards,
Customer Support Team

*This exchange is completely free - we cover all shipping costs.*""",

            'color_style': """Dear Valued Customer,

Thank you for reaching out about the color/style difference in your recent order.

I apologize that the item you received doesn't match what you expected from our website. This is definitely something we can resolve quickly.

**Color/Style Exchange Made Easy:**

1. **Simple Returns**: Use our online return portal [returns.company.com] with your order number
2. **Visual Confirmation**: Feel free to upload a photo to show the difference (optional)
3. **Correct Item Selection**: Choose the exact color/style you originally wanted
4. **Free Exchange**: No additional costs - we cover all return shipping

**Quality Promise**: We'll also review our product photos to ensure they accurately represent the actual item.

**Flexible Options:**
- Full exchange for the correct color/style
- Partial credit if you decide to keep the current item
- Complete refund if you prefer not to exchange

**Current Stock Check**: I can verify we have your preferred color/style in stock before processing the exchange.

Would you like me to help you select the exact item you were looking for? I can also check availability and expedite shipping.

Best regards,  
Customer Support Team

*All exchanges are processed at no cost to you.*""",

            'product_mismatch': """Dear Valued Customer,

Thank you for contacting us about receiving a different item than what you ordered.

I sincerely apologize for this mix-up in your order. This is entirely our error, and I want to resolve it immediately at no cost to you.

**Immediate Resolution Process:**

1. **Priority Processing**: I'm processing a replacement order for the correct item right now
2. **Free Return**: You'll receive a prepaid return label via email for the incorrect item  
3. **No Wait Required**: Your correct item ships today - don't wait to return the wrong item first
4. **Double-Check Verification**: I've personally verified the correct product details

**Your Correct Order**: Based on your original purchase, you should receive: [Product Name/Description]
**Expected Delivery**: [Timeframe - typically 2-3 business days]

**Our Guarantee**: This order receives priority handling and tracking. If there are any further issues, you get executive-level support and expedited shipping.

**Bonus**: As an apology for the inconvenience, you'll receive a discount code for your next purchase.

Can you confirm that the product description I have matches what you originally intended to order? I want to be absolutely certain we get it right this time.

Best regards,
Customer Support Team

*This entire resolution is free - we take full responsibility for the mix-up.*""",

            'general_return': """Dear Valued Customer,

Thank you for your return request. I'm happy to help make this process as simple as possible.

**Streamlined Return Process:**

1. **Online Portal**: Visit [returns.company.com] and enter your order number
2. **Flexible Options**: Choose between full refund or exchange for different item
3. **Free Return Shipping**: We provide a prepaid return label - no cost to you
4. **Quick Processing**: Refunds typically process within 3-5 business days after receipt

**Return Requirements**: Items must be returned within 30 days of delivery in original condition with tags attached (if applicable).

**Refund Details**: 
- Refunds go back to your original payment method
- Original shipping is refunded if the item was defective or our error
- You'll receive email confirmation once processing begins

**Need Assistance?** If you're considering an exchange instead, I can help you find a better-suited product or different size/style.

**Expedited Option**: For returns over $75, we offer free expedited processing upon request.

Is there anything specific about the return process I can clarify? I'm here to make this as easy as possible.

Best regards,
Customer Support Team

*Our 30-day return policy ensures you're completely satisfied with your purchase.*"""
        }
    
    def detect_exchange_request(self, email_content: str, 
                                classifier_complexity: float = None) -> ExchangeDetectionResult:
        """
        Analyze email to determine if it's an exchange request and how to handle it
        
        Args:
            email_content: The customer email text to analyze
            classifier_complexity: Complexity score from EmailClassifier (optional)
            
        Returns:
            ExchangeDetectionResult with detection analysis and recommendations
        """
        
        # Validate input early to avoid NoneType or unexpected-type errors
        if not email_content or not isinstance(email_content, str):
            return self._create_negative_result(
                ["Invalid email content provided"],
                classifier_complexity
            )
        
        # Check for empty or whitespace-only content after stripping
        if not email_content.strip():
            return self._create_negative_result(
                ["email_content is required - cannot be empty or whitespace-only"],
                classifier_complexity
            )
        
        email_lower = email_content.lower().strip()
        # ... rest of method ...
        
        # Initialize tracking variables
        detected_issues = []
        keywords_matched = []
        reasoning = []
        
        # 1. Check for exchange/return intent
        intent_score, intent_reasoning = self._detect_exchange_intent(email_lower)
        reasoning.extend(intent_reasoning)
        
        # If no exchange intent detected, it's not an exchange request
        if intent_score < 0.3:
            return self._create_negative_result(reasoning, classifier_complexity)
        
        # 2. Detect specific exchange types
        size_score, size_issues, size_keywords = self._detect_size_issues(email_lower)
        color_style_score, color_issues, color_keywords = self._detect_color_style_issues(email_lower)
        mismatch_score, mismatch_issues, mismatch_keywords = self._detect_product_mismatch(email_lower)
        
        # Combine detected issues and keywords
        detected_issues.extend(size_issues + color_issues + mismatch_issues)
        keywords_matched.extend(size_keywords + color_keywords + mismatch_keywords)
        
        # 3. Determine exchange type and confidence
        exchange_type, type_confidence = self._determine_exchange_type(
            size_score, color_style_score, mismatch_score, intent_score
        )
        
        # 4. Use existing classifier complexity or assess simple indicators
        if classifier_complexity is not None:
            complexity_level = self._determine_complexity_from_classifier(
                classifier_complexity, type_confidence
            )
            reasoning.append(f"Using classifier complexity score: {classifier_complexity:.2f}")
        else:
            complexity_level = self._assess_simple_complexity(email_lower, type_confidence)
            classifier_complexity = 0.5  # Default fallback
        
        # 5. Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            intent_score, type_confidence, complexity_level, len(detected_issues)
        )
        
        # 6. Decide if this should be auto-handled
        is_exchange_request = (
            exchange_type != ExchangeType.NOT_EXCHANGE and
            overall_confidence > 0.6 and
            complexity_level in [ExchangeComplexity.SIMPLE, ExchangeComplexity.MODERATE]
        )
        
        # 7. Select response template
        template = self._select_response_template(exchange_type, complexity_level) if is_exchange_request else None
        
        # 8. Generate final reasoning
        final_reasoning = self._generate_final_reasoning(
            is_exchange_request, exchange_type, complexity_level, overall_confidence, 
            reasoning, detected_issues
        )
        
        return ExchangeDetectionResult(
            is_exchange_request=is_exchange_request,
            exchange_type=exchange_type,
            complexity=complexity_level,
            confidence=overall_confidence,
            detected_issues=detected_issues,
            keywords_matched=keywords_matched,
            reasoning=final_reasoning,
            suggested_response_template=template,
            classifier_complexity=classifier_complexity
        )
    
    def _detect_exchange_intent(self, email_lower: str) -> Tuple[float, List[str]]:
        """Detect if email expresses intent to exchange or return something"""
        score = 0.0
        reasoning = []
        matches_found = []
        
        # Check exchange patterns
        for pattern in self.exchange_intent_patterns['exchange_requests']:
            if re.search(pattern, email_lower):
                score += 0.3
                matches_found.append(f"exchange: '{pattern}'")
        
        # Check return patterns
        for pattern in self.exchange_intent_patterns['return_requests']:
            if re.search(pattern, email_lower):
                score += 0.25
                matches_found.append(f"return: '{pattern}'")
        
        # Check process question patterns
        for pattern in self.exchange_intent_patterns['process_questions']:
            if re.search(pattern, email_lower):
                score += 0.2
                matches_found.append(f"process question: '{pattern}'")
        
        if matches_found:
            reasoning.append(f"Exchange/return intent detected: {len(matches_found)} patterns")
            reasoning.extend(matches_found[:2])  # Top 2 matches
        else:
            reasoning.append("No clear exchange/return intent found")
        
        return min(score, 1.0), reasoning
    
    def _detect_size_issues(self, email_lower: str) -> Tuple[float, List[str], List[str]]:
        """Detect size-related exchange requests"""
        score = 0.0
        issues = []
        keywords = []
        
        # Check explicit size patterns
        for pattern in self.size_patterns['explicit_size_mentions']:
            if re.search(pattern, email_lower):
                score += 0.4
                keywords.append(f"size pattern: {pattern}")
        
        # Check size keywords
        matched_keywords = [kw for kw in self.size_patterns['size_keywords'] if kw in email_lower]
        if matched_keywords:
            score += min(len(matched_keywords) * 0.15, 0.6)
            keywords.extend(matched_keywords[:3])
        
        if score > 0.3:
            issues.append("Size/fitting issue")
        
        return min(score, 1.0), issues, keywords
    
    def _detect_color_style_issues(self, email_lower: str) -> Tuple[float, List[str], List[str]]:
        """Detect color/style related exchange requests"""
        score = 0.0
        issues = []
        keywords = []
        
        # Check color patterns
        for pattern in self.color_style_patterns['color_mentions']:
            if re.search(pattern, email_lower):
                score += 0.4
                keywords.append(f"color: {pattern}")
        
        # Check style patterns
        for pattern in self.color_style_patterns['style_mentions']:
            if re.search(pattern, email_lower):
                score += 0.35
                keywords.append(f"style: {pattern}")
        
        # Check appearance keywords
        matched_keywords = [kw for kw in self.color_style_patterns['appearance_keywords'] if kw in email_lower]
        if matched_keywords:
            score += min(len(matched_keywords) * 0.1, 0.4)
            keywords.extend(matched_keywords[:2])
        
        if score > 0.3:
            issues.append("Color/style mismatch")
        
        return min(score, 1.0), issues, keywords
    
    def _detect_product_mismatch(self, email_lower: str) -> Tuple[float, List[str], List[str]]:
        """Detect general product mismatch issues"""
        score = 0.0
        issues = []
        keywords = []
        
        # Check mismatch patterns
        for pattern in self.product_mismatch_patterns['mismatch_phrases']:
            if re.search(pattern, email_lower):
                score += 0.35
                keywords.append(f"mismatch: {pattern}")
        
        # Check mismatch keywords
        matched_keywords = [kw for kw in self.product_mismatch_patterns['mismatch_keywords'] if kw in email_lower]
        if matched_keywords:
            score += min(len(matched_keywords) * 0.15, 0.5)
            keywords.extend(matched_keywords[:2])
        
        if score > 0.3:
            issues.append("Product mismatch")
        
        return min(score, 1.0), issues, keywords
    
    def _determine_exchange_type(self, size_score: float, color_style_score: float, 
                                mismatch_score: float, intent_score: float) -> Tuple[ExchangeType, float]:
        """Determine the primary exchange type and confidence"""
        
        scores = [
            (ExchangeType.SIZE_EXCHANGE, size_score),
            (ExchangeType.COLOR_STYLE, color_style_score), 
            (ExchangeType.PRODUCT_MISMATCH, mismatch_score)
        ]
        
        # Sort by score, highest first
        scores.sort(key=lambda x: x[1], reverse=True)
        top_type, top_score = scores[0]
        
        # If no specific type has good confidence, but intent is clear, use general return
        if top_score < 0.4 and intent_score > 0.5:
            return ExchangeType.GENERAL_RETURN, intent_score
        
        # If no clear type detected
        if top_score < 0.3:
            return ExchangeType.NOT_EXCHANGE, 0.0
        
        return top_type, top_score
    
    def _determine_complexity_from_classifier(self, classifier_complexity: float, 
                                            type_confidence: float) -> ExchangeComplexity:
        """Determine complexity level using EmailClassifier's complexity score"""
        
        # High complexity from classifier - needs escalation
        if classifier_complexity > 0.7:
            return ExchangeComplexity.COMPLEX
        
        # Medium complexity - can handle if high exchange confidence
        if classifier_complexity > 0.4:
            if type_confidence > 0.7:
                return ExchangeComplexity.MODERATE
            else:
                return ExchangeComplexity.COMPLEX
        
        # Low complexity - simple exchange
        return ExchangeComplexity.SIMPLE
    
    def _assess_simple_complexity(self, email_lower: str, type_confidence: float) -> ExchangeComplexity:
        """Fallback complexity assessment if classifier not available"""
        
        # Check for simple issue indicators
        simple_count = sum(1 for indicator in self.simple_issue_indicators if indicator in email_lower)
        
        if simple_count > 0:
            return ExchangeComplexity.SIMPLE
        
        # Base on type confidence
        if type_confidence > 0.8:
            return ExchangeComplexity.SIMPLE
        elif type_confidence > 0.6:
            return ExchangeComplexity.MODERATE
        else:
            return ExchangeComplexity.COMPLEX
    
    def _calculate_overall_confidence(self, intent_score: float, type_confidence: float,
                                    complexity_level: ExchangeComplexity, 
                                    detected_issues_count: int) -> float:
        """Calculate overall confidence in exchange detection"""
        
        # Base confidence from intent and type
        base_confidence = (intent_score * 0.4) + (type_confidence * 0.6)
        
        # Complexity adjustment
        complexity_multiplier = {
            ExchangeComplexity.SIMPLE: 1.0,
            ExchangeComplexity.MODERATE: 0.8,
            ExchangeComplexity.COMPLEX: 0.5
        }
        
        adjusted_confidence = base_confidence * complexity_multiplier[complexity_level]
        
        # Boost for multiple issue types detected
        if detected_issues_count > 1:
            adjusted_confidence += 0.1
        
        return min(adjusted_confidence, 1.0)
    
    def _select_response_template(self, exchange_type: ExchangeType, 
                                 complexity: ExchangeComplexity) -> Optional[str]:
        """Select appropriate response template"""
        
        # Don't auto-respond to complex issues
        if complexity == ExchangeComplexity.COMPLEX:
            return None
        
        template_mapping = {
            ExchangeType.SIZE_EXCHANGE: 'size_exchange',
            ExchangeType.COLOR_STYLE: 'color_style',
            ExchangeType.PRODUCT_MISMATCH: 'product_mismatch',
            ExchangeType.GENERAL_RETURN: 'general_return'
        }
        
        return template_mapping.get(exchange_type)
    
    def _generate_final_reasoning(self, is_exchange: bool, exchange_type: ExchangeType,
                                 complexity: ExchangeComplexity, confidence: float,
                                 reasoning: List[str], detected_issues: List[str]) -> List[str]:
        """Generate final reasoning for the decision"""
        
        final_reasoning = []
        
        if is_exchange:
            final_reasoning.append("EXCHANGE REQUEST DETECTED - AI can handle automatically")
            final_reasoning.append(f"Type: {exchange_type.value.replace('_', ' ').title()}")
            final_reasoning.append(f"Complexity: {complexity.value.title()}")
            final_reasoning.append(f"Confidence: {confidence:.2f}")
            if detected_issues:
                final_reasoning.append(f"Issues: {', '.join(detected_issues)}")
        else:
            final_reasoning.append("NOT AN EXCHANGE REQUEST - Continue normal processing")
            final_reasoning.append(f"Low confidence ({confidence:.2f}) or too complex")
        
        # Add key detection reasoning (limit to top 3)
        final_reasoning.extend(reasoning[:3])
        
        return final_reasoning
    
    def _create_negative_result(self, reasoning: List[str], 
                               classifier_complexity: float = None) -> ExchangeDetectionResult:
        """Create result for non-exchange emails"""
        
        reasoning.append("No exchange/return intent detected - processing normally")
        
        return ExchangeDetectionResult(
            is_exchange_request=False,
            exchange_type=ExchangeType.NOT_EXCHANGE,
            complexity=ExchangeComplexity.SIMPLE,
            confidence=0.0,
            detected_issues=[],
            keywords_matched=[],
            reasoning=reasoning,
            suggested_response_template=None,
            classifier_complexity=classifier_complexity or 0.0
        )
    
    def format_response(self, template_key: str, customer_name: str = "Valued Customer", 
                       **kwargs) -> str:
        """Format response template with customer details"""
        
        if template_key not in self.response_templates:
            return self.response_templates['general_return'].format(
                customer_name=customer_name, **kwargs
            )
        
        return self.response_templates[template_key].format(
            customer_name=customer_name, **kwargs
        )