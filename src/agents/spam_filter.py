"""
Smart Spam and Promotional Email Filter

This module provides sophisticated filtering for emails that don't require human responses:
- Newsletter subscriptions/confirmations  
- Marketing and promotional content
- System test emails
- Survey requests
- Webinar invitations
- Unsubscribe confirmations

Key Innovation: This runs BEFORE escalation logic to prevent promotional content 
from being incorrectly escalated to human agents.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import re
import string


class EmailDisposition(Enum):
    """How the email should be handled"""
    PROCESS_NORMALLY = "process_normally"      # Continue with normal AI pipeline
    AUTO_ACKNOWLEDGE = "auto_acknowledge"      # Send auto-acknowledgment
    AUTO_RESPOND = "auto_respond"             # Send template response
    DISCARD = "discard"                       # Silently discard (spam)
    UNSUBSCRIBE = "unsubscribe"               # Handle unsubscribe request


class SpamConfidence(Enum):
    """Confidence levels for spam/promotional detection"""
    DEFINITELY_SPAM = "definitely_spam"       # 95%+ confidence
    LIKELY_SPAM = "likely_spam"               # 80%+ confidence  
    POSSIBLY_SPAM = "possibly_spam"           # 60%+ confidence
    UNLIKELY_SPAM = "unlikely_spam"           # 40%+ confidence
    NOT_SPAM = "not_spam"                     # < 40% confidence


@dataclass
class FilterResult:
    """Result of spam/promotional filtering"""
    disposition: EmailDisposition
    confidence: SpamConfidence
    spam_score: float                         # 0.0 = definitely not spam, 1.0 = definitely spam
    detected_types: List[str]                 # Types of spam/promotional content detected
    auto_response_template: Optional[str]     # Template to use for auto-response
    reasoning: List[str]                      # Why this decision was made


class SmartSpamFilter:
    """
    Advanced spam and promotional email filter that identifies emails
    that don't need human intervention before they reach escalation logic.
    """
    
    def __init__(self):
        """Initialize the spam filter with detection patterns"""
        
        # Newsletter indicators (high confidence spam/promotional)
        self.newsletter_indicators = {
            'subject_patterns': [
                r'newsletter', r'weekly update', r'monthly digest',
                r'welcome to.*newsletter', r'subscription confirmed',
                r'thanks for subscribing', r'you\'re subscribed'
            ],
            'body_patterns': [
                r'thanks for subscribing', r'subscription.*confirmed',
                r'welcome to our newsletter', r'monthly newsletter',
                r'weekly update', r'you subscribed to',
                r'manage your subscription'
            ],
            'keywords': [
                'newsletter', 'subscription', 'unsubscribe', 'digest',
                'weekly update', 'monthly report', 'subscriber'
            ]
        }
        
        # Marketing and promotional indicators
        self.marketing_indicators = {
            'subject_patterns': [
                r'\d+%\s+off', r'special offer', r'limited time',
                r'sale ends', r'exclusive deal', r'don\'t miss',
                r'free shipping', r'buy now'
            ],
            'body_patterns': [
                r'\d+%\s+off', r'limited time offer', r'act now',
                r'special promotion', r'exclusive deal',
                r'click here to buy', r'shop now', r'get started today'
            ],
            'keywords': [
                'promotion', 'discount', 'sale', 'offer', 'deal',
                'coupon', 'savings', 'free shipping', 'buy now'
            ]
        }
        
        # Survey and feedback request indicators
        self.survey_indicators = {
            'subject_patterns': [
                r'feedback', r'survey', r'your opinion', r'rate us',
                r'tell us what you think', r'customer satisfaction'
            ],
            'body_patterns': [
                r'take.*survey', r'fill out.*survey', r'feedback survey',
                r'customer satisfaction survey', r'rate your experience',
                r'please take a moment', r'your opinion matters'
            ],
            'keywords': [
                'survey', 'feedback', 'opinion', 'rating', 'satisfaction',
                'questionnaire', 'poll', 'review'
            ]
        }
        
        # Webinar and event invitation indicators  
        self.event_indicators = {
            'subject_patterns': [
                r'webinar', r'invitation.*webinar', r'join us',
                r'upcoming event', r'you\'re invited', r'register now'
            ],
            'body_patterns': [
                r'join us.*webinar', r'webinar.*strategies',
                r'register now.*webinar', r'upcoming webinar',
                r'you\'re invited.*webinar', r'exclusive webinar'
            ],
            'keywords': [
                'webinar', 'event', 'invitation', 'register', 'join us',
                'seminar', 'workshop', 'conference'
            ]
        }
        
        # System test email indicators
        self.test_indicators = {
            'subject_patterns': [
                r'test', r'system test', r'test email', r'testing'
            ],
            'body_patterns': [
                r'this is a test', r'test email', r'system test',
                r'please disregard', r'testing.*system',
                r'verify.*functionality', r'email.*test'
            ],
            'keywords': [
                'test', 'testing', 'disregard', 'ignore', 'system test'
            ]
        }
        
        # Unsubscribe request indicators
        self.unsubscribe_indicators = {
            'subject_patterns': [
                r'unsubscribe', r'remove.*list', r'stop emails',
                r'opt out', r'cancel subscription'
            ],
            'body_patterns': [
                r'unsubscribe.*from', r'remove.*from.*list',
                r'stop sending', r'cancel.*subscription',
                r'opt out', r'no longer interested'
            ],
            'keywords': [
                'unsubscribe', 'remove', 'opt out', 'cancel',
                'stop sending', 'no longer want'
            ]
        }
        
        # Definite spam indicators (high confidence)
        self.spam_indicators = {
            'subject_patterns': [
                r'you.*won', r'congratulations.*winner', r'claim.*prize',
                r'urgent.*action', r'verify.*account.*immediately',
                r'suspended.*account', r'click.*immediately'
            ],
            'body_patterns': [
                r'you have won', r'claim your prize', r'winner.*lottery',
                r'urgent.*verify', r'account.*suspended',
                r'click here immediately', r'act now or lose'
            ],
            'keywords': [
                'winner', 'lottery', 'prize', 'urgent verification',
                'account suspended', 'click immediately', 'act now'
            ]
        }
        
        # Auto-response templates for different types
        self.auto_response_templates = {
            'newsletter': """Thank you for your newsletter subscription. We've received your confirmation and you're now subscribed to our updates.
            
If you have any questions about your subscription, please don't hesitate to contact us.

Best regards,
Customer Support Team""",
            
            'survey': """Thank you for your feedback request. We value your opinion and will consider your survey invitation.

For immediate support needs, please contact our customer service team directly.

Best regards,
Customer Support Team""",
            
            'event': """Thank you for the event invitation. We've received your webinar/event information.

For business inquiries or partnership discussions, please contact our business development team.

Best regards,
Customer Support Team""",
            
            'marketing': """Thank you for your marketing communication. We've received your promotional information.

For business partnerships or advertising inquiries, please contact our business development team.

Best regards,
Customer Support Team"""
        }
    
    def filter_email(self, email_content: str, subject: str = "") -> FilterResult:
        """
        Main filtering method - determines if email is spam/promotional
        and how it should be handled.
        
        Args:
            email_content: The email body content
            subject: Email subject line (optional)
            
        Returns:
            FilterResult with disposition and reasoning
        """
        
        email_lower = email_content.lower().strip()
        subject_lower = subject.lower().strip()
        combined_text = f"{subject_lower} {email_lower}"
        
        # CRITICAL: Check for business-critical issues that should NEVER be spam filtered
        if self._is_business_critical_issue(combined_text):
            return FilterResult(
                disposition=EmailDisposition.PROCESS_NORMALLY,
                confidence=SpamConfidence.NOT_SPAM,
                spam_score=0.0,
                detected_types=['business_critical'],
                auto_response_template=None,
                reasoning=['BUSINESS CRITICAL ISSUE - Bypassing spam filter for immediate processing']
            )
        
        # Run all detection methods
        detection_results = {
            'newsletter': self._detect_newsletter(combined_text, email_lower),
            'marketing': self._detect_marketing(combined_text, email_lower),
            'survey': self._detect_survey(combined_text, email_lower),
            'event': self._detect_event(combined_text, email_lower),
            'test': self._detect_test_email(combined_text, email_lower),
            'unsubscribe': self._detect_unsubscribe(combined_text, email_lower),
            'spam': self._detect_spam(combined_text, email_lower)
        }
        
        # Calculate overall spam score
        spam_score = self._calculate_spam_score(detection_results)
        
        # Determine disposition based on detection results
        disposition = self._determine_disposition(detection_results, spam_score)
        
        # Determine confidence level
        confidence = self._determine_confidence(spam_score, detection_results)
        
        # Get detected types
        detected_types = [type_name for type_name, score in detection_results.items() if score > 0.3]
        
        # Get auto-response template if applicable
        auto_response_template = self._get_auto_response_template(disposition, detected_types)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(disposition, detected_types, spam_score, detection_results)
        
        return FilterResult(
            disposition=disposition,
            confidence=confidence,
            spam_score=spam_score,
            detected_types=detected_types,
            auto_response_template=auto_response_template,
            reasoning=reasoning
        )
    
    def _detect_newsletter(self, combined_text: str, email_body: str) -> float:
        """Detect newsletter/subscription emails"""
        score = 0.0
        
        # Subject patterns
        for pattern in self.newsletter_indicators['subject_patterns']:
            if re.search(pattern, combined_text, re.IGNORECASE):
                score += 0.3
        
        # Body patterns
        for pattern in self.newsletter_indicators['body_patterns']:
            if re.search(pattern, email_body, re.IGNORECASE):
                score += 0.2
        
        # Keywords
        keyword_matches = sum(1 for keyword in self.newsletter_indicators['keywords'] 
                             if keyword in combined_text)
        score += min(keyword_matches * 0.1, 0.4)
        
        # Additional newsletter indicators
        if 'unsubscribe' in email_body:
            score += 0.2
        if re.search(r'manage.*subscription|subscription.*preferences', email_body):
            score += 0.3
        
        return min(score, 1.0)
    
    def _detect_marketing(self, combined_text: str, email_body: str) -> float:
        """Detect marketing/promotional emails"""
        score = 0.0
        
        # Subject patterns
        for pattern in self.marketing_indicators['subject_patterns']:
            if re.search(pattern, combined_text, re.IGNORECASE):
                score += 0.3
        
        # Body patterns  
        for pattern in self.marketing_indicators['body_patterns']:
            if re.search(pattern, email_body, re.IGNORECASE):
                score += 0.2
        
        # Keywords
        keyword_matches = sum(1 for keyword in self.marketing_indicators['keywords']
                             if keyword in combined_text)
        score += min(keyword_matches * 0.1, 0.4)
        
        # Marketing structure indicators
        if re.search(r'\d+%\s+off|\d+%\s+discount', combined_text):
            score += 0.3
        if re.search(r'limited time|act now|expires', combined_text):
            score += 0.2
        if 'call to action' in email_body.lower() or 'click here' in email_body.lower():
            score += 0.1
        
        return min(score, 1.0)
    
    def _detect_survey(self, combined_text: str, email_body: str) -> float:
        """Detect survey/feedback request emails"""
        score = 0.0
        
        # Subject patterns
        for pattern in self.survey_indicators['subject_patterns']:
            if re.search(pattern, combined_text, re.IGNORECASE):
                score += 0.3
        
        # Body patterns
        for pattern in self.survey_indicators['body_patterns']:
            if re.search(pattern, email_body, re.IGNORECASE):
                score += 0.2
        
        # Keywords
        keyword_matches = sum(1 for keyword in self.survey_indicators['keywords']
                             if keyword in combined_text)
        score += min(keyword_matches * 0.15, 0.4)
        
        # Survey-specific patterns
        if re.search(r'take.*moment.*fill', email_body, re.IGNORECASE):
            score += 0.2
        if 'your opinion' in email_body.lower():
            score += 0.2
            
        return min(score, 1.0)
    
    def _detect_event(self, combined_text: str, email_body: str) -> float:
        """Detect webinar/event invitation emails"""
        score = 0.0
        
        # Subject patterns
        for pattern in self.event_indicators['subject_patterns']:
            if re.search(pattern, combined_text, re.IGNORECASE):
                score += 0.3
        
        # Body patterns
        for pattern in self.event_indicators['body_patterns']:
            if re.search(pattern, email_body, re.IGNORECASE):
                score += 0.2
        
        # Keywords
        keyword_matches = sum(1 for keyword in self.event_indicators['keywords']
                             if keyword in combined_text)
        score += min(keyword_matches * 0.15, 0.4)
        
        # Event-specific indicators
        if 'register now' in email_body.lower():
            score += 0.2
        if re.search(r'join us.*friday|this friday', email_body, re.IGNORECASE):
            score += 0.2
        if 'marketing strategies' in email_body.lower():
            score += 0.1
            
        return min(score, 1.0)
    
    def _detect_test_email(self, combined_text: str, email_body: str) -> float:
        """Detect system test emails"""
        score = 0.0
        
        # Subject patterns
        for pattern in self.test_indicators['subject_patterns']:
            if re.search(pattern, combined_text, re.IGNORECASE):
                score += 0.4
        
        # Body patterns - test emails have very specific language
        for pattern in self.test_indicators['body_patterns']:
            if re.search(pattern, email_body, re.IGNORECASE):
                score += 0.3
        
        # Keywords
        keyword_matches = sum(1 for keyword in self.test_indicators['keywords']
                             if keyword in combined_text)
        score += min(keyword_matches * 0.2, 0.4)
        
        # Very specific test email phrases
        if 'please disregard' in email_body.lower():
            score += 0.4  # Strong indicator
        if 'verify the functionality' in email_body.lower():
            score += 0.3
            
        return min(score, 1.0)
    
    def _detect_unsubscribe(self, combined_text: str, email_body: str) -> float:
        """Detect unsubscribe requests"""
        score = 0.0
        
        # Subject patterns
        for pattern in self.unsubscribe_indicators['subject_patterns']:
            if re.search(pattern, combined_text, re.IGNORECASE):
                score += 0.4
        
        # Body patterns
        for pattern in self.unsubscribe_indicators['body_patterns']:
            if re.search(pattern, email_body, re.IGNORECASE):
                score += 0.3
        
        # Keywords
        keyword_matches = sum(1 for keyword in self.unsubscribe_indicators['keywords']
                             if keyword in combined_text)
        score += min(keyword_matches * 0.2, 0.4)
        
        return min(score, 1.0)
    
    def _detect_spam(self, combined_text: str, email_body: str) -> float:
        """Detect definite spam emails"""
        score = 0.0
        
        # Subject patterns
        for pattern in self.spam_indicators['subject_patterns']:
            if re.search(pattern, combined_text, re.IGNORECASE):
                score += 0.4
        
        # Body patterns
        for pattern in self.spam_indicators['body_patterns']:
            if re.search(pattern, email_body, re.IGNORECASE):
                score += 0.3
        
        # Keywords
        keyword_matches = sum(1 for keyword in self.spam_indicators['keywords']
                             if keyword in combined_text)
        score += min(keyword_matches * 0.2, 0.5)
        
        return min(score, 1.0)
    
    def _calculate_spam_score(self, detection_results: Dict[str, float]) -> float:
        """Calculate overall spam score from individual detections"""
        
        # Weights for different types of spam/promotional content
        weights = {
            'spam': 1.0,           # Definite spam gets highest weight
            'newsletter': 0.8,     # Newsletters are promotional but less spammy
            'marketing': 0.9,      # Marketing is promotional
            'survey': 0.6,         # Surveys are less spammy
            'event': 0.7,          # Events are promotional
            'test': 0.9,           # Test emails should be filtered
            'unsubscribe': 0.5     # Unsubscribe requests need handling
        }
        
        total_weighted_score = 0.0
        total_weights = 0.0
        
        for spam_type, score in detection_results.items():
            if score > 0:
                weight = weights.get(spam_type, 0.5)
                total_weighted_score += score * weight
                total_weights += weight
        
        if total_weights > 0:
            return min(total_weighted_score / total_weights, 1.0)
        
        return 0.0
    
    def _determine_disposition(self, detection_results: Dict[str, float], spam_score: float) -> EmailDisposition:
        """Determine how the email should be handled"""
        
        # Check for specific types first
        if detection_results.get('spam', 0) > 0.6:
            return EmailDisposition.DISCARD
        
        if detection_results.get('unsubscribe', 0) > 0.5:
            return EmailDisposition.UNSUBSCRIBE
        
        if detection_results.get('test', 0) > 0.7:
            return EmailDisposition.DISCARD  # Test emails should be discarded
        
        # For promotional content, decide between auto-respond and discard
        if spam_score > 0.7:
            # High confidence promotional - auto-respond
            if (detection_results.get('newsletter', 0) > 0.5 or 
                detection_results.get('survey', 0) > 0.5 or
                detection_results.get('event', 0) > 0.5):
                return EmailDisposition.AUTO_RESPOND
            else:
                return EmailDisposition.DISCARD  # Marketing/spam
        
        if spam_score > 0.4:
            # Medium confidence promotional - auto-acknowledge
            return EmailDisposition.AUTO_ACKNOWLEDGE
        
        # Low spam score - continue with normal processing
        return EmailDisposition.PROCESS_NORMALLY
    
    def _determine_confidence(self, spam_score: float, detection_results: Dict[str, float]) -> SpamConfidence:
        """Determine confidence level in spam detection"""
        
        max_detection_score = max(detection_results.values()) if detection_results else 0
        
        # High individual detection score = high confidence
        if max_detection_score > 0.8 and spam_score > 0.7:
            return SpamConfidence.DEFINITELY_SPAM
        
        if spam_score > 0.6:
            return SpamConfidence.LIKELY_SPAM
        
        if spam_score > 0.4:
            return SpamConfidence.POSSIBLY_SPAM
        
        if spam_score > 0.2:
            return SpamConfidence.UNLIKELY_SPAM
        
        return SpamConfidence.NOT_SPAM
    
    def _get_auto_response_template(self, disposition: EmailDisposition, detected_types: List[str]) -> Optional[str]:
        """Get appropriate auto-response template"""
        
        if disposition not in [EmailDisposition.AUTO_RESPOND, EmailDisposition.AUTO_ACKNOWLEDGE]:
            return None
        
        # Priority order for template selection
        if 'newsletter' in detected_types:
            return self.auto_response_templates['newsletter']
        elif 'survey' in detected_types:
            return self.auto_response_templates['survey']
        elif 'event' in detected_types:
            return self.auto_response_templates['event']
        elif 'marketing' in detected_types:
            return self.auto_response_templates['marketing']
        
        # Default acknowledgment
        return """Thank you for your email. We've received your message and will review it accordingly.

For immediate assistance, please contact our customer support team.

Best regards,
Customer Support Team"""
    
    def _generate_reasoning(self, disposition: EmailDisposition, detected_types: List[str], 
                          spam_score: float, detection_results: Dict[str, float]) -> List[str]:
        """Generate human-readable reasoning"""
        
        reasoning = []
        
        # Primary decision
        if disposition == EmailDisposition.DISCARD:
            reasoning.append("EMAIL DISCARDED - Identified as spam or test content")
        elif disposition == EmailDisposition.AUTO_RESPOND:
            reasoning.append("AUTO-RESPONSE SENT - Promotional/informational content")
        elif disposition == EmailDisposition.AUTO_ACKNOWLEDGE:
            reasoning.append("AUTO-ACKNOWLEDGMENT SENT - Likely promotional content")
        elif disposition == EmailDisposition.UNSUBSCRIBE:
            reasoning.append("UNSUBSCRIBE REQUEST - Processing removal")
        else:
            reasoning.append("NORMAL PROCESSING - Not identified as promotional content")
        
        # Detection details
        if detected_types:
            detected_str = ', '.join(detected_types)
            reasoning.append(f"Detected content types: {detected_str}")
        
        # Score context
        if spam_score > 0.7:
            reasoning.append(f"High promotional score: {spam_score:.2f}")
        elif spam_score > 0.4:
            reasoning.append(f"Moderate promotional score: {spam_score:.2f}")
        else:
            reasoning.append(f"Low promotional score: {spam_score:.2f}")
        
        # Strongest detection
        if detection_results:
            strongest_type = max(detection_results.items(), key=lambda x: x[1])
            if strongest_type[1] > 0.5:
                reasoning.append(f"Strongest match: {strongest_type[0]} ({strongest_type[1]:.2f})")
        
        return reasoning
    
    def _is_business_critical_issue(self, combined_text: str) -> bool:
        """
        Detect business-critical issues that should NEVER be spam filtered.
        These issues need immediate escalation and processing.
        """
        
        # Security and breach indicators
        security_patterns = [
            'security breach', 'hacked', 'unauthorized access', 'security concern',
            'suspicious activity', 'fraud alert', 'account compromised',
            'data breach', 'security vulnerability', 'identity theft'
        ]
        
        # System outage and critical failures
        system_critical_patterns = [
            'website down', 'site down', 'system down', 'service unavailable',
            'offline all morning', 'down all day', 'trying all day', 'hours trying',
            'can\'t place order', 'website has been', 'system outage', 'service disruption',
            'critical system error', 'system failure', 'database down'
        ]
        
        # Financial and billing critical issues
        financial_critical_patterns = [
            'payment failed but money deducted', 'charged twice', 'double charged',
            'billing error', 'wrong amount charged', 'payment went through but',
            'money deducted', 'duplicate charge', 'billing mistake'
        ]
        
        # Check for any business-critical patterns
        all_critical_patterns = security_patterns + system_critical_patterns + financial_critical_patterns
        
        for pattern in all_critical_patterns:
            if pattern in combined_text:
                return True
                
        return False