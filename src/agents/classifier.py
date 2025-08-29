"""
Custom Email Classification System - CORE LOGIC

THIS IS A CORE INNOVATION FILE - IMPLEMENT YOUR CUSTOM CLASSIFICATION ALGORITHM HERE

Key Requirements:
1. Create sophisticated multi-factor email classification (NOT just LLM calls)
2. Implement weighted keyword analysis
3. Add pattern matching and sentiment analysis  
4. Create confidence calculation based on multiple factors
5. Include historical accuracy tracking

This should showcase your algorithmic thinking and problem-solving skills.
"""

# TODO: IMPLEMENT YOUR CUSTOM CLASSIFICATION ALGORITHM HERE
# This is where you demonstrate innovation beyond simple LLM calls

# CORE LOGIC - IMPLEMENT MANUALLY

from dataclasses import dataclass
from typing import Dict, List
import re
import string

@dataclass
class CategoryProfile:
    name: str
    keywords: Dict[str, float]
    patterns: List[str]
    structural_boosts: Dict[str, float]
    sentiment_preference: float
    complexity_thresehold: float

  # Primary Business Categories
CUSTOMER_SUPPORT = CategoryProfile(
    name="customer_support",
    keywords={
        'support': 1.0, 'help': 1.0, 'question': 0.9, 'issue': 0.9,
        'problem': 0.8, 'assistance': 0.8, 'inquiry': 0.7, 'complaint':0.7,
        'feedback': 0.6, 'troubleshoot': 0.8
    },
    patterns=[
        r'need\s+help', r'customer\s+service', r'can\s+you\s+help',
        r'have\s+a\s+question', r'support\s+ticket'
    ],
    structural_boosts={'has_question_mark': 0.2, 'has_polite_language':0.1},
    sentiment_preference=0.0,
    complexity_thresehold=0.6
)

SALES_ORDER = CategoryProfile(
    name="sales_order",
    keywords={
        'order': 1.0, 'purchase': 1.0, 'buy': 0.9, 'invoice': 0.9,
        'quote': 0.8, 'product': 0.7, 'delivery': 0.8, 'shipment': 0.8,     
        'payment': 0.8, 'confirmation': 0.7
    },
    patterns=[
        r'order\s*#?\s*\d+', r'purchase\s+order', r'buy\s+\d+',
        r'invoice\s*#?\s*\d+', r'quote\s+for'
    ],
    structural_boosts={'has_order_number': 0.3, 'has_currency': 0.2,'has_quantity': 0.1},
    sentiment_preference=0.1,
    complexity_thresehold=0.7
)

MARKETING_PROMOTIONS = CategoryProfile(
    name="marketing_promotions",
    keywords={
        'promotion': 1.0, 'offer': 1.0, 'discount': 0.9, 'sale': 0.9,       
        'deal': 0.8, 'newsletter': 0.7, 'campaign': 0.8, 'special':
0.7,
        'coupon': 0.8, 'announcement': 0.6
    },
    patterns=[
        r'\d+%\s+off', r'special\s+offer', r'limited\s+time',
        r'newsletter', r'unsubscribe'
    ],
    structural_boosts={'has_percentage': 0.2, 'has_unsubscribe': 0.3,       
'has_cta': 0.1},
    sentiment_preference=0.3,
    complexity_thresehold=0.8
)

BILLING_FINANCE = CategoryProfile(
    name="billing_finance",
    keywords={
        'invoice': 1.0, 'billing': 1.0, 'payment': 1.0, 'receipt': 0.9,     
        'refund': 0.9, 'charge': 0.8, 'account': 0.7, 'statement': 0.8,     
        'balance': 0.8, 'transaction': 0.7
    },
    patterns=[
        r'invoice\s*#?\s*\d+', r'payment\s+(failed|declined)', r'refund\s+request',
        r'billing\s+statement', r'account\s+balance'
    ],
    structural_boosts={'has_invoice_number': 0.3, 'has_currency': 0.25,     
'has_date': 0.1},
    sentiment_preference=-0.1,
    complexity_thresehold=0.5
)

# Administrative Categories
INTERNAL_COMMUNICATION = CategoryProfile(
    name="internal_communication",
    keywords={
        'team': 1.0, 'update': 0.9, 'internal': 1.0, 'announcement':        
0.8,
        'policy': 0.8, 'hr': 0.9, 'colleague': 0.7, 'staff': 0.8,
        'employee': 0.8, 'department': 0.7
    },
    patterns=[
        r'team\s+meeting', r'internal\s+use', r'hr\s+policy',
        r'staff\s+announcement', r'department\s+update'
    ],
    structural_boosts={'has_internal_domain': 0.3, 'has_cc_multiple':       
0.2},
    sentiment_preference=0.0,
    complexity_thresehold=0.7
)

VENDOR_SUPPLIER = CategoryProfile(
    name="vendor_supplier",
    keywords={
        'vendor': 1.0, 'supplier': 1.0, 'procurement': 0.9, 'purchase order': 1.0,
        'rfq': 0.9, 'quotation': 0.9, 'contract': 0.8, 'agreement':
0.8,
        'delivery note': 0.8, 'invoice': 0.7
    },
    patterns=[
        r'purchase\s+order\s*#?\s*\d+', r'rfq\s*#?\s*\d+',
r'vendor\s+agreement',
        r'supplier\s+contract', r'delivery\s+note'
    ],
    structural_boosts={'has_po_number': 0.3, 'has_vendor_id': 0.2,
'has_contract_terms': 0.1},
    sentiment_preference=0.0,
    complexity_thresehold=0.6
)

LEGAL_COMPLIANCE = CategoryProfile(
    name="legal_compliance",
    keywords={
        'legal': 1.0, 'compliance': 1.0, 'contract': 0.9, 'agreement':      
0.9,
        'regulation': 0.8, 'policy': 0.7, 'terms': 0.8, 'conditions':       
0.8,
        'notice': 0.7, 'disclosure': 0.8
    },
    patterns=[
        r'legal\s+notice', r'compliance\s+requirement',
r'terms\s+and\s+conditions',
        r'legal\s+agreement', r'regulatory\s+compliance'
    ],
    structural_boosts={'has_legal_language': 0.3,
'has_signature_block': 0.2},
    sentiment_preference=0.0,
    complexity_thresehold=0.3  # Low threshold - escalate complex legal issues
)

TECHNICAL_IT = CategoryProfile(
    name="technical_it",
    keywords={
        'technical': 1.0, 'it': 1.0, 'system': 0.9, 'password': 0.9,        
        'reset': 0.8, 'error': 1.0, 'bug': 1.0, 'server': 0.9,
        'login': 0.9, 'support': 0.7
    },
    patterns=[
        r'error\s+(code|message)', r'system\s+(down|error)',
r'password\s+reset',
        r'server\s+(error|down)', r'login\s+(issue|problem)'
    ],
    structural_boosts={'has_error_code': 0.3, 'has_system_info': 0.2,       
'has_urgency': 0.1},
    sentiment_preference=-0.2,
    complexity_thresehold=0.4
)

# Specialized Categories
URGENT_HIGH_PRIORITY = CategoryProfile(
    name="urgent_high_priority",
    keywords={
        'urgent': 1.0, 'immediate': 1.0, 'asap': 1.0, 'important': 0.9,     
        'priority': 0.9, 'critical': 1.0, 'alert': 0.9, 'emergency':        
1.0,
        'action required': 1.0, 'respond': 0.7
    },
    patterns=[
        r'urgent.*please', r'asap|a\.s\.a\.p', r'emergency',
        r'critical.*issue', r'immediate.*attention'
    ],
    structural_boosts={'has_exclamation': 0.3, 'has_caps': 0.2,
'has_time_pressure': 0.2},
    sentiment_preference=-0.4,
    complexity_thresehold=0.3
)

SPAM_JUNK = CategoryProfile(
    name="spam_junk",
    keywords={
        'unsubscribe': 0.9, 'winner': 1.0, 'free': 0.8, 'prize': 1.0,       
        'lottery': 1.0, 'click here': 0.9, 'guaranteed': 0.8,
'risk-free': 0.8,
        'act now': 0.9, 'limited time': 0.7
    },
    patterns=[
        r'click\s+here', r'act\s+now', r'limited\s+time\s+offer',
        r'you\'ve\s+won', r'congratulations.*winner'
    ],
    structural_boosts={'has_suspicious_links': 0.4, 'has_all_caps':
0.3, 'has_multiple_exclamations': 0.2},
    sentiment_preference=0.5,  # Overly positive
    complexity_thresehold=0.9
)

NEWS_INFORMATION = CategoryProfile(
    name="news_information",
    keywords={
        'news': 1.0, 'update': 0.8, 'information': 0.9, 'report': 0.9,      
        'article': 0.9, 'newsletter': 0.8, 'insight': 0.7, 'trend':
0.7,
        'analysis': 0.8, 'brief': 0.7
    },
    patterns=[
        r'news\s+update', r'weekly\s+report', r'newsletter',
        r'industry\s+news', r'market\s+analysis'
    ],
    structural_boosts={'has_newsletter_format': 0.2, 'has_unsubscribe':     
0.1, 'has_links': 0.1},
    sentiment_preference=0.0,
    complexity_thresehold=0.8
)

SOCIAL_PERSONAL = CategoryProfile(
    name="social_personal",
    keywords={
        'party': 1.0, 'invitation': 1.0, 'congratulations': 0.9,
'birthday': 1.0,
        'wedding': 1.0, 'gathering': 0.8, 'celebration': 0.9, 'friend':     
0.7,
        'family': 0.7, 'social': 0.8
    },
    patterns=[
        r'you\'re\s+invited', r'birthday\s+party',
r'wedding\s+invitation',
        r'congratulations\s+on', r'celebration\s+for'
    ],
    structural_boosts={'has_rsvp': 0.3, 'has_date_time': 0.2,
'has_location': 0.1},
    sentiment_preference=0.6,
    complexity_thresehold=0.8
)

# Action-Based Categories
ACTION_REQUIRED = CategoryProfile(
    name="action_required",
    keywords={
        'action required': 1.0, 'please respond': 1.0, 'reply needed':      
1.0,
        'follow up': 0.8, 'pending': 0.8, 'awaiting': 0.8, 'reminder':      
0.9,
        'due': 0.9, 'deadline': 1.0, 'response needed': 1.0
    },
    patterns=[
        r'action\s+required', r'please\s+respond', r'reply\s+needed',       
        r'deadline.*\d+', r'due\s+by'
    ],
    structural_boosts={'has_deadline': 0.3, 'has_call_to_action': 0.2,      
'has_urgency': 0.2},
    sentiment_preference=-0.1,
    complexity_thresehold=0.5
)

FYI_INFORMATIONAL = CategoryProfile(
    name="fyi_informational",
    keywords={
        'fyi': 1.0, 'for your information': 1.0, 'just so you know':        
1.0,
        'no action needed': 1.0, 'update': 0.7, 'informational': 1.0,       
        'notification': 0.8, 'heads up': 0.9, 'reference': 0.7, 'note':     
0.6
    },
    patterns=[
        r'fyi', r'for\s+your\s+information', r'no\s+action\s+needed',       
        r'just\s+so\s+you\s+know', r'heads\s+up'
    ],
    structural_boosts={'has_fyi_marker': 0.3, 'has_no_questions': 0.2,      
'has_info_only': 0.1},
    sentiment_preference=0.0,
    complexity_thresehold=0.9
)

FOLLOW_UP = CategoryProfile(
    name="follow_up",
    keywords={
        'follow up': 1.0, 'reminder': 1.0, 'pending': 0.9, 'awaiting response': 1.0,
        'second request': 1.0, 'checking in': 0.9, 'update on': 0.8,        
        'status': 0.7, 'progress': 0.8, 'any news': 0.8
    },
    patterns=[
        r'follow\s+up\s+on', r'checking\s+in\s+on', r'any\s+update',        
        r'second\s+request', r'still\s+waiting'
    ],
    structural_boosts={'has_previous_reference': 0.3, 'has_timeline':       
0.2, 'has_gentle_nudge': 0.1},
    sentiment_preference=-0.1,
    complexity_thresehold=0.6
)

# NEW CATEGORIES FOR IMPROVED ESCALATION LOGIC
CUSTOMER_PRAISE = CategoryProfile(
    name="customer_praise",
    keywords={
        'thank you': 1.0, 'thanks': 1.0, 'appreciate': 1.0, 'grateful': 1.0,
        'excellent': 0.9, 'amazing': 0.9, 'great work': 1.0, 'love': 0.9,
        'fantastic': 0.9, 'wonderful': 0.9, 'perfect': 0.8, 'impressed': 0.9,
        'outstanding': 0.9, 'kudos': 1.0, 'well done': 0.9, 'awesome': 0.8,
        'brilliant': 0.8, 'pleased': 0.8, 'delighted': 0.9, 'satisfied': 0.8
    },
    patterns=[
        r'thank\s+you\s+for', r'thanks\s+for', r'appreciate\s+(how|the|your)',
        r'love\s+(the|your)', r'great\s+work', r'well\s+done', r'keep\s+up',
        r'really\s+(appreciate|love)', r'just\s+wanted\s+to\s+(say|thank)'
    ],
    structural_boosts={'has_positive_sentiment': 0.4, 'has_exclamation': 0.2},
    sentiment_preference=0.8,  # Strongly positive sentiment expected
    complexity_thresehold=0.9  # Very high threshold - don't escalate praise
)

FEATURE_SUGGESTIONS = CategoryProfile(
    name="feature_suggestions",
    keywords={
        'suggestion': 1.0, 'suggest': 1.0, 'could you add': 1.0, 'feature request': 1.0,
        'improvement': 0.9, 'enhance': 0.8, 'would be nice': 0.9, 'consider adding': 0.9,
        'recommend': 0.8, 'idea': 0.8, 'feedback': 0.7, 'propose': 0.8,
        'dark mode': 0.9, 'update': 0.6, 'upgrade': 0.6, 'add support': 0.9,
        'implement': 0.7, 'include': 0.6, 'option to': 0.7, 'ability to': 0.8
    },
    patterns=[
        r'could\s+you\s+(add|implement)', r'would\s+be\s+(nice|great)\s+if',
        r'suggestion\s*:', r'feature\s+request', r'one\s+idea', r'small\s+suggestion',
        r'consider\s+(adding|implementing)', r'how\s+about\s+(adding|including)'
    ],
    structural_boosts={'has_question_marks': 0.2, 'has_suggestions': 0.3},
    sentiment_preference=0.3,  # Mildly positive sentiment expected
    complexity_thresehold=0.8  # High threshold - don't escalate simple suggestions
)

PARTNERSHIP_BUSINESS = CategoryProfile(
    name="partnership_business",
    keywords={
        'partnership': 1.0, 'collaborate': 0.9, 'business opportunity': 1.0,
        'partner': 0.8, 'cooperation': 0.9, 'business proposal': 1.0,
        'joint venture': 1.0, 'strategic': 0.7, 'alliance': 0.9,
        'integration': 0.7, 'api access': 0.8, 'business development': 1.0,
        'digital agency': 0.9, 'company': 0.6, 'organization': 0.6,
        'discuss': 0.7, 'explore': 0.7, 'interested in': 0.8, 'contact about': 0.8
    },
    patterns=[
        r'partnership\s+(opportunities|with)', r'business\s+(opportunity|proposal)',
        r'digital\s+agency', r'would\s+like\s+to\s+(discuss|explore)',
        r'interested\s+in\s+(partnership|collaborating)', r'potential\s+(partnership|collaboration)',
        r'who\s+should\s+I\s+contact', r'discuss\s+(potential|opportunities)'
    ],
    structural_boosts={'has_formal_language': 0.3, 'has_signature_block': 0.2},
    sentiment_preference=0.1,  # Neutral to slightly positive
    complexity_thresehold=0.7  # Moderate threshold - route to partnerships team
)

SUBSCRIPTION_MANAGEMENT = CategoryProfile(
    name="subscription_management",
    keywords={
        'cancel': 1.0, 'subscription': 1.0, 'unsubscribe': 1.0, 'monthly': 0.8,
        'billing': 0.8, 'plan': 0.7, 'account': 0.7, 'membership': 0.8,
        'terminate': 0.9, 'end': 0.6, 'stop': 0.7, 'pause': 0.8,
        'modify': 0.7, 'change': 0.6, 'downgrade': 0.8, 'upgrade': 0.7,
        'immediately': 0.7, 'effective': 0.8, 'confirm': 0.7
    },
    patterns=[
        r'cancel.*subscription', r'unsubscribe.*from', r'cancel.*monthly',
        r'end.*membership', r'terminate.*account', r'stop.*billing',
        r'please\s+(cancel|confirm)', r'would\s+like\s+to\s+cancel'
    ],
    structural_boosts={'has_account_info': 0.2, 'has_clear_request': 0.3},
    sentiment_preference=-0.1,  # Slightly negative (cancellation requests)
    complexity_thresehold=0.7  # Moderate threshold - can often be auto-handled
)


class EmailClassifier():
    def __init__(self):
        # Create list of all category profiles  
        category_profiles = [
            CUSTOMER_SUPPORT, SALES_ORDER, MARKETING_PROMOTIONS, BILLING_FINANCE,
            INTERNAL_COMMUNICATION, VENDOR_SUPPLIER, LEGAL_COMPLIANCE, TECHNICAL_IT,
            URGENT_HIGH_PRIORITY, SPAM_JUNK, NEWS_INFORMATION, SOCIAL_PERSONAL,
            ACTION_REQUIRED, FYI_INFORMATIONAL, FOLLOW_UP,
            # NEW CATEGORIES FOR IMPROVED ESCALATION LOGIC
            CUSTOMER_PRAISE, FEATURE_SUGGESTIONS, PARTNERSHIP_BUSINESS, SUBSCRIPTION_MANAGEMENT]
            
        # Create dictionary mapping uppercase names to profiles for test compatibility
        self.categories = {}
        category_mapping = {
            'customer_support': 'CUSTOMER_SUPPORT',
            'technical_it': 'TECHNICAL_ISSUE',  # Map technical_it to TECHNICAL_ISSUE for tests
            'billing_finance': 'BILLING_INQUIRY',  # Map billing_finance to BILLING_INQUIRY for tests
            'refund_request': 'REFUND_REQUEST',  # We don't have this profile
            'complaint_negative': 'COMPLAINT_NEGATIVE',  # We don't have this profile
            'sales_order': 'SALES_ORDER',
            'marketing_promotions': 'MARKETING_PROMOTIONS',
            'legal_compliance': 'LEGAL_COMPLIANCE',
            'press_media': 'PRESS_MEDIA',  # We don't have this profile
            'vip_customer': 'VIP_CUSTOMER',  # We don't have this profile
            'spam_junk': 'SPAM_PROMOTIONAL',  # Map spam_junk to SPAM_PROMOTIONAL for tests
            'news_information': 'INFORMATIONAL',  # Map news_information to INFORMATIONAL for tests
            'internal_communication': 'INTERNAL_COMMUNICATION',
            'vendor_supplier': 'VENDOR_SUPPLIER',
            'urgent_high_priority': 'URGENT_HIGH_PRIORITY',
            'social_personal': 'SOCIAL_PERSONAL',
            'action_required': 'ACTION_REQUIRED',
            'fyi_informational': 'FYI_INFORMATIONAL',
            'follow_up': 'FOLLOW_UP',
            # NEW CATEGORIES MAPPINGS
            'customer_praise': 'CUSTOMER_PRAISE',
            'feature_suggestions': 'FEATURE_SUGGESTIONS', 
            'partnership_business': 'PARTNERSHIP_BUSINESS',
            'subscription_management': 'SUBSCRIPTION_MANAGEMENT',
            'other': 'OTHER'  # Fallback category
        }
        
        for profile in category_profiles:
            uppercase_name = category_mapping.get(profile.name, profile.name.upper())
            self.categories[uppercase_name] = profile
            
        # Also store the list for internal processing
        self.category_profiles = category_profiles

    def classify(self, email: str) -> dict:
        """
        Sophisticated multi-factor email classification algroithm

        This helps sort email before calling an LLM
        """

        # Extract all features once
        structural_features = self.detect_structural_features(email)
        sentiment_score = self.analyze_sentiment(email)
        complexity_score = self.calculate_complexity(email)

        # Score ALL categories
        category_scores = {}
        for category_profile in self.category_profiles:
            total_score = self.calculate_comprehensive_score(
                email, category_profile, structural_features, sentiment_score, complexity_score)
            category_scores[category_profile.name] = total_score
        
        # Determine winner and confidence
        best_category_name = max(category_scores, key=category_scores.get)
        best_score = category_scores[best_category_name]

        # Map to expected uppercase format for backward compatibility
        category_mapping = {
            'customer_support': 'CUSTOMER_SUPPORT',
            'technical_it': 'TECHNICAL_ISSUE',  # Map technical_it to TECHNICAL_ISSUE for tests
            'billing_finance': 'BILLING_INQUIRY',  # Map billing_finance to BILLING_INQUIRY for tests
            'refund_request': 'REFUND_REQUEST',  # We don't have this profile
            'complaint_negative': 'COMPLAINT_NEGATIVE',  # We don't have this profile
            'sales_order': 'SALES_ORDER',
            'marketing_promotions': 'MARKETING_PROMOTIONS',
            'legal_compliance': 'LEGAL_COMPLIANCE',
            'press_media': 'PRESS_MEDIA',  # We don't have this profile
            'vip_customer': 'VIP_CUSTOMER',  # We don't have this profile
            'spam_junk': 'SPAM_PROMOTIONAL',  # Map spam_junk to SPAM_PROMOTIONAL for tests
            'news_information': 'INFORMATIONAL',  # Map news_information to INFORMATIONAL for tests
            'internal_communication': 'INTERNAL_COMMUNICATION',
            'vendor_supplier': 'VENDOR_SUPPLIER',
            'urgent_high_priority': 'URGENT_HIGH_PRIORITY',
            'social_personal': 'SOCIAL_PERSONAL',
            'action_required': 'ACTION_REQUIRED',
            'fyi_informational': 'FYI_INFORMATIONAL',
            'follow_up': 'FOLLOW_UP',
            # NEW CATEGORIES MAPPINGS
            'customer_praise': 'CUSTOMER_PRAISE',
            'feature_suggestions': 'FEATURE_SUGGESTIONS', 
            'partnership_business': 'PARTNERSHIP_BUSINESS',
            'subscription_management': 'SUBSCRIPTION_MANAGEMENT',
            'other': 'OTHER'  # Fallback category
        }
        
        display_category_name = category_mapping.get(best_category_name, best_category_name.upper())

        # Calculate confidence based on score seperation 
        confidence = self.calculate_confidence(category_scores, best_score)

        # Generate reasoning
        reasoning = self.generate_classification_reasoning(
            email, best_category_name, category_scores, structural_features
        )

        # Convert category_scores to uppercase format for tests
        display_category_scores = {}
        for cat_name, score in category_scores.items():
            display_name = category_mapping.get(cat_name, cat_name.upper())
            display_category_scores[display_name] = score

        return{
            'category': display_category_name,
            'confidence': confidence,
            'category_scores': display_category_scores,
            'structural_features': structural_features,
            'sentiment_score': sentiment_score,
            'complexity_score': complexity_score,
            'reasoning': reasoning
        }

    def calculate_comprehensive_score(self, email, category_profile, structural_features, sentiment_score, complexity_score):     
        """
        Multi-factor scoring algorithm - your "secret sauce"
        """

        total_score = 0.0

        # 1. KEYWORD MATCHING (40% weight)
        keyword_score = self.calculate_keyword_score(email, category_profile.keywords)
        total_score += keyword_score * 0.4

        # 2. PATTERN MATCHING (25% weight)
        pattern_score = self.calculate_pattern_score(email, category_profile.patterns)
        total_score += pattern_score * 0.25

        # 3. STRUCTURAL FEATURES (20% weight)
        structural_score = self.calculate_structural_score(
            structural_features, category_profile.structural_boosts
        )
        total_score += structural_score * 0.2

        # 4. SENTIMENT MATCHING (10% weight)
        sentiment_match_score = self.calculate_sentiment_match(
            sentiment_score, category_profile.sentiment_preference
        )
        total_score += sentiment_match_score * 0.1

        # 5. COMPLEXITY HANDLING (5% weight)
        complexity_match_score = self.calculate_complexity_match(
            complexity_score, category_profile.complexity_thresehold
        )
        total_score += complexity_match_score * 0.05

        # 6. SENTIMENT-AWARE CATEGORY BOOSTS (NEW)
        sentiment_boost = self.calculate_sentiment_boost(
            sentiment_score, category_profile.name, email
        )
        total_score += sentiment_boost

        return total_score

    def detect_structural_features(self, email: str) -> Dict[str, bool]:
        results = {}
        cleared_email = email.lower().strip()
        import re

        # Simple character checks
        results['has_question_mark'] = '?' in email
        results['has_exclamation'] = '!' in email
        results['has_multiple_exclamations'] = '!!!' in email or email.count('!') > 2
        
        # Check for ALL CAPS (significant portion of text in caps)
        words = email.split()
        if len(words) > 0:
            caps_words = sum(1 for word in words if word.isupper() and len(word) > 1)
            results['has_caps'] = caps_words > len(words) * 0.3  # >30% caps words
            results['has_all_caps'] = caps_words > len(words) * 0.7  # >70% caps words
        else:
            results['has_caps'] = False
            results['has_all_caps'] = False

        # Simple word checks  
        polite_words = ['please', 'thank', 'sorry', 'could', 'would', 'kindly']
        results['has_polite_language'] = any(word in cleared_email for word in polite_words)
        
        urgency_words = ['urgent', 'asap', 'immediate', 'critical', 'emergency', 'rush']
        results['has_urgency'] = any(word in cleared_email for word in urgency_words)
        
        time_pressure_words = ['deadline', 'due by', 'expires', 'limited time', 'hurry']
        results['has_time_pressure'] = any(phrase in cleared_email for phrase in time_pressure_words)

        # Email structure checks
        results['has_no_questions'] = '?' not in email
        results['has_fyi_marker'] = any(marker in cleared_email for marker in ['fyi', 'for your information', 'heads up'])
        results['has_info_only'] = any(phrase in cleared_email for phrase in ['no action needed', 'for your information', 'fyi'])

        # Regex pattern checks - Numbers and IDs
        results['has_order_number'] = bool(re.search(r'order\s*#?\s*\d+', cleared_email))
        results['has_invoice_number'] = bool(re.search(r'invoice\s*#?\s*\d+', cleared_email))  
        results['has_tracking_number'] = bool(re.search(r'tracking\s*#?\s*\w+', cleared_email))
        results['has_po_number'] = bool(re.search(r'(purchase\s+order|po)\s*#?\s*\d+', cleared_email))
        results['has_vendor_id'] = bool(re.search(r'vendor\s*(id|#)\s*\w+', cleared_email))
        results['has_error_code'] = bool(re.search(r'error\s*(code|#)\s*\w+', cleared_email))

        # Contact information
        results['has_email_address'] = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', email))
        results['has_phone_number'] = bool(re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', email))

        # Financial and numerical patterns
        results['has_currency'] = bool(re.search(r'[$£€¥]\d+', email))
        results['has_percentage'] = bool(re.search(r'\d+%', email))
        results['has_quantity'] = bool(re.search(r'\b\d+\s*(items?|pieces?|units?|qty)\b', cleared_email))

        # Date and time patterns
        results['has_date'] = bool(re.search(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b', email))
        results['has_date_time'] = bool(re.search(r'\b\d{1,2}:\d{2}\s*(am|pm|AM|PM)?\b', email))
        results['has_deadline'] = bool(re.search(r'deadline.*\d+|due\s+by.*\d+', cleared_email))
        results['has_timeline'] = bool(re.search(r'(within|in)\s+\d+\s+(days?|weeks?|hours?)', cleared_email))

        # Marketing and promotional patterns  
        results['has_unsubscribe'] = 'unsubscribe' in cleared_email
        results['has_cta'] = any(cta in cleared_email for cta in ['click here', 'learn more', 'buy now', 'get started'])
        results['has_newsletter_format'] = bool(re.search(r'newsletter|weekly\s+update|monthly\s+report', cleared_email))
        results['has_suspicious_links'] = bool(re.search(r'(click\s+here|act\s+now|limited\s+time|free\s+offer)', cleared_email))

        # Social and event patterns
        results['has_rsvp'] = 'rsvp' in cleared_email
        results['has_location'] = bool(re.search(r'\b\d+\s+\w+\s+(street|st|avenue|ave|road|rd|drive|dr)\b', cleared_email))

        # Communication patterns
        results['has_call_to_action'] = any(phrase in cleared_email for phrase in ['please respond', 'action required', 'reply needed'])
        results['has_previous_reference'] = any(phrase in cleared_email for phrase in ['as discussed', 'following up', 'per our conversation'])
        results['has_gentle_nudge'] = any(phrase in cleared_email for phrase in ['just checking', 'friendly reminder', 'wanted to follow up'])

        # Technical and system patterns
        results['has_system_info'] = any(word in cleared_email for word in ['server', 'database', 'system', 'network', 'application'])
        
        # Legal and formal patterns
        results['has_legal_language'] = any(word in cleared_email for word in ['contract', 'agreement', 'terms', 'legal', 'compliance'])
        results['has_signature_block'] = bool(re.search(r'(best\s+regards|sincerely|kind\s+regards)', cleared_email))
        results['has_contract_terms'] = any(phrase in cleared_email for phrase in ['terms and conditions', 'agreement', 'contract'])

        # Internal communication patterns
        results['has_internal_domain'] = bool(re.search(r'@(company|internal|corp)\.', email))  # Adjust domain as needed
        results['has_cc_multiple'] = email.count('@') > 2  # Simple heuristic for multiple recipients

        # Content analysis
        results['has_links'] = bool(re.search(r'https?://|www\.', email))
        
        return results

    def analyze_sentiment(self, email: str) -> float:
        """
        Advanced sentiment analysis using multiple approaches
        
        Returns sentiment score from -1.0 (very negative) to +1.0 (very positive)
        Uses weighted combination of different sentiment indicators
        """
        import re
        
        email_lower = email.lower().strip()
        words = re.findall(r'\b\w+\b', email_lower)
        
        if not words:
            return 0.0
        
        total_sentiment = 0.0
        confidence_factors = []
        
        # 1. LEXICAL SENTIMENT ANALYSIS (40% weight)
        lexical_sentiment = self._calculate_lexical_sentiment(email_lower, words)
        total_sentiment += lexical_sentiment * 0.4
        confidence_factors.append(abs(lexical_sentiment))
        
        # 2. CONTEXTUAL SENTIMENT ANALYSIS (25% weight)  
        contextual_sentiment = self._calculate_contextual_sentiment(email_lower)
        total_sentiment += contextual_sentiment * 0.25
        confidence_factors.append(abs(contextual_sentiment))
        
        # 3. STRUCTURAL SENTIMENT INDICATORS (20% weight)
        structural_sentiment = self._calculate_structural_sentiment(email)
        total_sentiment += structural_sentiment * 0.2
        confidence_factors.append(abs(structural_sentiment))
        
        # 4. INTENSITY AND MODIFIERS (15% weight)
        intensity_sentiment = self._calculate_intensity_sentiment(email_lower, words)
        total_sentiment += intensity_sentiment * 0.15
        confidence_factors.append(abs(intensity_sentiment))
        
        # Normalize to [-1.0, 1.0] range
        final_sentiment = max(-1.0, min(1.0, total_sentiment))
        
        return final_sentiment
    
    def _calculate_lexical_sentiment(self, email_lower: str, words: list) -> float:
        """Calculate sentiment based on positive/negative word counts"""
        
        # Enhanced sentiment lexicons with weights
        positive_words = {
            # Strong positive (weight 1.0)
            'excellent': 1.0, 'outstanding': 1.0, 'perfect': 1.0, 'amazing': 1.0,
            'wonderful': 1.0, 'fantastic': 1.0, 'brilliant': 1.0, 'superb': 1.0,
            
            # Moderate positive (weight 0.7)
            'good': 0.7, 'great': 0.7, 'nice': 0.7, 'pleased': 0.7,
            'satisfied': 0.7, 'happy': 0.7, 'glad': 0.7, 'appreciate': 0.7,
            
            # Mild positive (weight 0.4)
            'okay': 0.4, 'fine': 0.4, 'thanks': 0.4, 'thank': 0.4,
            'helpful': 0.4, 'useful': 0.4, 'convenient': 0.4,
        }
        
        negative_words = {
            # Strong negative (weight -1.0)
            'terrible': -1.0, 'horrible': -1.0, 'awful': -1.0, 'disgusting': -1.0,
            'pathetic': -1.0, 'outrageous': -1.0, 'unacceptable': -1.0, 'furious': -1.0,
            
            # Moderate negative (weight -0.7)
            'bad': -0.7, 'poor': -0.7, 'disappointing': -0.7, 'frustrated': -0.7,
            'annoying': -0.7, 'unsatisfied': -0.7, 'unhappy': -0.7, 'angry': -0.7,
            
            # Mild negative (weight -0.4)
            'problem': -0.4, 'issue': -0.4, 'concern': -0.4, 'confused': -0.4,
            'difficult': -0.4, 'complicated': -0.4, 'wrong': -0.4,
        }
        
        sentiment_score = 0.0
        word_count = 0
        
        for word in words:
            if word in positive_words:
                sentiment_score += positive_words[word]
                word_count += 1
            elif word in negative_words:
                sentiment_score += negative_words[word]
                word_count += 1
        
        # Normalize by total words, not just sentiment words
        if len(words) > 0:
            return sentiment_score / len(words) * 10  # Scale up for better range
        
        return 0.0
    
    def _calculate_contextual_sentiment(self, email_lower: str) -> float:
        """Analyze sentiment based on phrases and context"""
        
        positive_phrases = {
            'thank you': 0.8, 'thanks for': 0.6, 'appreciate your': 0.7,
            'great job': 0.9, 'well done': 0.8, 'keep up': 0.6,
            'looking forward': 0.5, 'excited about': 0.8, 'love the': 0.7,
            'works perfectly': 1.0, 'exactly what': 0.6, 'better than expected': 0.9,
        }
        
        negative_phrases = {
            'not working': -0.8, 'does not work': -0.8, 'broken': -0.7,
            'waste of time': -1.0, 'money back': -0.6, 'completely useless': -1.0,
            'worst experience': -1.0, 'never again': -0.9, 'total disaster': -1.0,
            'fed up': -0.8, 'had enough': -0.7, 'sick of': -0.8,
            'what a joke': -0.8, 'ridiculous': -0.7, 'unbelievable': -0.6,
        }
        
        sentiment_score = 0.0
        
        # Check for positive phrases
        for phrase, weight in positive_phrases.items():
            if phrase in email_lower:
                sentiment_score += weight
        
        # Check for negative phrases
        for phrase, weight in negative_phrases.items():
            if phrase in email_lower:
                sentiment_score += weight  # weight is already negative
        
        return sentiment_score
    
    def _calculate_structural_sentiment(self, email: str) -> float:
        """Analyze sentiment based on punctuation and structure"""
        
        sentiment_score = 0.0
        
        # Exclamation marks analysis
        exclamation_count = email.count('!')
        if exclamation_count > 0:
            # Context matters - could be positive excitement or negative anger
            positive_context = any(word in email.lower() for word in 
                                 ['thank', 'great', 'excellent', 'amazing', 'wonderful'])
            negative_context = any(word in email.lower() for word in 
                                 ['terrible', 'awful', 'angry', 'frustrated', 'unacceptable'])
            
            if positive_context:
                sentiment_score += min(exclamation_count * 0.2, 0.8)  # Cap positive boost
            elif negative_context:
                sentiment_score -= min(exclamation_count * 0.2, 0.8)  # Cap negative impact
            else:
                sentiment_score += min(exclamation_count * 0.1, 0.4)  # Neutral excitement
        
        # ALL CAPS analysis (often indicates shouting/anger)
        words = email.split()
        if len(words) > 0:
            caps_ratio = sum(1 for word in words if word.isupper() and len(word) > 1) / len(words)
            if caps_ratio > 0.3:  # More than 30% caps words
                sentiment_score -= caps_ratio  # Negative indicator
        
        # Question marks (usually neutral to slightly negative - asking for help)
        question_count = email.count('?')
        if question_count > 2:
            sentiment_score -= question_count * 0.1  # Slight negative (confusion/frustration)
        
        # Ellipsis (...) often indicates hesitation or negative trailing off
        if '...' in email:
            sentiment_score -= 0.2
        
        return sentiment_score
    
    def _calculate_intensity_sentiment(self, email_lower: str, words: list) -> float:
        """Calculate sentiment based on intensifiers and modifiers"""
        
        # Intensifiers that amplify sentiment
        intensifiers = {
            'very': 1.5, 'extremely': 2.0, 'incredibly': 1.8, 'absolutely': 1.7,
            'totally': 1.6, 'completely': 1.8, 'utterly': 2.0, 'quite': 1.3,
            'really': 1.4, 'truly': 1.5, 'highly': 1.4, 'deeply': 1.6,
        }
        
        # Diminishers that reduce sentiment intensity
        diminishers = {
            'somewhat': 0.7, 'rather': 0.8, 'fairly': 0.8, 'slightly': 0.6,
            'a bit': 0.7, 'kind of': 0.7, 'sort of': 0.7, 'pretty': 0.8,
        }
        
        # Negators that flip sentiment
        negators = {'not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 'neither', 'nor'}
        
        sentiment_adjustments = []
        
        for i, word in enumerate(words):
            # Check for intensifiers before sentiment words
            if word in intensifiers and i < len(words) - 1:
                next_word = words[i + 1]
                if self._is_sentiment_word(next_word):
                    base_sentiment = self._get_word_sentiment(next_word)
                    if base_sentiment != 0:
                        intensified = base_sentiment * intensifiers[word]
                        sentiment_adjustments.append(intensified - base_sentiment)
            
            # Check for diminishers before sentiment words
            elif word in diminishers and i < len(words) - 1:
                next_word = words[i + 1]
                if self._is_sentiment_word(next_word):
                    base_sentiment = self._get_word_sentiment(next_word)
                    if base_sentiment != 0:
                        diminished = base_sentiment * diminishers[word]
                        sentiment_adjustments.append(diminished - base_sentiment)
            
            # Check for negators before sentiment words (within 2 words)
            elif word in negators:
                for j in range(i + 1, min(i + 3, len(words))):
                    if self._is_sentiment_word(words[j]):
                        base_sentiment = self._get_word_sentiment(words[j])
                        if base_sentiment != 0:
                            # Flip and slightly reduce intensity
                            negated = -base_sentiment * 0.8
                            sentiment_adjustments.append(negated - base_sentiment)
                        break
        
        return sum(sentiment_adjustments)
    
    def _is_sentiment_word(self, word: str) -> bool:
        """Check if word has sentiment value"""
        positive_words = {'excellent', 'good', 'great', 'wonderful', 'amazing', 'perfect', 'happy', 'pleased'}
        negative_words = {'terrible', 'bad', 'awful', 'horrible', 'angry', 'frustrated', 'disappointing'}
        return word in positive_words or word in negative_words
    
    def _get_word_sentiment(self, word: str) -> float:
        """Get base sentiment value for a word"""
        positive_words = {'excellent': 0.9, 'good': 0.6, 'great': 0.8, 'wonderful': 0.9, 
                         'amazing': 0.9, 'perfect': 1.0, 'happy': 0.7, 'pleased': 0.6}
        negative_words = {'terrible': -0.9, 'bad': -0.6, 'awful': -0.8, 'horrible': -0.9,
                         'angry': -0.7, 'frustrated': -0.6, 'disappointing': -0.7}
        
        return positive_words.get(word, negative_words.get(word, 0.0))
    
    def calculate_complexity(self, email: str) -> float:
        """
        Calculate email complexity score (0 = simple, 1 = very complex)
        
        Learning Concept: Text Complexity Analysis
        - We measure how difficult/complex an email is to handle
        - Simple emails can be auto-processed, complex ones need human review
        - Uses multiple indicators like length, technical terms, multiple issues
        """
        import re
        
        email_lower = email.lower().strip()
        words = email.split()
        
        if not words:
            return 0.0
            
        complexity = 0.0
        
        # 1. LENGTH COMPLEXITY (longer emails are more complex)
        word_count = len(words)
        if word_count > 200:
            complexity += 0.3
        elif word_count > 100:
            complexity += 0.2
        elif word_count > 50:
            complexity += 0.1
        
        # 2. SENTENCE COMPLEXITY (many sentences = more complex)
        sentence_count = len(re.split(r'[.!?]+', email))
        if sentence_count > 10:
            complexity += 0.2
        elif sentence_count > 5:
            complexity += 0.1
        
        # 3. QUESTION COMPLEXITY (multiple questions = more complex)
        question_count = email.count('?')
        if question_count > 3:
            complexity += 0.2
        elif question_count > 1:
            complexity += 0.1
        
        # 4. TECHNICAL LANGUAGE (technical terms increase complexity)
        technical_terms = [
            'api', 'ssl', 'database', 'server', 'configuration', 'integration',
            'authentication', 'authorization', 'encryption', 'protocol',
            'algorithm', 'framework', 'middleware', 'repository'
        ]
        tech_count = sum(1 for term in technical_terms if term in email_lower)
        complexity += min(tech_count * 0.1, 0.3)
        
        # 5. LEGAL/FORMAL LANGUAGE (increases complexity)
        legal_terms = [
            'contract', 'agreement', 'terms', 'conditions', 'liability',
            'compliance', 'regulation', 'lawsuit', 'attorney', 'legal'
        ]
        legal_count = sum(1 for term in legal_terms if term in email_lower)
        complexity += min(legal_count * 0.15, 0.4)
        
        # 6. MULTIPLE ISSUES INDICATOR
        issue_indicators = ['also', 'another', 'additionally', 'furthermore', 'moreover']
        issue_count = sum(1 for indicator in issue_indicators if indicator in email_lower)
        complexity += min(issue_count * 0.1, 0.2)
        
        # 7. URGENCY/ESCALATION LANGUAGE (adds complexity)
        escalation_terms = ['manager', 'supervisor', 'escalate', 'complaint', 'unacceptable']
        escalation_count = sum(1 for term in escalation_terms if term in email_lower)
        complexity += min(escalation_count * 0.15, 0.3)
        
        # Cap complexity at 1.0
        return min(complexity, 1.0)
    
    def calculate_keyword_score(self, email: str, keywords: dict) -> float:
        """
        Calculate score based on keyword matches with weights
        
        Learning Concept: Weighted Keyword Matching
        - Not all keywords are equally important
        - High-weight keywords (like 'refund') are strong signals
        - We also consider frequency - multiple mentions increase confidence
        """
        email_lower = email.lower().strip()
        words = re.findall(r'\b\w+\b', email_lower)
        
        if not words:
            return 0.0
            
        total_score = 0.0
        matches_found = 0
        
        # Check each keyword and its weight
        for keyword, weight in keywords.items():
            if keyword in email_lower:
                # Base score from weight
                score_contribution = weight
                
                # Bonus for multiple occurrences (but with diminishing returns)
                occurrence_count = email_lower.count(keyword)
                if occurrence_count > 1:
                    # Each additional occurrence adds 30% of original weight
                    bonus = (occurrence_count - 1) * weight * 0.3
                    score_contribution += bonus
                
                total_score += score_contribution
                matches_found += 1
        
        # Normalize by email length to prevent long emails from getting unfair advantage
        normalized_score = total_score / len(words) if words else 0.0
        
        # Scale up for better range (multiply by 10)
        scaled_score = normalized_score * 10
        
        # Cap the maximum score to prevent outliers
        return min(scaled_score, 5.0)
    
    def calculate_pattern_score(self, email: str, patterns: list) -> float:
        """
        Calculate score based on regex pattern matches
        
        Learning Concept: Pattern Recognition
        - Some emails have structural patterns that are strong indicators
        - Example: "order #12345" is a strong signal for order_status category
        - Regex patterns can detect formatted information like phone numbers, IDs
        """
        import re
        
        if not patterns:
            return 0.0
            
        pattern_matches = 0
        unique_patterns_matched = 0
        
        for pattern in patterns:
            try:
                matches = re.findall(pattern, email, re.IGNORECASE)
                if matches:
                    unique_patterns_matched += 1
                    pattern_matches += len(matches)  # Count multiple occurrences
            except re.error:
                # Skip invalid regex patterns
                continue
        
        # Score calculation:
        # - Each unique pattern matched = 0.5 points
        # - Each additional occurrence = 0.1 points
        base_score = unique_patterns_matched * 0.5
        frequency_bonus = max(0, pattern_matches - unique_patterns_matched) * 0.1
        
        total_score = base_score + frequency_bonus
        
        # Cap maximum score to prevent pattern-heavy categories from dominating
        return min(total_score, 2.0)
    
    def calculate_structural_score(self, structural_features: dict, structural_boosts: dict) -> float:
        """
        Calculate score based on structural features detected in email
        
        Learning Concept: Structural Analysis
        - Emails have structural elements beyond just words
        - Examples: phone numbers, email addresses, order numbers, etc.
        - These features can be strong indicators of email category
        """
        if not structural_features or not structural_boosts:
            return 0.0
            
        total_score = 0.0
        features_matched = 0
        
        # Add scores for each matching structural feature
        for feature_name, boost_value in structural_boosts.items():
            if structural_features.get(feature_name, False):
                total_score += boost_value
                features_matched += 1
        
        # Bonus for multiple structural features (indicates strong match)
        if features_matched > 2:
            total_score *= 1.2  # 20% bonus for rich feature match
        
        return min(total_score, 2.0)  # Cap maximum score
    
    def calculate_sentiment_match(self, email_sentiment: float, preferred_sentiment: float) -> float:
        """
        Calculate how well email sentiment matches category preference
        
        Learning Concept: Sentiment Matching
        - Different email categories expect different sentiment
        - Refund requests are usually negative, thank you emails are positive
        - We measure how close the email sentiment is to category expectation
        """
        # Calculate absolute difference between email sentiment and preferred sentiment
        sentiment_diff = abs(email_sentiment - preferred_sentiment)
        
        # Convert difference to similarity score (smaller diff = higher score)
        # Perfect match (diff=0) = score of 1.0
        # Maximum diff (diff=2.0) = score of 0.0
        similarity_score = max(0.0, 1.0 - (sentiment_diff / 2.0))
        
        return similarity_score
    
    def calculate_complexity_match(self, email_complexity: float, complexity_threshold: float) -> float:
        """
        Calculate how well email complexity matches category tolerance
        
        Learning Concept: Complexity Filtering
        - Some categories can handle complex emails, others cannot
        - Legal emails have low complexity tolerance (escalate complex ones)
        - Informational emails have high complexity tolerance
        """
        if email_complexity <= complexity_threshold:
            # Email complexity is within category tolerance
            # Closer to threshold = better match
            match_score = 0.5 + (email_complexity / complexity_threshold) * 0.5
            return min(match_score, 1.0)
        else:
            # Email is too complex for this category
            # Penalty increases with how much it exceeds threshold
            excess_complexity = email_complexity - complexity_threshold
            penalty = min(excess_complexity * 2, 1.0)  # Max penalty of 1.0
            return max(0.0, 0.5 - penalty)
    
    def calculate_sentiment_boost(self, sentiment_score: float, category_name: str, email: str) -> float:
        """
        Calculate sentiment-aware scoring boosts for specific categories
        
        NEW: Enhanced sentiment processing to reduce over-escalation
        - Praise emails get huge boost for CUSTOMER_PRAISE category
        - Positive suggestions get boost for FEATURE_SUGGESTIONS  
        - Clear positive sentiment prevents wrong categorization
        """
        
        boost = 0.0
        email_lower = email.lower()
        
        # CUSTOMER_PRAISE category gets massive boost for positive sentiment + thank you patterns
        if category_name == "customer_praise":
            if sentiment_score > 0.5:  # Positive sentiment
                boost += sentiment_score * 0.3  # Up to 0.3 boost for very positive
                
                # Extra boost for clear praise patterns
                praise_patterns = ['thank you', 'thanks', 'appreciate', 'great work', 'excellent', 'love']
                for pattern in praise_patterns:
                    if pattern in email_lower:
                        boost += 0.2  # Significant boost
                        break
                        
                # Additional boost for enthusiasm indicators
                if '!' in email and sentiment_score > 0.7:
                    boost += 0.15
        
        # FEATURE_SUGGESTIONS gets boost for positive suggestions
        elif category_name == "feature_suggestions":
            if sentiment_score > 0.2:  # Mildly positive or better
                suggestion_patterns = ['suggest', 'could you add', 'would be nice', 'idea', 'improve']
                for pattern in suggestion_patterns:
                    if pattern in email_lower:
                        boost += 0.2  # Good boost for suggestions
                        break
                        
                # Bonus for polite requests
                if any(word in email_lower for word in ['please', 'would', 'could']):
                    boost += 0.1
        
        # PARTNERSHIP_BUSINESS gets boost for formal, business-like language
        elif category_name == "partnership_business":
            if -0.2 <= sentiment_score <= 0.4:  # Neutral to mildly positive
                business_patterns = ['partnership', 'collaborate', 'business', 'discuss', 'opportunity']
                for pattern in business_patterns:
                    if pattern in email_lower:
                        boost += 0.15
                        break
                        
                # Boost for formal language
                if any(word in email_lower for word in ['interested in', 'would like to', 'potential']):
                    boost += 0.1
        
        # SUBSCRIPTION_MANAGEMENT gets small boost for clear requests
        elif category_name == "subscription_management":
            if -0.5 <= sentiment_score <= 0.2:  # Slightly negative to neutral (cancellations)
                cancel_patterns = ['cancel', 'unsubscribe', 'stop', 'terminate']
                for pattern in cancel_patterns:
                    if pattern in email_lower:
                        boost += 0.15
                        break
        
        # General positive sentiment boost for customer-facing categories
        elif category_name in ["customer_support", "sales_order"]:
            if sentiment_score > 0.6:  # Very positive
                boost += 0.1  # Small boost
        
        return min(boost, 0.5)  # Cap total boost at 0.5
    
    def calculate_confidence(self, category_scores: dict, best_score: float) -> float:
        """
        Calculate confidence based on score separation and absolute score
        
        Learning Concept: Confidence Calculation
        - High confidence when winner clearly beats other categories
        - Low confidence when scores are close together
        - Also consider absolute score - very low scores mean low confidence
        """
        if not category_scores or best_score <= 0:
            return 0.0
        
        # Get all scores and sort them
        all_scores = list(category_scores.values())
        all_scores.sort(reverse=True)
        
        # Calculate score separation (gap between 1st and 2nd place)
        if len(all_scores) >= 2:
            second_best = all_scores[1]
            score_separation = best_score - second_best
        else:
            score_separation = best_score
        
        # Base confidence from score separation (more generous)
        # Large separation = high confidence
        separation_confidence = min(score_separation / 0.5, 0.6)
        
        # Absolute score confidence (more generous)
        # Higher absolute scores indicate stronger matches  
        absolute_confidence = min(best_score / 3.0, 0.5)
        
        # Combined confidence
        total_confidence = separation_confidence + absolute_confidence
        
        # Boost confidence for reasonable scores
        if best_score > 0.5:
            total_confidence += 0.15  # Boost for decent matches
        if best_score > 1.0:
            total_confidence += 0.1   # Extra boost for good matches
        
        # Ensure confidence is between 0 and 1
        return min(max(total_confidence, 0.0), 1.0)
    
    def generate_classification_reasoning(self, email: str, best_category: str, 
                                        category_scores: dict, structural_features: dict) -> list:
        """
        Generate human-readable reasoning for the classification decision
        
        Learning Concept: Explainable AI
        - Users need to understand why the system made a decision
        - Transparency builds trust in automated systems
        - Helps with debugging and system improvement
        """
        reasoning = []
        best_score = category_scores[best_category]
        
        # Find the category profile for detailed analysis
        category_profile = None
        for profile in self.category_profiles:
            if profile.name == best_category:
                category_profile = profile
                break
        
        if not category_profile:
            return ["Classification completed with limited reasoning available"]
        
        # 1. Overall confidence reasoning
        if best_score > 3.0:
            reasoning.append(f"Strong match for {best_category} category (score: {best_score:.2f})")
        elif best_score > 1.5:
            reasoning.append(f"Moderate match for {best_category} category (score: {best_score:.2f})")
        else:
            reasoning.append(f"Weak match for {best_category} category (score: {best_score:.2f})")
        
        # 2. Keyword reasoning
        keyword_matches = []
        email_lower = email.lower()
        for keyword in category_profile.keywords.keys():
            if keyword in email_lower:
                keyword_matches.append(keyword)
        
        if keyword_matches:
            if len(keyword_matches) == 1:
                reasoning.append(f"Contains key term: '{keyword_matches[0]}'")
            else:
                reasoning.append(f"Contains {len(keyword_matches)} key terms: {', '.join(keyword_matches[:3])}")
        
        # 3. Pattern reasoning
        import re
        pattern_matches = 0
        for pattern in category_profile.patterns:
            try:
                if re.search(pattern, email, re.IGNORECASE):
                    pattern_matches += 1
            except re.error:
                continue
        
        if pattern_matches > 0:
            reasoning.append(f"Matches {pattern_matches} structural pattern(s)")
        
        # 4. Structural feature reasoning
        feature_matches = []
        for feature_name, boost in category_profile.structural_boosts.items():
            if structural_features.get(feature_name, False):
                feature_matches.append(feature_name.replace('has_', '').replace('_', ' '))
        
        if feature_matches:
            reasoning.append(f"Contains structural features: {', '.join(feature_matches)}")
        
        # 5. Competition reasoning
        sorted_scores = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_scores) >= 2:
            runner_up = sorted_scores[1]
            score_gap = best_score - runner_up[1]
            if score_gap > 1.0:
                reasoning.append(f"Clear winner over {runner_up[0]} (gap: {score_gap:.2f})")
            elif score_gap < 0.3:
                reasoning.append(f"Close competition with {runner_up[0]} (gap: {score_gap:.2f})")
        
        # 6. Fallback reasoning if no specific reasons found
        if len(reasoning) == 1:  # Only the overall score reasoning
            reasoning.append("Classification based on overall feature combination")
        
        return reasoning
