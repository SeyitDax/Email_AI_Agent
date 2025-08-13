"""
Unit tests for the EmailClassifier component.
Tests the sophisticated classification logic, sentiment analysis, and feature detection.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.agents.classifier import EmailClassifier, CategoryProfile


class TestEmailClassifier:
    """Test suite for EmailClassifier"""
    
    def test_classifier_initialization(self):
        """Test classifier initializes with correct categories"""
        classifier = EmailClassifier()
        
        assert len(classifier.categories) == 15
        assert 'CUSTOMER_SUPPORT' in classifier.categories
        assert 'TECHNICAL_ISSUE' in classifier.categories
        assert all(isinstance(profile, CategoryProfile) for profile in classifier.categories.values())
    
    def test_basic_classification(self, sample_emails):
        """Test basic email classification"""
        classifier = EmailClassifier()
        
        # Test customer support email
        result = classifier.classify(sample_emails['customer_support'])
        assert result['category'] == 'CUSTOMER_SUPPORT'
        assert 0.0 <= result['confidence'] <= 1.0
        assert 'category_scores' in result
        assert 'reasoning' in result
    
    def test_classification_confidence(self, sample_emails):
        """Test that classification confidence is properly calculated"""
        classifier = EmailClassifier()
        
        # Test multiple emails
        for email_type, email_content in sample_emails.items():
            result = classifier.classify(email_content)
            
            # Confidence should be a valid probability
            assert 0.0 <= result['confidence'] <= 1.0
            
            # Category scores should sum to approximately 1.0
            total_score = sum(result['category_scores'].values())
            assert 0.8 <= total_score <= 1.2  # Allow some rounding error
    
    def test_sentiment_analysis(self, sample_emails):
        """Test sentiment analysis functionality"""
        classifier = EmailClassifier()
        
        # Test positive sentiment
        positive_result = classifier.analyze_sentiment(sample_emails['compliment_positive'])
        assert positive_result > 0.3  # Should be positive
        
        # Test negative sentiment  
        negative_result = classifier.analyze_sentiment(sample_emails['complaint_negative'])
        assert negative_result < -0.3  # Should be negative
        
        # Test neutral sentiment
        neutral_result = classifier.analyze_sentiment(sample_emails['order_status'])
        assert -0.3 <= neutral_result <= 0.3  # Should be neutral
    
    def test_structural_feature_detection(self, sample_emails):
        """Test structural feature detection"""
        classifier = EmailClassifier()
        
        # Test order number detection
        order_email = "My order #12345 is delayed"
        features = classifier.detect_structural_features(order_email)
        assert features['has_order_number'] == True
        
        # Test question mark detection
        question_email = "Can you help me with this?"
        features = classifier.detect_structural_features(question_email)
        assert features['has_question_mark'] == True
        
        # Test phone number detection
        phone_email = "Please call me at 555-123-4567"
        features = classifier.detect_structural_features(phone_email)
        assert features['has_phone_number'] == True
        
        # Test email detection
        email_text = "Contact me at test@example.com"
        features = classifier.detect_structural_features(email_text)
        assert features['has_email'] == True
    
    def test_complexity_calculation(self, sample_emails):
        """Test complexity score calculation"""
        classifier = EmailClassifier()
        
        # Simple email should have low complexity
        simple_email = "Hello, thank you."
        simple_complexity = classifier.calculate_complexity(simple_email)
        assert 0.0 <= simple_complexity <= 0.4
        
        # Complex email should have high complexity
        complex_email = sample_emails['technical_complex']
        complex_complexity = classifier.calculate_complexity(complex_email)
        assert 0.4 <= complex_complexity <= 1.0
    
    def test_category_scoring(self):
        """Test category scoring logic"""
        classifier = EmailClassifier()
        
        # Test technical issue scoring
        tech_email = "The API is returning SSL certificate errors during handshake"
        scores = classifier._calculate_category_scores(tech_email, 
                                                      classifier.detect_structural_features(tech_email),
                                                      classifier.analyze_sentiment(tech_email),
                                                      classifier.calculate_complexity(tech_email))
        
        # Technical issue should score highly
        assert scores.get('TECHNICAL_ISSUE', 0) > 0.5
        
        # Other categories should score lower
        assert scores.get('COMPLIMENT_POSITIVE', 0) < 0.3
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        classifier = EmailClassifier()
        
        # Empty string
        result = classifier.classify("")
        assert result['category'] == 'OTHER_UNCATEGORIZED'
        assert result['confidence'] > 0.0
        
        # Very long email
        long_email = "word " * 1000
        result = classifier.classify(long_email)
        assert 'category' in result
        assert result['confidence'] > 0.0
        
        # Special characters
        special_email = "!@#$%^&*()_+-=[]{}|;':\"<>?,./`~"
        result = classifier.classify(special_email)
        assert 'category' in result
    
    def test_reasoning_generation(self, sample_emails):
        """Test that reasoning is generated for classifications"""
        classifier = EmailClassifier()
        
        for email_content in list(sample_emails.values())[:5]:  # Test subset for speed
            result = classifier.classify(email_content)
            
            # Should have reasoning
            assert 'reasoning' in result
            assert isinstance(result['reasoning'], list)
            assert len(result['reasoning']) > 0
            
            # Reasoning should be strings
            for reason in result['reasoning']:
                assert isinstance(reason, str)
                assert len(reason) > 0
    
    def test_category_profiles(self):
        """Test that category profiles are properly configured"""
        classifier = EmailClassifier()
        
        for category_name, profile in classifier.categories.items():
            # Each profile should have required attributes
            assert hasattr(profile, 'name')
            assert hasattr(profile, 'keywords')
            assert hasattr(profile, 'patterns')
            assert hasattr(profile, 'structural_boosts')
            assert hasattr(profile, 'sentiment_preference')
            assert hasattr(profile, 'complexity_threshold')
            
            # Keywords should be a dict with float weights
            assert isinstance(profile.keywords, dict)
            for keyword, weight in profile.keywords.items():
                assert isinstance(keyword, str)
                assert isinstance(weight, (int, float))
                assert weight > 0
            
            # Patterns should be a list of strings
            assert isinstance(profile.patterns, list)
            for pattern in profile.patterns:
                assert isinstance(pattern, str)
    
    def test_performance_benchmarks(self, sample_emails, performance_benchmarks):
        """Test that classifier meets performance benchmarks"""
        classifier = EmailClassifier()
        import time
        
        # Test processing speed
        start_time = time.time()
        results = []
        
        for email_content in sample_emails.values():
            result = classifier.classify(email_content)
            results.append(result)
        
        total_time = (time.time() - start_time) * 1000  # Convert to ms
        avg_time_per_email = total_time / len(sample_emails)
        
        # Should be reasonably fast
        assert avg_time_per_email < performance_benchmarks['max_processing_time_ms'] / 10
        
        # All results should have required fields
        for result in results:
            assert 'category' in result
            assert 'confidence' in result
            assert result['confidence'] >= 0.0
    
    def test_spam_detection_integration(self, sample_emails):
        """Test spam detection through classification"""
        classifier = EmailClassifier()
        
        # Spam email should be classified as spam
        spam_result = classifier.classify(sample_emails['spam_promotional'])
        assert spam_result['category'] == 'SPAM_PROMOTIONAL'
        
        # Spam should have distinctive structural features
        spam_features = classifier.detect_structural_features(sample_emails['spam_promotional'])
        assert spam_features.get('has_urgency_words', False) or spam_features.get('has_money_terms', False)
    
    @pytest.mark.parametrize("category", [
        'CUSTOMER_SUPPORT', 'TECHNICAL_ISSUE', 'BILLING_INQUIRY', 
        'REFUND_REQUEST', 'COMPLAINT_NEGATIVE'
    ])
    def test_category_specific_classification(self, category, sample_emails):
        """Test classification for specific categories"""
        classifier = EmailClassifier()
        
        # Find email for this category
        email_key = category.lower().replace('_', ' ').replace(' ', '_')
        if email_key in sample_emails:
            result = classifier.classify(sample_emails[email_key])
            
            # Should classify correctly or at least have high score for target category
            if result['category'] != category:
                # Check if target category has high score
                target_score = result['category_scores'].get(category, 0)
                assert target_score > 0.3, f"Category {category} should have higher score"