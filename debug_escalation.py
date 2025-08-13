#!/usr/bin/env python3
"""
Email Agent Escalation Debugger
Helps diagnose why emails are being escalated instead of processed
"""

import sys
import os
sys.path.insert(0, 'src')

from email_agent import EmailAgent
from src.agents.classifier import EmailClassifier
from src.agents.confidence_scorer import ConfidenceScorer
from src.agents.escalation_engine import EscalationEngine
from src.core.config import settings
import asyncio

def debug_email_processing(test_email: str):
    """Debug the complete email processing pipeline"""
    
    print("=== EMAIL AGENT ESCALATION DEBUGGER ===")
    print(f"Test Email: {test_email[:80]}...")
    print()
    
    # Check OpenAI Configuration
    print("1. CHECKING OPENAI CONFIGURATION")
    if settings.openai_api_key:
        key_preview = settings.openai_api_key[:10] + "..." + settings.openai_api_key[-4:]
        print(f"   [OK] OpenAI API Key: {key_preview}")
    else:
        print("   ❌ OpenAI API Key: NOT SET")
        print("   Fix: export OPENAI_API_KEY='your-key-here'")
        return
    
    print(f"   ✓ Model: {settings.openai_model}")
    print(f"   ✓ Max Tokens: {settings.openai_max_tokens}")
    print()
    
    # Test Classification
    print("2. TESTING EMAIL CLASSIFICATION")
    try:
        classifier = EmailClassifier()
        classification_result = classifier.classify(test_email)
        
        print(f"   ✓ Category: {classification_result['category']}")
        print(f"   ✓ Confidence: {classification_result['confidence']:.3f}")
        print(f"   ✓ Sentiment: {classification_result['sentiment_score']:.3f}")
        print(f"   ✓ Complexity: {classification_result['complexity_score']:.3f}")
        print()
        
    except Exception as e:
        print(f"   ❌ Classification failed: {e}")
        return
    
    # Test Confidence Analysis
    print("3. TESTING CONFIDENCE ANALYSIS")
    try:
        scorer = ConfidenceScorer()
        confidence_analysis = scorer.analyze_confidence(classification_result, test_email)
        
        print(f"   ✓ Overall Confidence: {confidence_analysis.overall_confidence:.3f}")
        print(f"   ✓ Risk Score: {confidence_analysis.risk_score:.3f}")
        print(f"   ✓ Recommended Action: {confidence_analysis.recommended_action.value}")
        
        if confidence_analysis.risk_factors:
            print(f"   ⚠ Risk Factors: {[rf.value for rf in confidence_analysis.risk_factors]}")
        print()
        
    except Exception as e:
        print(f"   ❌ Confidence analysis failed: {e}")
        return
    
    # Test Escalation Decision
    print("4. TESTING ESCALATION DECISION")
    try:
        escalation_engine = EscalationEngine()
        escalation_decision = escalation_engine.make_escalation_decision(
            test_email, classification_result, confidence_analysis
        )
        
        print(f"   📋 Should Escalate: {escalation_decision.should_escalate}")
        print(f"   📋 Priority: {escalation_decision.priority_level.value}")
        print(f"   📋 Escalation Score: {escalation_decision.escalation_score:.3f}")
        
        if escalation_decision.escalation_reasons:
            print(f"   📋 Escalation Reasons: {[er.value for er in escalation_decision.escalation_reasons]}")
            
        if escalation_decision.should_escalate:
            print(f"   ⚠ ESCALATION TRIGGERED - This is why your email gets escalated!")
            print(f"   📋 Assigned Team: {escalation_decision.assigned_team.value}")
            print("   📋 Reasoning:")
            for reason in escalation_decision.reasoning:
                print(f"      - {reason}")
        else:
            print("   ✓ No escalation required - should proceed to response generation")
        print()
        
    except Exception as e:
        print(f"   ❌ Escalation decision failed: {e}")
        return
    
    # Test Complete Pipeline
    print("5. TESTING COMPLETE PIPELINE")
    try:
        agent = EmailAgent()
        result = agent.process_email(test_email)
        
        print(f"   📧 Final Action: {result['action']}")
        print(f"   📧 Final Category: {result['category']}")
        print(f"   📧 Final Confidence: {result['confidence']:.3f}")
        
        if result['action'] == 'escalate':
            print("   ⚠ RESULT: Email was escalated")
        elif result['action'] == 'review':
            print("   ⚠ RESULT: Email needs human review")
        else:
            print("   ✓ RESULT: Email was processed normally")
            print(f"   📧 Response: {result.get('response', 'No response')[:100]}...")
        print()
        
    except Exception as e:
        print(f"   ❌ Complete pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Recommendations
    print("6. RECOMMENDATIONS")
    if escalation_decision.should_escalate:
        print("   🔧 Your emails are being escalated because:")
        for reason in escalation_decision.reasoning:
            print(f"      - {reason}")
        print()
        print("   🔧 To reduce escalations:")
        print("      - Lower escalation thresholds in config")
        print("      - Improve email content clarity")
        print("      - Check for VIP keywords being detected")
        print("      - Review risk factor detection")
    else:
        print("   ✓ System should be processing emails normally")
        print("   ✓ If still getting escalations, check OpenAI API errors")

def main():
    # Test with various email types
    test_emails = [
        "Hi, I need help with my account login issue. Can you please assist me?",
        "I want to cancel my subscription and get a refund for this month.",
        "This is CEO John Smith. Our system is down and needs immediate attention.",
        "Hi there, just wanted to say thanks for your great service!"
    ]
    
    if len(sys.argv) > 1:
        # Use provided email
        test_email = sys.argv[1]
        debug_email_processing(test_email)
    else:
        # Test multiple emails
        for i, email in enumerate(test_emails, 1):
            print(f"\n{'='*60}")
            print(f"TEST {i}/4")
            print('='*60)
            debug_email_processing(email)
            print()

if __name__ == "__main__":
    main()