"""
Test script to verify that legitimate customer service emails are still processed correctly
and not incorrectly filtered as spam
"""

import asyncio
from src.agents.responder import ResponseGenerator

async def test_legitimate_emails():
    """Test legitimate customer service emails to ensure they're not filtered"""
    
    response_gen = ResponseGenerator()
    
    # Legitimate customer service emails that should NOT be filtered
    legitimate_emails = [
        {
            'name': 'Order Status Inquiry',
            'content': '''Hi, I placed order #ORD-789123 three days ago and haven't received any tracking information. Can you please check the status? I'm getting worried it might be lost.'''
        },
        {
            'name': 'Billing Issue',
            'content': '''I was charged twice for my subscription this month - once on the 1st and again on the 15th. This is clearly an error. I need this duplicate charge refunded immediately.'''
        },
        {
            'name': 'Technical Support',
            'content': '''Your app keeps crashing every time I try to access my account dashboard. I've tried reinstalling but the problem persists. This is preventing me from managing my subscription.'''
        },
        {
            'name': 'Refund Request',
            'content': '''The product I received is completely different from what was advertised. The quality is terrible and it doesn't fit. I want a full refund and return label.'''
        }
    ]
    
    print("Testing Legitimate Customer Service Emails")
    print("=" * 50)
    
    for email in legitimate_emails:
        print(f"\nTesting: {email['name']}")
        print("-" * 30)
        
        try:
            result = await response_gen.generate_response(email['content'])
            
            # Check if it was properly processed (not spam filtered)
            was_spam_filtered = result.llm_model == 'spam_filter_system'
            
            print(f"Status: {result.status.value}")
            print(f"Category: {result.classification_result['category']}")
            print(f"Was Spam Filtered: {'YES' if was_spam_filtered else 'NO'}")
            print(f"LLM Model: {result.llm_model}")
            
            if result.escalation_decision and result.escalation_decision.should_escalate:
                print(f"Escalated: YES - {result.escalation_decision.escalation_reasons}")
            else:
                print("Escalated: NO")
                
            if result.response_text and not was_spam_filtered:
                print(f"Response: {result.response_text[:100]}...")
            
            print("Reasoning:")
            for reason in result.reasoning[:3]:  # Show first 3 reasons
                print(f"  • {reason}")
                
        except Exception as e:
            print(f"Error processing {email['name']}: {str(e)}")
    
    print("\n" + "=" * 50)
    print("Legitimate email testing completed!")

if __name__ == "__main__":
    asyncio.run(test_legitimate_emails())