"""
Test script for the new spam filter functionality
Tests the emails from the reference images that were incorrectly escalated
"""

import asyncio
from src.agents.responder import ResponseGenerator

async def test_spam_filter():
    """Test the spam filter with the problematic emails from the reference images"""
    
    # Initialize the response generator (includes spam filter)
    response_gen = ResponseGenerator()
    
    # Test emails from the reference images that were incorrectly escalated
    test_emails = [
        {
            'name': 'Newsletter Welcome',
            'content': '''From: Emily Rose emily_rose85@fakemail.net
Subject: Welcome to Our Newsletter!
Body: Hello! Thanks for subscribing to our monthly newsletter. Stay tuned for updates and special offers.'''
        },
        {
            'name': 'System Test Email', 
            'content': '''From: Test User test.user01@testingdomain.com
Subject: System Test Email
Body: This is a test email to verify the functionality of the email sorter system. Please disregard.'''
        },
        {
            'name': 'Feedback Survey',
            'content': '''From: Laura White laura.white99@mockmail.net
Subject: Feedback Request
Body: We value your opinion. Please take a moment to fill out our customer satisfaction survey.'''
        },
        {
            'name': 'Webinar Invitation',
            'content': '''From: Sarah Connor sarah_connor@myemail.co
Subject: Invitation to Webinar
Body: Join us this Friday for an exclusive webinar on marketing strategies. Register now to secure your spot!'''
        }
    ]
    
    print("Testing Spam Filter Integration")
    print("=" * 50)
    
    for email in test_emails:
        print(f"\nTesting: {email['name']}")
        print("-" * 30)
        
        try:
            # Process the email through the full pipeline
            result = await response_gen.generate_response(email['content'])
            
            # Display results
            print(f"Status: {result.status.value}")
            print(f"Category: {result.classification_result['category']}")
            print(f"Quality: {result.response_quality.value}")
            print(f"LLM Model: {result.llm_model}")
            
            if result.response_text:
                print(f"Response: {result.response_text[:100]}...")
            else:
                print("Response: (No response - email discarded)")
                
            print("Reasoning:")
            for reason in result.reasoning:
                print(f"  • {reason}")
                
        except Exception as e:
            print(f"Error processing {email['name']}: {str(e)}")
    
    print("\n" + "=" * 50)
    print("Spam filter testing completed!")

if __name__ == "__main__":
    asyncio.run(test_spam_filter())