"""
Test script for the new spam filter functionality
Tests the emails from the reference images that were incorrectly escalated
"""

import asyncio
import os
import traceback
from src.agents.responder import ResponseGenerator

async def run_spam_filter_harness():
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
            
            # Display results (guarded access to avoid AttributeError/KeyError)
            _status_obj = getattr(result, 'status', None)
            _status_value = getattr(_status_obj, 'value', None)
            status_str = (
                _status_value if isinstance(_status_value, str)
                else (str(_status_value) if _status_value is not None else "<missing>")
            )

            _cls = getattr(result, 'classification_result', None)
            if isinstance(_cls, dict):
                _category = _cls.get('category', "<missing>")
                category_str = _category if isinstance(_category, str) else str(_category)
            else:
                category_str = "<missing>"

            _quality_obj = getattr(result, 'response_quality', None)
            _quality_value = getattr(_quality_obj, 'value', None)
            quality_str = (
                _quality_value if isinstance(_quality_value, str)
                else (str(_quality_value) if _quality_value is not None else "<missing>")
            )

            _llm_model = getattr(result, 'llm_model', None)
            llm_model_str = (
                _llm_model if isinstance(_llm_model, str)
                else (str(_llm_model) if _llm_model is not None else "<missing>")
            )

            print(f"Status: {status_str}")
            print(f"Category: {category_str}")
            print(f"Quality: {quality_str}")
            print(f"LLM Model: {llm_model_str}")
            
            if result.response_text:
                print(f"Response: {result.response_text[:100]}...")
            else:
                print("Response: (No response - email discarded)")
                
            print("Reasoning:")
            for reason in result.reasoning:
                print(f"  • {reason}")
                
        except Exception as e:
            print(f"Error processing {email['name']}: {str(e)}")
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("Spam filter testing completed!")

if __name__ == "__main__":
    if os.getenv("RUN_SPAM_FILTER_HARNESS", "0") == "1":
        asyncio.run(run_spam_filter_harness())
    else:
        print("Set RUN_SPAM_FILTER_HARNESS=1 to run this interactive harness.")