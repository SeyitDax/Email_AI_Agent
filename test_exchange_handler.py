"""
Test script for the new exchange handler functionality
Tests the emails from reference files that should be handled by AI instead of escalated
"""

import asyncio
from src.agents.responder import ResponseGenerator

async def test_exchange_handler():
    """Test the exchange handler with problematic emails from reference files"""
    
    # Initialize the response generator (includes exchange handler)
    response_gen = ResponseGenerator()
    
    # Test emails from reference files that should be AI-handled, not escalated
    test_emails = [
        {
            'name': 'Size Exchange Request (Should AI Handle)',
            'content': '''From: Charlotte Hill charlotte.hill@temporarymail.com
Subject: Exchange request
Body: I bought the wrong size shoes. Could you guide me through the exchange process?''',
            'expected': 'AI should handle with size exchange template'
        },
        {
            'name': 'Color Mismatch (Should AI Handle)', 
            'content': '''From: Linda Brown linda.brown@fakemail.net
Subject: Wrong item sent
Body: I ordered a size M shirt but received a size L instead. How do I exchange it?''',
            'expected': 'AI should handle with size exchange template'
        },
        {
            'name': 'Product Mismatch (Should AI Handle)',
            'content': '''From: Peter Adams peter.adams@samplemail.io
Subject: Product not matching description
Body: The headphones I received are a different color than the ones shown on your website. Can I exchange them?''',
            'expected': 'AI should handle with color exchange template'
        },
        {
            'name': 'Simple Return Request (Should AI Handle)',
            'content': '''From: James Parker james.parker@testdomain.com
Subject: Product arrived without accessories
Body: The camera I ordered didn't include the charging cable or carrying case mentioned in the description.''',
            'expected': 'AI should handle with product mismatch template'
        }
    ]
    
    print("Testing Exchange Handler Integration")
    print("=" * 60)
    
    for email in test_emails:
        print(f"\nTesting: {email['name']}")
        print("-" * 40)
        print(f"Expected: {email['expected']}")
        
        try:
            # Process the email through the full pipeline
            result = await response_gen.generate_response(email['content'])
            
            # Check if it was handled by exchange handler
            was_exchange_handled = result.llm_model == 'exchange_handler_system'
            was_escalated = result.status.value == 'escalated'
            
            print(f"Status: {result.status.value}")
            print(f"Category: {result.classification_result['category']}")
            print(f"Exchange Handled: {'YES' if was_exchange_handled else 'NO'}")
            print(f"Was Escalated: {'YES' if was_escalated else 'NO'}")
            print(f"LLM Model: {result.llm_model}")
            print(f"Quality: {result.response_quality.value}")
            
            if result.response_text:
                print(f"Response Preview: {result.response_text[:150]}...")
            else:
                print("Response: (No response generated)")
                
            print("Reasoning:")
            for reason in result.reasoning[:4]:  # Show first 4 reasons
                print(f"  • {reason}")
            
            # Evaluate success
            if was_exchange_handled and not was_escalated:
                print("SUCCESS: Handled by AI exchange handler")
            elif was_escalated:
                print("ISSUE: Still being escalated")
            else:
                print("UNCLEAR: Different handling path")
                
        except Exception as e:
            print(f"Error processing {email['name']}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Exchange handler testing completed!")

if __name__ == "__main__":
    asyncio.run(test_exchange_handler())