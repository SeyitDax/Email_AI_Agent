"""
Test script for the enhanced team assignment logic
Tests emails based on reference files to ensure correct team routing
"""

import asyncio
from src.agents.responder import ResponseGenerator

async def test_team_assignments():
    """Test the enhanced team assignment logic with reference-based examples"""
    
    response_gen = ResponseGenerator()
    
    # Test cases based on reference files with expected vs actual team assignments
    test_emails = [
        {
            'name': 'Billing Error - Wrong Name on Receipt',
            'content': '''From: Sophia Rivera sophia.rivera@myemail.co
Subject: Wrong billing name on receipt
Body: My receipt shows someone else's name, even though the payment came from my account. Could you fix this?''',
            'expected_team': 'Billing Specialists',
            'should_escalate': True,
            'reference': 'This need to ESCLATE to Billing Specialist.png'
        },
        {
            'name': 'Payment Failed but Money Deducted',
            'content': '''From: Olivia Carter olivia.carter@customeremail.org
Subject: Payment failed but money deducted  
Body: My payment attempt showed an error, but my bank account was charged. Can you confirm if the order went through?''',
            'expected_team': 'Billing Specialists',
            'should_escalate': True,
            'reference': 'This needs to Esclate to Billing Specailist.png'
        },
        {
            'name': 'Website Down - Business Critical',
            'content': '''From: Henry Scott henry.scott@testingdomain.com
Subject: Website down
Body: I've been trying to place an order, but your website has been offline all morning. Is there an alternative?''',
            'expected_team': 'Management',
            'should_escalate': True,
            'reference': 'This needs to ESCLATE to Management.png'
        },
        {
            'name': 'Order Tracking Issue',
            'content': '''From: Hannah Brooks hannah.brooks@example.com
Subject: Order tracking link not working
Body: Hi, the tracking link in my shipment confirmation email gives me a "page not found" error. Could you send me the correct link?''',
            'expected_team': 'Technical Team',
            'should_escalate': True,
            'reference': 'This needs to esclate to Senior team not to management.png'
        },
        {
            'name': 'Order Cancellation Request',
            'content': '''From: Daniel Kim daniel.kim@dummyemail.org
Subject: Request for order cancellation
Body: Hi, I placed an order today but would like to cancel it before it ships. Order number is #2098.''',
            'expected_team': 'Technical Team',
            'should_escalate': True,
            'reference': 'This needs to esclate to technical team not the Senior team.png'
        },
        {
            'name': 'Warranty Replacement Request',
            'content': '''From: Samuel Wright samuel.wright@dummyemail.org
Subject: Replacement part request
Body: I purchased a blender from your store, and the lid broke. Can you send me a replacement part under warranty?''',
            'expected_team': 'Technical Team',
            'should_escalate': True,
            'reference': 'This needs to ESCLATE to Technical team not to Senior support.png'
        },
        {
            'name': 'System Security Concern - Critical',
            'content': '''From: Security Tester security@example.com
Subject: Suspicious Account Activity
Body: I noticed unauthorized login attempts on my account from unknown locations. This might be a security breach. Please investigate immediately.''',
            'expected_team': 'Senior Support',
            'should_escalate': True,
            'reference': 'Should escalate to Senior Team for security issues'
        },
        {
            'name': 'Login Issue - General Technical',
            'content': '''From: User Help user@example.com
Subject: Can't log into my account
Body: I forgot my password and the password reset email isn't coming through. Can you help me access my account?''',
            'expected_team': 'Technical Team',
            'should_escalate': True,
            'reference': 'Should route to Technical Team for login issues'
        }
    ]
    
    print("Testing Enhanced Team Assignment Logic")
    print("=" * 70)
    
    success_count = 0
    total_tests = len(test_emails)
    
    for email in test_emails:
        print(f"\nTesting: {email['name']}")
        print("-" * 50)
        print(f"Expected Team: {email['expected_team']}")
        print(f"Reference: {email['reference']}")
        
        try:
            result = await response_gen.generate_response(email['content'])
            
            # Extract team assignment from reasoning or escalation decision
            assigned_team = "Unknown"
            was_escalated = result.status.value == 'escalated'
            
            # Check if escalation decision exists and extract team
            if hasattr(result, 'escalation_decision') and result.escalation_decision:
                if hasattr(result.escalation_decision, 'assigned_team'):
                    team_enum = result.escalation_decision.assigned_team
                    assigned_team = team_enum.value.replace('_', ' ').title()
            
            # Also check reasoning for team assignment info
            team_from_reasoning = "Unknown"
            for reason in result.reasoning:
                if "team:" in reason.lower() or "assigned to" in reason.lower():
                    team_from_reasoning = reason
                    break
            
            print(f"Status: {result.status.value}")
            print(f"Was Escalated: {'YES' if was_escalated else 'NO'}")
            print(f"Assigned Team: {assigned_team}")
            if team_from_reasoning != "Unknown":
                print(f"Team Info: {team_from_reasoning}")
            
            # Check success criteria
            expected_team_key = email['expected_team'].lower().replace(' ', '_')
            actual_team_key = assigned_team.lower().replace(' ', '_')
            
            if was_escalated and expected_team_key in actual_team_key:
                print("SUCCESS: Correct team assignment!")
                success_count += 1
            elif was_escalated:
                print(f"PARTIAL: Escalated but wrong team (got {assigned_team})")
            else:
                print("ISSUE: Should have escalated but didn't")
            
            print("Key Reasoning:")
            for reason in result.reasoning[:3]:
                print(f"  • {reason}")
                
        except Exception as e:
            print(f"Error processing {email['name']}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print(f"Team Assignment Test Results: {success_count}/{total_tests} correct")
    print(f"Success Rate: {(success_count/total_tests)*100:.1f}%")
    
    if success_count == total_tests:
        print("EXCELLENT: All team assignments working correctly!")
    elif success_count >= total_tests * 0.8:
        print("GOOD: Most team assignments working correctly")
    else:
        print("NEEDS WORK: Several team assignment issues found")

if __name__ == "__main__":
    asyncio.run(test_team_assignments())