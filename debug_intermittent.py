#!/usr/bin/env python3
"""
Debug intermittent escalation issues
"""

import sys
import asyncio
sys.path.insert(0, 'src')

def test_multiple_runs(test_email, num_runs=5):
    """Test the same email multiple times to catch intermittent issues"""
    
    from email_agent import EmailAgent
    agent = EmailAgent()
    
    print(f"=== TESTING INTERMITTENT ISSUE ===")
    print(f"Email: {test_email}")
    print(f"Running {num_runs} times to catch intermittent behavior...\n")
    
    results = []
    
    for i in range(num_runs):
        print(f"Run {i+1}/{num_runs}:")
        
        try:
            result = agent.process_email(test_email)
            
            action = result.get('action', 'unknown')
            confidence = result.get('confidence', 'missing')
            reason = result.get('reason', 'no reason')
            
            print(f"  Action: {action}")
            print(f"  Confidence: {confidence}")
            
            if action == 'escalate':
                print(f"  Reason: {reason}")
                print("  [ESCALATED - This is the issue!]")
            else:
                print("  [PROCESSED NORMALLY]")
                
            results.append({
                'run': i+1,
                'action': action,
                'confidence': confidence,
                'success': action != 'escalate'
            })
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                'run': i+1,
                'action': 'error',
                'confidence': 0.0,
                'success': False
            })
            
        print()
    
    # Analysis
    escalated_runs = [r for r in results if r['action'] == 'escalate']
    successful_runs = [r for r in results if r['success']]
    
    print("=== ANALYSIS ===")
    print(f"Total runs: {num_runs}")
    print(f"Successful: {len(successful_runs)}")
    print(f"Escalated: {len(escalated_runs)}")
    print(f"Error rate: {(num_runs - len(successful_runs)) / num_runs * 100:.1f}%")
    
    if len(escalated_runs) > 0 and len(successful_runs) > 0:
        print("\n[CONFIRMED] Intermittent escalation issue detected!")
        print("This explains why you sometimes get 0% confidence.")
    elif len(escalated_runs) == num_runs:
        print("\n[CONSISTENT] All runs escalated - not intermittent")
    else:
        print("\n[STABLE] No escalations detected")

if __name__ == "__main__":
    test_emails = [
        "Hi, I need help with my account login issue.",
        "Can you help me reset my password?",
        "I want to cancel my subscription.",
        "Thank you for your great service!"
    ]
    
    if len(sys.argv) > 1:
        test_multiple_runs(sys.argv[1])
    else:
        for email in test_emails:
            test_multiple_runs(email, 3)
            print("\n" + "="*60 + "\n")