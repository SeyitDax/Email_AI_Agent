from email_agent import EmailAgent

# Initialize the agent
agent = EmailAgent()

# Test emails
test_emails = [
    "Hi, I ordered a laptop 3 days ago but haven't received any tracking information. Order #12345",
    "This product is broken! I want my money back immediately!",
    "Can you tell me more about the warranty on your smartphones?",
    "I'm having trouble logging into my account, it says my password is incorrect"
]

# Process each email
for email in test_emails:
    print("\n" + "="*50)
    print(f"EMAIL: {email}")
    print("-"*50)
    
    result = agent.process_email(email)
    
    print(f"CATEGORY: {result['category']}")
    print(f"ACTION: {result['action']}")
    print(f"CONFIDENCE: {result.get('confidence', 'N/A')}")
    print(f"\nRESPONSE:\n{result.get('response', result.get('reason', 'N/A'))}")