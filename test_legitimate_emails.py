"""
Test script to verify that legitimate customer service emails are still processed correctly
and not incorrectly filtered as spam
"""

import os
import unittest
from src.agents.responder import ResponseGenerator


@unittest.skipUnless(os.getenv("RUN_E2E_EMAIL_TESTS") == "1", "E2E email tests disabled")
class TestLegitimateEmails(unittest.IsolatedAsyncioTestCase):
    """Test legitimate customer service emails to ensure they're not filtered"""
    
    async def asyncSetUp(self):
        """Set up test fixtures"""
        self.response_gen = ResponseGenerator()
        
        # Legitimate customer service emails that should NOT be filtered
        self.legitimate_emails = [
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

    async def test_legitimate_emails_not_spam_filtered(self):
        """Test that legitimate customer service emails are not incorrectly filtered as spam"""
        
        for email in self.legitimate_emails:
            with self.subTest(email_type=email['name']):
                # Generate response for the email
                result = await self.response_gen.generate_response(email['content'])
                
                # Assert that the email was not spam filtered
                was_spam_filtered = result.llm_model == 'spam_filter_system'
                self.assertFalse(was_spam_filtered, 
                    f"Email '{email['name']}' was incorrectly filtered as spam")
                
                # Assert that we got a valid response
                self.assertIsNotNone(result, "Response should not be None")
                self.assertIsNotNone(result.status, "Response status should not be None")
                self.assertIsNotNone(result.classification_result, "Classification result should not be None")
                
                # Assert that the response has content (unless it was spam filtered)
                if not was_spam_filtered:
                    self.assertIsNotNone(result.response_text, 
                        f"Response text should not be None for '{email['name']}'")
                    self.assertGreater(len(result.response_text), 0,
                        f"Response text should not be empty for '{email['name']}'")
                
                # Assert that reasoning is provided
                self.assertIsNotNone(result.reasoning, "Reasoning should not be None")
                self.assertGreater(len(result.reasoning), 0, "Reasoning should not be empty")
                
                # Assert that classification category is present
                self.assertIn('category', result.classification_result,
                    "Classification result should contain a category")


if __name__ == "__main__":
    unittest.main()