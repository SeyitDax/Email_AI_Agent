"""
API endpoint tests for the FastAPI email processing routes.
Tests REST API functionality, request/response formats, and error handling.
"""

import pytest
import asyncio
from httpx import AsyncClient
from unittest.mock import patch, MagicMock, AsyncMock
from src.api.main import app


class TestEmailRoutes:
    """Test suite for email processing API routes"""
    
    @pytest.fixture
    async def client(self):
        """Create test client"""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac
    
    @pytest.fixture
    def mock_services(self):
        """Mock external services"""
        with patch('src.api.routes.emails.response_generator') as mock_rg, \
             patch('src.api.routes.emails.db_manager') as mock_db, \
             patch('src.api.routes.emails.email_service') as mock_email:
            
            # Mock response generator
            mock_result = MagicMock()
            mock_result.response_text = "Thank you for your inquiry."
            mock_result.status.value = 'approved'
            mock_result.response_quality.value = 'good'
            mock_result.quality_score = 0.85
            mock_result.generation_time_ms = 1500
            mock_result.total_tokens_used = 200
            mock_result.classification_result = {
                'category': 'CUSTOMER_SUPPORT',
                'confidence': 0.85,
                'category_scores': {'CUSTOMER_SUPPORT': 0.85}
            }
            mock_result.confidence_analysis.overall_confidence = 0.85
            mock_result.escalation_decision.should_escalate = False
            mock_result.reasoning = ["High confidence classification"]
            
            mock_rg.generate_response = AsyncMock(return_value=mock_result)
            
            # Mock database
            mock_db.store_email_record = AsyncMock(return_value="test-email-123")
            mock_db.store_classification_result = AsyncMock()
            mock_db.store_confidence_analysis = AsyncMock()
            mock_db.store_escalation_decision = AsyncMock()
            mock_db.store_response = AsyncMock(return_value="test-response-456")
            mock_db.store_performance_metric = AsyncMock()
            mock_db.get_performance_summary = AsyncMock(return_value={
                'total_emails': 100,
                'processed_emails': 95,
                'escalated_emails': 10,
                'avg_processing_time_ms': 1500,
                'avg_quality_score': 0.82
            })
            
            # Mock email service
            mock_email.analyze_email_content = MagicMock(return_value={
                'is_suspicious': False,
                'is_likely_spam': False,
                'spam_score': 0.1
            })
            
            yield {
                'response_generator': mock_rg,
                'db_manager': mock_db,
                'email_service': mock_email
            }
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = await client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    @pytest.mark.asyncio
    async def test_process_email_success(self, client, mock_services):
        """Test successful email processing"""
        email_data = {
            "email": {
                "subject": "Help with account",
                "body": "I need help resetting my password",
                "sender": "test@example.com",
                "sender_name": "Test User"
            }
        }
        
        response = await client.post("/emails/process", json=email_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "email_id" in data
        assert "status" in data
        assert "classification" in data
        assert "processing_time_ms" in data
    
    @pytest.mark.asyncio
    async def test_process_email_validation_error(self, client):
        """Test email processing with invalid data"""
        invalid_data = {
            "email": {
                "subject": "Test",
                # Missing required fields
            }
        }
        
        response = await client.post("/emails/process", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, client, mock_services):
        """Test batch email processing"""
        batch_data = {
            "emails": [
                {
                    "subject": "Help 1",
                    "body": "First email",
                    "sender": "test1@example.com",
                    "sender_name": "User 1"
                },
                {
                    "subject": "Help 2", 
                    "body": "Second email",
                    "sender": "test2@example.com",
                    "sender_name": "User 2"
                }
            ]
        }
        
        response = await client.post("/emails/batch", json=batch_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "batch_id" in data
        assert "total_emails" in data
        assert "successful" in data
        assert "failed" in data
        assert "results" in data
        assert len(data["results"]) == 2
    
    @pytest.mark.asyncio
    async def test_classify_only_endpoint(self, client, mock_services):
        """Test classification-only endpoint"""
        email_data = {
            "email": {
                "subject": "Technical Issue",
                "body": "The API is returning errors",
                "sender": "dev@company.com",
                "sender_name": "Developer"
            }
        }
        
        response = await client.post("/emails/classify", json=email_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "classification" in data
        assert "content_analysis" in data
        
        classification = data["classification"]
        assert "category" in classification
        assert "confidence" in classification
        assert "structural_features" in classification
    
    @pytest.mark.asyncio
    async def test_get_categories_endpoint(self, client):
        """Test get categories endpoint"""
        response = await client.get("/emails/categories")
        assert response.status_code == 200
        
        data = response.json()
        assert "categories" in data
        assert len(data["categories"]) > 0
        
        # Check category structure
        category = data["categories"][0]
        assert "id" in category
        assert "name" in category
        assert "description" in category
    
    @pytest.mark.asyncio
    async def test_performance_summary_endpoint(self, client, mock_services):
        """Test performance summary endpoint"""
        response = await client.get("/emails/performance/summary?hours=24")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "summary" in data
        assert "generated_at" in data
        
        summary = data["summary"]
        assert "total_emails" in summary
        assert "processed_emails" in summary
        assert "escalated_emails" in summary
        assert "performance_grade" in summary
        assert "recommendations" in summary
    
    @pytest.mark.asyncio
    async def test_performance_summary_validation(self, client, mock_services):
        """Test performance summary parameter validation"""
        # Invalid hours parameter
        response = await client.get("/emails/performance/summary?hours=200")
        assert response.status_code == 400
        
        # Valid parameters
        response = await client.get("/emails/performance/summary?hours=1")
        assert response.status_code == 200
        
        response = await client.get("/emails/performance/summary?hours=168")
        assert response.status_code == 200
    
    @pytest.mark.asyncio 
    async def test_api_key_authentication(self, client):
        """Test API key authentication"""
        # Test without API key (should work in test environment)
        response = await client.get("/emails/categories")
        assert response.status_code == 200
        
        # Test with API key
        headers = {"Authorization": "Bearer test-key"}
        response = await client.get("/emails/categories", headers=headers)
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_email_status_endpoint(self, client):
        """Test email status lookup endpoint"""
        response = await client.get("/emails/status/test-email-123")
        assert response.status_code == 200
        
        data = response.json()
        assert "email_id" in data
        assert "status" in data
        # Currently not implemented, so should return placeholder
        assert data["status"] == "not_implemented"
    
    @pytest.mark.asyncio
    async def test_error_handling(self, client):
        """Test API error handling"""
        # Test with completely invalid JSON
        response = await client.post(
            "/emails/process", 
            content="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    @pytest.mark.asyncio
    async def test_large_email_processing(self, client, mock_services):
        """Test processing of large emails"""
        large_body = "This is a test email. " * 1000  # ~23KB email
        
        email_data = {
            "email": {
                "subject": "Large Email Test",
                "body": large_body,
                "sender": "test@example.com",
                "sender_name": "Test User"
            }
        }
        
        response = await client.post("/emails/process", json=email_data)
        assert response.status_code == 200
        
        # Should process successfully despite size
        data = response.json()
        assert "email_id" in data
        assert "classification" in data
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, client, mock_services):
        """Test handling concurrent requests"""
        email_data = {
            "email": {
                "subject": "Concurrent Test",
                "body": "Testing concurrent processing",
                "sender": "test@example.com",
                "sender_name": "Test User"
            }
        }
        
        # Send 5 concurrent requests
        tasks = []
        for i in range(5):
            task = client.post("/emails/process", json=email_data)
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert "email_id" in data
    
    @pytest.mark.asyncio
    async def test_response_format_consistency(self, client, mock_services):
        """Test that responses have consistent format"""
        email_data = {
            "email": {
                "subject": "Format Test",
                "body": "Testing response format",
                "sender": "test@example.com", 
                "sender_name": "Test User"
            }
        }
        
        response = await client.post("/emails/process", json=email_data)
        assert response.status_code == 200
        
        data = response.json()
        
        # Check required fields
        required_fields = [
            "email_id", "status", "processing_time_ms", "processed_at"
        ]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Check data types
        assert isinstance(data["processing_time_ms"], int)
        assert data["processing_time_ms"] > 0
        assert isinstance(data["processed_at"], str)
    
    @pytest.mark.asyncio
    async def test_batch_processing_partial_failure(self, client, mock_services):
        """Test batch processing with some failures"""
        # Mock one failure
        mock_services['response_generator'].generate_response.side_effect = [
            MagicMock(status=MagicMock(value='approved')),  # Success
            Exception("Processing failed"),  # Failure
            MagicMock(status=MagicMock(value='approved'))   # Success
        ]
        
        batch_data = {
            "emails": [
                {"subject": "Email 1", "body": "First", "sender": "test1@example.com"},
                {"subject": "Email 2", "body": "Second", "sender": "test2@example.com"},
                {"subject": "Email 3", "body": "Third", "sender": "test3@example.com"}
            ]
        }
        
        response = await client.post("/emails/batch", json=batch_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["total_emails"] == 3
        assert data["successful"] >= 1  # At least one should succeed
        assert data["failed"] >= 1      # At least one should fail
    
    @pytest.mark.asyncio
    async def test_special_characters_handling(self, client, mock_services):
        """Test handling of emails with special characters"""
        special_email_data = {
            "email": {
                "subject": "Special chars: àáâãäåæçèéêë",
                "body": "Email with émojis 🚀 and spëcial charâcters: ñáéíóú",
                "sender": "tëst@exàmple.com",
                "sender_name": "Tést Üser"
            }
        }
        
        response = await client.post("/emails/process", json=special_email_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "email_id" in data
        assert "classification" in data
    
    @pytest.mark.asyncio
    async def test_empty_email_handling(self, client, mock_services):
        """Test handling of empty or minimal emails"""
        minimal_email_data = {
            "email": {
                "subject": "",
                "body": "",
                "sender": "test@example.com",
                "sender_name": "Test User"
            }
        }
        
        response = await client.post("/emails/process", json=minimal_email_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "email_id" in data
        # Should handle gracefully, possibly with low confidence
        if "classification" in data:
            assert "confidence" in data["classification"]