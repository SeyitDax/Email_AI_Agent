"""
Performance benchmark tests for the AI Email Agent system.
Tests speed, throughput, memory usage, and scalability.
"""

import pytest
import time
import asyncio
import psutil
import os
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch, MagicMock, AsyncMock

from src.agents.classifier import EmailClassifier
from src.agents.confidence_scorer import ConfidenceScorer 
from src.agents.responder import ResponseGenerator
from email_agent import EmailAgent


class TestPerformanceBenchmarks:
    """Performance benchmark test suite"""
    
    @pytest.fixture
    def mock_openai_fast(self):
        """Fast mock for OpenAI to test core logic performance"""
        with patch('src.agents.responder.ChatOpenAI') as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm.bind.return_value = mock_llm
            
            async def fast_response(messages):
                # Simulate fast API response
                await asyncio.sleep(0.01)  # 10ms delay
                mock_resp = MagicMock()
                mock_resp.content = "Thank you for your inquiry. We'll help you right away."
                return mock_resp
            
            mock_llm.ainvoke = fast_response
            mock_llm_class.return_value = mock_llm
            
            with patch('src.agents.responder.get_openai_callback') as mock_callback:
                callback_ctx = MagicMock()
                callback_ctx.total_tokens = 150
                callback_ctx.prompt_tokens = 100
                callback_ctx.completion_tokens = 50
                callback_ctx.total_cost = 0.0005
                callback_ctx.__enter__ = MagicMock(return_value=callback_ctx)
                callback_ctx.__exit__ = MagicMock(return_value=None)
                mock_callback.return_value = callback_ctx
                
                yield mock_llm
    
    def test_classifier_speed_benchmark(self, sample_emails, performance_benchmarks):
        """Test classifier processing speed"""
        classifier = EmailClassifier()
        
        start_time = time.time()
        results = []
        
        # Process all sample emails
        for email_content in sample_emails.values():
            result = classifier.classify(email_content)
            results.append(result)
        
        total_time = time.time() - start_time
        avg_time_per_email = (total_time / len(sample_emails)) * 1000  # Convert to ms
        
        print(f"Classifier average time: {avg_time_per_email:.2f}ms per email")
        
        # Should meet speed benchmark
        max_time_per_email = performance_benchmarks['max_processing_time_ms'] / 20  # Classifier should be fast
        assert avg_time_per_email < max_time_per_email, f"Classifier too slow: {avg_time_per_email:.2f}ms"
        
        # All results should be valid
        for result in results:
            assert 'category' in result
            assert 'confidence' in result
            assert 0.0 <= result['confidence'] <= 1.0
    
    def test_confidence_scorer_speed_benchmark(self, sample_emails, classification_results):
        """Test confidence scorer processing speed"""
        scorer = ConfidenceScorer()
        
        start_time = time.time()
        results = []
        
        # Process confidence analysis
        for email_content in list(sample_emails.values())[:10]:  # Test subset
            result = scorer.analyze_confidence(classification_results, email_content)
            results.append(result)
        
        total_time = time.time() - start_time
        avg_time_per_analysis = (total_time / len(results)) * 1000
        
        print(f"Confidence scorer average time: {avg_time_per_analysis:.2f}ms per analysis")
        
        # Should be reasonably fast
        assert avg_time_per_analysis < 100, f"Confidence scorer too slow: {avg_time_per_analysis:.2f}ms"
        
        # Results should be valid
        for result in results:
            assert 0.0 <= result.overall_confidence <= 1.0
            assert len(result.reasoning) > 0
    
    @pytest.mark.asyncio
    async def test_end_to_end_speed_benchmark(self, sample_emails, mock_openai_fast, performance_benchmarks):
        """Test complete pipeline speed"""
        generator = ResponseGenerator()
        
        start_time = time.time()
        results = []
        
        # Test subset of emails for speed
        test_emails = list(sample_emails.values())[:5]
        
        for email_content in test_emails:
            result = await generator.generate_response(email_content)
            results.append(result)
        
        total_time = time.time() - start_time
        avg_time_per_email = (total_time / len(test_emails)) * 1000
        
        print(f"End-to-end average time: {avg_time_per_email:.2f}ms per email")
        
        # Should meet benchmark (allowing for mock delay)
        assert avg_time_per_email < performance_benchmarks['max_processing_time_ms']
        
        # Check that processing times are recorded
        for result in results:
            assert result.generation_time_ms > 0
            assert result.generation_time_ms < performance_benchmarks['max_processing_time_ms']
    
    @pytest.mark.asyncio
    async def test_concurrent_processing_throughput(self, sample_emails, mock_openai_fast):
        """Test system throughput under concurrent load"""
        generator = ResponseGenerator()
        
        # Test concurrent processing
        num_concurrent = 10
        email_content = sample_emails['customer_support']
        
        start_time = time.time()
        
        # Create concurrent tasks
        tasks = []
        for _ in range(num_concurrent):
            task = generator.generate_response(email_content)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        throughput = num_concurrent / total_time  # emails per second
        
        print(f"Concurrent throughput: {throughput:.2f} emails/second")
        
        # Should handle reasonable concurrent load
        assert throughput > 5.0, f"Throughput too low: {throughput:.2f} emails/second"
        
        # All should succeed
        assert len(results) == num_concurrent
        for result in results:
            assert result.response_text is not None
    
    def test_memory_usage_benchmark(self, sample_emails):
        """Test memory usage under normal operation"""
        import gc
        gc.collect()  # Clean up before test
        
        # Get initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create components
        classifier = EmailClassifier()
        scorer = ConfidenceScorer() 
        agent = EmailAgent()
        
        # Process multiple emails
        for _ in range(3):  # Process each email multiple times
            for email_content in sample_emails.values():
                # Use legacy agent to avoid async complications
                result = agent.classify_email(email_content)
                confidence_result = scorer.analyze_confidence({
                    'category': 'CUSTOMER_SUPPORT',
                    'confidence': 0.8,
                    'category_scores': {'CUSTOMER_SUPPORT': 0.8},
                    'structural_features': {},
                    'sentiment_score': 0.0,
                    'complexity_score': 0.3
                }, email_content)
        
        # Check final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory usage: Initial={initial_memory:.1f}MB, Final={final_memory:.1f}MB, Increase={memory_increase:.1f}MB")
        
        # Memory increase should be reasonable (less than 100MB for this test)
        assert memory_increase < 100, f"Memory usage too high: {memory_increase:.1f}MB increase"
    
    def test_batch_processing_scalability(self, sample_emails):
        """Test scalability with batch processing"""
        agent = EmailAgent()
        
        # Test different batch sizes
        batch_sizes = [1, 5, 10, 20]
        times = []
        
        for batch_size in batch_sizes:
            emails_to_process = list(sample_emails.values()) * (batch_size // len(sample_emails) + 1)
            emails_to_process = emails_to_process[:batch_size]
            
            start_time = time.time()
            
            results = []
            for email_content in emails_to_process:
                result = agent.classify_email(email_content)  # Use sync method
                results.append(result)
            
            processing_time = time.time() - start_time
            times.append(processing_time)
            
            print(f"Batch size {batch_size}: {processing_time:.3f}s total, {(processing_time/batch_size*1000):.1f}ms per email")
        
        # Processing time per email should remain roughly constant (good scalability)
        time_per_email = [(times[i] / batch_sizes[i]) for i in range(len(batch_sizes))]
        
        # Check that time per email doesn't increase significantly with batch size
        if len(time_per_email) > 1:
            max_increase = max(time_per_email) / min(time_per_email)
            assert max_increase < 2.0, f"Performance degrades too much with batch size: {max_increase:.2f}x increase"
    
    def test_classification_accuracy_vs_speed_tradeoff(self, sample_emails):
        """Test accuracy vs speed tradeoff"""
        classifier = EmailClassifier()
        
        # Process emails and measure both speed and quality
        results = []
        times = []
        
        for email_content in sample_emails.values():
            start_time = time.time()
            result = classifier.classify(email_content)
            processing_time = time.time() - start_time
            
            results.append(result)
            times.append(processing_time * 1000)  # Convert to ms
        
        avg_time = sum(times) / len(times)
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        
        print(f"Accuracy vs Speed: {avg_confidence:.3f} confidence, {avg_time:.1f}ms avg time")
        
        # Should maintain good accuracy even with speed requirements
        assert avg_confidence > 0.6, f"Average confidence too low: {avg_confidence:.3f}"
        assert avg_time < 200, f"Average time too high: {avg_time:.1f}ms"
    
    @pytest.mark.asyncio
    async def test_error_recovery_performance(self, sample_emails, performance_benchmarks):
        """Test performance impact of error handling"""
        generator = ResponseGenerator()
        
        # Test with failing OpenAI calls
        with patch('src.agents.responder.ChatOpenAI', side_effect=Exception("API Error")):
            start_time = time.time()
            
            results = []
            for email_content in list(sample_emails.values())[:3]:  # Test subset
                result = await generator.generate_response(email_content)
                results.append(result)
            
            error_handling_time = time.time() - start_time
            avg_error_time = (error_handling_time / len(results)) * 1000
        
        print(f"Error handling average time: {avg_error_time:.2f}ms per email")
        
        # Error handling should be reasonably fast
        assert avg_error_time < performance_benchmarks['max_processing_time_ms']
        
        # Should all use fallbacks
        for result in results:
            assert result.fallback_used == True
            assert result.error_message is not None
    
    def test_component_initialization_time(self):
        """Test initialization time of components"""
        
        # Test classifier initialization
        start_time = time.time()
        classifier = EmailClassifier()
        classifier_init_time = (time.time() - start_time) * 1000
        
        # Test scorer initialization
        start_time = time.time()
        scorer = ConfidenceScorer()
        scorer_init_time = (time.time() - start_time) * 1000
        
        # Test agent initialization
        start_time = time.time()
        agent = EmailAgent()
        agent_init_time = (time.time() - start_time) * 1000
        
        print(f"Initialization times: Classifier={classifier_init_time:.1f}ms, "
              f"Scorer={scorer_init_time:.1f}ms, Agent={agent_init_time:.1f}ms")
        
        # Initialization should be fast
        assert classifier_init_time < 100, f"Classifier init too slow: {classifier_init_time:.1f}ms"
        assert scorer_init_time < 50, f"Scorer init too slow: {scorer_init_time:.1f}ms"
        # Agent init can be slower due to ResponseGenerator
        assert agent_init_time < 1000, f"Agent init too slow: {agent_init_time:.1f}ms"
    
    @pytest.mark.asyncio
    async def test_sustained_load_performance(self, sample_emails, mock_openai_fast):
        """Test performance under sustained load"""
        generator = ResponseGenerator()
        
        # Simulate sustained processing
        num_emails = 20
        email_content = sample_emails['customer_support']
        
        start_time = time.time()
        processing_times = []
        
        for i in range(num_emails):
            email_start = time.time()
            result = await generator.generate_response(email_content)
            email_time = (time.time() - email_start) * 1000
            processing_times.append(email_time)
            
            # Small delay between emails
            await asyncio.sleep(0.01)
        
        total_time = time.time() - start_time
        
        # Check for performance degradation over time
        first_half = processing_times[:num_emails//2]
        second_half = processing_times[num_emails//2:]
        
        avg_first_half = sum(first_half) / len(first_half)
        avg_second_half = sum(second_half) / len(second_half)
        
        print(f"Sustained load: First half={avg_first_half:.1f}ms, Second half={avg_second_half:.1f}ms")
        
        # Performance shouldn't degrade significantly
        degradation_ratio = avg_second_half / avg_first_half
        assert degradation_ratio < 1.5, f"Performance degraded too much: {degradation_ratio:.2f}x slower"
        
        # Overall throughput should be reasonable
        throughput = num_emails / total_time
        assert throughput > 2.0, f"Sustained throughput too low: {throughput:.2f} emails/second"
    
    def test_large_email_processing_performance(self):
        """Test performance with large emails"""
        classifier = EmailClassifier()
        
        # Create emails of different sizes
        base_email = "I need help with my account. "
        email_sizes = [
            (100, base_email * 10),      # ~2KB
            (500, base_email * 50),      # ~10KB  
            (1000, base_email * 100),    # ~20KB
            (2000, base_email * 200),    # ~40KB
        ]
        
        times = []
        for size_desc, email_content in email_sizes:
            start_time = time.time()
            result = classifier.classify(email_content)
            processing_time = (time.time() - start_time) * 1000
            times.append(processing_time)
            
            print(f"Email size ~{size_desc} words: {processing_time:.1f}ms")
            
            # Should still classify successfully
            assert 'category' in result
            assert result['confidence'] > 0.0
        
        # Processing time should scale reasonably with size
        if len(times) > 1:
            time_ratio = times[-1] / times[0]  # Largest vs smallest
            assert time_ratio < 5.0, f"Processing time scales poorly with size: {time_ratio:.2f}x increase"