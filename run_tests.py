"""
Test runner script for the AI Email Agent system.
Runs comprehensive tests and generates detailed reports.
"""

import sys
import os
import time
import pytest
import subprocess
from pathlib import Path


def main():
    """Run comprehensive test suite and generate reports"""
    
    print("AI Email Agent - Comprehensive Test Suite")
    print("=" * 60)
    
    # Ensure we're in the project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Add src to Python path for imports
    sys.path.insert(0, str(project_root / "src"))
    
    # Test configuration
    test_args = [
        "--verbose",
        "--tb=short",
        "--color=yes",
        "--durations=10",  # Show 10 slowest tests
        "--cov=src",       # Coverage for src directory
        "--cov-report=html:htmlcov",
        "--cov-report=term-missing",
        "--cov-report=xml",
        "--junit-xml=test-results.xml",
        "tests/"
    ]
    
    # Run different test categories
    test_categories = {
        "Unit Tests": {
            "path": "tests/unit/",
            "description": "Core component unit tests"
        },
        "Integration Tests": {
            "path": "tests/integration/",
            "description": "End-to-end pipeline integration tests"
        },
        "API Tests": {
            "path": "tests/api/",
            "description": "FastAPI endpoint tests"
        },
        "Performance Tests": {
            "path": "tests/performance/",
            "description": "Performance benchmarks and load tests",
            "markers": "-m not slow"  # Skip slow tests by default
        }
    }
    
    overall_start = time.time()
    all_results = {}
    
    for category_name, config in test_categories.items():
        print(f"\nRunning {category_name}")
        print(f"   {config['description']}")
        print("-" * 40)
        
        category_start = time.time()
        
        # Build test command
        cmd_args = [sys.executable, "-m", "pytest"] + test_args.copy()
        
        # Add markers if specified
        if "markers" in config:
            cmd_args.append(config["markers"])
        
        # Add specific test path
        cmd_args.append(config["path"])
        
        # Run tests
        try:
            result = subprocess.run(cmd_args, capture_output=True, text=True, timeout=300)
            category_time = time.time() - category_start
            
            all_results[category_name] = {
                'success': result.returncode == 0,
                'time': category_time,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
            
            if result.returncode == 0:
                print(f"[PASS] {category_name} ({category_time:.1f}s)")
            else:
                print(f"[FAIL] {category_name} ({category_time:.1f}s)")
                print("Error output:")
                print(result.stderr[-500:] if result.stderr else "No error output")
            
        except subprocess.TimeoutExpired:
            print(f"[TIMEOUT] {category_name} (5min limit)")
            all_results[category_name] = {
                'success': False,
                'time': 300,
                'error': 'Timeout after 5 minutes'
            }
        
        except Exception as e:
            print(f"[ERROR] {category_name}: {e}")
            all_results[category_name] = {
                'success': False,
                'time': time.time() - category_start,
                'error': str(e)
            }
    
    # Generate summary report
    print("\n" + "=" * 60)
    print("TEST SUMMARY REPORT")
    print("=" * 60)
    
    total_time = time.time() - overall_start
    passed_categories = sum(1 for r in all_results.values() if r['success'])
    total_categories = len(all_results)
    
    print(f"Overall Result: {passed_categories}/{total_categories} categories passed")
    print(f"Total Time: {total_time:.1f} seconds")
    print()
    
    for category, result in all_results.items():
        status = "[PASS]" if result['success'] else "[FAIL]"
        time_str = f"{result['time']:.1f}s"
        print(f"{status:<8} {category:<20} ({time_str})")
        
        if not result['success'] and 'error' in result:
            print(f"         Error: {result['error']}")
    
    # Generate detailed HTML report if coverage was generated
    if os.path.exists("htmlcov/index.html"):
        print(f"\nCoverage report generated: htmlcov/index.html")
    
    if os.path.exists("test-results.xml"):
        print(f"JUnit XML report: test-results.xml")
    
    # Run quick system validation
    print("\nQUICK SYSTEM VALIDATION")
    print("-" * 40)
    
    try:
        print("Testing basic imports...")
        from src.agents.classifier import EmailClassifier
        from src.agents.confidence_scorer import ConfidenceScorer
        from src.agents.escalation_engine import EscalationEngine
        from src.agents.responder import ResponseGenerator
        from email_agent import EmailAgent
        print("[PASS] All imports successful")
        
        print("Testing basic functionality...")
        agent = EmailAgent()
        test_result = agent.classify_email("Hello, I need help with my account.")
        assert 'category' in test_result
        assert 'confidence' in test_result
        print("[PASS] Basic functionality working")
        
        print("Testing API server startup...")
        try:
            from src.api.main import app
            print("[PASS] FastAPI app loads successfully")
        except Exception as e:
            print(f"[WARN] API startup warning: {e}")
        
    except Exception as e:
        print(f"[FAIL] System validation failed: {e}")
        return 1
    
    # Final recommendations
    print("\nRECOMMENDATIONS")
    print("-" * 40)
    
    if passed_categories == total_categories:
        print("All tests passed! System is ready for deployment.")
        print("   Consider running performance tests with real OpenAI API")
        print("   to validate end-to-end latency.")
    else:
        print("Some tests failed. Please address the following:")
        for category, result in all_results.items():
            if not result['success']:
                print(f"   - Fix {category} issues")
        
        print("   Run individual test categories to debug:")
        print("   python -m pytest tests/unit/ -v")
        print("   python -m pytest tests/integration/ -v")
    
    # Return appropriate exit code
    return 0 if passed_categories == total_categories else 1


if __name__ == "__main__":
    sys.exit(main())