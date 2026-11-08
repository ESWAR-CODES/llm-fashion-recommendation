"""
run_all_tests.py
Master Consolidated Test Suite Runner
LLM-Assisted Personalized Fashion Recommendation Using User Preferences and Semantic Retrieval

Runs all unit and integration test suites:
1. test_catalog.py
2. test_embeddings.py
3. test_profiles.py
4. test_ranking.py
5. test_llm_extraction.py
6. test_explanations.py
7. test_integration.py
"""

import os
import sys
import unittest
import time
from pathlib import Path

# Add fashion_project root to sys.path
TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_consolidated_test_suite() -> bool:
    print("=" * 80)
    print("CONSOLIDATED TEST SUITE EXECUTION")
    print("=" * 80)
    start_time = time.time()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Discover and add all tests from the tests directory
    suite.addTests(loader.discover(start_dir=str(TEST_DIR), pattern="test_*.py"))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"TESTS RUN: {result.testsRun} | FAILURES: {len(result.failures)} | ERRORS: {len(result.errors)} | TIME: {elapsed:.2f}s")
    print(f"OVERALL STATUS: {'ALL TESTS PASSED [PASS]' if result.wasSuccessful() else 'TESTS FAILED [FAIL]'}")
    print("=" * 80)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_consolidated_test_suite()
    sys.exit(0 if success else 1)
