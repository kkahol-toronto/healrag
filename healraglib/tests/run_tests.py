#!/usr/bin/env python3
"""
Test runner for HEALRAG library

Runs all unit tests and provides a summary of results.
"""

import unittest
import sys
import os

# Import from the parent package
from ..tests.test_storage_manager import run_tests as run_storage_tests
from ..tests.test_cli import run_cli_tests
from ..tests.test_content_manager import TestContentManager


def run_all_tests():
    """Run all test suites."""
    print("🚀 HEALRAG Test Suite")
    print("=" * 50)
    
    # Run storage manager tests
    print("\n📦 Testing StorageManager...")
    storage_success = run_storage_tests()
    
    # Run CLI tests
    print("\n🖥️  Testing CLI...")
    cli_success = run_cli_tests()
    
    # Run ContentManager tests
    print("\n📚 Testing ContentManager...")
    content_suite = unittest.TestLoader().loadTestsFromTestCase(TestContentManager)
    content_runner = unittest.TextTestRunner(verbosity=2)
    content_result = content_runner.run(content_suite)
    content_success = content_result.wasSuccessful()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   StorageManager: {'✅ PASSED' if storage_success else '❌ FAILED'}")
    print(f"   CLI: {'✅ PASSED' if cli_success else '❌ FAILED'}")
    print(f"   ContentManager: {'✅ PASSED' if content_success else '❌ FAILED'}")
    
    overall_success = storage_success and cli_success and content_success
    print(f"\n🎯 Overall: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    return overall_success


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 