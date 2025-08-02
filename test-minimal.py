#!/usr/bin/env python3
"""
Test script for BeatWizard minimal deployment
Verify all endpoints work without audio dependencies
"""

import requests
import json
import time

def test_endpoint(base_url, endpoint, method='GET', data=None, files=None):
    """Test a single endpoint"""
    url = f"{base_url}{endpoint}"
    
    try:
        if method == 'GET':
            response = requests.get(url, timeout=10)
        elif method == 'POST':
            if files:
                response = requests.post(url, files=files, data=data, timeout=10)
            else:
                response = requests.post(url, json=data, timeout=10)
        
        print(f"✅ {method} {endpoint}: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                if 'message' in result:
                    print(f"   📝 {result['message']}")
                if 'deployment_phase' in result:
                    print(f"   🚀 Phase: {result['deployment_phase']}")
            except:
                print(f"   📄 Response: {response.text[:100]}...")
        else:
            print(f"   ❌ Error: {response.text[:200]}...")
            
        return response.status_code == 200
        
    except Exception as e:
        print(f"❌ {method} {endpoint}: ERROR - {e}")
        return False

def run_minimal_tests(base_url):
    """Run all tests for minimal deployment"""
    print(f"🧪 Testing BeatWizard Minimal API: {base_url}")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 0
    
    # Test all GET endpoints
    get_endpoints = [
        "/",
        "/health", 
        "/api/info",
        "/api/demo",
        "/api/status"
    ]
    
    for endpoint in get_endpoints:
        total_tests += 1
        if test_endpoint(base_url, endpoint, 'GET'):
            tests_passed += 1
        print()
    
    # Test POST endpoints
    print("📤 Testing POST endpoints:")
    
    # Test echo endpoint
    total_tests += 1
    echo_data = {"test": "hello", "timestamp": time.time()}
    if test_endpoint(base_url, "/api/echo", 'POST', data=echo_data):
        tests_passed += 1
    print()
    
    # Test upload endpoint (without actual file)
    total_tests += 1
    print("📁 Testing upload endpoint (no file):")
    if test_endpoint(base_url, "/api/upload", 'POST'):
        # This should fail with 400, which is expected
        print("   ℹ️  Expected 400 error for missing file")
    print()
    
    # Summary
    print("=" * 60)
    print(f"🏁 Test Results: {tests_passed}/{total_tests} passed")
    
    if tests_passed == len(get_endpoints) + 1:  # All GET + echo POST
        print("🚀 ✅ ALL CRITICAL TESTS PASSED")
        print("✨ Ready for Railway deployment!")
        return True
    else:
        print("❌ Some tests failed - check endpoints")
        return False

def main():
    """Main test function"""
    environments = [
        ("Local", "http://localhost:8080"),
        # Add Railway URL here once deployed
        # ("Railway", "https://your-app.railway.app"),
    ]
    
    for env_name, base_url in environments:
        print(f"🌐 Testing {env_name} Environment")
        print(f"🔗 URL: {base_url}")
        print("-" * 60)
        
        success = run_minimal_tests(base_url)
        
        if success:
            print(f"✅ {env_name} is working perfectly!")
        else:
            print(f"❌ {env_name} has issues")
        
        print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    main()