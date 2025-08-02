#!/usr/bin/env python3
"""
Simple API test script for BeatWizard deployment
"""

import requests
import json
import time

def test_health_endpoint(base_url):
    """Test the health check endpoint"""
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health Check: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Service Status: {data.get('status')}")
            print(f"📊 Analyzer: {data.get('checks', {}).get('analyzer', 'Unknown')}")
        else:
            print(f"❌ Health check failed: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_formats_endpoint(base_url):
    """Test the supported formats endpoint"""
    try:
        response = requests.get(f"{base_url}/api/formats")
        print(f"\nSupported Formats: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Formats: {data.get('supported_formats')}")
            print(f"📏 Max size: {data.get('max_file_size_mb')}MB")
        else:
            print(f"❌ Formats check failed: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Formats check error: {e}")
        return False

def test_upload_endpoint(base_url, test_file_path=None):
    """Test the upload endpoint (requires a test audio file)"""
    if not test_file_path:
        print("\n⚠️  No test file provided - skipping upload test")
        return True
    
    try:
        with open(test_file_path, 'rb') as f:
            files = {'file': f}
            data = {
                'skill_level': 'beginner',
                'genre': 'electronic',
                'goals': 'streaming'
            }
            
            print(f"\n🔄 Testing upload with {test_file_path}...")
            response = requests.post(f"{base_url}/api/upload", files=files, data=data)
            
        print(f"Upload Test: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Analysis completed in {result.get('analysis_time', 'unknown')}s")
            print(f"📊 Analysis ID: {result.get('analysis_id')}")
        else:
            print(f"❌ Upload test failed: {response.text}")
        
        return response.status_code == 200
    except FileNotFoundError:
        print(f"❌ Test file not found: {test_file_path}")
        return False
    except Exception as e:
        print(f"❌ Upload test error: {e}")
        return False

def main():
    """Run all API tests"""
    print("🧪 BeatWizard API Test Suite")
    print("=" * 40)
    
    # Test different environments
    environments = [
        ("Local", "http://localhost:8080"),
        # Add your Railway URL here once deployed
        # ("Production", "https://your-app.railway.app"),
    ]
    
    for env_name, base_url in environments:
        print(f"\n🌐 Testing {env_name}: {base_url}")
        print("-" * 30)
        
        # Test health endpoint
        health_ok = test_health_endpoint(base_url)
        
        # Test formats endpoint
        formats_ok = test_formats_endpoint(base_url)
        
        # Test upload endpoint (optional)
        # Uncomment and provide a test file path
        # upload_ok = test_upload_endpoint(base_url, "path/to/test.mp3")
        
        # Summary
        if health_ok and formats_ok:
            print(f"\n✅ {env_name} tests passed!")
        else:
            print(f"\n❌ {env_name} tests failed!")

if __name__ == "__main__":
    main()