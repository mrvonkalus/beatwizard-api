#!/usr/bin/env python3
"""
Test your live BeatWizard API deployment
Works with Railway, Render, or any cloud platform
"""

import requests
import json
import time

def test_live_api(base_url):
    """Comprehensive test of live API deployment"""
    print(f"🧪 Testing Live API: {base_url}")
    print("=" * 60)
    
    tests = [
        ("Health Check", "GET", "/"),
        ("Detailed Health", "GET", "/health"),
        ("API Info", "GET", "/api/info"),
        ("Demo Analysis", "GET", "/api/demo"),
        ("Deployment Status", "GET", "/api/status"),
        ("Echo Test", "POST", "/api/echo", {"test": "deployment_success"}),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, method, endpoint, data in [(*t, None) if len(t) == 3 else t for t in tests]:
        try:
            url = f"{base_url}{endpoint}"
            
            if method == "GET":
                response = requests.get(url, timeout=10)
            else:
                response = requests.post(url, json=data, timeout=10)
            
            if response.status_code == 200:
                print(f"✅ {test_name}: SUCCESS")
                result = response.json()
                
                # Show key info
                if 'message' in result:
                    print(f"   💬 {result['message']}")
                if 'deployment_phase' in result:
                    print(f"   🚀 Phase: {result['deployment_phase']}")
                if 'mode' in result:
                    print(f"   🔧 Mode: {result['mode']}")
                
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED ({response.status_code})")
                print(f"   Error: {response.text[:100]}")
                
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
        
        print()
    
    # Summary
    print("=" * 60)
    print(f"🏁 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 🚀 DEPLOYMENT SUCCESSFUL! 🚀 🎉")
        print("✨ Your BeatWizard API is live and working perfectly!")
        print("\n🔗 Your API Endpoints:")
        print(f"   • Health: {base_url}/health")
        print(f"   • Info: {base_url}/api/info") 
        print(f"   • Demo: {base_url}/api/demo")
        print(f"   • Status: {base_url}/api/status")
        
        print("\n🎯 Next Steps:")
        print("   1. Add audio processing libraries")
        print("   2. Enable full BeatWizard analysis")
        print("   3. Connect to your frontend")
        
        return True
    else:
        print("❌ Some issues detected - check deployment logs")
        return False

def main():
    """Test different deployment URLs"""
    
    # Test different possible URLs
    urls_to_test = [
        ("Railway (provided)", "https://beatwizard-minimal-api-production.up.railway.app"),
        ("Railway (alt)", "https://beatwizard-api-production.up.railway.app"),
        # Add your Render URL here if you use it:
        # ("Render", "https://beatwizard-minimal-api.onrender.com"),
    ]
    
    print("🔍 Testing BeatWizard API Deployments")
    print("=" * 80)
    
    for platform, url in urls_to_test:
        print(f"\n🌐 Testing {platform}")
        print(f"🔗 URL: {url}")
        print("-" * 60)
        
        success = test_live_api(url)
        
        if success:
            print(f"✅ {platform} is working perfectly!")
            break
        else:
            print(f"❌ {platform} has issues")
        
        print("\n" + "=" * 80)
    
    print("\n💡 If all tests fail:")
    print("   1. Check Railway/Render dashboard for deployment errors")
    print("   2. Verify repository is connected correctly")
    print("   3. Check deployment logs for specific errors")
    print("   4. Try the backup deployment plan (deploy-to-render.md)")

if __name__ == "__main__":
    main()