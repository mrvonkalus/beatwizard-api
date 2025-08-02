#!/usr/bin/env python3
"""
Verify BeatWizard deployment readiness
Run this script before deploying to catch any issues
"""

import sys
import os
import importlib.util
from pathlib import Path

def check_file_exists(file_path, description):
    """Check if a required file exists"""
    if Path(file_path).exists():
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} - NOT FOUND")
        return False

def check_import(module_name):
    """Check if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✅ Import: {module_name}")
        return True
    except ImportError as e:
        print(f"❌ Import: {module_name} - {e}")
        return False

def check_environment():
    """Check environment variables"""
    required_vars = ['SECRET_KEY']
    optional_vars = ['OPENAI_API_KEY', 'CORS_ORIGINS']
    
    print("\n🔍 Environment Variables:")
    all_good = True
    
    for var in required_vars:
        if os.environ.get(var):
            print(f"✅ {var}: Set")
        else:
            print(f"⚠️  {var}: Not set (will use default)")
    
    for var in optional_vars:
        if os.environ.get(var):
            print(f"✅ {var}: Set")
        else:
            print(f"💡 {var}: Not set (optional)")
    
    return all_good

def main():
    """Run all deployment checks"""
    print("🔍 BeatWizard Deployment Verification")
    print("=" * 50)
    
    all_checks_passed = True
    
    # Check required files
    print("\n📁 Required Files:")
    required_files = [
        ("app.py", "Main application file"),
        ("requirements.txt", "Python dependencies"),
        ("railway.json", "Railway configuration"),
        ("Procfile", "Process configuration"),
        ("runtime.txt", "Python version"),
    ]
    
    for file_path, description in required_files:
        if not check_file_exists(file_path, description):
            all_checks_passed = False
    
    # Check critical imports
    print("\n📦 Critical Imports:")
    critical_imports = [
        "flask",
        "flask_cors", 
        "beatwizard",
        "loguru",
        "numpy",
        "librosa"
    ]
    
    for module in critical_imports:
        if not check_import(module):
            all_checks_passed = False
    
    # Check environment
    check_environment()
    
    # Check BeatWizard can initialize
    print("\n🎵 BeatWizard Initialization:")
    try:
        from beatwizard import EnhancedAudioAnalyzer
        analyzer = EnhancedAudioAnalyzer()
        print("✅ BeatWizard analyzer initialized successfully")
    except Exception as e:
        print(f"❌ BeatWizard initialization failed: {e}")
        all_checks_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    if all_checks_passed:
        print("🚀 ✅ ALL CHECKS PASSED - Ready for deployment!")
        print("\nNext steps:")
        print("1. git add . && git commit -m 'Ready for Railway'")
        print("2. git push origin main")
        print("3. Deploy on Railway")
        print("4. Set environment variables in Railway")
        sys.exit(0)
    else:
        print("❌ SOME CHECKS FAILED - Fix issues before deploying")
        print("\nPlease address the issues above and run this script again.")
        sys.exit(1)

if __name__ == "__main__":
    main()