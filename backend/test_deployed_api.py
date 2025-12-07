#!/usr/bin/env python3
"""
Test script for deployed Railway API endpoint.

Usage:
    python test_deployed_api.py --url https://your-api.up.railway.app
    python test_deployed_api.py --url https://your-api.up.railway.app --verbose
"""

import argparse
import io
import os
import sys
import time
from typing import Dict, Any
import requests


def create_test_image() -> bytes:
    """Create a test image for prediction testing."""
    # Try to use an actual test image from the test suite if available
    test_image_path = os.path.join(os.path.dirname(__file__), 'tests', 'images', 'happy.png')
    
    if os.path.exists(test_image_path):
        # Use a real test image that has a face
        with open(test_image_path, 'rb') as f:
            return f.read()
    else:
        # Fallback: Create a simple RGB image (won't have a face, but at least it's valid)
        try:
            from PIL import Image
            image = Image.new('RGB', (224, 224), color=(100, 150, 200))
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            return img_bytes.read()
        except ImportError:
            raise FileNotFoundError(
                "No test image found and PIL/Pillow not available. "
                "Please ensure backend/tests/images/happy.png exists or install Pillow."
            )


def test_health_endpoint(base_url: str, verbose: bool = False) -> Dict[str, Any]:
    """Test the /health endpoint."""
    print("\n" + "="*60)
    print("Testing Health Endpoint")
    print("="*60)
    
    url = f"{base_url}/health"
    start_time = time.time()
    
    try:
        response = requests.get(url, timeout=10)
        elapsed = (time.time() - start_time) * 1000
        
        if verbose:
            print(f"URL: {url}")
            print(f"Status Code: {response.status_code}")
            print(f"Response Time: {elapsed:.2f}ms")
            print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed")
            print(f"   Status: {data.get('status')}")
            print(f"   Model Loaded: {data.get('model_loaded')}")
            print(f"   Response Time: {elapsed:.2f}ms")
            return {"success": True, "data": data, "response_time_ms": elapsed}
        else:
            print(f"❌ Health check failed: Status {response.status_code}")
            return {"success": False, "status_code": response.status_code}
            
    except requests.exceptions.Timeout:
        print(f"❌ Health check timed out (may be cold start)")
        return {"success": False, "error": "timeout"}
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection error: {e}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"success": False, "error": str(e)}


def test_model_info_endpoint(base_url: str, verbose: bool = False) -> Dict[str, Any]:
    """Test the /info endpoint."""
    print("\n" + "="*60)
    print("Testing Model Info Endpoint")
    print("="*60)
    
    url = f"{base_url}/info"
    start_time = time.time()
    
    try:
        response = requests.get(url, timeout=10)
        elapsed = (time.time() - start_time) * 1000
        
        if verbose:
            print(f"URL: {url}")
            print(f"Status Code: {response.status_code}")
            print(f"Response Time: {elapsed:.2f}ms")
            print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model info retrieved")
            print(f"   Version: {data.get('model_version')}")
            print(f"   Architecture: {data.get('model_architecture')}")
            print(f"   Classes: {len(data.get('classes', []))} emotions")
            print(f"   Response Time: {elapsed:.2f}ms")
            return {"success": True, "data": data, "response_time_ms": elapsed}
        else:
            print(f"❌ Model info failed: Status {response.status_code}")
            return {"success": False, "status_code": response.status_code}
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"success": False, "error": str(e)}


def test_prediction_endpoint(base_url: str, verbose: bool = False) -> Dict[str, Any]:
    """Test the /predict endpoint with a sample image."""
    print("\n" + "="*60)
    print("Testing Prediction Endpoint")
    print("="*60)
    
    url = f"{base_url}/predict"
    
    # Get test image (prefer real test image with face)
    print("Loading test image...")
    image_bytes = create_test_image()
    
    # Determine file extension based on image type
    test_image_path = os.path.join(os.path.dirname(__file__), 'tests', 'images', 'happy.png')
    if os.path.exists(test_image_path):
        filename = 'test_image.png'
        content_type = 'image/png'
    else:
        filename = 'test_image.jpg'
        content_type = 'image/jpeg'
    
    # Prepare file upload
    files = {'file': (filename, image_bytes, content_type)}
    
    start_time = time.time()
    
    try:
        response = requests.post(url, files=files, timeout=30)
        elapsed = (time.time() - start_time) * 1000
        
        if verbose:
            print(f"URL: {url}")
            print(f"Status Code: {response.status_code}")
            print(f"Response Time: {elapsed:.2f}ms")
            print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"✅ Prediction successful")
                print(f"   Emotion: {data.get('emotion')}")
                print(f"   Confidence: {data.get('confidence'):.2%}")
                print(f"   Inference Time: {data.get('inference_time_ms', 0):.2f}ms")
                print(f"   Total Response Time: {elapsed:.2f}ms")
                if verbose:
                    print(f"   Probabilities: {data.get('probabilities')}")
                return {"success": True, "data": data, "response_time_ms": elapsed}
            else:
                print(f"⚠️  Prediction returned error: {data.get('message')}")
                return {"success": False, "data": data}
        else:
            error_data = response.json() if response.content else {}
            print(f"❌ Prediction failed: Status {response.status_code}")
            print(f"   Error: {error_data.get('message', 'Unknown error')}")
            return {"success": False, "status_code": response.status_code, "data": error_data}
            
    except requests.exceptions.Timeout:
        print(f"❌ Prediction timed out (may be processing large image)")
        return {"success": False, "error": "timeout"}
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"success": False, "error": str(e)}


def test_error_handling(base_url: str, verbose: bool = False) -> Dict[str, Any]:
    """Test error handling with invalid input."""
    print("\n" + "="*60)
    print("Testing Error Handling")
    print("="*60)
    
    url = f"{base_url}/predict"
    
    # Test with invalid file type
    print("Testing invalid file type...")
    files = {'file': ('test.txt', b'not an image', 'text/plain')}
    
    try:
        response = requests.post(url, files=files, timeout=10)
        data = response.json()
        
        # API returns 200 with success: false for validation errors
        if response.status_code == 200 and not data.get('success', True):
            error_type = data.get('error', '')
            if 'invalidfiletype' in error_type.lower() or 'file' in error_type.lower():
                print(f"✅ Error handling works correctly")
                print(f"   Error: {data.get('error')}")
                print(f"   Message: {data.get('message')}")
                return {"success": True, "data": data}
            else:
                print(f"⚠️  Unexpected error type: {error_type}")
                return {"success": False, "data": data}
        elif response.status_code == 400:
            # Also accept HTTP 400 if API is updated to return it
            data = response.json()
            print(f"✅ Error handling works correctly (HTTP 400)")
            print(f"   Error: {data.get('error')}")
            print(f"   Message: {data.get('message')}")
            return {"success": True, "data": data}
        else:
            print(f"⚠️  Unexpected response - Status: {response.status_code}, Success: {data.get('success')}")
            if verbose:
                print(f"   Response: {data}")
            return {"success": False, "status_code": response.status_code, "data": data}
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"success": False, "error": str(e)}


def test_api_docs(base_url: str) -> bool:
    """Check if API documentation is accessible."""
    print("\n" + "="*60)
    print("Checking API Documentation")
    print("="*60)
    
    docs_url = f"{base_url}/docs"
    
    try:
        response = requests.get(docs_url, timeout=5)
        if response.status_code == 200:
            print(f"✅ API docs accessible at: {docs_url}")
            return True
        else:
            print(f"⚠️  API docs returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"⚠️  Could not access API docs: {e}")
        return False


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(
        description="Test deployed Railway API endpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_deployed_api.py --url https://emotion-api.up.railway.app
  python test_deployed_api.py --url https://emotion-api.up.railway.app --verbose
        """
    )
    parser.add_argument(
        '--url',
        type=str,
        required=True,
        help='Base URL of the deployed API (e.g., https://emotion-api.up.railway.app)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show verbose output including full responses'
    )
    parser.add_argument(
        '--skip-prediction',
        action='store_true',
        help='Skip prediction test (useful if no face detection)'
    )
    
    args = parser.parse_args()
    
    # Normalize URL (remove trailing slash)
    base_url = args.url.rstrip('/')
    
    print("\n" + "="*60)
    print("Railway API Deployment Test")
    print("="*60)
    print(f"Testing API at: {base_url}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # Test health endpoint
    results['health'] = test_health_endpoint(base_url, args.verbose)
    
    # Test model info endpoint
    results['info'] = test_model_info_endpoint(base_url, args.verbose)
    
    # Test API docs
    results['docs'] = test_api_docs(base_url)
    
    # Test prediction endpoint (if not skipped)
    if not args.skip_prediction:
        results['prediction'] = test_prediction_endpoint(base_url, args.verbose)
    else:
        print("\n" + "="*60)
        print("Skipping Prediction Test")
        print("="*60)
        results['prediction'] = {"skipped": True}
    
    # Test error handling
    results['error_handling'] = test_error_handling(base_url, args.verbose)
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for k, v in results.items() 
                 if isinstance(v, dict) and v.get('success') is True)
    total = sum(1 for k, v in results.items() 
                if isinstance(v, dict) and 'success' in v)
    
    print(f"Tests Passed: {passed}/{total}")
    
    if results.get('health', {}).get('success'):
        print("✅ API is healthy and ready to use")
    else:
        print("❌ API health check failed - check deployment")
        sys.exit(1)
    
    if results.get('prediction', {}).get('success'):
        print("✅ Prediction endpoint working correctly")
    elif results.get('prediction', {}).get('skipped'):
        print("⚠️  Prediction test skipped")
    else:
        print("⚠️  Prediction endpoint may have issues")
    
    print(f"\nAPI Documentation: {base_url}/docs")
    print(f"Health Check: {base_url}/health")
    print(f"Model Info: {base_url}/info")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

