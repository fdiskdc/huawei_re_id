#!/usr/bin/env python3
"""
Test script to verify Redis caching and SHA256 hash-based task IDs.
"""
import requests
import time
import hashlib
import json

# Configuration
BASE_URL = "http://localhost:8000"
TEST_SEQUENCE = "ACGUACGUACGUACGUACGU"  # 20 nucleotides for quick testing

def generate_sha256_hash(sequence):
    """Generate SHA256 hash of RNA sequence."""
    return hashlib.sha256(sequence.encode('utf-8')).hexdigest()

def test_caching():
    """Test that caching works correctly."""
    print("=" * 60)
    print("Testing Redis Caching and SHA256 Task IDs")
    print("=" * 60)
    
    # Generate expected hash
    expected_job_id = generate_sha256_hash(TEST_SEQUENCE)
    print(f"\nTest sequence: {TEST_SEQUENCE}")
    print(f"Expected SHA256 hash (jobId): {expected_job_id}")
    
    # First request (cache miss)
    print("\n--- First Request (should be cache MISS) ---")
    start_time = time.time()
    response1 = requests.post(
        f"{BASE_URL}/mrmodn/api/v1/submit-task",
        json={
            "rnaSequence": TEST_SEQUENCE,
            "userId": "test_user"
        }
    )
    time1 = time.time() - start_time
    
    print(f"Status Code: {response1.status_code}")
    if response1.status_code == 200:
        data1 = response1.json()
        print(f"Received jobId: {data1.get('jobId')}")
        print(f"Time taken: {time1:.3f} seconds")
        
        # Verify jobId matches SHA256 hash
        if data1.get('jobId') == expected_job_id:
            print("✓ jobId matches SHA256 hash")
        else:
            print("✗ jobId does NOT match SHA256 hash!")
            return False
    else:
        print(f"Error: {response1.text}")
        return False
    
    # Wait a moment
    time.sleep(0.5)
    
    # Second request (cache hit)
    print("\n--- Second Request (should be cache HIT) ---")
    start_time = time.time()
    response2 = requests.post(
        f"{BASE_URL}/mrmodn/api/v1/submit-task",
        json={
            "rnaSequence": TEST_SEQUENCE,
            "userId": "test_user"
        }
    )
    time2 = time.time() - start_time
    
    print(f"Status Code: {response2.status_code}")
    if response2.status_code == 200:
        data2 = response2.json()
        print(f"Received jobId: {data2.get('jobId')}")
        print(f"Time taken: {time2:.3f} seconds")
        
        # Verify jobId matches
        if data2.get('jobId') == expected_job_id:
            print("✓ jobId matches SHA256 hash")
        else:
            print("✗ jobId does NOT match SHA256 hash!")
            return False
        
        # Verify results are identical
        if data1.get('classification') == data2.get('classification') and \
           data1.get('attention') == data2.get('attention'):
            print("✓ Results are identical (from cache)")
        else:
            print("✗ Results are different!")
            return False
        
        # Verify speedup
        speedup = time1 / time2 if time2 > 0 else 0
        print(f"Speedup factor: {speedup:.2f}x")
        if speedup > 2:
            print("✓ Significant speedup detected (> 2x)")
        else:
            print("⚠ Limited speedup (may be due to small sequence size)")
    else:
        print(f"Error: {response2.text}")
        return False
    
    print("\n--- Testing Different Sequence (new cache entry) ---")
    different_sequence = "GCUAGCUAGCUAGCUAGCUA"
    expected_job_id2 = generate_sha256_hash(different_sequence)
    print(f"Different sequence: {different_sequence}")
    print(f"Expected SHA256 hash: {expected_job_id2}")
    
    start_time = time.time()
    response3 = requests.post(
        f"{BASE_URL}/mrmodn/api/v1/submit-task",
        json={
            "rnaSequence": different_sequence,
            "userId": "test_user"
        }
    )
    time3 = time.time() - start_time
    
    print(f"Status Code: {response3.status_code}")
    if response3.status_code == 200:
        data3 = response3.json()
        print(f"Received jobId: {data3.get('jobId')}")
        print(f"Time taken: {time3:.3f} seconds")
        
        if data3.get('jobId') == expected_job_id2:
            print("✓ jobId matches SHA256 hash for different sequence")
        else:
            print("✗ jobId does NOT match SHA256 hash!")
            return False
    else:
        print(f"Error: {response3.text}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    try:
        success = test_caching()
        exit(0 if success else 1)
    except requests.exceptions.ConnectionError:
        print("\n✗ ERROR: Could not connect to server.")
        print("Make sure the Flask server is running on http://localhost:8000")
        exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
