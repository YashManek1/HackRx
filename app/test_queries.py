#!/usr/bin/env python3
"""
Test script for the enhanced query functionality.
Run this after ingesting a document to test step 4 implementation.
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:8000"

def test_basic_query():
    """Test the basic query endpoint"""
    print("🔍 Testing Basic Query...")
    
    query_data = {
        "query": "What is the policy coverage?",
        "top_k": 5
    }
    
    response = requests.post(f"{BASE_URL}/query", json=query_data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Query successful!")
        print(f"   Total candidates found: {result.get('total_candidates_found', 'N/A')}")
        print(f"   Filtered candidates: {result.get('filtered_candidates', 'N/A')}")
        print(f"   Results returned: {result.get('returned_results', 'N/A')}")
        
        for i, match in enumerate(result.get('matches', [])[:3]):  # Show first 3 results
            print(f"\n   📄 Result {i+1}:")
            print(f"      Rank: {match.get('rank', 'N/A')}")
            print(f"      Page: {match.get('page_number', 'N/A')}")
            print(f"      Chunk Type: {match.get('chunk_type', 'N/A')}")
            if 'similarity_score' in match:
                print(f"      Similarity Score: {match['similarity_score']:.4f}")
            if 'relevance_score' in match:
                print(f"      Relevance Score: {match['relevance_score']:.4f}")
            print(f"      Token Count: {match.get('token_count', 'N/A')}")
            print(f"      Content Preview: {match.get('content', '')[:100]}...")
        
        return True
    else:
        print(f"❌ Query failed: {response.status_code} - {response.text}")
        return False

def test_advanced_query():
    """Test the query endpoint with advanced filters"""
    print("\n🔍 Testing Advanced Query with Filters...")
    
    query_data = {
        "query": "insurance policy terms and conditions",
        "top_k": 3,
        "chunk_types": ["paragraph", "section"],
        "min_token_count": 50,
        "max_token_count": 300,
        "include_scores": True
    }
    
    response = requests.post(f"{BASE_URL}/query", json=query_data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Advanced query successful!")
        print(f"   Filters applied: {result.get('filters_applied', {})}")
        print(f"   Total candidates found: {result.get('total_candidates_found', 'N/A')}")
        print(f"   After filtering: {result.get('filtered_candidates', 'N/A')}")
        print(f"   Results returned: {result.get('returned_results', 'N/A')}")
        
        for i, match in enumerate(result.get('matches', [])):
            print(f"\n   📄 Result {i+1}:")
            print(f"      Rank: {match.get('rank', 'N/A')}")
            print(f"      Page: {match.get('page_number', 'N/A')}")
            print(f"      Chunk Type: {match.get('chunk_type', 'N/A')}")
            print(f"      Token Count: {match.get('token_count', 'N/A')}")
            if 'similarity_score' in match:
                print(f"      Similarity Score: {match['similarity_score']:.4f}")
            if 'relevance_score' in match:
                print(f"      Relevance Score: {match['relevance_score']:.4f}")
            print(f"      Content Preview: {match.get('content', '')[:100]}...")
        
        return True
    else:
        print(f"❌ Advanced query failed: {response.status_code} - {response.text}")
        return False

def test_index_stats():
    """Test basic query endpoint to check if index exists"""
    print("\n📊 Testing Index Availability...")
    
    # Test with a simple query to see if index is available
    query_data = {
        "query": "test",
        "top_k": 1
    }
    
    response = requests.post(f"{BASE_URL}/query", json=query_data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Index is available!")
        print(f"   Total candidates found: {result.get('total_candidates_found', 'N/A')}")
        print(f"   Filtered candidates: {result.get('filtered_candidates', 'N/A')}")
        
        # Show filter info
        filters = result.get('filters_applied', {})
        print(f"   Available chunk types: {filters.get('chunk_types', [])}")
        print(f"   Token range: {filters.get('min_token_count')} - {filters.get('max_token_count')}")
        
        return True
    elif response.status_code == 404:
        print(f"⚠️  Index not found. Please ingest a document first.")
        return False
    else:
        print(f"❌ Index check failed: {response.status_code} - {response.text}")
        return False

def test_different_queries():
    """Test with different types of queries"""
    print("\n🔍 Testing Different Query Types...")
    
    test_queries = [
        "What are the coverage limits?",
        "How to file a claim?",
        "What is excluded from the policy?",
        "Premium payment terms",
        "Policy renewal process"
    ]
    
    for query in test_queries:
        print(f"\n   Query: '{query}'")
        
        query_data = {
            "query": query,
            "top_k": 2
        }
        
        response = requests.post(f"{BASE_URL}/query", json=query_data)
        
        if response.status_code == 200:
            result = response.json()
            matches = result.get('matches', [])
            if matches:
                best_match = matches[0]
                print(f"   ✅ Best match (Page {best_match.get('page_number', 'N/A')}, "
                      f"Score: {best_match.get('relevance_score', 'N/A'):.4f}): "
                      f"{best_match.get('content', '')[:80]}...")
            else:
                print(f"   ⚠️  No matches found")
        else:
            print(f"   ❌ Query failed: {response.status_code}")
        
        time.sleep(0.5)  # Small delay between requests

def main():
    """Run all tests"""
    print("🚀 Starting Query System Tests")
    print("=" * 50)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/docs")
        if response.status_code != 200:
            print("❌ Server not responding. Please start the FastAPI server first:")
            print("   python main.py")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Please start the FastAPI server first:")
        print("   python main.py")
        return
    
    print("✅ Server is running")
    
    # Run tests
    success_count = 0
    total_tests = 3
    
    if test_index_stats():
        success_count += 1
    
    if test_basic_query():
        success_count += 1
    
    if test_advanced_query():
        success_count += 1
    
    test_different_queries()  # This always runs
    
    print("\n" + "=" * 50)
    print(f"🎯 Tests completed: {success_count}/{total_tests} successful")
    
    if success_count == total_tests:
        print("🎉 All tests passed! Your Step 4 implementation is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()
