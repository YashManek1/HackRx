#!/usr/bin/env python3
"""
Quick test for specific query issues
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_annual_return_query():
    """Test the specific query that was giving duplicate results"""
    print("🔍 Testing 'annual return' query...")
    
    # Test with debug endpoint first
    print("\n📊 Debug Analysis:")
    debug_data = {
        "query": "what is the annual return?",
        "top_k": 5
    }
    
    debug_response = requests.post(f"{BASE_URL}/query/debug", json=debug_data)
    
    if debug_response.status_code == 200:
        debug_result = debug_response.json()
        print(f"   Query terms: {debug_result.get('debug_results', [{}])[0].get('query_terms', [])}")
        
        for i, result in enumerate(debug_result.get('debug_results', [])[:5]):
            print(f"   Result {i+1}:")
            print(f"      Similarity Score: {result.get('similarity_score', 0):.4f}")
            print(f"      Page: {result.get('page', 'N/A')}")
            print(f"      Exact Matches: {result.get('exact_matches', [])}")
            print(f"      Content: {result.get('content_preview', '')[:80]}...")
            print()
    
    # Test with regular endpoint
    print("🎯 Regular Query Results:")
    query_data = {
        "query": "what is the annual return?",
        "top_k": 5
    }
    
    response = requests.post(f"{BASE_URL}/query", json=query_data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"   Total candidates found: {result.get('total_candidates_found', 'N/A')}")
        print(f"   After filtering: {result.get('filtered_candidates', 'N/A')}")
        print(f"   Returned results: {result.get('returned_results', 'N/A')}")
        print(f"   Deduplication applied: {result.get('deduplication_applied', False)}")
        
        matches = result.get('matches', [])
        unique_contents = set()
        
        for i, match in enumerate(matches):
            content_preview = match.get('content', '')[:100]
            
            print(f"\n   📄 Result {i+1}:")
            print(f"      Rank: {match.get('rank', 'N/A')}")
            print(f"      Page: {match.get('page_number', 'N/A')}")
            print(f"      Chunk Type: {match.get('chunk_type', 'N/A')}")
            print(f"      Similarity Score: {match.get('similarity_score', 'N/A'):.4f}")
            print(f"      Relevance Score: {match.get('relevance_score', 'N/A'):.4f}")
            print(f"      Exact Matches: {match.get('exact_matches', 'N/A')}")
            print(f"      Partial Matches: {match.get('partial_matches', 'N/A')}")
            print(f"      Content Preview: {content_preview}...")
            
            # Check for duplicates
            if content_preview in unique_contents:
                print(f"      ⚠️  DUPLICATE DETECTED!")
            else:
                unique_contents.add(content_preview)
        
        print(f"\n   📊 Analysis:")
        print(f"      Unique results: {len(unique_contents)}")
        print(f"      Total results: {len(matches)}")
        
        if len(unique_contents) < len(matches):
            print(f"      ❌ Still has duplicates!")
        else:
            print(f"      ✅ No duplicates found!")
    
    else:
        print(f"❌ Query failed: {response.status_code} - {response.text}")

def main():
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
    
    test_annual_return_query()

if __name__ == "__main__":
    main()
