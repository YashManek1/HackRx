#!/usr/bin/env python3
"""
Test the improved accuracy for the annual return query
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_annual_return_accuracy():
    """Test the specific query that was giving poor results"""
    print("🔍 Testing 'annual return' query accuracy...")
    
    query = "what is the annual return?"
    
    # Test with analysis endpoint first
    print("\n📊 Query Analysis:")
    analysis_data = {
        "query": query,
        "top_k": 5
    }
    
    analysis_response = requests.post(f"{BASE_URL}/query/analyze", json=analysis_data)
    
    if analysis_response.status_code == 200:
        analysis_result = analysis_response.json()
        
        query_analysis = analysis_result.get('query_analysis', {})
        print(f"   Query Intent: {query_analysis.get('likely_intent', 'unknown')}")
        print(f"   Query Terms: {query_analysis.get('query_terms', [])}")
        print(f"   Financial Keywords: {query_analysis.get('financial_keywords', [])}")
        
        print(f"\n   📄 Top Results Analysis:")
        for result in analysis_result.get('analysis_results', [])[:5]:
            print(f"      Rank {result.get('rank', 'N/A')}:")
            print(f"         Similarity Score: {result.get('similarity_score', 0):.4f}")
            print(f"         True Relevance: {result.get('true_relevance', 'unknown')}")
            print(f"         Primary Context: {result.get('context_analysis', {}).get('primary_context', 'unknown')}")
            print(f"         Exact Matches: {result.get('exact_matches', [])}")
            print(f"         Content: {result.get('content_preview', '')[:80]}...")
            print(f"         Explanation: {result.get('relevance_explanation', '')}")
            print()
    
    # Test with improved regular endpoint
    print("🎯 Improved Query Results:")
    query_data = {
        "query": query,
        "top_k": 5
    }
    
    response = requests.post(f"{BASE_URL}/query", json=query_data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"   Total candidates found: {result.get('total_candidates_found', 'N/A')}")
        print(f"   After filtering: {result.get('filtered_candidates', 'N/A')}")
        print(f"   Returned results: {result.get('returned_results', 'N/A')}")
        
        matches = result.get('matches', [])
        
        print(f"\n   📊 Results Summary:")
        high_relevance_count = 0
        financial_context_count = 0
        
        for i, match in enumerate(matches):
            content = match.get('content', '').lower()
            context_relevance = match.get('context_relevance', 1.0)
            
            # Check if this seems like a financial result
            financial_terms = ["profit", "investment", "rate", "percentage", "yield", "interest", "dividend", "roi", "performance", "gain"]
            has_financial_context = any(term in content for term in financial_terms)
            
            if has_financial_context:
                financial_context_count += 1
            
            if context_relevance > 0.8:
                high_relevance_count += 1
            
            print(f"\n      📄 Result {i+1}:")
            print(f"         Page: {match.get('page_number', 'N/A')}")
            print(f"         Similarity Score: {match.get('similarity_score', 'N/A'):.4f}")
            print(f"         Relevance Score: {match.get('relevance_score', 'N/A'):.4f}")
            print(f"         Context Relevance: {context_relevance:.2f}")
            print(f"         Exact Matches: {match.get('exact_matches', 'N/A')}")
            print(f"         Has Financial Context: {'Yes' if has_financial_context else 'No'}")
            print(f"         Content: {match.get('content', '')[:100]}...")
        
        # Calculate accuracy metrics
        accuracy_score = financial_context_count / len(matches) if matches else 0
        relevance_score = high_relevance_count / len(matches) if matches else 0
        
        print(f"\n   📈 Accuracy Metrics:")
        print(f"      Financial Context Results: {financial_context_count}/{len(matches)} ({accuracy_score:.1%})")
        print(f"      High Relevance Results: {high_relevance_count}/{len(matches)} ({relevance_score:.1%})")
        
        if accuracy_score > 0.6:
            print(f"      ✅ GOOD ACCURACY - Most results are financially relevant")
        elif accuracy_score > 0.3:
            print(f"      ⚠️  MEDIUM ACCURACY - Some relevant results found")
        else:
            print(f"      ❌ POOR ACCURACY - Few relevant results")
            
        # Provide recommendations
        print(f"\n   💡 Recommendations:")
        if accuracy_score < 0.5:
            print(f"      • Your document may not contain financial return information")
            print(f"      • Try queries like 'investment returns' or 'profit percentage'")
            print(f"      • Consider if this is an insurance document rather than investment document")
        
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
    test_annual_return_accuracy()

if __name__ == "__main__":
    main()
