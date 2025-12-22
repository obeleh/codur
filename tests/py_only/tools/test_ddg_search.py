import pytest
from codur.tools.duckduckgo import duckduckgo_search

def test_duckduckgo_search_rick_roll():
    """
    Test that searching for 'rick roll wikipedia' finds the Rickrolling page.
    """
    query = "rick roll wikipedia"
    expected_url_substring = "wikipedia.org/wiki/Rickrolling"
    
    # Increase max_results to be safe and try lite backend which seems more reliable in some envs
    results = duckduckgo_search(query, max_results=10, backend='lite')
    
    if not results:
        # Fallback to default backend if lite returns nothing
        results = duckduckgo_search(query, max_results=10)
    
    assert len(results) > 0, "No results returned from DuckDuckGo"
    
    found = False
    for res in results:
        url = res.get('href', '')
        if expected_url_substring in url:
            found = True
            break
            
    assert found, f"Could not find {expected_url_substring} in results"

if __name__ == "__main__":
    # If run directly, show results
    results = duckduckgo_search("rick roll wikipedia", max_results=10)
    for res in results:
        print(f"{res.get('title')}: {res.get('href')}")
