import json
import urllib.request
import sys

print("DEBUG: Importing modules...", flush=True)

def test_search(query, expected_category):
    url = "http://127.0.0.1:8000/search/text"
    payload = json.dumps({"query": query, "top_k": 5}).encode('utf-8')
    headers = {'Content-Type': 'application/json'}
    
    try:
        req = urllib.request.Request(url, data=payload, headers=headers)
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            
        print(f"\nQuery: '{query}' (Expected: {expected_category})", flush=True)
        passes = 0
        for i, item in enumerate(data):
            cat = item.get('category', 'Unknown')
            match = expected_category.lower() in cat.lower()
            status = "✅" if match else "❌"
            if match: passes += 1
            print(f"  {i+1}. [{status}] {cat} - {item.get('description', '')[:50]}...", flush=True)
            
        return passes == len(data) and len(data) > 0
        
    except Exception as e:
        print(f"Error testing '{query}': {e}", flush=True)
        return False

def main():
    print("Starting Search Quality Test...", flush=True)
    test_cases = [
        ("gold ring", "ring"),
        ("diamond necklace", "necklace"),
        ("pearl pendant", "necklace")
    ]

    score = 0
    for q, cat in test_cases:
        if test_search(q, cat):
            score += 1
            
    print(f"\nPassed {score}/{len(test_cases)} tests.", flush=True)

if __name__ == "__main__":
    main()
