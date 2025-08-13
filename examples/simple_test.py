#!/usr/bin/env python3
"""
Quick test using the Iris dataset.
"""

import requests

url = "http://localhost:8000/analyze"
data = {
    "url": "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv",
    "prompt": "Ths iris dataset is a data science classic. Perform exploratory data analysis and clustering."
}

print("Testing with Iris dataset...")
response = requests.post(url, data=data, timeout=86400)

if response.status_code == 200:
    with open("iris_report.html", "w") as f:
        f.write(response.text)
    print("✓ Test successful! Report saved to iris_report.html")
else:
    print(f"Error {response.status_code}: {response.text[:200]}")