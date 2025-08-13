#!/usr/bin/env python3
"""
Example: Analyze NVIDIA stock data with investment strategies.

Prerequisites:
- Server must be running: python3 main.py
- Run this script: python3 examples/analyze_nvidia.py
"""

import requests

# API endpoint and data
url = "http://localhost:8000/analyze"
data = {
    "url": "https://raw.githubusercontent.com/mystichronicle/NVIDIA-Stock-Analysis/refs/heads/main/data/NVDA.csv",
    "prompt": "List a few simple strategies and compare how they would have done on NVIDIA with a hypothetical $1,000 initial investment. Consider a few distinctive time periods."
}

print("Analyzing NVIDIA stock data...")
print("(This may take 1-2 minutes)\n")

# Make the request
response = requests.post(url, data=data, timeout=300)

if response.status_code == 200:
    # Save the HTML report
    with open("nvidia_analysis.html", "w") as f:
        f.write(response.text)
    print("✓ Analysis complete! Report saved to nvidia_analysis.html")
else:
    print(f"Error {response.status_code}: {response.text[:200]}")