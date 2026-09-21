#!/usr/bin/env python3
"""Diagnose the Gemini call. Run:  python test_gemini.py"""
import os, json, urllib.request, urllib.error

k = os.environ.get("GEMINI_API_KEY")
if not k:
    print("No GEMINI_API_KEY set in THIS terminal. Run: export GEMINI_API_KEY=...")
    raise SystemExit
print("key prefix:", k[:4], "| length:", len(k))

body = json.dumps({"contents": [{"parts": [{"text": "Say hi in one word."}]}]}).encode()
for model in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-flash-latest", "gemini-2.5-flash"]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    for label, hdr, u in [
        ("header", {"Content-Type": "application/json", "x-goog-api-key": k}, url),
        ("query",  {"Content-Type": "application/json"}, url + f"?key={k}"),
    ]:
        try:
            resp = urllib.request.urlopen(urllib.request.Request(u, data=body, headers=hdr), timeout=60).read().decode()
            print(f"\n✅ WORKS  ->  model={model}  auth={label}")
            print("response:", resp[:300])
            print(f"\nUse this: export GEMINI_MODEL={model}")
            raise SystemExit
        except urllib.error.HTTPError as e:
            print(f"[{model}/{label}] HTTP {e.code}: " + e.read().decode(errors='ignore')[:200])
        except Exception as e:
            print(f"[{model}/{label}] {type(e).__name__}: {e}")
print("\nAll attempts failed — copy everything above and send it back.")
