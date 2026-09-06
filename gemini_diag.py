import os
import sys

# Attempt to load Streamlit secrets if available
has_st_secrets = False
try:
    import streamlit as st
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
        has_st_secrets = True
except Exception:
    pass

def run_diagnostics():
    print("=" * 50)
    print("GEMINI API DIAGNOSTICS")
    print("=" * 50)

    # 1. Check SDK version
    try:
        import google.genai
        print(f"[OK] Google GenAI SDK is installed (version {getattr(google.genai, '__version__', 'unknown')}).")
    except ImportError:
        print("[FAIL] `google-genai` SDK is NOT installed. Install it with: pip install google-genai")
        return

    # 2. Check API Key presence
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        print(f"[OK] GEMINI_API_KEY is detected (Length: {len(api_key)}, Starts with: '{api_key[:4]}...').")
        if has_st_secrets:
            print("[INFO] Key was loaded from Streamlit secrets (.streamlit/secrets.toml).")
        else:
            print("[INFO] Key was loaded from environment variables.")
    else:
        print("[FAIL] GEMINI_API_KEY is NOT detected in environment or Streamlit secrets.")
        return

    # 3. Test API Request
    print("\n[TEST] Attempting to connect to Gemini API (Model: gemini-2.5-flash)...")
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents="Say 'Diagnostic successful' in exactly two words."
        )
        print(f"[SUCCESS] API returned: {response.text.strip()}")
    except Exception as e:
        print(f"\n[ERROR] API Request Failed:")
        print(f"{type(e).__name__}: {str(e)}")
        
        err_str = str(e).lower()
        if "400" in err_str or "invalid model" in err_str:
            print("-> HINT: The model 'gemini-3.0-flash' might not be available for this key. You may want to try 'gemini-1.5-flash'.")
        elif "401" in err_str or "unauthenticated" in err_str or "api key not valid" in err_str:
            print("-> HINT: The API key is invalid or revoked. Please regenerate it at Google AI Studio.")
        elif "403" in err_str or "permission" in err_str:
            print("-> HINT: The API key does not have permission to access this model.")

if __name__ == "__main__":
    run_diagnostics()
