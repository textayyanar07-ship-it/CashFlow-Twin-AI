import os
import json
import streamlit as st
from recommendations import get_rule_based_recommendations

# ── SDK import — supports google-genai (new) only ──────────────────────────
try:
    from google import genai as _genai
    _SDK_AVAILABLE = True
except ImportError:
    _genai = None
    _SDK_AVAILABLE = False


def get_api_key() -> str | None:
    """Load key from st.secrets first, then env var. Never expose it."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            v = st.secrets["GEMINI_API_KEY"]
            # Reject placeholder values
            if v and v not in ("YOUR_GEMINI_API_KEY_HERE", "", "None"):
                return v
    except Exception:
        pass
    v = os.environ.get("GEMINI_API_KEY", "")
    return v if v and v not in ("YOUR_GEMINI_API_KEY_HERE", "", "None") else None


def _safe_error(err: Exception, api_key: str) -> str:
    """Return error message string with the API key scrubbed out."""
    msg = str(err)
    if api_key and api_key in msg:
        msg = msg.replace(api_key, "[REDACTED]")
    # Trim to 200 chars for display safety
    return msg[:200]


def get_explanation_and_recommendations(context: dict) -> dict:
    """
    Returns a dict with keys:
        explanation  : str
        actions      : list[dict]
        gemini_used  : bool   ← True only when a live API call succeeded
        error        : str | None  ← safe error string if API failed
    """
    api_key = get_api_key()

    # ── No key → rule-based immediately ──────────────────────────────────────
    if not api_key:
        return _fallback(context, error=None, reason="no_key")

    # ── SDK not installed → rule-based ────────────────────────────────────────
    if not _SDK_AVAILABLE:
        return _fallback(context, error="google-genai SDK not installed", reason="no_sdk")

    # ── Attempt live Gemini call ──────────────────────────────────────────────
    MODELS = ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-2.5-flash"]
    prompt = _build_prompt(context)

    client = _genai.Client(api_key=api_key)

    for model in MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )
            text = response.text.strip()

            # Strip markdown fences if present
            for fence in ("```json", "```"):
                if text.startswith(fence):
                    text = text[len(fence):]
            if text.endswith("```"):
                text = text[:-3]

            result = json.loads(text.strip())

            # Validate expected keys exist
            if "explanation" not in result or "actions" not in result:
                raise ValueError("Response missing required keys")

            result["gemini_used"] = True
            result["error"] = None
            return result

        except json.JSONDecodeError as e:
            # Model responded but output wasn't JSON — try next model
            last_error = f"JSON parse error from {model}: {_safe_error(e, api_key)}"
            continue
        except Exception as e:
            safe_err = _safe_error(e, api_key)
            # 401 / 403 = bad key — no point trying more models
            if "401" in safe_err or "403" in safe_err or "UNAUTHENTICATED" in safe_err:
                return _fallback(
                    context,
                    error=f"API authentication failed (401/403). "
                          "Please regenerate your key at https://aistudio.google.com/app/apikey "
                          "and update .streamlit/secrets.toml",
                    reason="auth_error",
                )
            last_error = f"{model}: {safe_err}"
            continue

    # All models exhausted
    return _fallback(context, error=f"All models failed. Last: {last_error}", reason="api_error")


def _build_prompt(context: dict) -> str:
    cash    = context.get("current_cash", 0)
    runway  = context.get("runway_days", "Unknown")
    score   = context.get("risk_score", "Unknown")
    band    = context.get("risk_band", "Unknown")
    deficit = context.get("deficit_amount", 0)
    d_date  = context.get("deficit_date", "N/A")

    return f"""You are a financial advisor for a small Indian MSME business.
Current financial snapshot:
- Current Cash: Rs {cash:,.0f}
- Cash Runway: {runway} days
- Risk Score: {score} ({band} risk)
- Projected Deficit: Rs {deficit:,.0f} on {d_date}

Provide a short, plain-language analysis and exactly 3 prioritized actions.
Output ONLY valid JSON — no markdown fences, no extra text:
{{
    "explanation": "2-3 sentence plain English summary of the financial situation and key risks.",
    "actions": [
        {{"priority": "HIGH", "problem": "...", "action": "...", "impact": "..."}},
        {{"priority": "MEDIUM", "problem": "...", "action": "...", "impact": "..."}},
        {{"priority": "LOW", "problem": "...", "action": "...", "impact": "..."}}
    ]
}}"""


def _fallback(context: dict, error: str | None, reason: str) -> dict:
    recs = get_rule_based_recommendations(context)
    runway = context.get("runway_days", 60)
    cash   = context.get("current_cash", 0)
    band   = context.get("risk_band", "UNKNOWN")

    if reason == "no_key":
        explanation = (
            f"Running in rule-based advisor mode (no Gemini API key configured). "
            f"Your business has Rs {cash:,.0f} in cash with {runway} days of runway "
            f"at {band} risk. Set GEMINI_API_KEY in .streamlit/secrets.toml to enable AI analysis."
        )
    elif reason == "auth_error":
        explanation = (
            f"Gemini API authentication failed — your API key is invalid or expired. "
            f"Using rule-based analysis: Rs {cash:,.0f} cash, {runway}-day runway, {band} risk. "
            f"Please regenerate your key at aistudio.google.com."
        )
    else:
        explanation = (
            f"Using rule-based advisor (Gemini API unavailable). "
            f"Current position: Rs {cash:,.0f} cash, {runway}-day runway, {band} risk."
        )

    return {
        "explanation": explanation,
        "actions":     recs,
        "gemini_used": False,
        "error":       error,
    }
