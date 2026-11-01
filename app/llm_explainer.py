"""Optional explanations of derived DSP measurements via external LLM APIs.

Credentials are passed per call and never written to disk or logs. Only values
already filtered by the server's DSP allowlist are sent upstream.
"""

from __future__ import annotations

import json
import re
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


ANTHROPIC_API_VERSION = "2026-10-21"
PROVIDERS = {
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/models/"},
    "openai": {"url": "https://api.openai.com/v1/responses"},
    "nvidia": {"url": "https://integrate.api.nvidia.com/v1/chat/completions"},
    "openrouter": {"url": "https://openrouter.ai/api/v1/chat/completions"},
    "anthropic": {"url": "https://api.anthropic.com/v1/messages"},
}
PROVIDER_NAMES = {"gemini": "Google AI Studio", "openai": "OpenAI", "nvidia": "NVIDIA NIM", "openrouter": "OpenRouter", "anthropic": "Anthropic Claude"}
MODEL_LIST_URLS = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models",
    "openai": "https://api.openai.com/v1/models",
    "nvidia": "https://integrate.api.nvidia.com/v1/models",
    "openrouter": "https://openrouter.ai/api/v1/models?output_modalities=text",
    "anthropic": "https://api.anthropic.com/v1/models",
}
GEMINI_FREE_MODELS = {
    "gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite", "gemini-3-flash-preview",
    "gemini-2.5-flash", "gemini-2.5-flash-lite",
}
NVIDIA_FREE_ENDPOINTS = {
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
    "nvidia/nemotron-3.5-super-vl-120b-a12b",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
}
PROVIDER_ACCESS = {
    "gemini": {
        "key_steps": "Create a Gemini API key in Google AI Studio and use a Google project with Gemini API access. Free tier is limited to selected models and quotas; Google may use free-tier prompts to improve products.",
        "key_url": "https://aistudio.google.com/app/apikey",
        "models_url": "https://ai.google.dev/api/models",
        "pricing_url": "https://ai.google.dev/gemini-api/docs/pricing",
        "api_notes": "This key is used with the Gemini Developer API. Access to a listed model does not guarantee free quota or billing eligibility.",
    },
    "openai": {
        "key_steps": "Create a project API key on the OpenAI Platform. ChatGPT access and ChatGPT subscriptions are separate from API access and billing; add API billing or use eligible API credits. The API model catalog does not indicate a free tier.",
        "key_url": "https://platform.openai.com/api-keys",
        "models_url": "https://developers.openai.com/api/reference/resources/models/methods/list",
        "pricing_url": "https://openai.com/api/pricing/",
        "api_notes": "A ChatGPT web subscription is not an API key or general API credit. API usage is metered unless credits are available to the organization.",
    },
    "nvidia": {
        "key_steps": "Sign in to NVIDIA Build, create an NVIDIA API key, and use a model with a free hosted endpoint or an entitlement your account has. Free endpoints have provider limits and can change.",
        "key_url": "https://build.nvidia.com/",
        "models_url": "https://build.nvidia.com/models",
        "pricing_url": "https://build.nvidia.com/models",
        "api_notes": "The catalog list is checked using NVIDIA's OpenAI-compatible model endpoint. A free endpoint can still have usage, rate, or availability limits.",
    },
    "openrouter": {
        "key_steps": "Create an OpenRouter key. Choose a model marked FREE or the openrouter/free auto-router. Free-model access is subject to OpenRouter and upstream-provider rate limits and availability.",
        "key_url": "https://openrouter.ai/keys",
        "models_url": "https://openrouter.ai/models?max_price=0",
        "pricing_url": "https://openrouter.ai/pricing",
        "api_notes": "Use an OpenRouter key here, not a Google, OpenAI, or NVIDIA key. Models are ordered by price status; check details because upstream limits and terms differ.",
    },
    "anthropic": {
        "key_steps": "Create a Claude Console API key and make sure its organization/workspace has API access and available credits or billing. Claude web plans and API access are separate, though selected plans or promotions may include API credits. For a key that is not scoped to one workspace, provide the target workspace ID.",
        "key_url": "https://platform.claude.com/settings/keys",
        "models_url": "https://platform.claude.com/docs/en/api/models/list",
        "pricing_url": "https://platform.claude.com/docs/en/about-claude/pricing",
        "api_notes": "The model catalog lists active Claude API models. API usage is metered; eligible account credits can offset usage. Keys scoped to a workspace must belong to the workspace used for the request.",
    },
}
_MODEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,119}$")
_SYSTEM_INSTRUCTIONS = (
    "Explain audio DSP measurements for a student project in plain language. "
    "Use only the supplied numeric measurements. Do not identify a speaker, "
    "claim that a voice is human or synthetic, infer identity, or treat any "
    "measurement as proof. Explain likely limitations and keep the response concise."
)


def _provider_http_error(exc: HTTPError) -> ValueError:
    detail = _response_error_detail(exc)
    if exc.code in (401, 403):
        return ValueError("The provider rejected this key or its access. Check the selected provider, key type, and account permissions.")
    if exc.code == 429:
        return ValueError("The provider's model-list rate limit was reached. Wait and retry.")
    if exc.code in (500, 502, 503, 504):
        return ValueError(f"The provider model catalog is temporarily unavailable (HTTP {exc.code}). Try again later.")
    if detail:
        return ValueError(f"The provider model catalog rejected the request (HTTP {exc.code}): {detail}")
    return ValueError(f"The provider model catalog returned HTTP {exc.code}. Check the key and provider access.")


def _response_error_detail(exc: HTTPError) -> str:
    try:
        raw = exc.read(200_001)
        parsed = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return ""
    if not isinstance(parsed, dict):
        return ""
    error = parsed.get("error")
    if isinstance(error, dict):
        message = error.get("message")
    else:
        message = error or parsed.get("message")
    if not isinstance(message, str):
        return ""
    return " ".join(message.split())[:450]


def _request_catalog_page(provider: str, api_key: str, page_token: str | None = None, workspace_id: str = "") -> dict:
    url = MODEL_LIST_URLS[provider]
    if provider == "gemini":
        query = {"pageSize": "100"}
        if page_token:
            query["pageToken"] = page_token
        url += "?" + urlencode(query)
        headers = {"x-goog-api-key": api_key, "Accept": "application/json"}
    elif provider == "anthropic":
        query = {"limit": "100", "lifecycle": "active"}
        if page_token:
            query["after_id"] = page_token
        url += "?" + urlencode(query)
        headers = {"x-api-key": api_key, "anthropic-version": ANTHROPIC_API_VERSION, "Accept": "application/json"}
        if workspace_id:
            headers["anthropic-workspace-id"] = workspace_id
    else:
        headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
        if provider == "openrouter":
            headers["HTTP-Referer"] = "http://127.0.0.1"
            headers["X-Title"] = "AegisVoice"
    request = Request(url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=25) as response:
            raw = response.read(5_000_001)
    except HTTPError as exc:
        raise _provider_http_error(exc) from None
    except (URLError, TimeoutError, OSError):
        raise ValueError("Could not reach the provider model catalog. Check your internet connection and retry.") from None
    if len(raw) > 5_000_000:
        raise ValueError("The provider model catalog response was too large to read safely.")
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("The provider returned an unreadable model catalog.") from None
    if not isinstance(parsed, dict):
        raise ValueError("The provider returned an unexpected model catalog.")
    return parsed


def list_provider_models(provider: str, api_key: str, workspace_id: str = "") -> dict:
    if provider not in PROVIDERS:
        raise ValueError("Choose Google AI Studio, OpenAI, NVIDIA NIM, OpenRouter, or Anthropic Claude.")
    if not isinstance(api_key, str) or not api_key.strip() or len(api_key) > 500 or "\r" in api_key or "\n" in api_key:
        raise ValueError("Enter the selected provider's API key to load its model list.")
    if not isinstance(workspace_id, str) or len(workspace_id) > 120 or "\r" in workspace_id or "\n" in workspace_id:
        raise ValueError("Enter a valid Anthropic workspace ID or leave it blank for a workspace-scoped key.")

    items: list[dict] = []
    page_token = None
    for _ in range(10):
        response = _request_catalog_page(provider, api_key.strip(), page_token, workspace_id.strip())
        page = response.get("models") if provider == "gemini" else response.get("data")
        if not isinstance(page, list):
            raise ValueError("The provider response did not contain a model list.")
        items.extend(item for item in page if isinstance(item, dict))
        if provider == "gemini":
            page_token = response.get("nextPageToken")
        elif provider == "anthropic" and response.get("has_more"):
            page_token = response.get("last_id")
        else:
            page_token = None
        if not page_token:
            break

    models = []
    for item in items:
        model_id = item.get("baseModelId") or item.get("id") or item.get("name")
        if not isinstance(model_id, str):
            continue
        model_id = model_id.removeprefix("models/")
        display_name = str(item.get("displayName") or item.get("display_name") or item.get("name") or model_id)
        methods = item.get("supportedGenerationMethods") or []
        if provider == "gemini" and "generateContent" not in methods:
            continue
        if provider == "openrouter":
            architecture = item.get("architecture") if isinstance(item.get("architecture"), dict) else {}
            if "text" not in (architecture.get("output_modalities") or ["text"]):
                continue
        lowered = model_id.casefold()
        if any(token in lowered for token in ("embedding", "embed-", "rerank", "moderation", "transcribe", "whisper", "text-to-speech", "tts-", "-tts", "-live", "-image", "-video", "-audio", "-music", "lyria")):
            continue

        price_status = "unknown"
        price_note = "Check the provider pricing and your account limits; the model-list API does not report this model's complete billable status."
        if provider == "gemini":
            if model_id in GEMINI_FREE_MODELS or model_id.startswith("gemma-4-"):
                price_status = "free_tier"
                price_note = "Listed as free-tier eligible on Google's current pricing page; per-account quotas, eligibility, and terms still apply."
            else:
                price_status = "tier_dependent"
                price_note = "Google offers free access only for selected models and account quotas; check current model pricing."
        elif provider == "openai":
            price_status = "api_metered"
            price_note = "OpenAI API usage is metered; any organization credit may offset charges. ChatGPT free access is not API access."
        elif provider == "nvidia":
            if model_id in NVIDIA_FREE_ENDPOINTS:
                price_status = "free_endpoint"
                price_note = "NVIDIA currently lists a free hosted endpoint for this model; rate, account, and availability limits apply."
            else:
                price_status = "endpoint_dependent"
                price_note = "Check the NVIDIA model page for Free Endpoint status and current limits."
        elif provider == "openrouter":
            pricing = item.get("pricing") if isinstance(item.get("pricing"), dict) else {}
            prompt_price = pricing.get("prompt")
            completion_price = pricing.get("completion")
            try:
                model_is_free = float(prompt_price) == 0 and float(completion_price) == 0
            except (TypeError, ValueError):
                model_is_free = model_id.endswith(":free")
            if model_is_free or model_id.endswith(":free"):
                price_status = "free"
                price_note = "Catalog pricing is zero for this model variant; free requests still have provider rate and availability limits."
            else:
                price_status = "paid"
                price_note = "This model variant has nonzero or unreported pricing in OpenRouter's catalog."
        elif provider == "anthropic":
            price_status = "api_metered"
            price_note = "Claude API calls are metered; eligible account, trial, or promotional credits may offset usage. Claude web-plan access alone does not guarantee API access."
        models.append({
            "id": model_id,
            "name": display_name,
            "pricing_status": price_status,
            "pricing_note": price_note,
        })

    if provider == "openrouter" and not any(model["id"] == "openrouter/free" for model in models):
        models.append({"id": "openrouter/free", "name": "Free model auto-router", "pricing_status": "free",
                       "pricing_note": "Routes the request to an available free chat model. Free requests remain subject to OpenRouter and upstream limits."})
    if not models:
        raise ValueError("No text-generation models were returned for this key. Check model access or the provider catalog.")
    priority = {"free": 0, "free_tier": 0, "free_endpoint": 0, "paid": 1, "api_metered": 2, "tier_dependent": 2, "endpoint_dependent": 2, "unknown": 3}
    models.sort(key=lambda item: (priority.get(item["pricing_status"], 3), item["name"].casefold()))
    free_count = sum(item["pricing_status"] in {"free", "free_tier", "free_endpoint"} for item in models)
    returned_models = models[:1500]
    return {
        "provider": PROVIDER_NAMES[provider],
        "models": returned_models,
        "model_count": len(returned_models),
        "total_model_count": len(models),
        "models_truncated": len(returned_models) < len(models),
        "free_model_count": free_count,
        "access": PROVIDER_ACCESS[provider],
        "catalog_note": "Models are discovered from the selected provider after you request a refresh. A catalog entry does not guarantee account entitlement, free quota, or successful inference; availability, billing, and rate limits can differ by account, region, and date.",
    }


def explain_measurements(provider: str, model: str, api_key: str, measurements: dict, workspace_id: str = "") -> str:
    if provider not in PROVIDERS:
        raise ValueError("Choose Google AI Studio, NVIDIA NIM, OpenRouter, OpenAI, or Anthropic Claude.")
    if not api_key or not api_key.strip() or len(api_key) > 500 or "\r" in api_key or "\n" in api_key:
        raise ValueError("Enter a valid provider API key for this request.")
    if not model or not _MODEL_PATTERN.fullmatch(model):
        raise ValueError("Enter a valid model name.")
    if not isinstance(workspace_id, str) or len(workspace_id) > 120 or "\r" in workspace_id or "\n" in workspace_id:
        raise ValueError("Enter a valid Anthropic workspace ID or leave it blank for a workspace-scoped key.")
    if not isinstance(measurements, dict) or not measurements or len(measurements) > 80:
        raise ValueError("No supported derived measurements were supplied.")

    prompt = "Explain these derived audio measurements. They are not a verdict:\n" + json.dumps(
        measurements, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    )
    if provider == "gemini":
        url = PROVIDERS[provider]["base_url"] + quote(model, safe="") + ":generateContent"
        body = {
            "systemInstruction": {"parts": [{"text": _SYSTEM_INSTRUCTIONS}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 450},
        }
        headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    elif provider == "openai":
        url = PROVIDERS[provider]["url"]
        body = {"model": model, "instructions": _SYSTEM_INSTRUCTIONS, "input": prompt,
                "max_output_tokens": 450, "store": False}
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    elif provider == "anthropic":
        url = PROVIDERS[provider]["url"]
        body = {"model": model, "system": _SYSTEM_INSTRUCTIONS,
                "messages": [{"role": "user", "content": prompt}], "max_tokens": 450}
        headers = {"x-api-key": api_key, "anthropic-version": ANTHROPIC_API_VERSION, "Content-Type": "application/json"}
        if workspace_id.strip():
            headers["anthropic-workspace-id"] = workspace_id.strip()
    else:
        url = PROVIDERS[provider]["url"]
        body = {"model": model, "messages": [{"role": "system", "content": _SYSTEM_INSTRUCTIONS},
                                               {"role": "user", "content": prompt}],
                "max_tokens": 450, "temperature": 0.2, "stream": False}
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        if provider == "openrouter":
            headers["HTTP-Referer"] = "http://127.0.0.1"
            headers["X-Title"] = "AegisVoice"

    request = Request(url, data=json.dumps(body, separators=(",", ":")).encode("utf-8"), headers=headers, method="POST")
    try:
        with urlopen(request, timeout=35) as response:
            raw = response.read(1_000_001)
    except HTTPError as exc:
        if exc.code in (401, 403):
            raise ValueError("The provider rejected this API key. Check the key and its access to the selected model.") from None
        if exc.code == 429:
            raise ValueError("The provider rate limit was reached. Wait a moment and try again.") from None
        if exc.code in (500, 502, 503, 504):
            raise ValueError(
                f"The selected provider is temporarily unavailable (HTTP {exc.code}). "
                "This does not by itself mean the API key is wrong. Try again later, or select another available model."
            ) from None
        detail = _response_error_detail(exc)
        if detail:
            raise ValueError(f"The selected provider rejected the request (HTTP {exc.code}): {detail}") from None
        raise ValueError(f"The selected provider returned HTTP {exc.code}. Check the model name and provider status.") from None
    except (URLError, TimeoutError, OSError):
        raise ValueError("Could not connect to the selected provider. Check your internet connection and try again.") from None
    if len(raw) > 1_000_000:
        raise ValueError("The provider response was too large to read safely.")
    try:
        result = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("The provider returned an unreadable response.") from None

    if provider == "gemini":
        candidates = result.get("candidates", [])
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        answer = "\n".join(part.get("text", "") for part in parts if isinstance(part, dict))
    elif provider == "openai":
        chunks = []
        for item in result.get("output", []):
            if isinstance(item, dict):
                chunks.extend(part.get("text", "") for part in item.get("content", [])
                              if isinstance(part, dict) and part.get("type") == "output_text")
        answer = "\n".join(chunks)
    elif provider == "anthropic":
        answer = "\n".join(
            item.get("text", "") for item in result.get("content", [])
            if isinstance(item, dict) and item.get("type") == "text"
        )
    else:
        choices = result.get("choices", [])
        message = choices[0].get("message", {}) if choices else {}
        content = message.get("content", "")
        if isinstance(content, list):
            answer = "\n".join(item.get("text", "") for item in content if isinstance(item, dict))
        else:
            answer = content if isinstance(content, str) else ""
    answer = answer.strip()
    if not answer:
        raise ValueError("The provider returned no explanation. Try another model.")
    return answer[:6000]
