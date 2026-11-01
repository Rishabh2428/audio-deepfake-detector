# Provider API access and model discovery

AegisVoice's optional online explanation sends a small allowlisted set of numeric DSP measurements to the selected provider after the user checks the send-consent box. It does not send the audio, filename, history identifier, local classifier score, label, or verdict. The API key is supplied for one request, used in memory, and not written to the database or configuration files.

The **Load model catalog** action is separate. It sends the entered key to the selected provider's model-list endpoint so that the backend can return current model IDs and any available pricing metadata. It does not send audio or DSP values. The dropdown is populated from the live catalog instead of a hardcoded short list. Catalog visibility does not prove that a model will have free quota or that a generation request will be accepted.

## Shared requirements

Each provider uses a different API key and authentication format. A ChatGPT login, Google account password, or key from one provider cannot authenticate with another provider. In general, the account needs API access, a valid provider-specific key, access to the selected model, and any required billing or credits. Quotas and region restrictions are provider/account-specific. Never commit an API key to source code or share screenshots that show it.

To use one: select the provider, follow its **API key setup** link, paste that provider's key into AegisVoice, check the separate catalog consent, then press **Load model catalog**. Choose a listed model and check its price/access label. To request an explanation, check the separate measurements consent and press **Explain evidence**. Catalog lookup sends only the key; explanation sends only allowlisted numeric DSP measurements. The browser calls AegisVoice's local backend, which then uses the provider's own authentication, model-catalog endpoint, and generation endpoint. Keys are not saved to files or history.

## Google Gemini API

- Create a key in [Google AI Studio](https://aistudio.google.com/app/apikey).
- The backend calls `GET https://generativelanguage.googleapis.com/v1beta/models` and keeps models that support `generateContent`.
- Catalog calls use Google's `x-goog-api-key` header; explanation calls use the Gemini `:generateContent` endpoint.
- Google offers a free tier for selected models with limits. The app marks current known free-tier text models when they appear in the catalog. Verify the current [Gemini pricing page](https://ai.google.dev/gemini-api/docs/pricing); quotas, eligibility, and model pricing can change. Google says free-tier content may be used to improve products.
- API key documentation: [list models](https://ai.google.dev/api/models).

## OpenAI API (GPT models)

- Create a project key on the [OpenAI Platform](https://platform.openai.com/api-keys).
- The backend calls `GET https://api.openai.com/v1/models` and lists the models returned for that API organization.
- Requests use `Authorization: Bearer …` and the Responses API. The list may include models that are visible to the organization but unavailable for the selected operation; an actual explanation request is the final compatibility check.
- ChatGPT web access and subscription billing are separate from API access. The API model list does not expose a free-tier flag; AegisVoice labels these models as API-metered, while noting that account credits may offset usage. See [OpenAI API billing](https://help.openai.com/en/articles/9039756-managing-billing-for-chatgpt-and-the-api-platform) and [API pricing](https://openai.com/api/pricing/).

## NVIDIA NIM (Nemotron and other supported models)

- Sign in to [NVIDIA Build](https://build.nvidia.com/) and create an NVIDIA API key.
- The backend calls NVIDIA's OpenAI-compatible model-list endpoint and returns text-generation model IDs from that endpoint. The catalog may not reflect which hosted endpoints your key can call.
- Requests use the NVIDIA API key as a Bearer token and the NIM chat-completions endpoint.
- NVIDIA marks some hosted models as **Free Endpoint**. AegisVoice labels its current known Nemotron free endpoints when present, but free endpoint limits and availability can change. Check the model's [NVIDIA catalog page](https://build.nvidia.com/models) before use; some models or deployment choices may require separate entitlement or paid infrastructure.

## OpenRouter

- Create a separate key at [OpenRouter Keys](https://openrouter.ai/keys).
- The backend calls `GET https://openrouter.ai/api/v1/models?output_modalities=text`, uses the returned pricing fields, and moves zero-priced text models to the top. The `openrouter/free` route remains available as a manual model ID.
- Requests use an OpenRouter key as a Bearer token with its OpenAI-compatible chat-completions endpoint. The `openrouter/free` auto-router is also listed when it is not returned by the catalog.
- Free models and upstream providers can change, fail over, or impose rate limits. Check [OpenRouter pricing and plan limits](https://openrouter.ai/pricing).

## Anthropic Claude API

- Create an API key in the [Claude Console](https://platform.claude.com/settings/keys). A Claude web account or subscription is not automatically an API key. The API uses a separate Console organization/workspace and metered usage; trial or eligible plan credits can offset costs.
- The backend calls `GET https://api.anthropic.com/v1/models`, requests active models, and follows Anthropic's pagination. It sends `x-api-key` and `anthropic-version` headers.
- A key scoped to one workspace usually needs no extra setting. For a key that can act across workspaces, enter the target `anthropic-workspace-id` in the optional field; the same workspace is used for catalog lookup and explanation.
- Check [Anthropic API key setup](https://platform.claude.com/docs/en/get-api-key), [Models API](https://platform.claude.com/docs/en/api/models/list), and [API pricing](https://platform.claude.com/docs/en/about-claude/pricing). Claude API usage is metered; available credits depend on account eligibility.

## Free-model labels

The app distinguishes **FREE**, **FREE-TIER ELIGIBLE**, **FREE ENDPOINT**, **PAID**, **API METERED**, **TIER DEPENDENT**, **CHECK ENDPOINT**, and **PRICE UNKNOWN**. The dropdown sorts known free options first. Some providers do not expose pricing in their catalog; their entries are marked as metered or tier-dependent instead of being called free. These labels describe provider catalog/pricing data, not a guarantee about an individual's remaining quota, region, billing status, or successful inference. Re-load the catalog after changing a key or workspace and check linked provider pricing before sending a generation request.
