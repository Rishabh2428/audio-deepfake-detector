const $ = (selector) => document.querySelector(selector);
const state = { currentId: null, currentReport: null, currentRecord: null, currentAnalyzedFile: null, maxUploadMb: 50, classifierAvailable: false, explanationBusy: false, modelCatalogBusy: false, modelCatalogProvider: null, modelCatalog: [], resembleBusy: false, resembleDetectionId: null, resembleReady: false, resembleQuestionId: null, resembleQuestionText: '' };

function showView(name) {
  document.querySelectorAll('.view').forEach((view) => { view.hidden = view.id !== `view-${name}`; });
  document.querySelectorAll('.tab').forEach((tab) => {
    const active = tab.dataset.view === name;
    tab.classList.toggle('active', active);
    if (active) tab.setAttribute('aria-current', 'page'); else tab.removeAttribute('aria-current');
  });
  if (name === 'history') loadHistory();
  if (name === 'training') loadTrainingStatus();
  if (name === 'method') history.replaceState(null, '', '#method');
  else history.replaceState(null, '', '#top');
}

document.querySelectorAll('.tab').forEach((tab) => tab.addEventListener('click', () => showView(tab.dataset.view)));
document.querySelectorAll('[data-open-view]').forEach((link) => link.addEventListener('click', (event) => {
  event.preventDefault(); showView(link.dataset.openView);
}));

async function checkHealth() {
  const badge = $('#service-state');
  try {
    const response = await fetch('/api/status');
    if (!response.ok) throw new Error('Local service unavailable');
    const status = await response.json();
    state.classifierAvailable = Boolean(status.classifier_available);
    state.maxUploadMb = Number(status.max_upload_mb) || 50;
    badge.textContent = status.classifier_available ? 'Experimental DSP model ready' : 'DSP only · train a model';
    badge.classList.remove('error'); badge.classList.add('ready');
    $('#model-analysis-mode').textContent = status.classifier_available ? 'DSP + local ML' : 'Descriptive DSP';
    $('#model-score-mode').textContent = status.classifier_available ? '1–11 index · uncalibrated' : 'Not available';
    $('#model-decision-mode').textContent = status.classifier_available ? `${Math.round((status.model_threshold ?? 0.5) * 100)}% cutoff` : 'INCONCLUSIVE';
    if (status.classifier_available) {
      $('#model-title').textContent = `${status.model_algorithm || 'Experimental model'} ready`;
      $('#model-status-copy').textContent = `${status.model_feature_set || 'Basic'} feature set · ${status.model_feature_count || 11} model inputs.`;
      $('#model-description').textContent = 'The 1–11 synthetic-evidence index is uncalibrated and depends on your labels. Its decision cutoff is adjustable in Train model.';
    } else {
      $('#model-title').textContent = 'No trained classifier';
      $('#model-status-copy').textContent = status.analysis_mode === 'stdlib_wav_dsp_only'
        ? 'WAV-only fallback is running. Start the full service to load the saved classifier and use compressed formats.'
        : 'No saved classifier was found. Import labeled audio and train a local model.';
      $('#model-description').textContent = 'DSP measurements remain descriptive; without a classifier, the decision is INCONCLUSIVE.';
    }
    const formats = Array.isArray(status.accepted_formats) ? status.accepted_formats.join(' · ') : 'Audio';
    const fileTypes = document.querySelector('.file-types');
    if (fileTypes) fileTypes.textContent = `${formats} · up to ${status.max_upload_mb} MB · ${status.max_duration_seconds} seconds`;
    $('#model-threshold').value = Math.round((Number(status.model_threshold) || 0.5) * 100);
    $('#threshold-value').textContent = `${$('#model-threshold').value}%`;
    $('#save-threshold').disabled = !state.classifierAvailable;
    updateExplainButton();
    await loadTrainingStatus();
  } catch {
    badge.textContent = 'Local service unavailable'; badge.classList.add('error');
  }
}

const fileInput = $('#audio-file');
const fileLabel = $('#selected-file');
const audioPreview = $('#audio-preview');
let previewUrl = null;
let previewBusy = false;
let previewRequest = null;
const analyzeButton = $('#analyze-button');
const dropzone = $('#dropzone');
dropzone.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    fileInput.click();
  }
});
fileInput.addEventListener('change', () => {
  state.currentAnalyzedFile = null;
  updateResembleButton();
  const file = fileInput.files?.[0];
  if (previewRequest) previewRequest.abort();
  previewRequest = null; previewBusy = false;
  fileLabel.hidden = !file;
  if (file) fileLabel.textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  audioPreview.hidden = !file;
  audioPreview.pause(); audioPreview.removeAttribute('src');
  audioPreview.load();
  $('#replay-preview').hidden = !file; $('#replay-preview').disabled = true;
  $('#prepare-preview').hidden = !file;
  $('#prepare-preview').disabled = false;
  $('#prepare-preview').textContent = 'Retry playable preview';
  $('#preview-status').textContent = '';
  const tooLarge = file && file.size > state.maxUploadMb * 1024 * 1024;
  analyzeButton.disabled = !file || tooLarge || Boolean(file);
  $('#upload-status').textContent = tooLarge ? `This file is larger than the ${state.maxUploadMb} MB analysis limit.` : '';
  $('#explain-locally').disabled = true;
  $('#local-explanation').hidden = true;
  if (file && !tooLarge) preparePlayablePreview();
});

async function preparePlayablePreview() {
  const file = fileInput.files?.[0];
  if (!file || previewBusy || analyzeButton.dataset.running === 'true') return;
  previewBusy = true;
  const controller = new AbortController(); previewRequest = controller;
  const button = $('#prepare-preview'); button.disabled = true;
  analyzeButton.disabled = true;
  $('#preview-status').textContent = 'Preparing a playable preview on this device…';
  try {
    const form = new FormData(); form.append('file', file, file.name);
    const response = await fetch('/api/preview', { method: 'POST', body: form, signal: controller.signal });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error?.error?.message || 'Could not prepare this preview.');
    }
    const blob = await response.blob();
    if (blob.size < 44 || !fileInput.files?.[0] || fileInput.files[0] !== file) return;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(new Blob([blob], { type: 'audio/wav' }));
    audioPreview.pause(); audioPreview.removeAttribute('src'); audioPreview.load();
    const ready = waitForPreviewAudio(audioPreview);
    audioPreview.src = previewUrl; audioPreview.load();
    await ready;
    $('#preview-status').textContent = 'Audio decoded locally. Use Play or Replay to listen.';
  } catch (error) {
    if (error.name === 'AbortError') return;
    $('#preview-status').textContent = error.message || 'Preview could not be decoded.';
  } finally {
    if (previewRequest === controller) {
      previewRequest = null; previewBusy = false; button.disabled = false;
      const selected = fileInput.files?.[0];
      analyzeButton.disabled = !selected || selected.size > state.maxUploadMb * 1024 * 1024;
    }
  }
}
function waitForPreviewAudio(audio) {
  if (audio.readyState >= HTMLMediaElement.HAVE_FUTURE_DATA) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const cleanup = () => { clearTimeout(timer); audio.removeEventListener('canplay', ready); audio.removeEventListener('error', failed); };
    const ready = () => { cleanup(); resolve(); };
    const failed = () => { cleanup(); reject(new Error('Your browser could not open the prepared audio preview.')); };
    const timer = setTimeout(() => { cleanup(); reject(new Error('The preview took too long to load. Choose Retry playable preview.')); }, 12000);
    audio.addEventListener('canplay', ready, { once: true }); audio.addEventListener('error', failed, { once: true });
  });
}
$('#prepare-preview').addEventListener('click', preparePlayablePreview);
audioPreview.addEventListener('canplay', () => {
  $('#replay-preview').disabled = false;
  $('#preview-status').textContent = 'Preview ready · decoded to browser-safe WAV on this device.';
});
audioPreview.addEventListener('error', () => {
  $('#replay-preview').disabled = true;
  if (fileInput.files?.length && !previewBusy) $('#preview-status').textContent = 'The preview could not be opened. Choose Retry playable preview.';
});
$('#replay-preview').addEventListener('click', async () => {
  try {
    audioPreview.pause(); audioPreview.currentTime = 0;
    await audioPreview.play();
    $('#preview-status').textContent = 'Playing from the beginning.';
  } catch {
    $('#preview-status').textContent = 'Press Play in the audio controls to start playback.';
  }
});

function makeWavFile(samples, sampleRate, name) {
  const buffer = new ArrayBuffer(44 + samples.length * 2), view = new DataView(buffer);
  const ascii = (offset, value) => [...value].forEach((character, index) => view.setUint8(offset + index, character.charCodeAt(0)));
  ascii(0, 'RIFF'); view.setUint32(4, 36 + samples.length * 2, true); ascii(8, 'WAVE');
  ascii(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  ascii(36, 'data'); view.setUint32(40, samples.length * 2, true);
  samples.forEach((sample, index) => view.setInt16(44 + index * 2, Math.max(-32768, Math.min(32767, Math.round(sample * 32767))), true));
  return new File([buffer], name, { type: 'audio/wav' });
}
$('#synthetic-test-tone').addEventListener('click', () => {
  const rate = 16000, duration = 2.2, count = Math.floor(rate * duration), samples = new Float32Array(count);
  for (let index = 0; index < count; index++) {
    const time = index / rate, fade = Math.min(1, time / 0.03, (duration - time) / 0.03);
    samples[index] = 0.18 * fade * (Math.sin(2 * Math.PI * 220 * time) + 0.55 * Math.sin(2 * Math.PI * 660 * time));
  }
  const transfer = new DataTransfer(); transfer.items.add(makeWavFile(samples, rate, 'synthetic_test_tone.wav'));
  fileInput.files = transfer.files; fileInput.dispatchEvent(new Event('change', { bubbles: true }));
  $('#preview-status').textContent = 'Test tone generated in this browser; it is not speech and will not be used for training.';
});
['dragenter', 'dragover'].forEach((type) => dropzone.addEventListener(type, (event) => { event.preventDefault(); dropzone.classList.add('dragging'); }));
['dragleave', 'drop'].forEach((type) => dropzone.addEventListener(type, (event) => { event.preventDefault(); dropzone.classList.remove('dragging'); }));
dropzone.addEventListener('drop', (event) => {
  const file = event.dataTransfer?.files?.[0];
  if (!file) return;
  const transfer = new DataTransfer(); transfer.items.add(file); fileInput.files = transfer.files;
  fileInput.dispatchEvent(new Event('change', { bubbles: true }));
});

$('#upload-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const file = fileInput.files?.[0];
  if (!file) return;
  analyzeButton.disabled = true;
  analyzeButton.dataset.running = 'true';
  $('#prepare-preview').disabled = true;
    $('#upload-status').textContent = 'Decoding audio and calculating DSP measurements…';
  try {
    const form = new FormData(); form.append('file', file, file.name);
    const response = await fetch('/api/analyze', { method: 'POST', body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data?.error?.message || 'Analysis could not be completed.');
    renderAnalysis(data, file);
    $('#upload-status').textContent = 'Measurements saved in local history.';
    updateHistoryCount();
    $('#result-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (error) {
    $('#upload-status').textContent = error.message || 'Local service unavailable.';
  } finally {
    analyzeButton.dataset.running = 'false';
    $('#prepare-preview').disabled = !fileInput.files?.length;
    analyzeButton.disabled = !fileInput.files?.length || fileInput.files[0].size > state.maxUploadMb * 1024 * 1024;
  }
});

function formatNumber(value, digits = 2) {
  return Number.isFinite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: digits }) : '—';
}
function addMetric(label, value, unit = '') {
  const item = document.createElement('div'); item.className = 'metric';
  const name = document.createElement('div'); name.className = 'metric-label'; name.textContent = label;
  const amount = document.createElement('div'); amount.className = 'metric-value'; amount.textContent = value;
  if (unit) { const unitNode = document.createElement('span'); unitNode.className = 'metric-unit'; unitNode.textContent = unit; amount.append(unitNode); }
  item.append(name, amount); return item;
}
function addEvidence(container, label, value, unit = '') {
  const item = document.createElement('div'); item.className = 'evidence-item';
  const name = document.createElement('span'); name.textContent = label;
  const amount = document.createElement('strong'); amount.textContent = value;
  if (unit) { const suffix = document.createElement('small'); suffix.textContent = unit; amount.append(suffix); }
  item.append(name, amount); container.append(item);
}
function addEvidenceHeading(container, text) {
  const heading = document.createElement('h4'); heading.className = 'evidence-group-heading'; heading.textContent = text; container.append(heading);
}
const PROVIDER_GUIDES = {
  gemini: {
    key_label: 'Google AI Studio API key',
    key_placeholder: 'Paste your Gemini API key',
    text: 'Create a Gemini API key in Google AI Studio and use a Google project with Gemini API access. Free access applies only to selected models and quotas; free-tier prompts may be used to improve Google products. A model appearing in the catalog does not guarantee that your project has free quota.',
    links: [['Create key', 'https://aistudio.google.com/app/apikey'], ['Model API', 'https://ai.google.dev/api/models'], ['Pricing and free tier', 'https://ai.google.dev/gemini-api/docs/pricing']],
  },
  openai: {
    key_label: 'OpenAI Platform API key',
    key_placeholder: 'Paste your OpenAI API key',
    text: 'Create a project API key on the OpenAI Platform. ChatGPT web access and subscriptions are separate from API billing. OpenAI’s model-list endpoint does not mark models as free; API calls are metered unless your API organization has credits.',
    links: [['Create API key', 'https://platform.openai.com/api-keys'], ['Model list API', 'https://developers.openai.com/api/reference/resources/models/methods/list'], ['API pricing', 'https://openai.com/api/pricing/']],
  },
  nvidia: {
    key_label: 'NVIDIA NIM API key',
    key_placeholder: 'Paste your NVIDIA API key',
    text: 'Create an API key in NVIDIA Build. Choose a hosted model endpoint available to your account. Free endpoint access is model-specific and can have rate or availability limits; the list marks known free endpoints and links the catalog for current status.',
    links: [['Create key / try models', 'https://build.nvidia.com/'], ['NVIDIA model catalog', 'https://build.nvidia.com/models'], ['NVIDIA API guide', 'https://docs.api.nvidia.com/nim/reference/llm-apis']],
  },
  openrouter: {
    key_label: 'OpenRouter API key',
    key_placeholder: 'Paste your OpenRouter API key',
    text: 'Create an OpenRouter key; do not use a Google, OpenAI, or NVIDIA key here. Free model variants are labeled from the live catalog pricing. Free requests still depend on OpenRouter and upstream model rate limits, availability, and account rules.',
    links: [['Create key', 'https://openrouter.ai/keys'], ['Free models', 'https://openrouter.ai/models?max_price=0'], ['Plans and request limits', 'https://openrouter.ai/pricing']],
  },
  anthropic: {
    key_label: 'Anthropic Claude API key',
    key_placeholder: 'Paste your Claude Console API key',
    text: 'Create a key in the Claude Console and use an organization/workspace with API access. Claude web access and API billing are separate; eligible accounts or promotions may include API credits. API requests are metered, so check your available credits and pricing first. If the key is not scoped to a workspace, enter that workspace ID below.',
    links: [['Create key', 'https://platform.claude.com/settings/keys'], ['Models API', 'https://platform.claude.com/docs/en/api/models/list'], ['API pricing and credits', 'https://platform.claude.com/docs/en/about-claude/pricing']],
  },
};

function renderProviderGuide(provider, access = null) {
  const guide = PROVIDER_GUIDES[provider];
  const section = $('#provider-access-guide');
  section.hidden = !guide;
  if (!guide) return;
  $('#provider-access-text').textContent = access
    ? [access.key_steps, access.api_notes].filter(Boolean).join(' ')
    : guide.text;
  $('#provider-key-label').textContent = guide.key_label;
  $('#explanation-api-key').placeholder = guide.key_placeholder;
  const links = $('#provider-access-links'); links.replaceChildren();
  const catalogLinks = access ? [
    ['Create API key', access.key_url],
    ['Model catalog details', access.models_url],
    ['Pricing and limits', access.pricing_url],
  ].filter(([, href]) => href) : guide.links;
  catalogLinks.forEach(([label, href]) => {
    const anchor = document.createElement('a'); anchor.href = href; anchor.target = '_blank'; anchor.rel = 'noreferrer'; anchor.textContent = `${label} ↗`; links.append(anchor);
  });
}

function providerPriceLabel(model) {
  return ({ free: 'FREE', free_tier: 'FREE-TIER ELIGIBLE', free_endpoint: 'FREE ENDPOINT', paid: 'PAID', api_metered: 'API METERED', tier_dependent: 'TIER DEPENDENT', endpoint_dependent: 'CHECK ENDPOINT', unknown: 'PRICE UNKNOWN' })[model?.pricing_status] || 'NOT CHECKED';
}

function updateModelPricingNote() {
  const modelId = $('#explanation-model').value;
  const found = state.modelCatalog.find((model) => model.id === modelId);
  $('#model-pricing-note').textContent = found
    ? `${providerPriceLabel(found)} · ${found.pricing_note}`
    : 'Load the selected provider’s model catalog after entering the provider’s own key.';
}

function updateExplainButton() {
  const provider = $('#explanation-provider').value;
  const local = provider === 'local';
  document.querySelectorAll('.provider-field').forEach((field) => { field.hidden = local; });
  $('#anthropic-workspace-wrap').hidden = local || provider !== 'anthropic';
  $('#explanation-consent-wrap').hidden = local;
  $('#explanation-privacy').hidden = local;
  $('#provider-access-guide').hidden = local;
  $('#local-explanation-status').textContent = local
    ? 'Local explanations stay on this device. Online providers are optional and require your explicit approval.'
    : 'Online explanation sends an allowlisted set of derived measurements after you check the consent box.';
  $('#load-provider-models').disabled = local || state.modelCatalogBusy || !$('#explanation-api-key').value.trim() || !$('#model-discovery-consent').checked;
  const allowed = !state.explanationBusy && !state.modelCatalogBusy && Boolean(state.currentReport) && (local || ($('#explanation-consent').checked && $('#explanation-api-key').value.trim().length > 0 && $('#explanation-model').value.trim().length > 0));
  $('#explain-locally').disabled = !allowed;
  updateModelPricingNote();
}

function populateProviderModels(result, provider) {
  state.modelCatalog = result.models || [];
  state.modelCatalogProvider = provider;
  const select = $('#explanation-model'); select.replaceChildren();
  for (const model of state.modelCatalog) {
    const option = document.createElement('option'); option.value = model.id;
    option.textContent = `${model.name} · ${providerPriceLabel(model)}`; select.append(option);
  }
  select.disabled = false;
  const preferred = state.modelCatalog.find((model) => ['free', 'free_tier', 'free_endpoint'].includes(model.pricing_status)) || state.modelCatalog[0];
  if (preferred) select.value = preferred.id;
  const freeCount = Number(result.free_model_count) || 0;
  const summary = freeCount
    ? `${result.model_count} text-generation models loaded · ${freeCount} marked free/free-tier eligible. Verify account quota and pricing before using one.`
    : `${result.model_count} models loaded · no model was confirmed free by this catalog response. See the provider-specific pricing note.`;
  const truncationNote = result.models_truncated ? ` Showing ${result.model_count} of ${result.total_model_count} models; free options are sorted first.` : '';
  $('#model-catalog-status').textContent = `${summary}${truncationNote} ${result.catalog_note || ''}`;
  renderProviderGuide(provider, result.access);
  updateExplainButton();
}
function explainEvidenceLocally(report) {
  const dsp = report.dsp || {}, time = dsp.time_domain || {}, frequency = dsp.frequency_domain || {};
  const prosody = dsp.prosodic || {}, phase = dsp.phase || {}, decision = report.decision || {};
  const notes = [];
  notes.push(`The recording is ${formatNumber(time.duration_seconds, 2)} seconds long. Its measured input-quality score is ${report.quality?.score_0_to_100 ?? 'not available'} out of 100; this rates recording conditions, not whether the voice is genuine.`);
  if (Number.isFinite(prosody.mean_f0_hz)) notes.push(`The estimated average pitch is ${formatNumber(prosody.mean_f0_hz, 1)} Hz. Jitter (${formatNumber(prosody.jitter_local_proxy_percent, 3)}%) and shimmer (${formatNumber(prosody.shimmer_local_proxy_percent, 3)}%) are short-frame estimates that vary with the speaker, words, and recording.`);
  else notes.push('Pitch was not estimated reliably from voiced frames, so pitch-based cues should not be interpreted for this recording.');
  notes.push(`Spectral entropy is ${formatNumber(frequency.spectral_entropy_normalized_mean, 3)} and flatness is ${formatNumber(frequency.spectral_flatness_mean, 4)}. These describe how energy is distributed across frequencies; neither value by itself identifies a human or synthetic source.`);
  if (Number.isFinite(phase.mean_residual_phase_variation_radians)) notes.push(`The phase-aware STFT measure is ${formatNumber(phase.mean_residual_phase_variation_radians, 3)} radians. Codec conversion, editing, microphones, and noise can also change phase measurements.`);
  if (Number.isFinite(decision.score_index_1_to_11)) notes.push(`The trained ${decision.model_algorithm || 'local'} model produced an experimental synthetic-evidence index of ${decision.score_index_1_to_11}/11. The current cutoff is ${Math.round((decision.threshold ?? 0.5) * 100)}%; the index is uncalibrated and is not a probability.`);
  else if (decision.classifier_status === 'withheld_quality') notes.push(`The classifier did not score this recording: ${decision.classifier_block_reasons?.join('; ') || 'a speech-quality gate was triggered'}.`);
  else notes.push('No trained local classifier was available for this recording, so the DSP measurements do not produce a human-versus-synthetic result.');
  notes.push('Use these measurements as inspection evidence only. Similar patterns can occur in genuine and generated speech, and the model depends on the examples used to train it.');
  return notes;
}
$('#explanation-provider').addEventListener('change', () => {
  const provider = $('#explanation-provider').value;
  state.modelCatalog = []; state.modelCatalogProvider = null;
  const modelSelect = $('#explanation-model'); modelSelect.replaceChildren(new Option('Load the model catalog to choose a model…', '')); modelSelect.disabled = true;
  $('#explanation-api-key').value = '';
  $('#anthropic-workspace-id').value = '';
  $('#explanation-consent').checked = false;
  $('#model-discovery-consent').checked = false;
  $('#model-catalog-status').textContent = '';
  $('#load-provider-models').textContent = 'Load model catalog';
  renderProviderGuide(provider);
  updateExplainButton();
});
$('#explanation-consent').addEventListener('change', updateExplainButton);
$('#model-discovery-consent').addEventListener('change', updateExplainButton);
$('#explanation-api-key').addEventListener('input', () => {
  if (state.modelCatalogProvider) {
    state.modelCatalog = []; state.modelCatalogProvider = null;
    const modelSelect = $('#explanation-model'); modelSelect.replaceChildren(new Option('Key changed — load the catalog again…', '')); modelSelect.disabled = true;
    $('#model-catalog-status').textContent = 'The key changed. Load the catalog again to check models for this key.';
  }
  updateExplainButton();
});
$('#anthropic-workspace-id').addEventListener('input', () => {
  if (state.modelCatalogProvider === 'anthropic') {
    state.modelCatalog = []; state.modelCatalogProvider = null;
    const modelSelect = $('#explanation-model'); modelSelect.replaceChildren(new Option('Workspace changed — load the catalog again…', '')); modelSelect.disabled = true;
    $('#model-catalog-status').textContent = 'The workspace changed. Load the catalog again for this workspace.';
  }
  updateExplainButton();
});
$('#explanation-model').addEventListener('change', updateExplainButton);
$('#load-provider-models').addEventListener('click', async () => {
  const provider = $('#explanation-provider').value;
  const apiKey = $('#explanation-api-key').value.trim();
  const workspaceId = $('#anthropic-workspace-id').value.trim();
  if (provider === 'local' || !apiKey || !$('#model-discovery-consent').checked || state.modelCatalogBusy) return;
  state.modelCatalogBusy = true;
  const button = $('#load-provider-models'); button.textContent = 'Loading catalog…';
  $('#model-catalog-status').textContent = 'Checking this provider’s text models and any published free/pricing details…';
  updateExplainButton();
  try {
    const response = await fetch('/api/providers/models', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ provider, api_key: apiKey, workspace_id: workspaceId, consent: true }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result?.error?.message || 'Could not load this provider catalog.');
    if (provider !== $('#explanation-provider').value || apiKey !== $('#explanation-api-key').value.trim() || workspaceId !== $('#anthropic-workspace-id').value.trim()) {
      throw new Error('The provider or key changed while loading the catalog. Load it again with the current selection.');
    }
    populateProviderModels(result, provider);
  } catch (error) {
    $('#model-catalog-status').textContent = error.message || 'Could not load this provider catalog.';
  } finally {
    state.modelCatalogBusy = false; button.textContent = 'Load model catalog'; updateExplainButton();
  }
});
$('#explain-locally').addEventListener('click', async () => {
  if (!state.currentReport || $('#explain-locally').disabled) return;
  const output = $('#local-explanation');
  output.replaceChildren();
  output.hidden = false;
  const provider = $('#explanation-provider').value;
  if (provider !== 'local') {
    const button = $('#explain-locally'), keyInput = $('#explanation-api-key');
    const apiKey = keyInput.value;
    state.explanationBusy = true; button.disabled = true; button.textContent = 'Requesting explanation…';
    output.textContent = 'Sending the derived DSP measurements you approved…';
    try {
      const response = await fetch('/api/explain', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider, model: $('#explanation-model').value.trim(), api_key: apiKey, workspace_id: $('#anthropic-workspace-id').value.trim(), consent: $('#explanation-consent').checked, dsp: state.currentReport.dsp || {} }) });
      const result = await response.json();
      if (!response.ok) throw new Error(result?.error?.message || 'The provider request failed.');
      output.textContent = `${result.provider} · ${result.model}\n\n${result.explanation}\n\nSent measurements: ${result.sent_fields.join(', ')}`;
    } catch (error) {
      output.textContent = error.message || 'The provider request failed.';
    } finally {
      state.explanationBusy = false;
      keyInput.value = '';
      button.textContent = 'Explain evidence';
      updateExplainButton();
    }
    return;
  }
  const list = document.createElement('ul');
  explainEvidenceLocally(state.currentReport).forEach((text) => { const item = document.createElement('li'); item.textContent = text; list.append(item); });
  output.append(list);
});
$('#copy-dsp-for-chat').addEventListener('click', async () => {
  if (!state.currentReport) return;
  const payload = {
    task: 'Explain these DSP measurements in plain language. Do not claim they prove whether a voice is human or synthetic. Do not invent a probability. Note that non-speech or quality-gated audio has no voice-class result.',
    dsp_measurements: state.currentReport.dsp || {},
    quality_checks: state.currentReport.quality || {},
    classifier_result: state.currentReport.decision || {},
  };
  try {
    await navigator.clipboard.writeText(JSON.stringify(payload, null, 2));
    $('#copy-dsp-status').textContent = 'Measurements copied. Paste them into the chat you choose.';
  } catch {
    $('#copy-dsp-status').textContent = 'Clipboard access was blocked by this browser. Use the local explanation or enable clipboard permission.';
  }
});

function updateResembleButton() {
  const consent = $('#resemble-consent').checked;
  const key = $('#resemble-api-key').value.trim();
  const intelligence = $('#resemble-intelligence').checked;
  $('#resemble-infer-intelligence').disabled = !intelligence;
  if (!intelligence) $('#resemble-infer-intelligence').checked = false;
  $('#run-resemble').disabled = !state.currentAnalyzedFile || !consent || !key || state.resembleBusy;
  $('#check-resemble').disabled = !state.resembleDetectionId || !key || state.resembleBusy;
  $('#clear-resemble-key').disabled = !key || state.resembleBusy;
  $('#ask-resemble').disabled = !state.resembleReady || !state.resembleDetectionId || Boolean(state.resembleQuestionId) || !key || !$('#resemble-question-consent').checked || !$('#resemble-question').value.trim() || state.resembleBusy;
  $('#check-resemble-answer').disabled = !state.resembleQuestionId || !key || state.resembleBusy;
}

function resembleValue(value) {
  if (value === null || value === undefined || value === '') return 'No result returned';
  if (typeof value === 'string') return value;
  return JSON.stringify(value, null, 2);
}

function showResembleResult(result) {
  const output = $('#resemble-result'); output.hidden = false;
  const pending = result.pending || result.status === 'processing';
  if (pending) {
    state.resembleReady = false;
    $('#resemble-ask-panel').hidden = true;
    state.resembleDetectionId = result.detection_id;
    $('#check-resemble').hidden = false;
    $('#check-resemble').disabled = false;
    const stages = [`Detect: ${result.status || 'processing'}`];
    if (result.intelligence_status) stages.push(`Intelligence: ${result.intelligence_status}`);
    if (result.watermark?.status) stages.push(`Watermark: ${result.watermark.status}`);
    output.textContent = `Resemble is still processing. ${stages.join(' · ')}. Detection ID: ${result.detection_id || 'not returned'}. Keep the token in the field and use Check result status.`;
    return;
  }
  const report = [
    `${result.provider || 'Resemble Detect'} · ${result.verdict || 'INCONCLUSIVE'}`,
    `Job status: ${result.status || 'unknown'}`,
  ];
  if (result.provider_error) report.push(`Provider message: ${result.provider_error}`);
  if (result.provider_label) report.push(`Provider label: ${result.provider_label}`);
  if (Number.isFinite(result.aggregated_score)) report.push(`Aggregated score: ${(result.aggregated_score * 100).toFixed(2)}% · raw ${result.aggregated_score.toFixed(4)} (provider score; not a calibrated probability)`);
  if (Number.isFinite(result.consistency)) report.push(`Consistency: ${result.consistency.toFixed(4)}`);
  if (Array.isArray(result.chunk_scores)) {
    const scores = result.chunk_scores.map((score) => Number(score)).map((score) => Number.isFinite(score) ? `${(score * 100).toFixed(2)}%` : 'unavailable');
    report.push(`Per-segment scores: ${scores.join(', ')}`);
  }
  report.push(`Zero-retention requested: ${result.zero_retention_requested ? 'yes' : 'no'}`);
  if (result.file_deleted_at) report.push(`Provider reports file deletion at: ${result.file_deleted_at}`);
  if (result.intelligence !== undefined && result.intelligence !== null) {
    report.push(`INTELLIGENCE REPORT · ${result.intelligence_status || 'returned'}\n${formatResembleIntelligence(result.intelligence)}`);
  }
  if (result.watermark) report.push(`WATERMARK / SYNTHID\n${resembleValue(result.watermark)}`);
  if (result.audio_source_tracing) report.push(`AUDIO SOURCE TRACING\n${resembleValue(result.audio_source_tracing)}`);
  if (result.c2pa_manifest) report.push(`CONTENT CREDENTIALS (C2PA)\n${resembleValue(result.c2pa_manifest)}`);
  if (result.visualization) report.push(`DEEPFAKE VISUALIZATION\n${resembleValue(result.visualization)}`);
  if (result.warning) report.push(result.warning);
  output.textContent = report.join('\n\n');
  state.resembleDetectionId = result.detection_id || state.resembleDetectionId;
  state.resembleReady = result.status === 'completed' && Boolean(state.resembleDetectionId);
  $('#check-resemble').hidden = true;
  $('#resemble-ask-panel').hidden = !state.resembleReady;
  $('#resemble-question-status').textContent = '';
  $('#resemble-consent').checked = false;
  $('#resemble-intelligence').checked = false;
  $('#resemble-watermark').checked = false;
  $('#resemble-source-tracing').checked = false;
  $('#resemble-visualize').checked = false;
  $('#resemble-infer-intelligence').checked = false;
  updateResembleButton();
}

function formatResembleIntelligence(value) {
  const experts = Array.isArray(value) ? value : [value];
  return experts.map((expert, index) => {
    if (!expert || typeof expert !== 'object') return `Expert ${index + 1}: ${resembleValue(expert)}`;
    const lines = [`Expert ${index + 1} · ${expert.status || 'report'}`];
    if (expert.error_message) lines.push(`Status message: ${expert.error_message}`);
    const description = expert.description;
    if (description && typeof description === 'object' && !Array.isArray(description)) {
      Object.entries(description).forEach(([key, detail]) => {
        const title = key.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
        lines.push(`${title}:\n${resembleValue(detail)}`);
      });
    } else if (description !== null && description !== undefined) {
      lines.push(resembleValue(description));
    }
    return lines.join('\n\n');
  }).join('\n\n');
}
async function resembleRequest(url, options) {
  const response = await fetch(url, options);
  const result = await response.json();
  if (!response.ok) throw new Error(result?.error?.message || 'Resemble Detect request failed.');
  return result;
}
$('#resemble-consent').addEventListener('change', updateResembleButton);
$('#resemble-api-key').addEventListener('input', updateResembleButton);
$('#resemble-question').addEventListener('input', updateResembleButton);
$('#resemble-question-consent').addEventListener('change', updateResembleButton);
$('#clear-resemble-key').addEventListener('click', () => {
  $('#resemble-api-key').value = '';
  $('#resemble-question-consent').checked = false;
  updateResembleButton();
});
['resemble-intelligence', 'resemble-watermark', 'resemble-source-tracing', 'resemble-visualize', 'resemble-infer-intelligence'].forEach((id) => {
  $(`#${id}`).addEventListener('change', updateResembleButton);
});
$('#run-resemble').addEventListener('click', async () => {
  const file = state.currentAnalyzedFile, key = $('#resemble-api-key').value.trim();
  if (!file || !key || !$('#resemble-consent').checked || state.resembleBusy) return;
  state.resembleReady = false; state.resembleDetectionId = null; state.resembleQuestionId = null;
  $('#resemble-ask-panel').hidden = true; $('#resemble-conversation').replaceChildren();
  $('#check-resemble-answer').hidden = true;
  const form = new FormData(); form.append('file', file, file.name); form.append('api_key', key); form.append('consent', 'true');
  form.append('intelligence', String($('#resemble-intelligence').checked));
  form.append('detect_watermark', String($('#resemble-watermark').checked));
  form.append('audio_source_tracing', String($('#resemble-source-tracing').checked));
  form.append('infer_from_intelligence', String($('#resemble-infer-intelligence').checked));
  form.append('visualize', String($('#resemble-visualize').checked));
  state.resembleBusy = true; updateResembleButton();
  $('#resemble-result').hidden = false; $('#resemble-result').textContent = 'Uploading audio to Resemble and waiting for its result…';
  try { showResembleResult(await resembleRequest('/api/external-detect/resemble', { method: 'POST', body: form })); }
  catch (error) { $('#resemble-result').textContent = error.message || 'Resemble Detect request failed.'; }
  finally { state.resembleBusy = false; updateResembleButton(); }
});
$('#check-resemble').addEventListener('click', async () => {
  const key = $('#resemble-api-key').value.trim();
  if (!state.resembleDetectionId || !key || state.resembleBusy) return;
  state.resembleBusy = true; updateResembleButton();
  $('#resemble-result').textContent = 'Checking with Resemble…';
  try {
    const result = await resembleRequest(`/api/external-detect/resemble/${encodeURIComponent(state.resembleDetectionId)}`, { headers: { Authorization: `Bearer ${key}` } });
    showResembleResult(result);
  } catch (error) { $('#resemble-result').textContent = error.message || 'Could not check this Resemble result.'; }
  finally { state.resembleBusy = false; updateResembleButton(); }
});

function displayResembleAnswer(result) {
  const status = $('#resemble-question-status');
  if (['pending', 'processing'].includes(result.status)) {
    state.resembleQuestionId = result.question_id;
    status.textContent = `Resemble is preparing the answer (${result.status}). Select Check answer in a few seconds.`;
    $('#check-resemble-answer').hidden = false;
    updateResembleButton();
    return;
  }
  state.resembleQuestionId = null;
  $('#check-resemble-answer').hidden = true;
  if (result.status === 'failed') {
    status.textContent = result.error_message || 'Resemble could not generate an answer.';
  } else {
    const exchange = document.createElement('article'); exchange.className = 'resemble-exchange';
    const question = document.createElement('h5'); question.textContent = result.query || state.resembleQuestionText;
    const answer = document.createElement('p'); answer.textContent = result.answer || 'No answer was returned.';
    exchange.append(question, answer); $('#resemble-conversation').append(exchange);
    $('#resemble-question').value = '';
    status.textContent = 'Answer received from Resemble Intelligence.';
  }
  updateResembleButton();
}

$('#ask-resemble').addEventListener('click', async () => {
  const question = $('#resemble-question').value.trim(), key = $('#resemble-api-key').value.trim();
  if (!question || !key || !state.resembleReady || !$('#resemble-question-consent').checked || state.resembleBusy) return;
  state.resembleQuestionText = question; state.resembleBusy = true; updateResembleButton();
  $('#resemble-question-status').textContent = 'Sending your question to Resemble Intelligence…';
  try {
    const result = await resembleRequest(`/api/external-detect/resemble/${encodeURIComponent(state.resembleDetectionId)}/intelligence`, {
      method: 'POST', headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' }, body: JSON.stringify({ query: question, consent: true }),
    });
    displayResembleAnswer(result);
  } catch (error) { $('#resemble-question-status').textContent = error.message || 'Could not ask Resemble Intelligence.'; }
  finally { state.resembleBusy = false; updateResembleButton(); }
});

$('#check-resemble-answer').addEventListener('click', async () => {
  const key = $('#resemble-api-key').value.trim();
  if (!state.resembleQuestionId || !key || state.resembleBusy) return;
  state.resembleBusy = true; updateResembleButton();
  $('#resemble-question-status').textContent = 'Checking the answer…';
  try {
    const result = await resembleRequest(`/api/external-detect/resemble/${encodeURIComponent(state.resembleDetectionId)}/intelligence/${encodeURIComponent(state.resembleQuestionId)}`, {
      headers: { Authorization: `Bearer ${key}` },
    });
    displayResembleAnswer(result);
  } catch (error) { $('#resemble-question-status').textContent = error.message || 'Could not check the Intelligence answer.'; }
  finally { state.resembleBusy = false; updateResembleButton(); }
});

function svgLineChart(values, { xMin = 0, xMax = null, yMin = null, yMax = null, xLabel = '', yLabel = '', stroke = '#b6e5c3', fill = false } = {}) {
  const width = 600, height = 175, left = 49, right = 10, top = 8, bottom = 26;
  const valid = values.filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
  if (!valid.length) return '<p class="empty-state">No plotted measurements.</p>';
  xMax = xMax ?? Math.max(...valid.map((p) => p.x)); yMin = yMin ?? Math.min(...valid.map((p) => p.y)); yMax = yMax ?? Math.max(...valid.map((p) => p.y));
  if (xMax === xMin) xMax = xMin + 1; if (yMax === yMin) { yMax += 1; yMin -= 1; }
  const sx = (x) => left + (x - xMin) / (xMax - xMin) * (width - left - right);
  const sy = (y) => top + (yMax - y) / (yMax - yMin) * (height - top - bottom);
  const path = valid.map((p, index) => `${index ? 'L' : 'M'}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' ');
  const baseY = height - bottom;
  const area = fill ? `<path d="${path} L${sx(valid.at(-1).x).toFixed(1)},${baseY} L${sx(valid[0].x).toFixed(1)},${baseY} Z" fill="${stroke}" opacity=".1"/>` : '';
  const yTicks = [0, 0.5, 1].map((fraction) => {
    const y = top + fraction * (height - top - bottom); const value = yMax - fraction * (yMax - yMin);
    return `<line x1="${left}" x2="${width - right}" y1="${y}" y2="${y}" stroke="#26372d"/><text x="${left - 7}" y="${y + 3}" text-anchor="end">${formatNumber(value, 1)}</text>`;
  }).join('');
  return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${yLabel} by ${xLabel}"><title>${yLabel} by ${xLabel}</title>${yTicks}${area}<path d="${path}" fill="none" stroke="${stroke}" stroke-width="1.7" vector-effect="non-scaling-stroke"/><text x="${(left + width - right) / 2}" y="${height - 3}" text-anchor="middle">${xLabel}</text><text transform="translate(12 ${(height - bottom + top) / 2}) rotate(-90)" text-anchor="middle">${yLabel}</text></svg>`;
}

function drawSpectrogram(spectrogram) {
  const canvas = $('#spectrogram-chart');
  const frames = spectrogram?.values_u8;
  const bands = spectrogram?.frequency_hz;
  if (!Array.isArray(frames) || !frames.length || !Array.isArray(bands) || !bands.length) {
    canvas.width = 900; canvas.height = 240;
    const context = canvas.getContext('2d', { alpha: false });
    context.fillStyle = '#080d18'; context.fillRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = '#9aaac0'; context.font = '14px sans-serif'; context.textAlign = 'center';
    context.fillText('No saved spectrogram for this older report', canvas.width / 2, canvas.height / 2);
    canvas.setAttribute('aria-label', 'Spectrogram was not saved with this older report. Reanalyze the audio to generate one.');
    return;
  }
  const columns = Math.min(frames.length, 600), rows = bands.length;
  canvas.width = columns; canvas.height = rows;
  const context = canvas.getContext('2d', { alpha: false });
  const pixels = context.createImageData(columns, rows);
  for (let x = 0; x < columns; x++) {
    const source = frames[Math.min(frames.length - 1, Math.floor(x * frames.length / columns))];
    for (let band = 0; band < rows; band++) {
      const intensity = Math.max(0, Math.min(1, Number(source?.[band] || 0) / 255));
      const low = intensity < 0.5, blend = low ? intensity * 2 : (intensity - 0.5) * 2;
      const start = low ? [8, 13, 24] : [47, 194, 194];
      const end = low ? [47, 194, 194] : [255, 181, 91];
      const offset = ((rows - 1 - band) * columns + x) * 4;
      pixels.data[offset] = start[0] + (end[0] - start[0]) * blend;
      pixels.data[offset + 1] = start[1] + (end[1] - start[1]) * blend;
      pixels.data[offset + 2] = start[2] + (end[2] - start[2]) * blend;
      pixels.data[offset + 3] = 255;
    }
  }
  context.putImageData(pixels, 0, 0);
  const duration = spectrogram.time_seconds?.at(-1) ?? 0;
  canvas.setAttribute('aria-label', `Log-frequency STFT spectrogram to ${formatNumber(duration, 1)} seconds. Color shows power relative to this recording; it is not a deepfake score.`);
}

function renderAnalysis(data, analyzedFile = null) {
  state.currentId = data.analysis_id; state.currentReport = data.report; state.currentRecord = data;
  state.currentAnalyzedFile = analyzedFile;
  state.resembleDetectionId = null; state.resembleReady = false; state.resembleQuestionId = null; state.resembleQuestionText = '';
  $('#resemble-api-key').value = ''; $('#resemble-consent').checked = false;
  $('#resemble-question-consent').checked = false; $('#resemble-question').value = '';
  $('#resemble-question-status').textContent = ''; $('#resemble-conversation').replaceChildren();
  $('#resemble-intelligence').checked = false; $('#resemble-watermark').checked = false;
  $('#resemble-source-tracing').checked = false; $('#resemble-infer-intelligence').checked = false;
  $('#resemble-visualize').checked = false;
  $('#resemble-result').hidden = true; $('#resemble-result').textContent = ''; $('#resemble-ask-panel').hidden = true;
  $('#check-resemble').hidden = true;
  updateResembleButton();
  const report = data.report; const dsp = report.dsp; const time = dsp.time_domain; const freq = dsp.frequency_domain;
  $('#result-section').hidden = false;
  $('#copy-dsp-for-chat').disabled = false;
  $('#copy-dsp-status').textContent = '';
  $('#explanation-consent').checked = false;
  $('#model-discovery-consent').checked = false;
  $('#explanation-api-key').value = '';
  state.modelCatalog = []; state.modelCatalogProvider = null;
  $('#explanation-model').replaceChildren(new Option('Load the model catalog to choose a model…', '')); $('#explanation-model').disabled = true;
  $('#model-catalog-status').textContent = '';
  updateExplainButton();
  $('#local-explanation').hidden = true;
  $('#local-explanation').replaceChildren();
  $('#result-time').textContent = new Date(data.created_at).toLocaleString();
  $('#result-title').textContent = report.decision.label;
  $('#result-reason').textContent = report.decision.reason;
  const score = report.decision.score, scoreIndex = report.decision.score_index_1_to_11, cutoff = Math.round((report.decision.threshold ?? 0.5) * 100);
  $('#result-score').innerHTML = Number.isFinite(scoreIndex) ? `${scoreIndex}<span>/ 11</span>` : '—<span>/ 11</span>';
  $('#result-score-note').textContent = Number.isFinite(scoreIndex) ? `Experimental synthetic-evidence index · synthetic raw output ${score.toFixed(3)} · raw decision cutoff ${(report.decision.threshold ?? 0.5).toFixed(2)}. Not a probability.` : report.decision.classifier_status === 'withheld_quality' ? `No voice probability is shown because this signal failed the speech quality gate (${(report.decision.classifier_block_reasons || []).join('; ')}). A generated test tone is tonal sound, not human or synthetic speech.` : 'No trained detector is loaded. A score cannot be reported.';
  $('#result-score-tag').textContent = Number.isFinite(scoreIndex) ? `EXPERIMENTAL · RAW CUTOFF ${(report.decision.threshold ?? 0.5).toFixed(2)}` : 'UNAVAILABLE · NOT A ZERO SCORE';
  const hasScore = Number.isFinite(score);
  const humanRaw = hasScore ? (1 - score) * 100 : null, syntheticRaw = hasScore ? score * 100 : null;
  $('#human-output-value').textContent = hasScore ? (1 - score).toFixed(3) : '—';
  $('#synthetic-output-value').textContent = hasScore ? score.toFixed(3) : '—';
  $('#human-output-bar').style.width = hasScore ? `${humanRaw}%` : '0%';
  $('#synthetic-output-bar').style.width = hasScore ? `${syntheticRaw}%` : '0%';
  $('#human-output-card').classList.toggle('is-leading', report.decision.label === 'LIKELY_HUMAN');
  $('#synthetic-output-card').classList.toggle('is-leading', report.decision.label === 'LIKELY_SYNTHETIC');
  $('#class-output-summary').textContent = hasScore
    ? `${report.decision.label === 'LIKELY_HUMAN' ? 'Model leans HUMAN' : report.decision.label === 'LIKELY_SYNTHETIC' ? 'Model leans SYNTHETIC' : 'Model result withheld'} · raw cutoff ${(report.decision.threshold ?? 0.5).toFixed(2)}. Raw class outputs are uncalibrated and are not probabilities.`
    : report.decision.classifier_status === 'withheld_quality' ? 'No class comparison: the quality gate withheld model inference.' : 'No class comparison: train a local classifier first.';
  const stages = {
    decode: { label: `${String(report.source.format || 'audio').toUpperCase()} decoded`, state: 'is-complete' },
    validate: { label: report.quality.warnings?.length ? 'checked · notices' : 'passed', state: report.quality.warnings?.length ? 'is-warning' : 'is-complete' },
    dsp: { label: 'measured locally', state: 'is-complete' },
    model: report.decision.classifier_status === 'experimental_user_trained'
      ? { label: report.decision.model_algorithm || 'classifier scored', state: 'is-complete' }
      : report.decision.classifier_status === 'withheld_quality'
        ? { label: 'withheld by quality gate', state: 'is-warning' }
        : { label: 'not trained', state: 'is-skipped' },
    report: { label: 'Saved', state: 'is-complete' },
  };
  for (const [step, details] of Object.entries(stages)) {
    const item = document.querySelector(`[data-pipeline-step="${step}"]`);
    item.classList.remove('is-complete', 'is-warning', 'is-skipped'); item.classList.add(details.state);
    $(`#pipeline-${step}-label`).textContent = details.label;
  }
  $('#pipeline-result-note').textContent = report.decision.classifier_status === 'experimental_user_trained'
    ? 'Audio was decoded and validated; DSP ran locally, the trained classifier scored it, and the report was saved to History.'
    : report.decision.classifier_status === 'withheld_quality'
      ? 'Audio was decoded and validated; DSP ran locally and the report was saved. The classifier was withheld by the quality gate.'
      : 'Audio was decoded and validated; DSP ran locally and the report was saved. No classifier is configured, so no class score was produced.';
  const warnings = $('#quality-warnings'); warnings.replaceChildren();
  report.quality.warnings.forEach((message) => { const item = document.createElement('div'); item.className = 'warning'; item.textContent = message; warnings.append(item); });
  const qualityScore = report.quality.score_0_to_100;
  const qualityRing = $('#input-quality-ring');
  if (Number.isFinite(qualityScore)) {
    qualityRing.style.setProperty('--quality', `${qualityScore}%`);
    qualityRing.setAttribute('aria-valuenow', qualityScore);
    qualityRing.setAttribute('aria-label', `Technical input quality ${qualityScore} out of 100; not model confidence`);
    $('#input-quality-value').textContent = qualityScore;
    $('#input-quality-note').textContent = report.quality.meaning;
    const factors = $('#input-quality-factors'); factors.replaceChildren();
    const notes = report.quality.factors?.length ? report.quality.factors : ['No penalty from the listed technical-quality checks.'];
    notes.forEach((note) => { const item = document.createElement('li'); item.textContent = note; factors.append(item); });
  }
  const metrics = $('#metric-grid'); metrics.replaceChildren(
    addMetric('Duration', formatNumber(time.duration_seconds, 2), 'sec'),
    addMetric('RMS level', time.rms_dbfs === null ? '−∞' : formatNumber(time.rms_dbfs, 1), 'dBFS'),
    addMetric('Peak', time.peak_dbfs === null ? '−∞' : formatNumber(time.peak_dbfs, 1), 'dBFS'),
    addMetric('Centroid', formatNumber(freq.spectral_centroid_mean_hz, 0), 'Hz'),
    addMetric('Flatness', formatNumber(freq.spectral_flatness_mean, 3)),
    addMetric('Entropy', formatNumber(freq.spectral_entropy_normalized_mean, 3)),
    addMetric('85% roll-off', formatNumber(freq.spectral_rolloff_85_mean_hz, 0), 'Hz'),
    addMetric('Low-energy frames', formatNumber(time.low_energy_frame_fraction_relative_minus_40_db * 100, 1), '%'),
    addMetric('Zero-crossing rate', formatNumber(freq.zero_crossing_rate_mean, 4)),
    addMetric('Spectral flux', formatNumber(freq.spectral_flux_mean, 4)),
    addMetric('Clipped samples', formatNumber(time.clipping_sample_fraction_at_0_999 * 100, 3), '%'),
    addMetric('DC offset', formatNumber(time.dc_offset_mean_amplitude, 5)),
    addMetric('Power below 4 kHz', formatNumber(freq.power_below_4khz_fraction * 100, 1), '%'),
    addMetric('Original sample rate', formatNumber(report.source.original_sample_rate_hz, 0), 'Hz'),
    addMetric('Stereo fold-down level', formatNumber(report.source.downmix_rms_ratio, 3)),
  );
  const evidence = $('#evidence-grid'); evidence.replaceChildren();
  const decision = report.decision;
  const modelFeatures = [
    ['RMS level', time.rms_dbfs, 'dBFS', 2], ['Peak level', time.peak_dbfs, 'dBFS', 2],
    ['DC offset', time.dc_offset_mean_amplitude, '', 5], ['Clipping fraction', time.clipping_sample_fraction_at_0_999 * 100, '%', 3],
    ['Low-energy frames', time.low_energy_frame_fraction_relative_minus_40_db * 100, '%', 2],
    ['Spectral centroid', freq.spectral_centroid_mean_hz, 'Hz', 1], ['Spectral flatness', freq.spectral_flatness_mean, '', 4],
    ['Spectral entropy', freq.spectral_entropy_normalized_mean, '', 4], ['85% spectral roll-off', freq.spectral_rolloff_85_mean_hz, 'Hz', 1],
    ['Zero-crossing rate', freq.zero_crossing_rate_mean, '', 5], ['Spectral flux', freq.spectral_flux_mean, '', 5],
  ];
  const advancedModel = decision.feature_set === 'advanced';
  addEvidenceHeading(evidence, advancedModel ? `Classifier inputs · ${decision.feature_count || 38} extended DSP/cepstral features` : 'Classifier inputs · 11 basic DSP features');
  for (const [label, value, unit, digits] of modelFeatures) addEvidence(evidence, label, formatNumber(value, digits), unit);
  addEvidenceHeading(evidence, advancedModel ? 'Additional model inputs · selected extended model' : 'Additional DSP descriptors · not used by this saved model');
  const pro = dsp.prosodic || {}, ext = dsp.extended_spectral || {}, bands = ext.band_power_fractions || {}, temporal = dsp.temporal_modulation || {};
  [
    ['Mean pitch estimate', pro.mean_f0_hz, 'Hz', 1], ['Pitch variation proxy', pro.pitch_variation_proxy_pct, '%', 3],
    ['Frame-level amplitude variation proxy', pro.frame_level_variation_proxy_pct, '%', 2],
    ['Autocorrelation harmonicity proxy', pro.autocorrelation_harmonicity_proxy_mean, '', 3],
    ['Autocorrelation HNR proxy', pro.autocorrelation_hnr_proxy_mean_db, 'dB', 2],
    ['Voiced-frame fraction', pro.voiced_frame_fraction * 100, '%', 1],
    ['Spectral flux variation', ext.spectral_flux_std, '', 5], ['Spectral spread', ext.spectral_spread_mean_hz, 'Hz', 1],
    ['Spectral slope', ext.spectral_slope_mean_db_per_octave, 'dB/oct', 2], ['Spectral crest', ext.spectral_crest_mean, '', 2],
    ['Mean frame RMS', ext.frame_rms_mean_amplitude, '', 5], ['Frame RMS variation', ext.frame_rms_std_amplitude, '', 5],
    ['Amplitude modulation index', temporal.amplitude_modulation_index, '', 3], ['Peak envelope modulation (2–8 Hz)', temporal.peak_modulation_frequency_2_to_8_hz, 'Hz', 2],
    ['Envelope power in 2–8 Hz band', Number.isFinite(temporal.power_fraction_2_to_8_hz_of_0_5_to_20_hz) ? temporal.power_fraction_2_to_8_hz_of_0_5_to_20_hz * 100 : null, '%', 1],
    ['Low-band power (<500 Hz)', bands.low_below_500_hz * 100, '%', 1],
    ['Mid-band power (500–2000 Hz)', bands.mid_500_to_2000_hz * 100, '%', 1],
    ['High-band power (≥2000 Hz)', bands.high_at_or_above_2000_hz * 100, '%', 1],
  ].forEach(([label, value, unit, digits]) => addEvidence(evidence, label, formatNumber(value, digits), unit));
  const cepstral = dsp.cepstral || {}, phase = dsp.phase || {};
  for (const family of ['mfcc', 'lfcc', 'cqcc_style']) {
    const label = family === 'cqcc_style' ? 'CQT-style cepstra' : family.toUpperCase();
    const coefficients = cepstral[family]?.mean_coefficients || [];
    coefficients.slice(1, 5).forEach((value, index) => addEvidence(evidence, `${label} coefficient C${index + 1}`, formatNumber(value, 3)));
  }
  addEvidence(evidence, 'Residual STFT phase variation', formatNumber(phase.mean_residual_phase_variation_radians, 4), 'rad');
  addEvidence(evidence, 'Residual phase jump ratio', formatNumber((phase.residual_phase_jump_ratio || 0) * 100, 2), '%');
  addEvidence(evidence, 'Group-delay outlier ratio', formatNumber((phase.group_delay_outlier_ratio || 0) * 100, 2), '%');
  if (Number.isFinite(decision.score_index_1_to_11)) {
    addEvidence(evidence, 'Synthetic evidence index', `${decision.score_index_1_to_11} / 11`, `decision cutoff ${Math.round((decision.threshold ?? 0.5) * 100)}%`);
    addEvidence(evidence, 'Raw model output', formatNumber(decision.score, 4), 'uncalibrated · not a probability');
  } else if (decision.classifier_status === 'withheld_quality') addEvidence(evidence, 'Classifier score', 'Withheld', 'speech-quality gate');
  else addEvidence(evidence, 'Classifier score', 'Unavailable', 'DSP only');
  $('#evidence-note').textContent = Number.isFinite(decision.score_index_1_to_11)
    ? advancedModel ? 'The selected model used basic DSP plus MFCC, LFCC, CQT-style cepstra, prosody, spectral, and phase features. The index is uncalibrated and does not identify the audio source.' : 'The local model used the first 11 basic DSP values. The added cepstral, phase, and prosodic measurements are shown for inspection only. The index is uncalibrated.'
    : decision.classifier_status === 'withheld_quality' ? 'These measurements describe the signal. The local model was available, but a quality warning prevented its inference.' : 'These measurements describe the signal only. No human-versus-AI classifier ran for this file.';
  const wave = dsp.waveform.samples.map((y, index) => ({ x: index * dsp.waveform.time_step_seconds, y }));
  $('#waveform-chart').innerHTML = svgLineChart(wave, { yMin: -1, yMax: 1, xLabel: 'Time (seconds)', yLabel: 'Amplitude', stroke: '#b6e5c3', fill: true });
  const psd = dsp.power_spectral_density;
  const spectrum = psd.frequency_hz.map((x, index) => ({ x, y: psd.power_db_per_hz[index] }));
  $('#spectrum-chart').innerHTML = svgLineChart(spectrum, { xMin: 0, xMax: psd.frequency_hz.at(-1), yLabel: 'dB / Hz', xLabel: 'Frequency (Hz)', stroke: '#e5bf77' });
  $('#spectrum-meta').textContent = `${psd.frequency_resolution_hz.toFixed(1)} Hz bins · ${psd.units}`;
  drawSpectrogram(dsp.spectrogram);
  const details = $('#technical-details'); details.replaceChildren();
  const config = dsp.configuration;
  const section = (title) => { const heading = document.createElement('h4'); heading.textContent = title; details.append(heading); return document.createElement('dl'); };
  const row = (list, name, value) => { const term = document.createElement('dt'); term.textContent = name; const description = document.createElement('dd'); description.textContent = value; list.append(term, description); };
  const source = section('Input and preprocessing');
  row(source, 'Container / codec', `${report.source.format} / ${report.source.subtype}`);
  row(source, 'Original audio', `${report.source.original_sample_rate_hz.toLocaleString()} Hz · ${report.source.original_channels} channel(s) · ${formatNumber(time.duration_seconds, 2)} sec`);
  row(source, 'Analysis pipeline', `Mono · ${config.sample_rate_hz.toLocaleString()} Hz · ${config.frame_duration_ms} ms ${config.window} frames · ${config.hop_duration_ms} ms hop · ${config.fft_size}-point FFT`);
  row(source, 'Stereo fold-down RMS ratio', formatNumber(report.source.downmix_rms_ratio, 4));
  row(source, 'File SHA-256', data.file_sha256);
  details.append(source);
  const modelMetrics = decision.model_metrics;
  if (modelMetrics) {
    const evaluation = section('Saved model and held-out evaluation');
    row(evaluation, 'Classifier', `${decision.model_algorithm || 'User-trained classifier'}; positive class: synthetic`);
    row(evaluation, 'Feature schema', `${decision.feature_set || 'basic'} · ${decision.feature_count || 11} features`);
    row(evaluation, 'Dataset class counts', `${modelMetrics.class_counts.human} human / ${modelMetrics.class_counts.synthetic} synthetic`);
    row(evaluation, 'Train / holdout', `${modelMetrics.train_files} / ${modelMetrics.holdout_files} files; ${modelMetrics.train_groups} / ${modelMetrics.holdout_groups} groups`);
    row(evaluation, 'Saved model fit', `${modelMetrics.final_fit_examples || 'Not recorded'} labeled feature vectors`);
    row(evaluation, 'Accuracy / balanced accuracy', `${(modelMetrics.holdout_accuracy * 100).toFixed(1)}% / ${(modelMetrics.holdout_balanced_accuracy * 100).toFixed(1)}%`);
    row(evaluation, 'ROC AUC', formatNumber(modelMetrics.holdout_auc, 3));
    if (Number.isFinite(modelMetrics.holdout_eer)) row(evaluation, 'Equal error rate (held-out)', `${(modelMetrics.holdout_eer * 100).toFixed(1)}% · evaluation metric only`);
    if (Number.isFinite(modelMetrics.holdout_false_accept_rate_at_0_5) && Number.isFinite(modelMetrics.holdout_false_reject_rate_at_0_5)) row(evaluation, 'False accept / reject at 0.50', `${(modelMetrics.holdout_false_accept_rate_at_0_5 * 100).toFixed(1)}% / ${(modelMetrics.holdout_false_reject_rate_at_0_5 * 100).toFixed(1)}%`);
    row(evaluation, 'Confusion matrix', `Rows true [human, synthetic]; columns predicted [human, synthetic]: ${JSON.stringify(modelMetrics.confusion_matrix_rows_true_human_synthetic_cols_pred_human_synthetic)}`);
    row(evaluation, 'Score meaning', `Synthetic-evidence index from 1 to 11, mapped from the raw uncalibrated model output. Current decision cutoff: ${Math.round((decision.threshold ?? 0.5) * 100)}%. Not a calibrated probability.`);
    row(evaluation, 'Quality gate', decision.classifier_block_reasons?.length ? decision.classifier_block_reasons.join('; ') : 'Only unusable or non-speech inputs block inference; lesser quality warnings stay advisory.');
    details.append(evaluation);
  } else {
    const status = section('Classifier status'); row(status, 'Model result', 'No trained local classifier was used. DSP measurements above are descriptive only.'); details.append(status);
  }
  const caveats = section('Limitations');
  [...report.quality.warnings, ...report.limitations].forEach((text, index) => row(caveats, `Note ${index + 1}`, text));
  details.append(caveats);
  $('#delete-current').onclick = deleteCurrent;
  $('#export-report-html').onclick = exportCurrentReportHtml;
}

async function updateHistoryCount() {
  try { const response = await fetch('/api/history?limit=1'); const data = await response.json(); $('#history-count').textContent = data.total; } catch { /* status is shown separately */ }
}
async function loadHistory() {
  const host = $('#history-list'); host.replaceChildren();
  try {
    const response = await fetch('/api/history?limit=50'); if (!response.ok) throw new Error('Could not load history.');
    const data = await response.json(); $('#history-count').textContent = data.total;
    $('#history-empty').hidden = data.items.length > 0;
    data.items.forEach((record) => {
      const row = document.createElement('div'); row.className = 'history-item';
      const name = document.createElement('div'); name.className = 'history-name'; name.textContent = record.filename;
      const date = document.createElement('div'); date.className = 'history-meta'; date.textContent = new Date(record.created_at).toLocaleString();
      const duration = document.createElement('div'); duration.className = 'history-meta'; duration.textContent = `${formatNumber(record.duration_seconds, 2)} sec`;
      const decision = document.createElement('div'); decision.className = 'history-decision'; decision.textContent = record.decision;
      const view = document.createElement('button'); view.type = 'button'; view.className = 'history-action'; view.textContent = 'Open'; view.setAttribute('aria-label', `Open analysis for ${record.filename}`); view.onclick = () => openHistoryRecord(record.analysis_id);
      const remove = document.createElement('button'); remove.type = 'button'; remove.className = 'history-action'; remove.textContent = 'Delete'; remove.setAttribute('aria-label', `Delete analysis for ${record.filename}`); remove.onclick = () => deleteRecord(record.analysis_id);
      const confirmHuman = document.createElement('button'); confirmHuman.type='button'; confirmHuman.className='history-action history-confirm'; confirmHuman.textContent='Confirm human'; confirmHuman.onclick=()=>confirmHistoryLabel(record.analysis_id,'human',confirmHuman);
      const confirmSynthetic = document.createElement('button'); confirmSynthetic.type='button'; confirmSynthetic.className='history-action history-confirm'; confirmSynthetic.textContent='Confirm AI'; confirmSynthetic.onclick=()=>confirmHistoryLabel(record.analysis_id,'synthetic',confirmSynthetic);
      const unverified = document.createElement('button'); unverified.type='button'; unverified.className='history-action history-confirm'; unverified.textContent='Unverified'; unverified.onclick=()=>confirmHistoryLabel(record.analysis_id,'uncertain',unverified);
      const review=document.createElement('div');review.className='history-review-actions';review.append(confirmHuman,confirmSynthetic,unverified);
      row.append(name, date, duration, decision, view, remove, review); host.append(row);
    });
  } catch (error) { $('#history-empty').hidden = false; $('#history-empty').textContent = error.message; }
}
async function confirmHistoryLabel(id,label,button){
  button.disabled=true;
  try{const r=await fetch(`/api/history/${encodeURIComponent(id)}/confirm`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({label})});const d=await r.json();if(!r.ok)throw new Error(d?.error?.message||'Could not save label.');button.textContent=label==='uncertain'?'Excluded':'Confirmed';$('#training-progress').textContent=label==='uncertain'?'Run marked unverified and excluded from training.':'Confirmed label added to the training pool.';await updateHistoryCount();await loadTrainingStatus();}
  catch(e){button.textContent=e.message;}
  finally{button.disabled=false;}
}
async function openHistoryRecord(id) {
  try {
    const response = await fetch(`/api/history/${encodeURIComponent(id)}`); const data = await response.json();
    if (!response.ok) throw new Error(data?.error?.message || 'Could not open this report.');
    renderAnalysis(data); showView('analyze'); $('#result-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (error) { $('#history-empty').hidden = false; $('#history-empty').textContent = error.message; }
}
async function deleteRecord(id) {
  if (!window.confirm('Delete this saved report? The source audio is not stored by AegisVoice.')) return;
  const response = await fetch(`/api/history/${encodeURIComponent(id)}`, { method: 'DELETE' });
  if (!response.ok) { alert('The report could not be deleted.'); return; }
  if (state.currentId === id) { state.currentId = null; $('#result-section').hidden = true; }
  await loadHistory();
}
async function deleteCurrent() { if (state.currentId) await deleteRecord(state.currentId); }

function parseCsv(text) {
  const lines = text.replace(/^\uFEFF/, '').split(/\r?\n/).filter((line) => line.trim());
  const cells = (line) => { const out=[]; let value='',quoted=false; for(let i=0;i<line.length;i++){const c=line[i];if(c==='"'&&quoted&&line[i+1]==='"'){value+='"';i++;}else if(c==='"')quoted=!quoted;else if(c===','&&!quoted){out.push(value.trim());value='';}else value+=c;}out.push(value.trim());return out; };
  if (!lines.length) return new Map();
  const heads=cells(lines[0]).map((v)=>v.toLowerCase()); const fi=heads.findIndex((v)=>['filename','file','path','filepath'].includes(v)); const li=heads.findIndex((v)=>['label','class','category'].includes(v)); const gi=heads.findIndex((v)=>['group','group_id','speaker','recording_id'].includes(v));
  if(fi<0||li<0) throw new Error('CSV needs filename/path and label columns.');
  const map=new Map(); for(const line of lines.slice(1)){const row=cells(line); if(row.length<=Math.max(fi,li))continue;const label=normalizeLabel(row[li]);if(!label)continue;const path=row[fi].replaceAll('\\','/').toLowerCase();map.set(path,{label,group:gi>=0&&row[gi]?row[gi]:null});map.set(path.split('/').at(-1),{label,group:gi>=0&&row[gi]?row[gi]:null});} return map;
}
function normalizeLabel(value){const v=String(value||'').trim().toLowerCase();if(['real','human','bonafide','genuine','authentic','0'].includes(v))return'human';if(['fake','synthetic','spoof','deepfake','ai','generated','1'].includes(v))return'synthetic';return null;}
async function loadTrainingStatus(){
  try {
    const r = await fetch('/api/training/status'); const d = await r.json();
    if (!r.ok) throw new Error(d?.error?.message || 'Training tools require full mode. Start the full local service to use the model and compressed audio formats.');
    const source = $('#training-source').value;
    const all = { examples: d.examples || {}, groups: d.groups || {}, advanced_examples: d.advanced_feature_examples_by_label || {}, advanced_groups: d.advanced_feature_groups || {} };
    const history = d.confirmed_history || {};
    const pool = source === 'confirmed_history' ? { examples: history.examples || {}, groups: history.groups || {}, advanced_examples: history.advanced_examples || {}, advanced_groups: history.advanced_groups || {} } : all;
    const e = all.examples, g = all.groups, human = e.human || 0, synthetic = e.synthetic || 0;
    const selectedFeatureSet = $('#feature-set').value;
    const selectedExamples = selectedFeatureSet === 'advanced' ? pool.advanced_examples : pool.examples;
    const selectedGroups = selectedFeatureSet === 'advanced' ? pool.advanced_groups : pool.groups;
    const trainHuman = selectedExamples.human || 0, trainSynthetic = selectedExamples.synthetic || 0;
    const humanGroups = selectedGroups.human || 0, syntheticGroups = selectedGroups.synthetic || 0;
    $('#training-counts').textContent = `${source === 'confirmed_history' ? 'Confirmed History' : 'All labels'} · ${selectedFeatureSet}: ${trainHuman} human / ${trainSynthetic} AI · ${humanGroups} / ${syntheticGroups} recording groups · ${d.confirmed_history_examples || 0} confirmed History labels total`;
    $('#model-dataset-count').textContent = `${human} human / ${synthetic} AI`;
    const ready = trainHuman + trainSynthetic >= 16 && humanGroups >= 4 && syntheticGroups >= 4;
    $('#train-model').disabled = !ready;
    $('#compare-models').disabled = !ready;
    $('#training-warning').textContent = ready
      ? `Training minimum met for ${source === 'confirmed_history' ? 'confirmed History labels' : 'the combined label pool'} with ${selectedFeatureSet} features: ${trainHuman + trainSynthetic} recordings and at least four groups per class. Use verified labels; this small corpus cannot establish real-world accuracy.`
      : `${source === 'confirmed_history' ? 'Confirmed History labels' : 'The selected label pool'} needs at least 16 ${selectedFeatureSet} recordings and 4 recording groups in each class. Current: ${trainHuman + trainSynthetic} recordings; ${humanGroups} human groups; ${syntheticGroups} AI groups.`;
    if (d.model_ready) $('#model-analysis-mode').textContent = `DSP + ${d.active_algorithm || 'local ML'}`;
    if (d.model) {
      const m = d.model, matrix = m.confusion_matrix_rows_true_human_synthetic_cols_pred_human_synthetic;
      const cm = Array.isArray(matrix) ? matrix.map((line) => line.join(' / ')).join(' · ') : 'not available';
      const algorithm = d.active_algorithm || 'Unknown model', featureSet = d.active_feature_set || 'basic';
      $('#training-results').innerHTML = `<p><b>Active model:</b> ${escapeHTML(algorithm)} · ${escapeHTML(featureSet)} features · ${escapeHTML(d.active_training_source === 'confirmed_history' ? 'confirmed History only' : 'all imported and confirmed labels')}</p><div class="train-metrics"><span>Holdout accuracy <b>${(m.holdout_accuracy * 100).toFixed(1)}%</b></span><span>Balanced accuracy <b>${(m.holdout_balanced_accuracy * 100).toFixed(1)}%</b></span><span>Holdout ROC AUC <b>${Number.isFinite(m.holdout_auc) ? m.holdout_auc.toFixed(3) : '—'}</b></span><span>Equal error rate <b>${Number.isFinite(m.holdout_eer) ? (m.holdout_eer * 100).toFixed(1) + '%' : 'Not recorded'}</b></span><span>False accept / reject at 0.50 <b>${Number.isFinite(m.holdout_false_accept_rate_at_0_5) ? (m.holdout_false_accept_rate_at_0_5 * 100).toFixed(1) + '% / ' + (m.holdout_false_reject_rate_at_0_5 * 100).toFixed(1) + '%' : 'Not recorded'}</b></span><span>Class counts <b>${m.class_counts.human} human / ${m.class_counts.synthetic} AI</b></span><span>Train / holdout files <b>${m.train_files} / ${m.holdout_files}</b></span><span>Train / holdout groups <b>${m.train_groups} / ${m.holdout_groups}</b></span><span>Final model fit examples <b>${m.final_fit_examples || 'Not recorded'}</b></span><span>Confusion matrix <b>${cm}</b></span></div><p class="model-warning">Experimental, uncalibrated score. EER is calculated on the held-out portion for evaluation only; it is not used to set the model threshold. Only ${m.holdout_files} held-out recordings were evaluated, so these results do not establish performance on new speakers or generators.</p>`;
    } else {
      $('#training-results').textContent = 'No trained classifier is saved yet. Import labeled recordings, then train the model.';
    }
  } catch(e) { $('#training-warning').textContent = e.message; }
}
$('#import-training').addEventListener('click',async()=>{
  const button=$('#import-training'); if(button.disabled)return; button.disabled=true;
  try {
  const files=[...Array.from($('#training-folder').files||[],f=>({file:f,relative:f.webkitRelativePath||f.name,manualLabel:null})),...Array.from($('#human-files').files||[],f=>({file:f,relative:f.name,manualLabel:'human'})),...Array.from($('#synthetic-files').files||[],f=>({file:f,relative:f.name,manualLabel:'synthetic'}))].filter(x=>/\.(wav|flac|ogg|oga|opus|aiff|aif|mp3|m4a|mp4|aac|wma|webm|mka)$/i.test(x.file.name));
  if(!files.length){$('#training-progress').textContent='Choose a folder or audio files first.';return;}
  let mapping=new Map();try{const csv=$('#training-csv').files?.[0];if(csv)mapping=parseCsv(await csv.text());}catch(e){$('#training-progress').textContent=e.message;return;}
  let added=0, duplicates=0, skipped=0;const dataset=$('#dataset-name').value.trim()||'User supplied dataset';
  for(let i=0;i<files.length;i++){const file=files[i].file,relative=files[i].relative.replaceAll('\\','/');let record=mapping.get(relative.toLowerCase())||mapping.get(file.name.toLowerCase());let label=record?.label||files[i].manualLabel;
    if(!label){const parts=relative.toLowerCase().split('/');if(parts.some(x=>['real','human','bonafide','genuine'].includes(x)))label='human';else if(parts.some(x=>['fake','synthetic','spoof','deepfake','ai'].includes(x)))label='synthetic';}
    if(!label){skipped++;continue;}const group=record?.group||file.name.replace(/\.[^.]+$/,'');const form=new FormData();form.append('file',file,file.name);form.append('label',label);form.append('group_id',group);form.append('dataset_name',dataset);
    $('#training-progress').textContent=`Importing ${i+1}/${files.length}: ${file.name}`;try{const r=await fetch('/api/training/samples',{method:'POST',body:form});const d=await r.json();if(!r.ok)throw new Error(d?.error?.message||`Could not import ${file.name}`);if(d.added)added++;else if(d.duplicate)duplicates++;}catch(e){$('#training-progress').textContent=`Stopped at ${file.name}: ${e.message}`;await loadTrainingStatus();return;}
  }
  $('#training-progress').textContent=`Imported ${added} new clips; ${duplicates} duplicates already existed; ${skipped} unlabeled files were skipped.`;await loadTrainingStatus();
  } finally { button.disabled=false; }
});
$('#feature-set').addEventListener('change', loadTrainingStatus);
$('#training-source').addEventListener('change', loadTrainingStatus);
$('#train-model').addEventListener('click',async()=>{const button=$('#train-model');button.disabled=true;$('#training-progress').textContent='Training the selected classifier on grouped DSP features…';try{const source=$('#training-source').value;const r=await fetch('/api/model/train',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({algorithm:$('#model-algorithm').value,feature_set:$('#feature-set').value,source})});const d=await r.json();if(!r.ok)throw new Error(d?.error?.message||'Training failed.');$('#training-progress').textContent=`${d.algorithm} trained locally using ${d.feature_set} features from ${source==='confirmed_history'?'confirmed History results':'all imported and confirmed labels'}.`;await loadTrainingStatus();await checkHealth();}catch(e){$('#training-progress').textContent=e.message;}finally{button.disabled=false;}});

$('#compare-models').addEventListener('click',async()=>{
  const button=$('#compare-models');button.disabled=true;const host=$('#model-comparison');host.textContent='Running all five model families on the same group-held-out split…';
  try{
    const r=await fetch('/api/model/compare',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({algorithm:$('#model-algorithm').value,feature_set:$('#feature-set').value,source:$('#training-source').value})});const d=await r.json();if(!r.ok)throw new Error(d?.error?.message||'Model comparison failed.');
    const fragment=document.createDocumentFragment();const summary=document.createElement('p');summary.textContent=`${d.labeled_examples} labeled recordings · ${d.feature_count} ${d.feature_set} features. ${d.warning}`;fragment.append(summary);
    const table=document.createElement('table');table.className='comparison-table';const head=document.createElement('tr');for(const label of ['Model','Accuracy','Balanced accuracy','ROC AUC','EER','Confusion matrix']){const th=document.createElement('th');th.textContent=label;head.append(th);}table.append(head);
    for(const item of d.models||[]){const tr=document.createElement('tr');const values=item.error?[item.name,'Unavailable','Unavailable','Unavailable','Unavailable',item.error]:[item.name,`${(item.metrics.holdout_accuracy*100).toFixed(1)}%`,`${(item.metrics.holdout_balanced_accuracy*100).toFixed(1)}%`,Number.isFinite(item.metrics.holdout_auc)?item.metrics.holdout_auc.toFixed(3):'—',Number.isFinite(item.metrics.holdout_eer)?`${(item.metrics.holdout_eer*100).toFixed(1)}%`:'—',item.metrics.confusion_matrix_rows_true_human_synthetic_cols_pred_human_synthetic.map((row)=>row.join(' / ')).join(' · ')];for(const value of values){const td=document.createElement('td');td.textContent=value;tr.append(td);}table.append(tr);}
    fragment.append(table);host.replaceChildren(fragment);
  }catch(e){host.textContent=e.message||'Comparison failed.';}finally{button.disabled=false;}
});

$('#reset-model').addEventListener('click',async()=>{
  if(!window.confirm('Reset the active trained model? Your imported and confirmed labels will remain available for retraining.'))return;
  const button=$('#reset-model');button.disabled=true;
  try{const r=await fetch('/api/model',{method:'DELETE'});const d=await r.json();if(!r.ok)throw new Error(d?.error?.message||'Could not reset the model.');$('#training-progress').textContent=d.reset?'Trained model removed. Your labels are preserved.':'No saved model was present. Your labels are preserved.';await loadTrainingStatus();await checkHealth();}
  catch(e){$('#training-progress').textContent=e.message||'Could not reset the model.';}finally{button.disabled=false;}
});

$('#model-threshold').addEventListener('input', () => { $('#threshold-value').textContent = `${$('#model-threshold').value}%`; });
$('#save-threshold').addEventListener('click', async () => {
  const button = $('#save-threshold'); button.disabled = true;
  try {
    const response = await fetch('/api/model/threshold', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ threshold: Number($('#model-threshold').value) / 100 }) });
    const data = await response.json();
    if (!response.ok) throw new Error(data?.error?.message || 'Could not save the threshold.');
    $('#training-progress').textContent = `Decision threshold saved at ${Math.round(data.threshold * 100)}%. New analyses will use it.`;
    $('#model-decision-mode').textContent = `${Math.round(data.threshold * 100)}% cutoff`;
  } catch (error) {
    $('#training-progress').textContent = error.message || 'Could not save the threshold.';
  } finally { button.disabled = !state.classifierAvailable; }
});

checkHealth(); updateHistoryCount();
const requestedView = new URLSearchParams(location.search).get('view');
if (requestedView === 'training') showView('training');
else if (location.hash === '#method') showView('method');
