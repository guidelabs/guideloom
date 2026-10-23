<!-- Adapted from the supplied DataloaderModesVisualization component. -->
<script>
  let { streamWithMetadataDescription, streamDescription, documentsDescription} = $props();

  const TEXTS = [
    "The Sun is a very large hot ball of glowing gas at the center of our solar system.",
    "A horse is a large, four-legged animal. It has a long face and strong legs with hooves.",
  ];
  const DOC_IDS = ["facts/sun", "facts/horse"];
  const CONCEPTS = [
    ["Astronomical object", "Energy source", "High temperature"],
    ["Mammal", "Large body size", "Quadruped"],
  ];

  // Tokenize: each word with its prefix space is a token
  function tokenize(text) {
    const words = text.split(/\s+/).filter(Boolean);
    return words.map((w, i) => (i === 0 ? w : " " + w));
  }

  const documents = $derived(
    TEXTS.map((text, i) => ({
      id: DOC_IDS[i],
      tokens: tokenize(text),
      concepts: CONCEPTS[i],
    }))
  );

  const streamTokens = $derived(
    documents.flatMap((d) => [...d.tokens, "<eos>"])
  );
  const totalTokens = $derived(streamTokens.length);

  // Mode: 'stream' | 'documents' | 'stream-with-metadata'
  let mode = $state("stream-with-metadata");

  // Stream / stream-with-metadata: window and offset (hover updates selection, persists)
  let windowSize = $state(8);
  let offsetIndex = $state(0);

  // Documents: selected document index (hover updates selection, persists)
  let selectedDocIndex = $state(0);

  function clampInteger(value, min, max, fallback = min) {
    const number = Number(value);
    return Number.isFinite(number)
      ? Math.min(max, Math.max(min, Math.trunc(number)))
      : fallback;
  }

  const effectiveWindowSize = $derived(clampInteger(windowSize, 2, totalTokens, 8));
  const effectiveDocIndex = $derived(clampInteger(selectedDocIndex, 0, documents.length - 1));

  // Only allow observations that have a full window of tokens
  const maxOffsetIndex = $derived(
    Math.max(0, Math.floor(totalTokens / effectiveWindowSize) - 1)
  );
  const effectiveOffsetIndex = $derived(clampInteger(offsetIndex, 0, maxOffsetIndex));
  const windowStart = $derived(effectiveOffsetIndex * effectiveWindowSize);
  const windowEnd = $derived(Math.min(windowStart + effectiveWindowSize, totalTokens));

  $effect(() => {
    if (offsetIndex > maxOffsetIndex) offsetIndex = maxOffsetIndex;
  });

  // For stream-with-metadata: which metadata objects are in the window
  // Metadata spans cover doc tokens + its <eos> (so <eos> selects metadata for that doc)
  const metadataInWindow = $derived.by(() => {
    let idx = 0;
    const result = [];
    for (let d = 0; d < documents.length; d++) {
      const len = documents[d].tokens.length;
      const start = idx;
      const end = idx + len + 1; // include <eos>
      const overlaps = windowStart < end && windowEnd > start;
      result.push({ docIndex: d, start, end, overlaps });
      idx = end;
    }
    return result;
  });

  function isTokenSelected(tokenIndex) {
    return tokenIndex >= windowStart && tokenIndex < windowEnd;
  }

  function getTokenDocIndex(tokenIndex) {
    let idx = 0;
    for (let d = 0; d < documents.length; d++) {
      const docLen = documents[d].tokens.length;
      if (tokenIndex <= idx + docLen) return d; // doc tokens + <eos>
      idx += docLen + 1;
    }
    return documents.length - 1;
  }

  function getOffsetIndexForToken(tokenIndex) {
    return Math.floor(tokenIndex / effectiveWindowSize);
  }

  // Hover updates selection and persists (no revert on blur - avoids flickering)
  function handleTokenHover(tokenIndex) {
    if (mode === "stream" || mode === "stream-with-metadata") {
      const idx = getOffsetIndexForToken(tokenIndex);
      offsetIndex = Math.min(idx, maxOffsetIndex);
    }
  }

  function handleOffsetSlotHover(slotIndex) {
    if (mode === "stream" || mode === "stream-with-metadata") {
      offsetIndex = Math.min(slotIndex, maxOffsetIndex);
    }
  }

  function handleDocBlockHover(docIdx) {
    if (mode === "documents") {
      selectedDocIndex = docIdx;
    }
  }
</script>

<div class="container">
  <div class="controls">
    <div class="mode-select">
      <span class="mode-label">Mode:</span>
      <div class="mode-buttons">
        <button
          type="button"
          class="mode-btn"
          class:active={mode === "stream-with-metadata"}
          aria-pressed={mode === "stream-with-metadata"}
          onclick={() => (mode = "stream-with-metadata")}
        >
          stream-with-metadata
        </button>
        <button
          type="button"
          class="mode-btn"
          class:active={mode === "documents"}
          aria-pressed={mode === "documents"}
          onclick={() => (mode = "documents")}
        >
          documents
        </button>
        <button
          type="button"
          class="mode-btn"
          class:active={mode === "stream"}
          aria-pressed={mode === "stream"}
          onclick={() => (mode = "stream")}
        >
          stream
        </button>
      </div>
    </div>

    {#if mode === "stream" || mode === "stream-with-metadata"}
      <div class="stream-controls">
        <div>
          <label for="observation-index">Observation index:</label>
          <input
            id="observation-index"
            type="number"
            inputmode="numeric"
            step="1"
            bind:value={offsetIndex}
            onblur={() => (offsetIndex = effectiveOffsetIndex)}
            min="0"
            max={maxOffsetIndex}
          />
        </div>
        <div>
          <label for="window-size">Window size:</label>
          <input
            id="window-size"
            type="number"
            inputmode="numeric"
            step="1"
            bind:value={windowSize}
            onblur={() => (windowSize = effectiveWindowSize)}
            min="2"
            max={totalTokens}
          />
        </div>
      </div>
    {:else}
      <div class="doc-controls">
        <div>
          <label for="observation-index-doc">Observation index:</label>
          <input
            id="observation-index-doc"
            type="number"
            inputmode="numeric"
            step="1"
            bind:value={selectedDocIndex}
            onblur={() => (selectedDocIndex = effectiveDocIndex)}
            min="0"
            max={documents.length - 1}
          />
        </div>
      </div>
    {/if}
  </div>

  <hr />

  {#if mode === "stream-with-metadata"}
    <p class="mode-desc">
      <strong>Stream with metadata:</strong>
      {streamWithMetadataDescription}
    </p>
  {:else if mode === "stream"}
    <p class="mode-desc">
      <strong>Stream:</strong>
      {streamDescription}
    </p>
  {:else}
    <p class="mode-desc">
      <strong>Documents:</strong>
      {documentsDescription}
    </p>
  {/if}

  {#if mode === "stream-with-metadata"}
    <h4 class="vis-title">Metadata (document ID + concepts)</h4>
    <div class="metadata-blocks">
      {#each metadataInWindow as meta, i}
        <div
          class="meta-block"
          class:selected={meta.overlaps}
          class:grayed={!meta.overlaps}
        >
          <span class="meta-id">{documents[meta.docIndex].id}</span>
          <span class="meta-concepts">
            {documents[meta.docIndex].concepts.join(", ")}
          </span>
        </div>
      {/each}
    </div>
  {/if}

  <h4 class="vis-title">
    {#if mode === "stream-with-metadata"}
      Token stream
    {:else if mode === "stream"}
      Token stream
    {:else}
      Documents (document ID + tokens + concepts)
    {/if}
  </h4>

  {#if mode === "documents"}
    <div class="documents-view">
      {#each documents as doc, docIdx}
        <div
          class="doc-block"
          class:selected={docIdx === effectiveDocIndex}
          class:grayed={docIdx !== effectiveDocIndex}
          role="button"
          aria-pressed={docIdx === effectiveDocIndex}
          tabindex="0"
          onmouseenter={() => handleDocBlockHover(docIdx)}
          onfocus={() => handleDocBlockHover(docIdx)}
          onclick={() => handleDocBlockHover(docIdx)}
          onkeydown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              handleDocBlockHover(docIdx);
            }
          }}
        >
          <div class="doc-header">
            <span class="doc-id">{doc.id}</span>
            <span class="doc-concepts">{doc.concepts.join(", ")}</span>
          </div>
          <div class="token-row">
            {#each doc.tokens as token}
              <span
                class="token"
                class:selected={docIdx === effectiveDocIndex}
                class:grayed={docIdx !== effectiveDocIndex}
              >
                {token === " " ? "␣" : token}
              </span>
            {/each}
          </div>
        </div>
      {/each}
    </div>
  {:else}
    <div class="stream-view">
      <div class="offset-row">
        <span class="offset-label">Observation index:</span>
        <div class="offset-slots">
        {#each Array.from({ length: maxOffsetIndex + 1 }) as _, slotIdx}
          <button
            type="button"
            class="offset-slot"
            class:active={slotIdx === effectiveOffsetIndex}
            aria-pressed={slotIdx === effectiveOffsetIndex}
            aria-label={`Observation ${slotIdx}`}
            onmouseenter={() => handleOffsetSlotHover(slotIdx)}
            onfocus={() => handleOffsetSlotHover(slotIdx)}
            onclick={() => handleOffsetSlotHover(slotIdx)}
          >
            {slotIdx}
          </button>
        {/each}
        </div>
      </div>
      <div class="token-stream">
        {#each streamTokens as token, ti}
          {@const docIdx = getTokenDocIndex(ti)}
          {@const selected = isTokenSelected(ti)}
          {@const prevDocIdx = ti > 0 ? getTokenDocIndex(ti - 1) : -1}
          {@const isEos = token === "<eos>"}
          {@const omitted = ti >= (maxOffsetIndex + 1) * effectiveWindowSize}
          <button
            type="button"
            class="token"
            class:selected
            disabled={omitted}
            aria-pressed={selected}
            title={omitted ? "Omitted: does not fill a complete window" : `Token ${ti}`}
            class:grayed={!selected}
            class:doc-boundary={!isEos && docIdx > 0 && prevDocIdx !== docIdx}
            onmouseenter={() => handleTokenHover(ti)}
            onfocus={() => handleTokenHover(ti)}
            onclick={() => handleTokenHover(ti)}
          >
            {token === " " ? "␣" : token}
          </button>
        {/each}
      </div>
    </div>
  {/if}

  <div class="legend">
    <span class="legend-item selected">Selected</span>
    <span class="legend-item grayed">Unselected (grayed)</span>
    {#if mode === "stream" || mode === "stream-with-metadata"}
      <span class="legend-note">Hover, click, or focus tokens/windows to select a sample. Trailing tokens without a full window are omitted.</span>
    {:else}
      <span class="legend-note">Hover, click, or focus a document to select it.</span>
    {/if}
  </div>
</div>

<style>
  .vis-title {
    margin: 1em 0 0.2em 0 !important;
    padding: 0;
    font-weight: bold;
  }

  .container {
    margin: 1.5rem 0;
    min-width: 0;
    color: var(--md-default-fg-color);
    padding: 1.5rem;
    background: var(--md-default-bg-color, #fff);
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
  }

  .controls {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 1.5rem;
    margin-bottom: 1rem;
  }

  .mode-select .mode-label,
  .stream-controls label,
  .doc-controls label {
    display: block;
    font-size: 0.9em;
    color: var(--md-default-fg-color, #4b5563);
    margin-bottom: 0.25rem;
  }

  .mode-buttons {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
  }

  .mode-btn {
    padding: 0.4rem 0.75rem;
    border: 1px solid var(--md-default-fg-color--lightest, #e5e7eb);
    border-radius: 6px;
    background: var(--md-code-bg-color, #fafafa);
    font-size: 0.85em;
    cursor: pointer;
  }

  .mode-btn:hover {
    background: var(--md-code-bg-color, #f3f4f6);
  }

  .mode-btn.active {
    background: var(--md-accent-fg-color, #6366f1);
    color: var(--md-accent-bg-color, white);
    border-color: var(--md-accent-fg-color, #6366f1);
  }

  .stream-controls,
  .doc-controls {
    display: flex;
    align-items: flex-end;
    gap: 1rem;
  }

  .stream-controls input,
  .doc-controls input {
    width: 6ch;
    padding: 0.3rem 0.5rem;
    border: 1px solid var(--md-default-fg-color--lightest, #e5e7eb);
    border-radius: 4px;
  }

  .mode-desc {
    font-size: 0.9em;
    color: var(--md-default-fg-color--light, #6b7280);
    margin: 0.5rem 0 1rem 0;
  }

  .metadata-blocks {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 1rem;
  }

  .meta-block {
    padding: 0.4rem 0.75rem;
    border-radius: 6px;
    border: 1px solid var(--md-default-fg-color--lightest, #e5e7eb);
    font-size: 0.85em;
  }

  .meta-block.selected {
    background: color-mix(in srgb, var(--md-accent-fg-color) 15%, var(--md-default-bg-color));
    border-color: var(--md-accent-fg-color, #6366f1);
  }

  .meta-block.grayed {
    background: var(--md-code-bg-color, #f3f4f6);
    color: var(--md-default-fg-color--lighter, #9ca3af);
  }

  .meta-id {
    font-weight: 600;
    display: block;
    margin-bottom: 0.2rem;
  }

  .meta-concepts {
    font-size: 0.9em;
    color: var(--md-default-fg-color--light, #6b7280);
  }

  .meta-block.grayed .meta-concepts {
    color: var(--md-default-fg-color--lighter, #9ca3af);
  }

  .documents-view {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .doc-block {
    border: 1px solid var(--md-default-fg-color--lightest, #e5e7eb);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    background: var(--md-code-bg-color, #fafafa);
    cursor: pointer;
  }

  .doc-block.selected {
    background: color-mix(in srgb, var(--md-accent-fg-color) 15%, var(--md-default-bg-color));
    border-color: var(--md-accent-fg-color, #6366f1);
  }

  .doc-block.grayed {
    opacity: 0.5;
    background: var(--md-code-bg-color, #f3f4f6);
  }

  .doc-header {
    margin-bottom: 0.5rem;
    font-size: 0.9em;
  }

  .doc-id {
    font-weight: 600;
    color: var(--md-default-fg-color, #4b5563);
  }

  .doc-concepts {
    color: var(--md-default-fg-color--light, #6b7280);
    margin-left: 0.5rem;
  }

  .token-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.15rem;
  }

  .stream-view {
    border: 1px solid var(--md-default-fg-color--lightest, #e5e7eb);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    background: var(--md-code-bg-color, #fafafa);
    margin: 1rem 0;
    position: relative;
  }

  .offset-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
    flex-wrap: wrap;
  }

  .offset-label {
    font-size: 0.85em;
    color: var(--md-default-fg-color--light, #6b7280);
  }

  .offset-slots {
    display: flex;
    gap: 0.2rem;
    flex-wrap: wrap;
  }

  .offset-slot {
    width: 1.8em;
    height: 1.8em;
    line-height: 1.8em;
    text-align: center;
    font-size: 0.8em;
    border-radius: 4px;
    cursor: pointer;
    background: var(--md-default-fg-color--lightest, #e5e7eb);
    color: var(--md-default-fg-color--light, #6b7280);
    border: none;
    padding: 0;
    font-family: inherit;
  }

  .offset-slot:hover,
  .offset-slot.active {
    background: var(--md-accent-fg-color, #6366f1);
    color: var(--md-accent-bg-color, white);
  }

  .token-stream {
    display: flex;
    flex-wrap: wrap;
    gap: 0.15rem;
    align-items: baseline;
  }

  .token {
    padding: 0.15rem 0.25rem;
    border-radius: 4px;
    font-size: 0.9em;
    font-family: var(--md-code-font-family, monospace);
    border: none;
    background: transparent;
    cursor: pointer;
  }

  .token.selected {
    background: var(--md-accent-fg-color, #6366f1);
    color: var(--md-accent-bg-color, white);
  }

  .token.grayed {
    background: var(--md-default-fg-color--lightest, #e5e7eb);
    color: var(--md-default-fg-color--lighter, #9ca3af);
  }

  .token.doc-boundary {
    border-left: 2px solid var(--md-accent-fg-color, #6366f1);
    margin-left: 0.1rem;
  }

  .token-stream .token {
    cursor: pointer;
  }

  .token-stream .token:hover {
    filter: brightness(1.1);
  }

  .legend {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 1rem;
    margin-top: 1rem;
    font-size: 0.85em;
    color: var(--md-default-fg-color--light, #6b7280);
  }

  .legend-item.selected {
    color: var(--md-accent-fg-color, #6366f1);
    font-weight: 500;
  }

  .legend-item.grayed {
    color: var(--md-default-fg-color--lighter, #9ca3af);
  }

  .legend-note {
    font-style: italic;
  }
  button:focus-visible,
  .doc-block:focus-visible,
  input:focus-visible {
    outline: 2px solid var(--md-accent-fg-color);
    outline-offset: 2px;
  }

  .token:disabled {
    cursor: default;
    opacity: 0.55;
  }

  @media (max-width: 600px) {
    .container { padding: 1rem; }
    .stream-view { padding: 0.75rem; }
    .stream-controls { flex-wrap: wrap; }
  }
</style>
