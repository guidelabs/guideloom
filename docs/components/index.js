import { mount, unmount } from "svelte";
import StorageFormats from "./StorageFormats.svelte";

let target;
let component;

function initialize() {
  const next = document.querySelector("[data-storage-formats]");
  if (next === target) return;
  if (component) unmount(component);
  target = next;
  component = undefined;
  if (!target) return;
  target.replaceChildren();
  component = mount(StorageFormats, {
    target,
    props: {
      streamWithMetadataDescription: "Read fixed-length token windows together with the document IDs and concept lists attached to overlapping spans. A window can cross document boundaries.",
      streamDescription: "Read fixed-length windows from the concatenated tokens. This example stores tokens alone; plain streams can also include fixed-size fields such as a document ID per token.",
      documentsDescription: "Read one complete document, including its ID, tokens, and concepts. Documents keep their boundaries; your collator handles padding when making batches.",
    },
  });
}

// Also remount when Zensical changes pages through instant navigation.
if (typeof document$ !== "undefined") {
  document$.subscribe(initialize);
} else if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initialize, { once: true });
} else {
  initialize();
}
