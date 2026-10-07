# Paper sample identifiers

Internal dish keys (`dish_id`, D11, D12, …) remain stable. A separate sample ID
stores the handwritten paper identifier. Editing this ID never changes masks,
measurements or seed review decisions.

After measurement and export, the web server schedules a background DeepSeek
vision request for each paper crop. Only nearby paper regions from the original
photograph are sent; the full photograph is not uploaded. Failed OCR does not
invalidate the completed analysis. Existing results can be read with **Paper
labels → Read labels**. Each dish has one editable ID beneath its thumbnail. Enter or leaving the
field saves changes. A per-dish spinner indicates recognition in progress;
completed automatic IDs display Auto, human edits display Saved, and failures
are shown in red. IDs update individually as recognition completes. Saved human IDs always take precedence over later model transcriptions.

The model is `deepseek-flash` via the official Chat Completions endpoint, with
base64 JPEG inputs and JSON output. Machine sample IDs use exactly four digits,
an underscore and one ordinary digit (e.g. `0475_1`). Leading zeros are preserved;
circled suffixes are converted to ordinary digits. Original writing and secondary
lines are retained in `raw_text`. At the user's request, unclear digits are filled
with the model's best guess and the resulting ID is applied immediately, without
a mandatory review step. If the model leaves an ID empty, an explicit fallback
uses the four-digit image name and dish position (or `0000` when unavailable).
Legibility, inference and fallback notes are retained as provenance; these IDs
are automatic assignments, not human-confirmed transcriptions.
Model legibility is self-reported, not an experimentally calibrated confidence.
Paper localization uncertainty and duplicate model IDs are recorded in metadata.
Human edits always override automatic assignments.

Outputs include `sample_labels.json` and native paper crops. CSV downloads retain
numeric `dish_id` and add `dish_label`, `sample_id`, `sample_id_candidate`,
`sample_label_status`, and `paper_label_text`. Candidate and confirmed status are
explicit, so OCR is not misrepresented as verified metadata.

Credentials are configured in **Settings → Integrations**, or read from the
`GRAINMASTER_LABEL_API_KEY` / `DEEPSEEK_API_KEY` environment variables. Stored credentials
are kept separately under the local workbench state directory and are ignored by Git.
No credential is written into analysis results.
Transcriptions are cached by crop contents, model and prompt version to avoid
repeated paid requests. API failures can be retried via **Read labels**.

References: [DeepSeek vision](https://api-docs.deepseek.com/guides/vision/),
[JSON output](https://api-docs.deepseek.com/guides/json_mode/).
