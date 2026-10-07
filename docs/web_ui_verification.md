# Web workbench verification

Verified on 2026-10-07 with the local service at 127.0.0.1:8765.

- Full test suite: `python -m pytest -q` — 86 passed.
- Frontend syntax: `node --check web/app.js` — passed (Node 20.18.0).
- Review adapter loaded all 36 existing photos: 172 dishes, 4009 candidates;
  finite measurements. Record: `artifacts/web_ui/verification/adapter_batch.json`.
- Real browser loaded 0462, displayed 4 dish thumbnails and 53 candidates.
- Clicking D11 focused its crop; S01 showed its mask, ID, 6.68 mm length,
  3.15 mm width, 16.87 mm² area and Lab 73.06 / 4.91 / 28.90.
- Confirming S01 changed confirmed count to 1; discarding changed retained count
  from 53 to 52; restoring pending recovered 53 and zero confirmed. A reload
  preserved restored state. Original masks and pipeline CSVs were unchanged.
- Two candidates have specific shape warnings (D11/S05 and D12/S02).
- Browser file chooser imported the actual raw 0462.jpg. The copied bytes equal
  the raw file. A fresh run, ID `upload_141a4fd45f2e4a2cafb26d75b06fa31c`, completed
  spatial/color calibration and produced 4 dishes and 53 classical instances.
- Browser downloaded its seed CSV: 53 rows, with review state/provenance.
- Browser displayed all four dish summary rows, lengths, widths, areas and Lab.
- 390 px responsive check: document scroll width 375 px, no page horizontal
  overflow; summary uses a scrollable table. Temporary viewport reset afterward.
- Browser console inspection returned no errors.

API tests additionally cover invalid uploads, actual failed analysis of an image
without a ruler, unknown IDs, invalid review modes, persistence, exports and local
origin checks. API artifacts and tests use isolated temporary review directories.

Current limitations: classical masks may merge touching seeds; priority warnings
are heuristics; mask drawing/splitting is not implemented. No seed model training
or annotation is claimed. Human inclusion decisions do not validate absolute
scale or color accuracy. Lab means use equal seed weights, not pooled pixel weights.
