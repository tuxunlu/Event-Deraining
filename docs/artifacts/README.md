# Published artifact sources

These are the exact page bodies behind the two published EvORSP explainers. They
live here because the previous generators were written to `/tmp` and lost to a
cleanup, which left two published pages asserting a claim we had already refuted
and no way to edit them.

| file | published at |
|---|---|
| `evorsp_explained.html` | https://claude.ai/code/artifact/588476d4-519d-4739-8e3b-026906cfe6e1 |
| `evorsp_pipeline.html`  | https://claude.ai/code/artifact/7251b2e8-c608-4bc3-9100-d0bc0d32e2e1 |
| `evorsp_master_table.html` | https://claude.ai/code/artifact/0ead6033-0f37-44a8-b679-fb84d0dff6b5 |

Each file is a page *body*: it is wrapped in `<!doctype html><head>…</head><body>`
at publish time, so it starts at `<title>` and carries its own `<style>`. To
update a page, edit the file and republish it to the SAME url above — publishing
without the url creates a second artifact instead of updating the original.

## What was corrected (2026-08-12)

Both pages claimed the temporal frontend's ungated residual band means the
network "cannot delete the static scene". `evorsp/eval/adversarial_dc.py`
refutes this: attacked on its own the decomposition is immovable (1.0x), but the
full network still suppresses static input ~530x, and deleting the protected
band moves that to ~363x -- i.e. the band contributes none of the protection.

`evorsp/model/rsp_guard3d.py` implements a version that does hold,
`logit = alpha*res + C*tanh(z/C)`, giving `logit > alpha - C` for any weights.
Verified: 3.9x suppression, minimum logit exactly 2.000 at alpha=4, C=2.
It costs 0.007 (weak floor) to 0.046 (strong floor) event-DA -- see the tables
on either page. Raw run output: `evorsp/checkpoints/adversarial_dc_result.txt`.
