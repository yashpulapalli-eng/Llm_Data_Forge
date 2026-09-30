# data/

This folder is a **local staging area only** — its contents are gitignored (see root `.gitignore`) because raw/curated LLM training data is far too large to version in git.

| Folder | Contents |
|---|---|
| `raw/` | Original, unprocessed source corpus/corpora (Phase 1) |
| `filtered/` | Quality-filtered, non-deduplicated variant (Phase 2) |
| `deduplicated/` | Deduplicated, non-filtered variant (Phase 3) |
| `filtered_deduplicated/` | Filtered **and** deduplicated variant (Phase 3) |

Document where the actual data lives (cloud bucket, cluster storage, etc.) and how to reproduce each folder's contents in the corresponding phase report, not here.
