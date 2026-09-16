# Meta-Memory QA Dataset Placeholder

This folder reserves the Stage 2 dataset shape for QA-driven meta-memory updates.

Version 6.1 does not run QA evaluation yet. It only keeps the schema direction so later versions can attach `evaluation_signals`, compare a formation strategy against a baseline, and generate stricter `MetaMemoryUpdateCandidate` objects.

Planned flow:

```text
workspace + query + qa_items
-> formation plan
-> memory artifacts
-> answer/evidence evaluation
-> evaluation_signals
-> meta-memory update candidate
```

See `sample_schema.json` for the draft fields.
