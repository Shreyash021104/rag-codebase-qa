# Contributing

## `main` is protected

Every change goes through a pull request — CI must pass and a review is required
before anything merges, including from the maintainer's own tooling. Fork the repo,
branch off `main`, open a PR into `main`.

## Before you open a PR

1. **Read the README's "hardest decisions" section.** It explains why chunking is
   AST-aware, why retrieval is hybrid (vector + BM25 + RRF), and why the LLM step is
   optional. A PR that reverts one of those without addressing the underlying reason
   will bounce.
2. **If you change chunking or retrieval, re-run the eval** (`scripts/eval.py`) and
   include the before/after top-k numbers in your PR description. Retrieval quality is
   the whole point of this project — changes to it need to be measured, not vibed.
3. **Keep secrets out of commits.** Config comes from environment variables / `.env`
   (gitignored). Never commit an API key or connection string.

## Good first contributions

- Add a language to the chunker's extension map (`app/chunker.py`) and a test question
  for it in the eval set.
- Incremental re-indexing: only re-chunk files whose content hash changed.
- Stream the LLM response for lower perceived latency.

## License

MIT
