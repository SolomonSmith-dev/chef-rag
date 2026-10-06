---
title: chef-rag
emoji: 🍳
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# chef-rag demo

Professional kitchen Q&A answered only from public-domain sources, with citations and a
retrieval-scores panel. Hybrid BM25 + dense retrieval, cross-encoder rerank, refusal when
the sources do not cover the question. Question text is never logged.

Source: https://github.com/SolomonSmith-dev/chef-rag

`POST /query` with `{"question": "..."}` returns `answer`, `citations`, `scores`.
