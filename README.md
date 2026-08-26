# RAG + Recommendation Engine

A RAG (retrieval-augmented generation) query service combined with a hybrid
recommender, built as a FastAPI application. This README is deliberately
specific about what has actually been run and verified versus what is
still a template you need to adapt -- see "Verification status" below
before treating this as done.

## What's actually been verified (not just written)

Everything below was executed in a sandbox during development, not just
read for correctness:

- Full test suite: **28/28 tests pass** (`pytest tests/`), covering the
  recommender's training loop, the vector store's fallback and eviction
  behavior, the semantic cache's expiry handling, the rate limiter's
  Redis-down fallback, and the FastAPI app's auth/CORS/error-handling.
- The FastAPI app boots and serves real requests via `TestClient` with
  **zero external services running** -- Redis, Pinecone, and every LLM
  provider are all absent, and every component falls back correctly
  (verified via captured logs showing the fallback path actually engaging,
  not just existing in the code).
- A full RAG query, end to end, using real seeded catalog data: embedding,
  retrieval, ranking, context building, generation (via a local
  no-API-key-required provider), and groundedness scoring all produced
  real, inspectable output.
- The recommender's training loop was run on synthetic data and the loss
  was confirmed to actually decrease across epochs (not a stub).
- The RAG evaluation script (the CI quality gate) was run standalone and
  produced real faithfulness/relevancy numbers against a generated eval set.
- Several real bugs were found and fixed *during this verification*, not
  just reasoned about: a vector store bug where retrieved document text was
  nested under `metadata` but every downstream consumer read it from the
  top level (silently empty context/groundedness until fixed), and a
  predictor bug where querying recommendations before a model was trained
  returned an opaque 502 instead of gracefully degrading to an empty list.

## What is NOT verified (be aware before calling this "done")

- **Docker build was not executed.** No Docker daemon was available in the
  build sandbox. `requirements.txt` matches exactly what was pip-installed
  and tested, so risk is low, but run `docker build -f docker/Dockerfile .`
  yourself before trusting it in CI.
- **Kubernetes manifests were validated as syntactically correct YAML with
  consistent resource references, not applied to a real cluster.** Test
  with `kubectl apply --dry-run=server` against a real cluster first.
- **CI pipeline (`(.github/workflows/ci.yml`) was not executed on GitHub
  Actions** -- the individual commands it runs (pytest, the eval script,
  data generation) were verified locally, but the workflow YAML itself,
  secrets wiring, and multi-job orchestration have not been run.
- **No real production data.** `scripts/generate_sample_data.py` produces
  clearly-synthetic data so the project is runnable out of the box. Replace
  it with your actual catalog/interactions before training a model you
  intend to serve real traffic with.
- **Real LLM provider calls (OpenAI/Anthropic/Google) were not exercised**
  -- no API keys were available in this environment. The provider-calling
  code paths are implemented and lazily imported, but only the local-echo
  test provider was actually invoked end to end.
- **Load testing, security penetration testing, and a practiced
  incident/rollback runbook do not exist.** These require real
  infrastructure, real traffic, and organizational process -- they cannot
  be produced as chat-delivered files.

## Architecture

```
Request -> AuthMiddleware -> RateLimitMiddleware -> RAG/Recommendation route
                                                            |
                            RAGPipeline: cache check -> retrieve -> rank
                            -> build context -> route LLM -> generate
                            -> check groundedness -> score confidence
```

Key design decision: **heavy ML dependencies (torch, sentence-transformers,
transformers) are optional**, not required to boot. The default embedder is
a deterministic scikit-learn `HashingVectorizer` (`TfidfEmbedder`), and the
default recommender backend is plain-numpy matrix factorization. This means
the whole system installs and runs in seconds without a multi-GB model
download -- important for CI, for local dev, and for anyone evaluating this
without a GPU. Swap in real semantic embeddings for production by installing
the `semantic-embeddings` extra and setting `embedder.backend:
sentence-transformers` in config.

## Quick start

```bash
make install          # core deps only, no torch
make seed-data         # generates data/items.parquet, interactions.parquet, rag_test.parquet
make train              # trains a real recommender, saves to models/hybrid_v1
make test                # runs the full verified test suite
make run                  # starts the API on :8000 (Redis optional, falls back gracefully)
```

Or with Docker (build not executed in this environment -- please verify):

```bash
make seed-data
docker compose up --build
```

## Configuration

Config is loaded from environment variables prefixed `RAG_` (nested keys
via `__`, e.g. `RAG_AUTH__SECRET_KEY`), plus a handful of variables read
directly by name in specific modules (`REDIS_HOST`, `OPENAI_API_KEY`, etc).
See `.env.example` for the full list and which code path reads each one.

## What's a stub vs. real

| Component | Status |
|---|---|
| RAG pipeline (retrieve/rank/generate/groundedness) | Real, tested end-to-end |
| Recommender (matrix factorization) | Real, actually trains and learns |
| Vector store (in-memory) | Real, bounded, no fabrication -- verified with tests |
| Vector store (Pinecone backend) | Implemented, lazily imported, **not exercised** (no API key) |
| Semantic cache | Real, tested including the expiry-during-iteration edge case |
| Rate limiting (Redis + fallback) | Real, tested including the Redis-down path |
| Auth (JWT) | Real, tested |
| LLM providers (OpenAI/Anthropic/Google) | Implemented, lazily imported, **not exercised** |
| LLM local-echo provider | Real, used for all end-to-end testing in this repo |
| MLflow experiment tracking / model registry | Implemented with safe fallback, **not exercised** (no MLflow server) |
| CI pipeline | Written, individual steps verified locally, **workflow itself not run on GH Actions** |
| Kubernetes manifests | Valid YAML, internally consistent, **not applied to a real cluster** |
| Docker build | Requirements verified via direct pip install, **`docker build` itself not run** |
| Load testing / security review / runbooks | **Not included** -- out of scope for a chat-delivered project |

## Project structure

```
src/
  core/
    embeddings.py          # pluggable embedder (tfidf default, sentence-transformers optional)
    models/hybrid.py       # recommender: numpy matrix factorization + content scoring
    rag/                   # pipeline, cache, context builder, ranker, groundedness
    retrieval/              # vector store, multi-stage retriever
    generation/              # LLM router, response generator (multi-provider + local-echo)
  api/                       # FastAPI app, middleware, routes, schemas, DI container
  data/                       # user history, feature store (Redis-backed)
  serving/                     # predictor (async-safe), cache manager
  evaluation/                   # RAG evaluator (faithfulness/relevancy/precision)
  mlops/                          # experiment tracking, model registry (MLflow, optional)
  monitoring/                      # Prometheus metrics
tests/
  unit/                              # component tests, including regression tests for
                                       # specific bugs found and fixed during development
  integration/                        # full FastAPI app tests via TestClient
scripts/
  generate_sample_data.py             # synthetic data for local dev/CI
  train_model.py                       # trains and saves the recommender
  seed_vectors.py                       # seeds a standalone vector store
  evaluate_rag.py                        # CI quality gate
  wait_for_services.sh                    # actually invoked by docker/entrypoint.sh
docker/                                   # Dockerfile, entrypoint.sh
k8s/                                       # Deployment, Service, HPA, PDB, NetworkPolicy
.github/workflows/ci.yml                    # lint -> test -> eval gate -> build -> deploy
```