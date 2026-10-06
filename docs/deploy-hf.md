# Deploy the demo to a Hugging Face Space (run on your Mac)

The build sandbox could not reach huggingface.co and has no HF token, so nothing was
pushed. Run these steps yourself. Commands use full paths for `ls` and `cp -f`.

Prerequisite: `docs/corpus-runbook.md` passed (the Space build downloads the same
corpus with the same script, so the URLs must already work).

## 1. Assemble the Space directory

```bash
uv run python scripts/prepare_space.py --out build/hf-space
```
Expect: `prepared N files in build/hf-space (K original notes included)`.
Any file in `data/raw/original/` is published with the Space. Move out notes you do
not want public before running.

```bash
/bin/ls build/hf-space
```
Expect: `Dockerfile  LICENSE  README.md  data  pyproject.toml  scripts  src  uv.lock`.

## 2. Optional local container test

```bash
docker build -t chef-rag-demo build/hf-space
docker run --rm -p 7860:7860 -e OPENROUTER_API_KEY chef-rag-demo
```
The `-e OPENROUTER_API_KEY` form passes your shell's value without writing it anywhere.
Expect the build to end with `ingest` output (`indexed N chunks (dim=384)`) and the
container to log `Uvicorn running on http://0.0.0.0:7860`. Open http://localhost:7860.

## 3. Create the Space and push

```bash
huggingface-cli login            # paste your HF token when prompted
huggingface-cli repo create chef-rag --type space --space_sdk docker
git clone https://huggingface.co/spaces/SolomonSmith-dev/chef-rag build/hf-space-repo
/bin/cp -fR build/hf-space/. build/hf-space-repo/
cd build/hf-space-repo && git add -A && git commit -m "deploy chef-rag demo" && git push && cd ../..
```
Expect: `git push` succeeds and the Space page shows "Building".

## 4. Add the secret (do not paste it into a terminal)

Space page, Settings, "Variables and secrets", New secret:

- Name `OPENROUTER_API_KEY`, value your key.

Optional variables: `DEMO_DAILY_TOKEN_CAP` (default 200000 tokens/day, about 55
queries), `DEMO_RATE_PER_MIN` (5), `DEMO_RATE_PER_DAY` (40), `DEMO_MODEL`
(default `anthropic/claude-haiku-4.5`). Optional secrets `LANGFUSE_PUBLIC_KEY` and
`LANGFUSE_SECRET_KEY` send traces to Langfuse; without them traces stay on the Space's
ephemeral disk and contain no question text.

Then Settings, "Factory reboot".

## 5. Done checks (can fail)

```bash
uv run python scripts/check_demo.py --url https://solomonsmith-dev-chef-rag.hf.space; echo "exit=$?"
```
Expect: `OK` and `exit=0`. Possible `FAIL` lines: `/healthz returned 502` (still
building; wait and retry), `no retrieval scores returned`, `answer has no citations`.

```bash
uv run python scripts/check_demo.py --url https://solomonsmith-dev-chef-rag.hf.space --rate-limit; echo "exit=$?"
```
Expect: `OK`. This sends 13 queries and fails if no HTTP 429 appears. Run it once;
it uses a few thousand tokens and may exhaust the per-IP daily limit for your address.

Budget check: set the variable `DEMO_DAILY_TOKEN_CAP` to `1`, reboot, ask one
question (answered), ask a second. The second must return the message "Demo budget
reached for today". Then set the variable back and reboot.

## 6. Link it

Replace the README placeholder under "Demo" with the Space URL, commit, push.
