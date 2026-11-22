# Improvement plan: from "depression detector" to a safe, well-built check-in tool

Status: **approved** (2026-11-22). Gate answers: check-in companion; ELIZA + optional self-hosted LLM (no cloud APIs); Streamlit local web app; safety fix merged first.

## Where the project is today

| Area | Finding | Evidence |
| --- | --- | --- |
| Safety | No crisis handling. The ELIZA reflection turns "I want to die" into "Why do you want to die?". | `src/chatbot.py` `respond()` — fixed on branch `fix/crisis-safety` (Phase 0). |
| Model validity | The classifier does not detect depression. "Not depressed" data is 100% HappyDB *happy moments*; "depressed" is first-person distress posts. It learns "happy moment vs anything else". | Trained Multinomial NB (acc 0.83) labels *"My manager scheduled a meeting about the quarterly budget on Tuesday."* as **depressed**. |
| Evaluation | 30 test documents per class; one random split; accuracy only. The README's accuracy table reads as a benchmark. | `classifier.py` output. |
| Data | `depressed.txt` provenance and license undocumented (looks like scraped forum posts — personal text). | `data/` |
| Repo hygiene | 33 MB Windows `.exe` and ~4.6 MB of PDFs in the tree; `License` (non-standard name); no tests, CI, linting, packaging, CONTRIBUTING, SECURITY, CHANGELOG, issue templates. | `Presentation/`, `p1.pdf`, `report.pdf` |
| Architecture | Two scripts glued by `output.txt`. The model is retrained on every prediction. Conversation, scoring and I/O are mixed. | `src/` |
| Privacy | The full conversation is written to `output.txt` in plain text by default. | `chatbot.py run()` |
| UX | CLI only; ends with a single binary verdict ("shows signs of depression") and no guidance. | — |

## Target

A small, honest, open-source **mental-health check-in companion**:
a supportive conversation, a **validated screening questionnaire (PHQ-9)** for the actual
result, clear next steps and crisis resources, everything local by default. The original
NLP classifier stays as a clearly labelled **research/educational mode**, not the verdict.

## Phases

### Phase 0: Safety fix (ready)
Verified when: crisis phrases never get reflected; tests green.
Branch: `fix/crisis-safety` (commit 8762782) — `src/safety.py`, 28 tests.

### Phase 1: Clean, open-source-grade repository (do first)
Verified when: `pytest`, `ruff check`, `ruff format --check`, `mypy` pass locally and in CI; repo
under 1 MB of tracked non-data files; a new contributor can install and run from the README alone.

#### Step 1.1: Repo hygiene (S)
- Remove `Presentation/*.exe`, `Presentation.pdf`, `p1.pdf`, `report.pdf` from the tree; attach
  the report PDF to a GitHub release and link it. Rename `License` → `LICENSE`.
- Add `CONTRIBUTING.md`, `SECURITY.md` (incl. "this is not a crisis service"), `CODE_OF_CONDUCT.md`,
  `CHANGELOG.md`, issue/PR templates, `.editorconfig`.
- Blocked by: none

#### Step 1.2: Packaging + tooling (S)
- `pyproject.toml` (PEP 621, src layout `src/moodcheck/`), console entry
  point, pinned dev deps; ruff (lint + format), mypy, pytest + coverage; pre-commit.
- GitHub Actions CI: lint, type-check, tests on Python 3.10–3.13.
- Blocked by: 1.1

#### Step 1.3: Architecture refactor (M)
Split by responsibility, each module testable on its own:
```
src/<pkg>/
  safety.py        crisis detection + resources (Phase 0)
  conversation/    engine interface; eliza.py (current bot)
  screening/       phq9.py (Phase 2)
  research/        corpus loading, preprocessing, classifier, evaluation
  storage.py       opt-in local session saving
  cli.py           Typer/argparse CLI wiring only
```
- Train once, save the model (`joblib`) with a version stamp; predict loads it.
- Transcript saving becomes **opt-in** (`--save`), default off.
- Behaviour preserved; tests for each module.
- Blocked by: 1.2

#### Step 1.4: Honest README + data card (S)
- Rewrite the README: what it is / is not, quickstart, architecture diagram, safety section.
- `data/README.md` (data card): sources, licenses, known biases, the happy-moments confound.
- Remove the benchmark-style accuracy table; report Phase 3 evaluation instead.
- Blocked by: 1.3

### Phase 2: Validated check-in and guidance (core feature)
Verified when: PHQ-9 scoring matches the published bands in unit tests; item 9 > 0 always
shows crisis resources; results screen shows next steps.

#### Step 2.1: PHQ-9 module (S)
- The 9 items, 0–3 answers, total 0–27, published severity bands (cut points 5/10/15/20,
  Kroenke, Spitzer & Williams 2001 — re-verify against the primary source at build time).
  Original credit line kept.
- Any non-zero answer to item 9 (self-harm thoughts) → crisis resources, regardless of total.
- Blocked by: 1.3

#### Step 2.2: Guidance (S)
- Per band: plain-language meaning, suggested next step (e.g. "talk to a GP" from moderate up),
  evidence-based self-care tips sourced and cited (NHS / NIMH pages), locale helplines.
- Never a diagnosis; wording reviewed against the disclaimer.
- Blocked by: 2.1

#### Step 2.3: Flow (S)
- Conversation → offer the check-in → results + guidance → optional save. Works in CLI.
- Blocked by: 2.1, 2.2

### Phase 3: Make the research mode honest (S–M)
Verified when: an evaluation report shows the confound and a fixed baseline.
- Add a neutral control set (everyday non-emotional text) so "not depressed" ≠ "happy";
  stratified k-fold, precision/recall/F1, confusion matrix; a probe set of neutral sentences
  that must not be called "depressed".
- If no properly licensed clinical dataset is available (see Not yet specified), document the
  limits rather than chase accuracy.
- Blocked by: 1.3

### Phase 4: Better conversation, fully local (M)
Verified when: the engine interface has two implementations, both pass the shared
safety tests; the bot no longer parrots; no network traffic leaves the host during a session.
- Engine interface; improved ELIZA (more patterns, short memory, no reflection of harmful text).
- Optional **self-hosted LLM engine** through a local Ollama server (`/api/chat` on
  `localhost:11434`, [docs](https://docs.ollama.com/api/chat.md)), with a supportive-listener
  system prompt. The safety screen runs on input **and** model output. ELIZA is the fallback when
  no local model is running. Endpoint configurable but defaults to localhost; refuse non-local
  endpoints unless explicitly allowed.
- No third-party/cloud LLM APIs in this project.
- Blocked by: 1.3

### Phase 5: Friendly interface (M)
Verified when: a first-time user can complete chat → check-in → results without reading docs.
- Local Streamlit web app: chat pane, check-in form, results page with guidance and
  helplines always visible, "your data stays on this device" notice. Accessible (WCAG AA).
- Blocked by: 2.3, 4

## Privacy and HIPAA

Self-hosting keeps every conversation on the machine running the tool. That is a strong privacy
safeguard, but it does **not** by itself make anything "HIPAA compliant":

- HIPAA applies to covered entities (providers, health plans, clearinghouses) and their business
  associates. A consumer using a tool on their own is generally outside HIPAA
  ([HHS](https://www.hhs.gov/hipaa/for-professionals/privacy/guidance/access-right-health-apps-apis/index.html),
  [FTC](https://www.ftc.gov/business-guidance/resources/mobile-health-apps-interactive-tool)).
- If a covered entity deploys it, compliance comes from that organisation's safeguards (access
  control, audit logs, encryption at rest, risk analysis, BAAs with vendors), not from the code.

What the project will do: local-only processing by default, no telemetry, opt-in and
encrypted-at-rest session storage, a `docs/privacy.md` that states exactly what is stored and
where, and a "deploying in a clinical setting" section listing the safeguards an operator must add.
What it will never do: claim to be HIPAA compliant or certified.

## Decision log
| Decision | Chosen option | Why |
| --- | --- | --- |
| Result shown to the user | PHQ-9 score + guidance, not the ML label | The ML label is invalid (see table); PHQ-9 is validated and free to reproduce. |
| Safety screen placement | Before every reply, every engine | One shared guarantee, testable once. |
| Transcript storage | Opt-in, local only | Mental-health text is sensitive. |
| Product framing | Check-in companion; ML = research mode | Gate answer 2026-11-22. |
| Conversation engine | Improved ELIZA default; optional self-hosted LLM via Ollama; no cloud APIs | User decision 2026-11-22 (replaces the earlier Claude-API answer): conversations must stay on infrastructure we control. |
| Interface | Streamlit local web app (CLI kept) | Gate answer; fastest path to a friendly UI. |
| Package name | `moodcheck` (GitHub repo name unchanged for now) | Renaming the repo changes public URLs; can be done later. |
| Large binaries | Removed from the tree, not from git history | A history rewrite would break existing clones; the size is a one-time clone cost. |

## Not yet specified
- Phase 3: is there a properly licensed labelled dataset we may use (e.g. research datasets that
  need a data-use agreement)? Fog — researched at Phase 3 pickup.
- Phase 4: which local model (size vs. quality on a laptop CPU/GPU) — benchmark candidates at pickup.
- Phase 2: which countries' helplines to list beyond the US + findahelpline.com.

## Always / Never
- Always: crisis resources reachable from every screen; disclaimer that this is not a diagnosis;
  tests for every safety path; cite sources for any health claim.
- Never: diagnose, give medication advice, store or send conversations without explicit consent,
  claim clinical accuracy, claim HIPAA compliance, send conversation text to third-party APIs,
  ship the ML label as a verdict.

## Progress
| Phase | Status |
| --- | --- |
| 0 Safety | ready (branch `fix/crisis-safety`) |
| 1 Repo + architecture | not started |
| 2 PHQ-9 + guidance | not started |
| 3 Honest research mode | not started |
| 4 Conversation | not started |
| 5 Interface | not started |
