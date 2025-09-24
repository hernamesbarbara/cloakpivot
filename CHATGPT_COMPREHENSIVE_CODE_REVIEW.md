
# CloakPivot Multi‑Session Codebase Assessment Plan
*A technical memo and protocols for a coding agent to identify superfluous code/files, non‑DRY implementations, irrelevant code paths, and disorganized modules. The goal is to produce a bulletproof, ordered change plan—**not** to implement changes.*

---

## 0) Scope & Objectives
**Target**: the `cloakpivot/` Python package and its tests.  
**Outcome**: a tightly scoped, reproducible set of refactor proposals (rename/split/remove/unify) with risk assessment, test impact, and precise steps so a future agent (or maintainer) can confidently implement them.

This memo defines:
- Review principles and heuristics to spot **superfluous** or **irrelevant** files/code paths.
- A **multi‑session protocol** for managing context over time.
- A **triage order** (hotspots first) derived from the line-count inventory.
- **Documentation artifacts** to be produced along the way (issue list, diagrams, diffs, test impact notes).
- A **change plan** template with explicit acceptance criteria.

> **Important**: Do not change code. Only read, measure, tag, and propose.

---

## 1) Inputs & Quick Repo Signals

### 1.1 Provided line‑count snapshots (selected highlights)
**Top code hotspots (≥ ~500 LOC):**
- `cloakpivot/masking/presidio_adapter.py` — **1310**
- `cloakpivot/masking/applicator.py` — **861**
- `cloakpivot/unmasking/document_unmasker.py` — **772**
- `cloakpivot/core/analyzer.py` — **597**
- `cloakpivot/core/surrogate.py` — **632**
- `cloakpivot/core/normalization.py` — **618**
- `cloakpivot/masking/engine.py` — **611**
- `cloakpivot/core/results.py` — **563**
- `cloakpivot/unmasking/anchor_resolver.py` — **561**
- `cloakpivot/core/policy_loader.py` — **554**
- `cloakpivot/loaders.py` — **543**
- `cloakpivot/core/validation.py` — **515**
- (Many others in the 200‑500 LOC range.)

**Test inventory:** 9,646 LOC across comprehensive unit/integration test files.

> **Signals:** The very large modules and duplicate‑sounding adapters (e.g., *masking* and *unmasking* `presidio_adapter.py`) suggest **DRY** opportunities, potential **leaky boundaries**, and **responsibility overload**.

---

## 2) Review Principles & Heuristics

1. **Single Responsibility & Cohesion**
   - A module or class should do one thing well. If a file exceeds ~400‑600 LOC, check if it bundles unrelated responsibilities or large private helper ecosystems that belong elsewhere.

2. **DRY & Cross‑cutting Utilities**
   - Search for near‑identical functions/algorithms across `masking/*`, `unmasking/*`, `core/*`. Favor shared utilities and protocols (e.g., `formats/`, `core/types.py`, `masking/protocols.py`).

3. **Dead/Irrelevant Code Paths**
   - Identify: (a) unreferenced public functions, (b) features guarded by obsolete flags, (c) adapters for engines no longer used, (d) compatibility shims that duplicate newer abstractions.

4. **Boundary Integrity**
   - Confirm clean separation between **document I/O**, **policy/strategy**, **detection/anonymization**, **surrogates**, **validation**, and **CLI**. Flag bidirectional import tangles, circular dependencies, or all‑knowing “god” modules.

5. **Error Handling & Normalization**
   - Large `error_handling.py`, `normalization.py`, and `validation.py` may harbor misc utilities. Ensure they are **principled**, not dumping grounds.

6. **Public API Stability**
   - Identify the **public surface area** (import paths used by tests and examples). Any rename/split must document migration steps and deprecation notes.

7. **Tests as Truth**
   - Treat tests as the **contract**. Any proposal must classify impacted tests (broken/updated/unchanged) and suggest minimal change strategies.

8. **Performance & Memory**
   - Note algorithmic hotspots in long files (multi‑pass traversals, O(n²) merges, heavy string ops) but defer optimization changes—only log risks and opportunities.

---

## 3) Tagging & Evidence Protocol

Use the following tags in all notes, issues, and TODOs. Each finding should include: **file:line(s)**, **short quote/snippet**, **reasoning**, and **proposed action**.

- **[DRY]** Duplicate/near‑duplicate code.  
- **[REDUNDANT]** Unused or superseded by newer APIs.  
- **[DEAD]** Unreachable / not referenced by code or tests.  
- **[LEAKY-BOUNDARY]** Cross‑layer calls or tight coupling.  
- **[OVERSIZED]** File > ~600 LOC; candidate for split.  
- **[MIXED-RESPONSIBILITY]** Multiple concerns in one module.  
- **[SHOULD-BE-PRIVATE]** Public API that should be internal.  
- **[SPEC-NEEDED]** Behavior unclear; needs owner/product decision.  
- **[TEST-GAP]** Behavior lacks tests.  
- **[DOC-GAP]** Key behavior lacks docstrings/README notes.

**Evidence format example:**

```
Finding: [DRY]
Location: cloakpivot/masking/presidio_adapter.py:220-318
Mirror:   cloakpivot/unmasking/presidio_adapter.py:180-276
Snippet:  normalize_entity_spans(...), map_operator_config(...)
Why:      Same algorithm with minor param diffs.
Propose:  Extract to core/presidio_common.py (internal), parametrize.
Impact:   Update 4 call-sites. No behavior change. Tests: 6 files.
Risk:     Low; add unit tests for the extracted helper.
```

All findings should be captured in a machine‑diffable list (`docs/review/findings.csv`) and human‑readable notebook (`docs/review/ASSESSMENT.md`).

---

## 4) Multi‑Session Workflow (Context Management)

This repo is large. Work must be staged and resumable. Follow this order and produce artifacts per session.

### Session 1 — Baseline & Inventory
**Goals**
- Map **public API** (imports used by tests/examples/CLI).
- Generate metrics: LOC by module, function counts, import graph (fan‑in/fan‑out), duplicate detection, unused code report.
- Produce initial **Hotspot Triage List** (Top 20 files by (LOC × fan‑in)).

**Commands (suggested)**
- LOC: `find cloakpivot -name '*.py' -print0 | xargs -0 wc -l | sort -nr`
- Imports graph (examples): `pip install snakefood` or `pip install pydeps`; then `pydeps cloakpivot --max-bacon=2 --noshow --output docs/review/deps.svg`
- Dead code: `pip install vulture` ; `vulture cloakpivot --min-confidence 90 > docs/review/vulture.txt`
- Duplicates: `pip install jscpd` ; `jscpd --languages "python" --reporters json --output docs/review/jscpd`
- Complexity: `pip install radon` ; `radon cc -s cloakpivot > docs/review/complexity.txt`

**Artifacts**
- `docs/review/BASELINE.md` (metrics + public API summary)
- `docs/review/deps.svg`
- `docs/review/vulture.txt`
- `docs/review/jscpd/report.json`
- `docs/review/complexity.txt`

### Session 2 — DRY Sweep & Boundary Audit
**Goals**
- Focus on pairs with similar names/roles:  
  - `masking/presidio_adapter.py` **vs** `unmasking/presidio_adapter.py`  
  - `document/*` **vs** `core/*` mappers/extractors  
  - `engine.py` and `engine_builder.py` split of concerns  
- Tag all [DRY], [LEAKY-BOUNDARY], [MIXED-RESPONSIBILITY] findings.

**Artifacts**
- `docs/review/DRY_MAP.md` — table of duplicates with candidate extraction target and call‑site count.
- `docs/review/BOUNDARIES.md` — each layer’s intended responsibilities and the cross‑calls to prune.

### Session 3 — Dead/Redundant/Legacy
**Goals**
- Review `compat.py`, `wrappers.py`, and older shims in `core/*` for **[REDUNDANT]** or **[DEAD]** paths.
- Mark functions/classes that have **no references** or **only legacy callers**.
- Align with tests—if a path has no tests and no external imports, propose removal or quarantine.

**Artifacts**
- `docs/review/DEPRECATIONS.md` — items slated for removal/quarantine with rationale and safe removal steps.

### Session 4 — Oversized Modules → Split Plans
**Goals**
- For each **[OVERSIZED]** module (esp. the 8 listed above), design a no‑behavior‑change **split plan**:
  - Identify cohesive sub‑areas.
  - Draft new file names and internal import edges.
  - Define migration shims (re‑exports) to avoid breaking public API.

**Artifacts**
- `docs/review/SPLIT_PLANS/*.md` — one doc per module with exact move map and re‑export plan.

### Session 5 — Test Impact & Risk Grid
**Goals**
- For each proposed change, map **affected test files** and risk level (Low/Med/High).  
- Identify **missing tests** that should be added first to lock current behavior.

**Artifacts**
- `docs/review/TEST_IMPACT.md` — matrix of change‑proposal → impacted tests.
- `docs/review/TEST_GAPS.md` — list of missing coverage.

### Session 6 — Final Change Set & Acceptance Criteria
**Goals**
- Collate a sequenced change set with chunkable PRs. Each PR should be small, reversible, and covered by tests.

**Artifacts**
- `docs/review/CHANGESET_PLAN.md` — ordered list of PRs with acceptance criteria.
- `docs/review/TRACKING_TODO.md` — master checklist to execute later.

---

## 5) Hotspot‑First Triage (Initial Targets)

Prioritize in this order (reason: LOC, centrality, naming collisions, test surface):

1. **`masking/presidio_adapter.py` (1310 LOC)** — Very large; likely blends strategy mapping, span transforms, engine orchestration. Expect multiple **[MIXED-RESPONSIBILITY]** and **[DRY]** with unmasking counterpart.
2. **`masking/applicator.py` (861 LOC)** — Core mutation logic; candidate to split into **pre‑processing**, **application**, **post‑validation** helpers.
3. **`unmasking/document_unmasker.py` (772 LOC)** — Large orchestration; review overlap with `masking/engine.py` and `unmasking/engine.py`.
4. **`core/surrogate.py` (632 LOC)** — Ensure generators and policies are modular; move shared utilities to `core/` submodules.
5. **`core/normalization.py` (618 LOC)** — Confirm clear scope (input vs internal vs output normalization). Flag grab‑bag utilities.
6. **`masking/engine.py` (611 LOC)** and **`core/results.py` (563 LOC)** — Engine/result coupling: ensure result types live in `core/types.py` or `core/results.py` consistently.
7. **`unmasking/anchor_resolver.py` (561 LOC)** & **`core/policy_loader.py` (554 LOC)** — Check DRY with `loaders.py` and `core/validation.py`.
8. **`core/analyzer.py` (597 LOC)** — Ensure responsibilities distinct from `core/detection.py` & `document/extractor.py`.

For each, produce a **SPLIT_PLAN** with sections: *Rationale*, *New Modules*, *Move Map*, *Re‑exports*, *Call‑site Updates*, *Test Impact*, *Risks*.

---

## 6) Documentation & Traceability Requirements

- **Every finding** gets a **stable ID**: `FIND-0001`, `FIND-0002`, … (reference in all docs/PR plans).
- Maintain a **CSV ledger**: `docs/review/findings.csv` with columns:
  - `id, tag, file, start_line, end_line, summary, proposal, risk, impacted_tests`
- Keep a **running changelog of proposals (no code)** in `docs/review/ASSESSMENT.md`.
- All diagrams and metrics must be stored under `docs/review/` with timestamps in filenames (not in content), e.g., `deps_2025-09-24.svg`.

---

## 7) Proposed Folder/Layout Improvements (No Code Changes Yet)

- Introduce `cloakpivot/presidio_common/` or `cloakpivot/core/presidio_common.py` for shared adapter utilities referenced by both masking and unmasking.
- Ensure public types (used by callers/tests) live in `core/types.py` or `core/results.py`, and keep **protocols** in `masking/protocols.py` only if they are masking‑specific; otherwise move to `core/protocols.py`.
- Confirm that `document/` handles **I/O and structure extraction** exclusively; **no policy/strategy** logic should leak in.
- Evaluate `wrappers.py` and `compat.py` for **[REDUNDANT]** patterns; consider quarantining in `compat/` with explicit deprecation plan.

---

## 8) Checklists

### 8.1 DRY Checklist
- [ ] Same function names across modules?  
- [ ] Same algorithm with different param defaults?  
- [ ] Repeat of span/anchor normalization between masking/unmasking?  
- [ ] Duplicated error formatting/validation code?

### 8.2 Dead/Redundant Checklist
- [ ] Unreferenced functions/classes (`vulture` report).  
- [ ] Legacy adapters not used by CLI/tests.  
- [ ] Flags/env vars that no longer alter behavior.  
- [ ] Shadowed utilities replaced by `core/*` equivalents.

### 8.3 Boundary Checklist
- [ ] Imports crossing from `document/*` → `core/*` only (not vice versa).  
- [ ] `engine/*` depends on `core/*` and `masking/*`/`unmasking/*`, not the other way around.  
- [ ] `cli/*` only orchestrates top‑level API; no business logic.

### 8.4 Oversized File Split Checklist
- [ ] Identify cohesive groups (3–7 functions/classes per group).  
- [ ] Define new module names.  
- [ ] Plan re‑exports for backward compatibility.  
- [ ] Map call‑site import updates.  
- [ ] Unit tests to pin the moved behavior before split.

---

## 9) Risk Management & Acceptance Criteria

For each proposal, include:
- **No behavior change** statement (what must remain identical).  
- **Pass criteria**: existing tests green; new pinning tests added where gaps exist.  
- **Rollback**: re‑export toggle or revert plan.  
- **Size limit**: single PR ≤ 300 LOC diff where possible.

---

## 10) Deliverables (What the Future Agent Will Receive)

1. `docs/review/BASELINE.md` — metrics, graphs, public API map.  
2. `docs/review/DRY_MAP.md` — duplications and unifications.  
3. `docs/review/BOUNDARIES.md` — target architecture and violations.  
4. `docs/review/DEPRECATIONS.md` — dead/legacy items with removal steps.  
5. `docs/review/SPLIT_PLANS/*.md` — one per oversized module.  
6. `docs/review/TEST_IMPACT.md` — matrix of change → tests.  
7. `docs/review/TEST_GAPS.md` — tests to add before refactors.  
8. `docs/review/CHANGESET_PLAN.md` — ordered PR plan with acceptance criteria.  
9. `docs/review/TRACKING_TODO.md` — master checklist (IDs linked to findings).

---

## 11) Appendix — Helpful One‑liners (No Changes Performed)

- Largest files first:  
  `find cloakpivot -name '*.py' -print0 | xargs -0 wc -l | sort -nr | head -n 30`
- Functions & classes inventory:  
  `grep -RInE '^(def|class)\s' cloakpivot | sed 's/:/ | /' > docs/review/symbols.txt`
- Public imports used in tests:  
  `grep -RInE '^from\s+cloakpivot|^import\s+cloakpivot' tests | sort > docs/review/test_imports.txt`

> **Reminder:** This memorandum defines *process* and *artifacts*, not code edits. Follow it to produce an actionable, low‑risk refactor plan for CloakPivot.
