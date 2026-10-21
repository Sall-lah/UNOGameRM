# UNOGameRM: Risk Calibration and Cross-Turn Resource Allocation in a Budget-Aware MCTS & PlugMem UNO Agent

> **Research Title (EN):** *Evaluating Risk Calibration and Cross-Turn Resource Allocation in a Budget-Aware Search and Memory LLM Agent in the Game of UNO*  
> **Judul Penelitian (ID):** *Evaluasi Kalibrasi Risiko dan Alokasi Sumber Daya Lintas Giliran pada Agen LLM dengan Pencarian Berbatas Anggaran dan Memori dalam Permainan UNO*

**UNOGameRM** is a deterministic, zero-leakage experimental research platform built around a modified 2-player non-stacking UNO engine ([`theSanguss/UNO.py`](https://github.com/theSanguss/UNO.py)). It evaluates how a Large Language Model (LLM) agent—augmented with **PlugMem** intra-round three-tier memory and **MARS** budget-aware imperfect-information Monte Carlo Tree Search (MCTS)—calibrates short-horizon risk predictions and allocates tactical card resources across turns under strict wall-clock latency budgets.

---

## Table of Contents

1. [Research Questions (RQ1 – RQ3)](#1-research-questions-rq1--rq3)
2. [End-to-End Three-Zone Architecture](#2-end-to-end-three-zone-architecture)
3. [Repository Structure](#3-repository-structure)
4. [Engineering Blueprints & OpenSpec Capabilities](#4-engineering-blueprints--openspec-capabilities)
5. [Getting Started & Configuration](#5-getting-started--configuration)
6. [Experiment Runbook & Statistical Analysis CLI](#6-experiment-runbook--statistical-analysis-cli)
7. [Testing & Verification](#7-testing--verification)
8. [Interactive Human Terminal Mode & Upstream Attribution](#8-interactive-human-terminal-mode--upstream-attribution)

---

## 1. Research Questions (RQ1 – RQ3)

| ID | Research Dimension | Core Evaluation Focus | Primary Metrics & Estimators |
| :--- | :--- | :--- | :--- |
| **RQ1** | **Conditional Short-Horizon Risk Calibration** | How accurately and well-calibrated does the agent estimate conditional probabilities of suffering a forced penalty draw ($p_{t,A} \leftrightarrow y_{t,A}$) or an immediate opponent round win ($p_{t,B} \leftrightarrow y_{t,B}$) over the horizon $H_t = [t, t+1)$ after locking action $a_t$? | Brier Score, Expected Calibration Error (ECE, 10 equal-width bins), Clipped Log-Loss, and Horizon Stratum 1 deterministic sanity check ($y_{t,A} = y_{t,B} = 0$) with round-level cluster-bootstrap 95% CIs. |
| **RQ2** | **Cross-Turn Resource Allocation & Tactical Alignment** | Does the agent's risk assessment align with its cross-turn conservation and deployment of high-leverage action/wild cards ($\text{Skip}$, $\text{Reverse}$, $+2$, $\text{Wild}$, $\text{Wild}+4$) and resulting hand-size trajectories? | Spearman rank correlation ($\rho_S$), action/wild usage rate across risk regimes ($\hat{p}_t \ge \tau_{\text{risk}}$ vs. $< \tau_{\text{risk}}$), next-turn hand-size delta ($\Delta |H_{t+1}^{\text{agent}}|$), and round draw efficiency. |
| **RQ3** | **Memory & Search Grounding of Risk Estimates** | How strongly do **PlugMem** opponent constraints ($\text{Exp}_t$) and **MARS MCTS** lookahead statistics ($N, Q$, rollout penalty/win rates) ground the LLM's risk predictions and rationales, and when do residual miscalibrations occur? | Prediction–outcome residuals ($r_{t,A} = p_{t,A} - y_{t,A}$, $r_{t,B} = p_{t,B} - y_{t,B}$), adverse-event contrast by memory/search state, and the automated 4-criterion rationale validity rubric. |

---

## 2. End-to-End Three-Zone Architecture

To enforce **zero hidden-state leakage** (*no future or opponent-hand leakage*) and **strict pre-dispatch commitment** (*no post-hoc rationalization*), the platform is partitioned into three isolated zones:

<img width="1600" height="682" alt="Three-Zone Architecture" src="https://github.com/user-attachments/assets/3bd2b59c-d72c-4b6a-aba9-0e7a94fe505b" />

### Core Subsystem Highlights

- **Zone 1 (`src/core/`, `src/env/`)**:
  - [`UNOCard` & `UNOAction`](src/core/cards.py): Immutable, hashable value types producing byte-identical string encodings to `UNO.py` (`card.Card.__str__`) and full 108-card deck construction.
  - [`HeadlessGame`](src/env/headless_game.py) & [`ModifiedUNOServerPlatform`](src/env/uno_engine.py): Deterministic headless port of `UNO.py` with seeded RNG injection, fixed `AIPlayer` intellect, automatic `"uno"` declaration on $2 \to 1$ transitions, and typed [`HorizonServerEvent`](src/env/events.py) telemetry separating penalty draws ($A_{\text{pen}}$) from voluntary draws ($A_{\text{nop}}$).
  - [`ClientServerBarrier`](src/env/barrier.py): Strips hidden opponent cards ($H_t^{\text{opp}}$) and draw-pile ordering, emitting an immutable [`LegalObservation`](src/env/barrier.py) ($o_t$) and computing the unseen card pool $U_t$ ($108 - |H_t^{\text{agent}}| - |D_t^{\text{public}}|$).
- **Zone 2 (`src/memory/`, `src/search/`, `src/llm/`, `src/harness/`)**:
  - [`UNOPlugMem`](src/memory/plugmem.py): Intra-round 3-tier memory graph (Episodic $G^E$, Semantic $G^S$, Procedural $G^P$) tracking opponent draw constraints (`LACKS_COLOR`, `LACKS_VALUE`) and wild colour declarations with exponential confidence decay ($\gamma = 0.75$) and soft determinization weights $\omega(c) = \max(0.01, 1 - 0.85 \cdot w_\tau)$.
  - [`call1_prior`](src/llm/call1_prior.py) & [`call2_risk`](src/llm/call2_risk.py): Strict Pydantic-validated two-call LLM interface with multi-tiered fallback ladders. Call #1 prunes actions and assigns $P_{\text{prior}}$; Call #2 estimates conditional risks $(p_{t,A}, p_{t,B})$ for the **already locked** action $a_t$.
  - [`MARSBudgetAwareMCTS`](src/search/mars_mcts.py): Open-loop imperfect-information MCTS sampling determinizations $(\hat{H}^{\text{opp}}, \hat{D}^{\text{draw}})$ from $U_t$, guided by PUCT and efficiency-modulated rewards $R(v) = G(v) \cdot [\max(t_{\text{sim}}, \epsilon)/L]^w$.
- **Zone 3 (`src/audit/`, `src/analysis/`, `src/experiment/`)**:
  - [`Stage1FreezeLogger`](src/audit/stage1.py), [`HorizonMonitor`](src/audit/horizon.py), & [`Stage2Evaluator`](src/audit/stage2.py): Cryptographically freezes pre-execution decisions (`flush` + `os.fsync` + SHA-256) prior to server dispatch, classifies horizon strata (Stratum 1: $0$ opp turns; Stratum 2: $1$ opp turn; Stratum 3: $\ge 2$ opp turns), and enforces $t_{\text{freeze}} < t_{\text{dispatch}} \le t_{\text{eval}}$.

---

## 3. Repository Structure

```text
UNOGameRM/
├── config.yaml                  # Master experimental & agent configuration
├── environment.yml              # Conda environment specification (Python 3.12)
├── main.py                      # Preserved upstream interactive human CLI game
├── uno.py / card.py / deck.py   # Upstream UNO.py modules (untouched for human play)
├── player.py / cli_tools.py     # Upstream AIPlayer & terminal helpers
├── docs/                        # Comprehensive mathematical & engineering blueprints
│   ├── README.md                # Master implementation guide
│   ├── 01_architecture_and_information_barrier.md
│   ├── 02_uno_platform_modifications.md
│   ├── 03_plugmem_memory_specification.md
│   ├── 04_mars_budget_mcts_specification.md
│   ├── 05_llm_prompts_and_schemas.md
│   ├── 06_two_stage_audit_protocol.md
│   ├── 07_evaluation_and_statistical_analysis.md
│   └── 08_experiment_runbook_and_pilot.md
├── openspec/                    # OpenSpec specifications & archived change history
│   ├── specs/                   # 12 canonical capability specifications
│   └── changes/archive/         # Completed & verified OpenSpec changes (1 -> 9)
├── src/                         # Modular research platform implementation
│   ├── config.py                # Pydantic v2 configuration loader & freeze validator
│   ├── core/                    # UNOCard, UNOAction, 108-card deck & string parser
│   ├── env/                     # HeadlessGame, ModifiedUNOServerPlatform, Barrier, Events
│   ├── memory/                  # PlugMem graph (G^E, G^S, G^P), retrieval, decay, refiner
│   ├── search/                  # Determinization over U_t, 2P forward model, MARS MCTS
│   ├── llm/                     # LLMClient protocol, FakeLLMClient, prompts, schemas, Call #1/#2
│   ├── audit/                   # Stage 1 freeze logger, HorizonMonitor, Stage 2 evaluator, JSONL
│   ├── harness/                 # AgentHarness per-turn pipeline & RoundRunner
│   ├── analysis/                # RQ1/RQ2/RQ3 evaluators, cluster bootstrap, rubric, CLI
│   └── experiment/              # Validation suite, pilot, calibration, sample-size, main run, CLI
└── tests/                       # 191 automated unit & end-to-end integration tests
    ├── analysis/                # Statistical metrics, bootstrap, RQ1-RQ3, & CLI tests
    ├── experiment/              # Config freeze, validation suite, pilot/calibration, & main tests
    ├── harness/                 # Turn budget, non-play decisions, ordering, & round runner tests
    ├── llm/                     # Client, prompt rendering, schema validation, & fallback tests
    ├── search/                  # Determinization, forward model, UCT/reward, & fallback tests
    └── test_*.py                # Core card model, engine adapter, barrier, PlugMem, & audit tests
```

---

## 4. Engineering Blueprints & OpenSpec Capabilities

Every subsystem is backed by both an engineering blueprint in [`docs/`](docs/) and a formal specification in [`openspec/specs/`](openspec/specs/):

| Zone / Layer | Engineering Blueprint (`docs/`) | Formal Specification (`openspec/specs/`) | Implementation Package (`src/`) |
| :--- | :--- | :--- | :--- |
| **Architecture & Barrier** | [`01_architecture_and_information_barrier.md`](docs/01_architecture_and_information_barrier.md) | [`information-barrier`](openspec/specs/information-barrier/spec.md) | [`src/env/barrier.py`](src/env/barrier.py) |
| **Headless UNO Engine** | [`02_uno_platform_modifications.md`](docs/02_uno_platform_modifications.md) | [`uno-card-model`](openspec/specs/uno-card-model/spec.md), [`uno-engine-adapter`](openspec/specs/uno-engine-adapter/spec.md), [`uno-horizon-events`](openspec/specs/uno-horizon-events/spec.md) | [`src/core/cards.py`](src/core/cards.py), [`src/env/`](src/env/) |
| **PlugMem Memory** | [`03_plugmem_memory_specification.md`](docs/03_plugmem_memory_specification.md) | [`plugmem-memory`](openspec/specs/plugmem-memory/spec.md) | [`src/memory/`](src/memory/) |
| **MARS Budget-Aware MCTS** | [`04_mars_budget_mcts_specification.md`](docs/04_mars_budget_mcts_specification.md) | [`mars-mcts`](openspec/specs/mars-mcts/spec.md) | [`src/search/`](src/search/) |
| **LLM Decision Calls** | [`05_llm_prompts_and_schemas.md`](docs/05_llm_prompts_and_schemas.md) | [`llm-decision-calls`](openspec/specs/llm-decision-calls/spec.md) | [`src/llm/`](src/llm/) |
| **Two-Stage Audit Protocol** | [`06_two_stage_audit_protocol.md`](docs/06_two_stage_audit_protocol.md) | [`audit-logging`](openspec/specs/audit-logging/spec.md), [`horizon-labeling`](openspec/specs/horizon-labeling/spec.md), [`agent-harness`](openspec/specs/agent-harness/spec.md) | [`src/audit/`](src/audit/), [`src/harness/`](src/harness/) |
| **Statistical Evaluation** | [`07_evaluation_and_statistical_analysis.md`](docs/07_evaluation_and_statistical_analysis.md) | [`evaluation-analysis`](openspec/specs/evaluation-analysis/spec.md) | [`src/analysis/`](src/analysis/) |
| **Experiment Runbook** | [`08_experiment_runbook_and_pilot.md`](docs/08_experiment_runbook_and_pilot.md) | [`experiment-runbook`](openspec/specs/experiment-runbook/spec.md) | [`src/experiment/`](src/experiment/) |

---

## 5. Getting Started & Configuration

### Prerequisites

- **Python**: `3.12+` (required for `UNO.py`'s nested f-string syntax and modern typing).
- **Conda / Mamba**: Recommended for reproducible scientific dependencies (`numpy`, `pandas`, `scipy`, `pyarrow`, `pydantic`, `httpx`).

### Environment Installation

```bash
# 1. Create the Conda environment from environment.yml
conda env create -f environment.yml

# 2. Activate the environment
conda activate uno-research
```

### LLM Endpoint & Environment Variables

All experimental parameters are centralized in [`config.yaml`](config.yaml) and validated via [`src/config.py`](src/config.py). By default, the LLM client targets an OpenAI-compatible `/v1/chat/completions` endpoint (such as local vLLM, Ollama, llama.cpp, or OpenAI API):

```powershell
# PowerShell
$env:OPENAI_API_KEY = "your-api-key-here"
```

```bash
# Bash / Zsh
export OPENAI_API_KEY="your-api-key-here"
```

> [!IMPORTANT]
> Never commit API keys or `.env` files to version control. [`src/config.py`](src/config.py) enforces a security validator that rejects literal secrets (`sk-...`) inside `config.yaml` and reads credentials exclusively from the environment variable named in `llm.api_key_env`.

---

## 6. Experiment Runbook & Statistical Analysis CLI

### 6.1 Running the Experiment Runbook (`src.experiment.run`)

The experiment pipeline ([`src/experiment/run.py`](src/experiment/run.py)) orchestrates the four sequential experimental stages defined in [`docs/08_experiment_runbook_and_pilot.md`](docs/08_experiment_runbook_and_pilot.md):

```bash
# Phase 1: Run pre-pilot architectural & protocol validation checks
# (Verifies information barrier, 2P non-stacking turn retention, A_pen vs A_nop, auto-UNO, & timestamp ordering)
python -m src.experiment.run validate --config config.yaml

# Phase 2: Run exploratory pilot rounds (R_pilot = 15 rounds -> data/pilot/dataset_drt.jsonl)
python -m src.experiment.run pilot --config config.yaml

# Phase 3: Calibrate latency budgets (P50/P90/P99), check JSON compliance (>= 95%),
#          and compute the recommended main-run sample size R
python -m src.experiment.run calibrate --config config.yaml

# Phase 4: Execute the frozen main experiment (supports automatic crash-safe resume & writes run_manifest.json)
python -m src.experiment.run main --config config.yaml

# Optional: Override the number of rounds for main execution
python -m src.experiment.run main --config config.yaml --rounds 100
```

### 6.2 Running Statistical Evaluation (`src.analysis.run`)

Once decision records (`dataset_drt.jsonl`) are collected, run the evaluation suite ([`src/analysis/run.py`](src/analysis/run.py)) to compute RQ1, RQ2, and RQ3 metrics, round-level cluster-bootstrap confidence intervals, the non-fallback vs. fallback sensitivity split, and the stratified rationale audit sample:

```bash
python -m src.analysis.run \
  --dataset data/main_experiment/dataset_drt.jsonl \
  --output-dir reports \
  --bootstrap-b 2000 \
  --bootstrap-seed 42 \
  --tau-risk 0.5
```

**Generated Report Artifacts in `reports/`:**
- `evaluation_report.json` — Complete JSON report with metadata, primary subset metrics, and sensitivity subset metrics.
- `rq1_primary_calibration.csv` & `rq1_sensitivity_calibration.csv` — Overall and per-stratum Brier, ECE, and Log-Loss with 95% cluster-bootstrap CIs.
- `rq2_primary_alignment.csv` & `rq2_sensitivity_alignment.csv` — Cross-turn resource conservation metrics and Spearman correlations across risk regimes.
- `rationale_audit_sample.csv` — Stratified sample of decision rationales scored against the 4-criterion automated validity rubric.

---

## 7. Testing & Verification

The repository includes **191 automated unit and integration tests** across 52 test modules using deterministic `FakeLLMClient` doubles (requiring zero external network access):

```bash
# Run the complete test suite
pytest -q

# Run specific subsystem test suites
pytest tests/test_cards.py tests/test_deck_and_parse.py tests/test_barrier.py
pytest tests/test_plugmem_*.py
pytest tests/test_stage1.py tests/test_stage2.py tests/test_horizon*.py
pytest tests/search/
pytest tests/llm/
pytest tests/harness/
pytest tests/analysis/
pytest tests/experiment/
```

---

## 8. Interactive Human Terminal Mode & Upstream Attribution

The original interactive terminal UNO game is preserved unmodified so you can play UNO directly in your terminal using the Rich TUI:

```bash
python main.py
```

### Upstream Credits

- **Original Game Engine & Rich TUI:** [`theSanguss/UNO.py`](https://github.com/theSanguss/UNO.py) by **theSanguss**.
- All Zone 1 engine adaptations are implemented via non-invasive subclassing and hooks in [`src/env/`](src/env/) so that [`main.py`](main.py), [`uno.py`](uno.py), [`card.py`](card.py), [`deck.py`](deck.py), [`player.py`](player.py), and [`cli_tools.py`](cli_tools.py) remain 100% compatible with the upstream human gameplay experience.
