# Virtual Data Center & Canary Detection: Implementation Status & Audit

**Document Version:** 2.1.0 (Versioned Scenarios, One-Command Demo, Scaling Benchmarks, Publication Plots & Technical Limitations Complete)  
**Date:** 2026-11-14  
**Status:** COMPLETE — Versioned scenario suite (`canary_ssm/scenarios/`, 8 canonical scenarios), one-command local demo startup (`scripts/demo.py`, `canary-ssm demo`), separate persistence of 5 distinct artifacts (`manifest.json`, `telemetry.csv`, `model_policy.json`, `recommendations.csv`, `evaluator_results.json`), fleet scaling benchmark across 12 to 384 servers with hardware profiling (`results/runtime_scaling.csv`), publication figures generated strictly from CSV tables (`results/plots/`), comprehensive technical limitations document (`docs/LIMITATIONS.md`), and preserved original benchmark configs verified. Exit gate passed: fresh local runs reproduce telemetry and detector outputs with exact parity. Full fast test suite: 229 passed, 1 skipped, 1 deselected in 104.39s with zero regressions.  

---

## 1. Executive Summary

This document tracks the incremental implementation of the Virtual Data Center in `canary_ssm`.

- **Prompt 0 (Audit & Architecture Freeze):** Completed. Architecture frozen in `docs/virtual_world_architecture.md`.
- **Prompt 1 (Virtual Fleet & Clock):** Completed. Configurable `Fleet`, `Rack`, `Server` records, default 4-rack 24-server topology, deterministic discrete `SimulatedClock`, priority `EventQueue`, and `VirtualWorld` engine implemented. Verified by 13 unit tests (`tests/test_world.py`, all passing in 0.71s).
- **Prompt 2 (Realistic Multi-Metric Telemetry):** Completed. `TelemetryRecord` contract, dynamic server numerical states (`ServerResourceState`), and `HealthyTelemetryEngine` implemented. Emits request counts, CPU utilization, process memory, log latency, and error counts under fleet-wide demand, diurnal cycles, rack-shared nuisance variations, and hardware class offsets. Enforces strict bounds and anti-leakage firewall. Verified by 8 unit tests (`tests/test_telemetry.py`, all passing in 0.34s).
- **Prompt 3 (Fault Engine & Evaluator Truth Store):** Completed. `FaultEngine`, typed infrastructure vs. release faults (`SERVER_OUTAGE`, `RACK_NETWORK_LATENCY`, `FLEET_DEMAND_SPIKE`, `CPU_SATURATION`, `PROCESS_MEMORY_LEAK`, `ERROR_REGRESSION`, `RELEASE_LATENCY_REGRESSION`), resource pressure coupling (Normal, Pressure, Thrashing, OOM crash/restart), independent RNG streams, and `EvaluatorFaultTruthStore` implemented. Verified by 6 unit tests (`tests/test_faults.py`, all passing in 0.42s).
- **Prompt 4 (Release Registry & Deployment Lifecycle):** Completed. Versioned `ReleaseRegistry`, `ReleaseArtifact` (SHA-256 hash, strict unexecuted label, size/type limits), `BehaviorProfile` (CLEAN, MEMORY_LEAK, CPU_REGRESSION, ERROR_REGRESSION, LATENCY_REGRESSION), `validate_canary_allocation` (enforcing control coexistence and rejecting confounded pure-canary racks and pure-canary classes), `DeploymentManager` (deploy canary, rollback with explicit process restart / heap clearing semantics, idempotent command IDs), declarative sample manifests (`configs/releases/`), CLI `release` subcommand, and 8 unit tests (`tests/test_deployments.py`, all passing in 0.34s).
- **Dashboard & API Server (Prompt 4b/5):** Completed. FastAPI REST API (`canary_ssm/api/`) with `SimulatorService` managing world state, deployments, and background stepping. Modern dark-themed 5-page Single Page Application (`canary_ssm/dashboard/`: World, Releases, Fault Lab, Monitoring, Evaluation) with zero external build dependencies. Verified by 7 unit tests (`tests/test_api.py`, all passing in 0.92s).
- **Hierarchical Detection & Concrete Acceptance Demos:** Completed. Dual Advisory Engine (`InfrastructureAdvisory` vs. `CanaryReleaseAdvisory`), same-rack baseline control contrast adjustment, memory pressure channel coupling, and 4 concrete demo scenarios (`canary_ssm/studies/acceptance_scenarios.py`). Verified by 5 unit tests (`tests/test_acceptance_scenarios.py`, all passing in 5.28s).
- **Time Slicing & Streaming Residuals Pipeline:** Completed. `canary_ssm/telemetry/slicing.py` (`StreamSliceBuilder`, `SlicingProtocol`, `PrePeriodBaseline`, `TimeSlice`, `SliceAuditLog`, `infer_release_cause`) and nested rack adjustment in `canary_ssm/factor.py`. Strict causality (`arrival_time <= decision_timestamp`), deterministic duplicate resolution, immutable past alerts, exclusion of exposed hosts from controls, pre-period parameter fitting ($t < t_0$), nested fleet factor and rack nuisance estimation with zero-sum identifiability ($\sum_j W_j \hat{\gamma}_j = 0$), closed-form shared estimation uncertainty propagation ($\text{Cov}(A_t)$ in `cov_U`), coverage protocol abstention (`INSUFFICIENT_RACK_CONTROLS`, `INSUFFICIENT_CLASS_CONTROLS`, `INSUFFICIENT_CANARY_COVERAGE`), multi-metric residuals (latency, memory, CPU, errors) and masks, absent rack controls cause inference blocker, and causal invariance (no future leakage). Verified by 13 unit tests (`tests/test_slicing.py`, all passing in 0.52s).
- **Generic Kalman Engine & Unified Streaming Detectors:** Completed. `canary_ssm/kalman.py` and `canary_ssm/detectors.py` (`StreamingDetector`, `OnlineEWMA`, `OnlineCUSUM`, `OnlineTopKCUSUM`, `OnlineIndependentSSM`, `OnlinePooledSSM`, `OnlineHierarchicalSSM`, `create_streaming_detector`). Scalar fixture verified ($m_1=0.6875, P_1=0.34375, m_2=126/151, P_2=71/302$); tiny multivariate case verified against independent textbook reference; Joseph form maintains PSD symmetry under long missing gaps and correlated observation noise; parsimonious AR noise fitting separates process noise from measurement uncertainty; pre-period noise states $(m, P)$ carried into rollout; proper nonzero effect prior initialized at $t_0$; complete pooling SSM correctly propagates $P_\theta$ and cross-covariances; and causal streaming parity verified across all detectors. Verified by 15 unit tests (`tests/test_kalman_and_detectors.py`, all passing in 2.54s).
- **Hierarchical State-Space Model & Pooling Dynamics:** Completed. `canary_ssm/hierarchy.py` and `canary_ssm/studies/study_hierarchical_pooling.py`. Deterministic Helmert basis ($B^T B = I_{n-G}$, $C^T B = 0$, $L = [C \quad B]$ full rank $n$), singleton class support ($m=1 \implies \text{shape }(1, 0)$), state $s = [\theta, b, e]$, transition $F = \text{blockdiag}(\rho_g I_G, \rho_h I_{n-G}, \Phi_e)$, observation $H = [C, B, I]$, effect extraction $L_{\text{full}} = [C, B, 0]$. Outputs class effects $\theta_t$, host deviations $B b_t$ (summing to 0 within every class), effect means $a_{m, t}$, full posterior covariance $a_{\text{cov}, t}$, and host log latency effect exceedance probability $\Pr(a_{it} > \log(1.02))$. Release score is strictly $\max_i \log p_{it}$ (monotone max-host risk, not a joint probability). Closed-form implied host covariance and pooling sensitivity ($\text{Corr} \to 1$ as $\tau \to 0$; $\text{Corr} = (m-1)/(2m-1)$ at $\tau=1$; $\text{Corr} = 0$ at $\tau=\sqrt{m}$; $\text{Corr} \to -1/(m-1)$ as $\tau \to \infty$). 4 demonstration scenarios implemented and verified: (1) Class-wide harm, (2) Sparse harm, (3) Infrastructure rack shock, and (4) Rollout harm followed by rollback. All diagnostic traces saved to `results/table_pooling_diagnostic_traces.csv` (3,120 rows). Verified by 10 unit tests (`tests/test_hierarchy_pooling.py`, all passing in 2.22s).
- **Prompt 9 (Memory Growth, GC Dynamics & Multi-Channel Policy):** Completed. `canary_ssm/memory.py` and `canary_ssm/studies/study_memory_channel.py`. Operational boundary $\kappa_M = 0.50$ MB/min, persistence window $H = 15$ ticks, GC allocation/sweep cycles (`gc_period_ticks=15, gc_amplitude_mb=300.0`), allocator warmup ramps (`warmup_ticks=15, warmup_amplitude_mb=400.0`), physical headroom / TTE guardrail, causal difference MA(1) model with Joseph updates, 5 verified operational fixtures, and Bonferroni-Šidák joint null calibration ($\text{FWER} \le 0.05$). Verified by 9 unit tests (`tests/test_memory_channel.py`, all passing in 0.69s).
- **Prompt 10 (Independent Detector Service & UI Monitoring Integration):** Completed. `canary_ssm/detector_service.py`, `canary_ssm/api/routes.py`, `canary_ssm/api/service.py`, `canary_ssm/dashboard/index.html`, `app.js`, `style.css`. Complete firewall decoupling (zero imports of `simulate.Truth`, `metrics`, or truth files; AST verified). Exposes host effect estimates with analytical 95% CIs ($\hat{a}_i \pm 1.96 \cdot \text{SE}(a_i)$), class effects ($\hat{\theta}_g \pm \text{SE}$), host risk ($\Pr(a_i > \log(1.02))$), release score ($\max_i \log \Pr$), model and policy IDs (`MODEL-HSSM-V1`, `POLICY-SIDAK-0.05`), data eligibility, memory drift rate ($\hat{g}_i \pm \text{SE}$), capacity headroom MB, TTE min, operational capacity guardrail, and dual advisory states. Monitoring view updated with 2 dedicated SSM canvas charts (Latency Effect with 95% CI band and Memory Drift Rate with $\kappa_M$ threshold), first alert vertical marker lines, interactive server drill-down modal, explicit manual rollback confirmation modal, opt-in automated rollback mode preserving benchmark traces with censoring tracking, and graceful disconnection fallback (`DETECTOR NOT CONNECTED` / `DETECTOR UNAVAILABLE`) preserving telemetry charts. Verified by 8 unit tests in `tests/test_detector_service.py` and 13 integration tests in `tests/test_api.py`. Full test suite: 213 fast tests passing in 90.54s.

---

## 2. Source-Backed Responsibility Inventory

Below is the detailed mapping of repository components against the target Virtual Data Center responsibilities:

| Responsibility Domain | Status | Existing Files & Functions | Description & Scope in Current Codebase | Delta Needed for Virtual Data Center |
|---|---|---|---|---|
| **world** | **Implemented** | `canary_ssm/world/` (`models.py`, `clock.py`, `world.py`) | Configurable `Fleet`, `Rack`, `Server` records (default 4 racks x 6 servers = 24 servers). `SimulatedClock` discrete simulation time, `EventQueue` min-heap, `VirtualWorld` orchestrator with `step(N)`, `reset()`, and bounded `fast_forward()`. CLI `world` subcommand. | Ready for Deployments and Dashboard. |
| **workloads** | **Implemented** | `canary_ssm/world/telemetry.py` (`HealthyTelemetryEngine`) | Generates fleet log-workload $x_t \sim \text{AR}(1)$, diurnal cycles, host elasticity $\beta_i x_t$, request volume scaling, and queuing delay. | Integrated into world simulation. |
| **faults** | **Implemented** | `canary_ssm/faults/` (`models.py`, `truth.py`, `engine.py`) | Explicit infrastructure vs. release faults, scope isolation, resource pressure memory leak coupling, OOM crash/restart behavior, independent RNG streams, and `EvaluatorFaultTruthStore`. | Wire into Developer Fault Lab API/UI. |
| **telemetry** | **Implemented** | `canary_ssm/telemetry/` (`records.py`), `canary_ssm/world/telemetry.py` | Multi-metric `TelemetryRecord` (`run_id`, `tick`, `event_time`, `arrival_time`, `server_id`, `rack_id`, `class_id`, `version`, `exposure`, `request_count`, `latency_ms`, `memory_mb`, `cpu_fraction`, `error_count`, `observed`). Bounded, anti-leakage firewall, low-volume uncertainty. | Streaming buffer and API routes. |
| **deployments** | **Implemented** | `canary_ssm/deployments/` (`models.py`, `allocation.py`, `manager.py`, `samples.py`) | Versioned `ReleaseRegistry`, `ReleaseArtifact` (SHA-256, strictly unexecuted label), `BehaviorProfile`, `validate_canary_allocation` (blocks pure-canary rack/class confounding), `DeploymentManager` (deploy, rollback with/without restart, idempotent command IDs), sample manifests, CLI commands. | Ready for Dashboard Releases page and API endpoints. |
| **detectors** | **Implemented** | `canary_ssm/factor.py` (`adjust()`), `canary_ssm/kalman.py` (`KalmanFilter`), `canary_ssm/hierarchy.py` (`EffectDetector`), `canary_ssm/baselines.py`, `canary_ssm/baselines_extra.py`, `canary_ssm/oracles.py` | Full Joseph-form Kalman filter, orthonormal class-effect basis, rank-1 factor adjustment, 11 baselines, 3 oracles. | Preserve existing latency SSM; in later increment, add same-rack control contrast adjustment and memory growth channel. |
| **policy** | **Implemented** | `canary_ssm/decision.py` (`calibrate_threshold()`, `first_alarm()`, `advisory_replay()`, lines 17–100) | Order-statistic null calibration, latched alarm, one-crossing advisory replay. | Separate dual advisory states: `InfrastructureAdvisory` (NORMAL/RACK_DEGRADED/FLEET_OUTAGE) vs. `CanaryReleaseAdvisory` (RUNNING/PAUSE/ROLLED_BACK/INSUFFICIENT_DATA). |
| **api** | **Implemented** | `canary_ssm/api/` (`app.py`, `routes.py`, `service.py`) | FastAPI REST API exposing simulation control (`start`, `pause`, `step`, `reset`, `speed`), server/rack topology, release creation/validation/deployment/rollback, manual fault injection, ground truth inspection, and telemetry timeline. | CLI `dashboard` command. |
| **dashboard** | **Implemented** | `canary_ssm/dashboard/` (`index.html`, `style.css`, `app.js`) | 5-page Single Page Application (1. World, 2. Releases, 3. Fault Lab, 4. Monitoring, 5. Evaluation). Vanilla ES6 JS + modern dark CSS, pure HTML5 Canvas charts, zero build dependencies. Served statically by FastAPI. | Integrated into dashboard web server. |
| **evaluation** | **Implemented** | `canary_ssm/metrics.py` (`event_truth()`, `classify()`, `localization()`, lines 27–100), `canary_ssm/experiments.py`, `canary_ssm/report.py` | Comprehensive evaluator-only metrics (capped delay, localization F1/AUC, Brier score, Beta-Binomial null validation). | Wire into Evaluation dashboard page for post-run scenario replay and table downloads. |

---

## 3. Preservation of Existing Run Commands

All existing execution modes and CLI workflows must be preserved throughout implementation:

1. **Unit Test Suite:**
   ```powershell
   # Run full test suite (excluding slow legacy reproduction test):
   python -m pytest -q -m "not slow"
   ```
   *Verified status:* `152 passed, 1 skipped, 1 deselected in 91.94s`.

2. **PowerShell Task Runner (`tasks.ps1`):**
   - `.\tasks.ps1 test` -> Runs pytest suite.
   - `.\tasks.ps1 smoke` -> Runs 5-stage pipeline with `configs/smoke.yaml`.
   - `.\tasks.ps1 plain-pipeline` -> Runs end-to-end simulation, fit, calibration, validation, evaluation.
   - `.\tasks.ps1 step-pipeline` -> Runs step-regime pipeline.
   - `.\tasks.ps1 audit` -> Runs frozen calibration audit.
   - `.\tasks.ps1 report` -> Generates markdown summary report.

3. **Core CLI (`canary_ssm.cli` / `canary-ssm`):**
   ```powershell
   python -m canary_ssm.cli simulate --config configs/smoke.yaml --role smoke --out data_smoke/smoke
   python -m canary_ssm.cli fit --data data_smoke/dev --out artifacts_smoke/model --grid-stride 4 --allow-small-dev
   python -m canary_ssm.cli calibrate --data data_smoke/calibration --model artifacts_smoke/model --out artifacts_smoke/policy
   python -m canary_ssm.cli validate-null --data data_smoke/null_validation --model artifacts_smoke/model --policy artifacts_smoke/policy
   python -m canary_ssm.cli evaluate --data data_smoke/test --model artifacts_smoke/model --policy artifacts_smoke/policy --out results_smoke
   ```

---

## 4. Verified Smoke Command Output (Prompt 0 Gate)

To satisfy the Prompt 0 exit gate requirement ("small runnable smoke command"), the following CLI simulation was executed using the existing repository engine:

```powershell
python -m canary_ssm.cli simulate --config configs/smoke.yaml --role smoke --out data_smoke/smoke
```

**Recorded Output:**
```json
{
  "role": "smoke",
  "regime": null,
  "n_pairs": 12,
  "per_family": null,
  "family_counts": {"sparse_single": 8, "baseline": 16},
  "paired": true,
  "seed_purpose": "smoke",
  "seed_base": 10000,
  "seed_ranges": [{"purpose": "smoke", "family_idx": 0, "count": 12, "min_seed": 10000, "max_seed": 10011}],
  "config_hash": "30f05aee6fd99c1d",
  "config": {
    "sim": {},
    "roles": {"smoke": {"pairs": 12}, "dev": {"pairs": 60}, "calibration": {"pairs": 60}, "null_validation": {"pairs": 60}, "test": {"pairs": 24}},
    "families": [
      {"name": "baseline", "weight": 0.6, "overrides": {"n_affected": 3, "effect_pct": 3.0, "duration": 45}},
      {"name": "sparse_single", "weight": 0.4, "overrides": {"n_affected": 1, "effect_pct": 4.0, "duration": 45}}
    ]
  },
  "config_path": "configs/smoke.yaml",
  "n_rollouts": 24,
  "manifest_only": false,
  "compact": false,
  "disk_free_bytes": 782875721728
}
```

---

## 5. Migration & Integration Map

Implementation will proceed across five sequential increments. Each increment is verified by focused unit and smoke tests before proceeding:

```
[Prompt 0: Audit & Architecture Freeze]  <=== COMPLETE
       │
       ▼
[Increment 1: World First]
  ├── Add `canary_ssm/world/` (fleet.py, rack.py, server.py, clock.py)
  ├── Add `canary_ssm/workloads/` (traffic.py, capacity.py)
  ├── Verification: 4 racks x 6 servers simulation runs 300 discrete minutes
  └── Gate: All existing pytest tests continue to pass
       │
       ▼
[Increment 2: Telemetry & Faults]
  ├── Add `canary_ssm/telemetry/` (metrics.py, buffer.py, schema updates)
  ├── Add `canary_ssm/faults/` (fault_lab.py, injections.py: memory leak, rack slowdown)
  ├── Verification: Telemetry emits all 6 metrics; memory leak coupled to pressure
  └── Gate: Pure simulation tests verify fault schedules and truth isolation
       │
       ▼
[Increment 3: Deployment & Dashboard]
  ├── Add `canary_ssm/deployments/` (registry.py, allocation.py, identifiability.py)
  ├── Add `canary_ssm/api/` (routes.py, server.py)
  ├── Add `canary_ssm/dashboard/` (index.html, app.js, style.css)
  ├── Verification: 5-page UI serves world state; dry-run blocks pure-canary racks
  └── Gate: Browser and API smoke checks verify unexecuted artifact labels
       │
       ▼
[Increment 4: Hierarchical SSM Adaptation]
  ├── Integrate rack nuisance adjustment into `factor.py`
  ├── Add memory growth channel to detector suite
  ├── Calibrate joint multi-channel policy
  ├── Verification: Dual advisory outputs (Infrastructure vs Canary Release)
  └── Gate: Null calibration verified on multi-channel detectors
       │
       ▼
[Increment 5: Validation Suite & Acceptance Demo]
  ├── Run Scenario A (Memory Leak on 2 canaries) -> Pause recommended
  ├── Run Scenario B (Rack network slowdown) -> Rack degraded, Canary running
  ├── Run Scenario C (Compound faults + Identifiability stress) -> Uncertain / Insufficient Data
  └── Final Gate: Full demo scenario verified and documented in results
```

---

## 6. Prompt 0 Exit Gate Checklist

- [x] **Source-backed inventory:** Full mapping of current files/functions to world, workloads, faults, telemetry, deployments, detectors, policy, api, dashboard, evaluation responsibilities documented with file paths and line numbers.
- [x] **Architecture document:** `docs/virtual_world_architecture.md` created, establishing two distinct hierarchies, observation decomposition, multi-metric channels, resource pressure coupling, dual advisory states, and explicit disclaimers regarding numerical simulation vs. code execution.
- [x] **Implementation status document:** `docs/implementation_status.md` created with verified test logs, smoke outputs, and migration roadmap.
- [x] **Preservation of run commands:** Existing CLI commands and `tasks.ps1` targets audited and verified.
- [x] **Small runnable smoke command:** Executed and recorded (`python -m canary_ssm.cli simulate --config configs/smoke.yaml --role smoke --out data_smoke/smoke`).
- [x] **No inference changes:** No statistical models, filters, or inference logic modified in this increment.

---

## 7. Prompt 1 Exit Gate Checklist & Verification

- [x] **Configurable Fleet, Rack, and Server records:** Implemented in `canary_ssm/world/models.py`. Default configuration generates 4 racks of 6 servers (24 servers total) with balanced hardware/service classes (3 class 0, 3 class 1 per rack).
- [x] **Constraint validation:** Enforces positive capacities (`memory_capacity_mb > 0`, `cpu_cores > 0`, `base_latency_ms > 0`) and rejects duplicate server IDs both within a rack and across the entire fleet with descriptive errors.
- [x] **SimulatedClock & EventQueue:** Discrete integer time ($t \ge 0$) completely independent of wall-clock time. Priority min-heap event queue ordered deterministically by `(tick, priority, seq)`.
- [x] **VirtualWorld Engine:** Provides `start()`, `pause()`, `step(n)`, `reset()`, and bounded `fast_forward()`.
- [x] **Step Equivalence:** Advancing $N$ ticks in one call (`step(N)`) executes identical per-tick transitions and yields identical state as calling `step(1)` $N$ times.
- [x] **Replay Determinism:** Calling `reset()` restores the exact initial world configuration and seed, reproducing the identical state and event trajectory bitwise.
- [x] **Topology CLI & Inspection:** CLI subcommand `python -m canary_ssm.cli world` supports `--topology`, `--step <N>`, `--fast-forward <T>`, `--inspect`, and `--json`.
- [x] **Unit & Regression Verification:** 13 new unit tests in `tests/test_world.py` passed in 0.71s. Regression suite of 43 tests passed in 6.06s with zero failures.

---

## 8. Prompt 2 Exit Gate Checklist & Verification

- [x] **Multi-Metric TelemetryRecord Contract:** Implemented in `canary_ssm/telemetry/records.py`. Contains `run_id`, `tick`, `event_time`, `arrival_time`, `server_id`, `rack_id`, `class_id`, `version`, `exposure`, `request_count`, `latency_ms`, `memory_mb`, `cpu_fraction`, `error_count`, and `observed`.
- [x] **Strict Anti-Leakage Firewall:** Verified that `TelemetryRecord` does NOT contain `fault_type`, `injected_severity`, or any hidden ground-truth fields.
- [x] **Latency Statistic Declaration:** Explicitly documented in `TelemetryRecord.__doc__` that `latency_ms` represents the estimated geometric mean response time in ms over the discrete 1-minute aggregation interval ($\exp(\text{mean log-latency})$), and does not represent an empirical request-level p95/p99 tail metric.
- [x] **Dynamic Server Numerical States:** Implemented `ServerResourceState` in `canary_ssm/world/models.py` tracking workload demand, request count, CPU fraction, memory level, latency, log-latency, error count, and AR(1) temporal noise states.
- [x] **Healthy Telemetry Generator:** Implemented `HealthyTelemetryEngine` in `canary_ssm/world/telemetry.py`. Incorporates fleet-wide demand shocks ($x_t$), diurnal cycle, fleet common factor ($f_t$), rack-shared nuisance shifts ($\gamma_j(t)$), hardware class offsets, and request volume queuing effects.
- [x] **Low-Volume Measurement Uncertainty:** Measurement noise standard deviation is scaled inversely with request volume ($\sigma_{\text{meas}} \propto 1 / \sqrt{\max(1, N_{it})}$). When request volume is 0 (offline), error count is 0.
- [x] **Strict Numerical Bounds:** Enforced CPU $\in [0.0, 1.0]$, latency $> 0.0 \text{ ms}$, error count $\in [0, \text{request\_count}]$, and memory $\in [100.0, M_{\text{cap}}]$.
- [x] **No Label-to-Status Shortcuts:** Verified that baseline controls and canaries share the exact same physical response functions under normal workload changes.
- [x] **Shared Load & Rack Shift Sensitivity:** Verified that fleet-wide workload increases elevate requests, CPU, and latency across all servers, and rack-specific shifts affect all hosts in that rack identically.
- [x] **Deterministic Replay:** Verified that two independent runs with identical seed produce bitwise identical telemetry record series.
- [x] **Unit & Regression Verification:** 8 unit tests in `tests/test_telemetry.py` passed in 0.34s; 51 regression tests passed in 6.26s.

---

## 9. Prompt 3 Exit Gate Checklist & Verification

- [x] **Explicit Infrastructure vs. Release Fault Types:** Implemented in `canary_ssm/faults/models.py`. Supports infrastructure modes (`SERVER_OUTAGE`, `RACK_NETWORK_LATENCY`, `FLEET_DEMAND_SPIKE`, `CPU_SATURATION`) and release modes (`PROCESS_MEMORY_LEAK`, `ERROR_REGRESSION`, `RELEASE_LATENCY_REGRESSION`).
- [x] **Scope Rules & Isolation:** Verified that rack faults affect all servers in the target rack (controls and canaries), while release faults ONLY affect exposed instances of `target_version` and optional `susceptible_class_id`.
- [x] **No Pre-Exposure Release Effect:** Verified that before exposure tick (or before version deployment), release faults have zero effect on any server.
- [x] **Resource Pressure Coupling for Memory Leaks:** Verified that memory leak accumulates process memory over time. Normal zone (<70% capacity) has no downstream effect; Pressure zone (70-90%) elevates GC CPU and adds minor latency jitter; Thrashing zone (90-100%) triggers severe paging latency and error spikes; OOM threshold (100%) triggers simulated crash, offline downtime gap, and restart memory reset.
- [x] **Independent RNG Streams:** Disjoint sub-streams for `workload`, `noise`, `fault_scheduling`, `susceptibility`, and `missingness` using `np.random.SeedSequence`.
- [x] **Paired Null/Harm Exogenous Stream Sharing:** Verified that paired null and harm runs share bitwise identical exogenous workload and noise streams, differing solely by the injected fault effects.
- [x] **EvaluatorFaultTruthStore:** Strictly isolated from detector code. Records all active fault intervals, affected sets, outage intervals, and OOM crash events.
- [x] **Overlapping Fault Composition & Recovery:** Verified that overlapping rack latency and memory leak compose additively in log latency, and system recovers to healthy baseline when faults expire.
- [x] **Unit & Regression Verification:** 6 new unit tests in `tests/test_faults.py` passed in 0.42s; 57 regression tests passed in 7.35s with zero failures.

---

## 10. Prompt 4 Exit Gate Checklist & Verification

- [x] **Source Artifact Metadata & Unexecuted Guard:** Implemented in `canary_ssm/deployments/models.py`. Stores `artifact_filename`, SHA-256 checksum, size limits (10MB max), extension whitelist (`.py`, `.whl`, etc.). Strictly enforces `is_executed=False` and `execution_status="UNEXECUTED_SIMULATION_PROFILE"`.
- [x] **Simulation Behavior Profiles:** Fully decoupled from static source code upload. Supports `CLEAN`, `MEMORY_LEAK`, `CPU_REGRESSION`, `ERROR_REGRESSION`, and `LATENCY_REGRESSION` profiles with configurable severity, duration, and class susceptibility.
- [x] **Strict Anti-Leakage Separation:** Detector modules cannot inspect artifact contents, behavior profiles, or fault configurations.
- [x] **Identifiability Guard & Confounding Rejection:** Implemented in `canary_ssm/deployments/allocation.py`. Blocks invalid allocations:
  - Empty allocations or unknown server IDs.
  - All servers allocated as canary (0 controls).
  - Pure-canary rack (`CONFUSED_RACK_CONFOUNDING`): Entire rack running only canaries with 0 same-rack controls.
  - Pure-canary class (`CONFUSED_CLASS_CONFOUNDING`): Entire hardware class running only canaries with 0 same-class controls.
  - Default allocation algorithm selects 1 canary per rack, alternating classes across racks for balanced control representation.
- [x] **Deployment Lifecycle & Multi-Version Coexistence:** Implemented in `canary_ssm/deployments/manager.py`. Supports deploying versioned releases to designated canaries while control hosts retain baseline version. Telemetry records reflect real-time software version tag per host.
- [x] **Targeted Release Fault Injection:** Release faults generated by behavior profiles affect only exposed canaries matching version and susceptibility criteria; same-rack baseline controls remain unaffected.
- [x] **Rollback Lifecycle Semantics:** Supports `rollback_canary`:
  - `restart_process=False`: Reverts version tag without resetting heap; accumulated resident memory persists (memory does not disappear solely because version label changes).
  - `restart_process=True`: Reverts version tag and resets resident heap to baseline nominal level (process restart).
- [x] **Command Idempotency:** All create, deploy, and rollback operations track unique `command_id` values to prevent double-execution.
- [x] **Declarative Sample Manifests:** Stored in `configs/releases/`: `sample_manifest_clean.json`, `sample_manifest_memory_leak.json`, `sample_manifest_latency_regression.json`, `sample_manifest_cpu_regression.json`, `sample_manifest_error_regression.json`.
- [x] **CLI Release Commands:** `canary-ssm release --list`, `--create`, `--validate`, `--default-allocation`, `--sample-manifests`.
- [x] **Unit & Regression Verification:** 8 new unit tests in `tests/test_deployments.py` passed in 0.34s; full non-slow regression suite (140 tests) passed in 108.23s with zero failures.

---

## 11. API & Web Dashboard Verification

- [x] **SimulatorService Architecture:** Implemented in `canary_ssm/api/service.py`. Manages active `VirtualWorld`, `DeploymentManager`, background auto-step thread, and real-time multi-metric aggregation.
- [x] **FastAPI REST API:** Implemented in `canary_ssm/api/routes.py` and `canary_ssm/api/app.py`.
  - `/api/world/status`, `/api/world/control` (play, pause, step, reset, speed).
  - `/api/world/servers`, `/api/world/racks`.
  - `/api/releases` (list, create, validate, deploy, rollback).
  - `/api/faults/inject`, `/api/faults/truth` (developer fault lab view).
  - `/api/monitoring/latest`, `/api/monitoring/timeline`, `/api/monitoring/advisory`.
  - `/api/evaluation/summary`.
- [x] **5-Page Single Page Application:** Implemented in `canary_ssm/dashboard/` (`index.html`, `style.css`, `app.js`):
  1. *World:* Real-time rack cards with 24 server tiles, CPU/RAM/latency meters, online indicators, play/pause/step/reset/speed controls.
  2. *Releases:* Versioned artifact upload, unexecuted label banner, sample manifest loader, behavior profile configuration, interactive canary allocation matrix, identifiability check, deploy & rollback actions.
  3. *Fault Lab:* Developer view badge (ground truth isolated from detector), manual fault injection form, active ground truth fault history table.
  4. *Monitoring:* Dual Advisory status card, real-time Canvas charts for latency contrast, process memory with resource pressure thresholds (70%, 90%, 100% capacity), CPU fraction, request volume & error counts.
  5. *Evaluation:* Scenario truth vs. detector decision scorecard, time-to-detect metrics, CSV/JSON export buttons.
- [x] **Zero Build Dependencies:** Pure HTML5, modern dark CSS, and vanilla ES6 JS; served directly by FastAPI without node/npm/webpack.
- [x] **CLI Dashboard Subcommand:** `canary-ssm dashboard --host 127.0.0.1 --port 8000`.
- [x] **Unit & Integration Verification:** 7 unit tests in `tests/test_api.py` passed in 0.92s.

---

## 12. Concrete Acceptance Scenarios & Detection Verification

Implemented in `canary_ssm/studies/acceptance_scenarios.py` and verified in `tests/test_acceptance_scenarios.py`:

- [x] **Scenario 1 (Clean Baseline Rollout):**
  - 180 pre-exposure ticks + 120 exposure ticks with clean release deployed to 4 canaries (1 per rack).
  - Result: `InfrastructureAdvisory: NORMAL`, `CanaryReleaseAdvisory: RUNNING`, release score < threshold (0.022 < 0.40). Correct clean classification.
- [x] **Scenario 2 (Canary Memory Leak):**
  - 180 pre-exposure ticks + 120 exposure ticks with memory leak (+35 MB/min) on 2 canaries (Class 0).
  - Modeled memory accumulates (+2,090 MB over controls).
  - Result: `InfrastructureAdvisory: NORMAL`, `CanaryReleaseAdvisory: PAUSE RECOMMENDED`. Correct canary-specific fault detection.
- [x] **Scenario 3 (Rack Network Slowdown):**
  - 180 pre-exposure ticks + 120 exposure ticks with rack network latency fault (+0.30 log latency) on `rack-0` affecting all servers in `rack-0`.
  - Same-rack baseline controls estimate $\hat{\gamma}_{\text{rack-0}}(t)$, absorbing the rack nuisance.
  - Result: `InfrastructureAdvisory: RACK_DEGRADED: rack-0`, `CanaryReleaseAdvisory: RUNNING`. Avoids false pause; correctly identifies infrastructure degradation.
- [x] **Scenario 4A (Compound Faults):**
  - Both rack network latency on `rack-0` AND canary memory leak on canaries.
  - Result: `InfrastructureAdvisory: RACK_DEGRADED: rack-0`, `CanaryReleaseAdvisory: PAUSE RECOMMENDED`. Correctly reports both advisories simultaneously.
- [x] **Scenario 4B (Pure-Canary Rack Confounding):**
  - Pure-canary rack (all 6 servers in `rack-0` allocated as canaries, leaving 0 same-rack controls).
  - Allocation validator flags `CONFUSED_RACK_CONFOUNDING`.
  - Under evaluation, detector reports `CanaryReleaseAdvisory: INSUFFICIENT DATA / CONFLICT` due to unidentifiable confounding.
- [x] **Unit & Regression Verification:** 5 unit tests in `tests/test_acceptance_scenarios.py` passed in 1.19s; full regression suite (`152 passed, 1 skipped, 1 deselected in 91.94s`).

---

## 13. Prompt 6 Exit Gate Checklist & Verification (Time Slicing & Streaming Residuals Pipeline)

- [x] **Causal Streaming Time-Slice Buffer:** Implemented in `canary_ssm/telemetry/slicing.py` (`StreamSliceBuilder`). Enforces strict causal receipt: slices formed strictly at decision tick $t$ using records where $\text{arrival\_time} \le t$.
- [x] **Audit & Ingestion Policy:** Duplicate records logged and deduplicated (`DUPLICATE_DROP`), late/stale arrivals logged without mutating historical slices (`STALE_LOG_ONLY`), missing records logged and masked.
- [x] **Pre-Period Estimator & Control Stratification:** Implemented `PrePeriodBaseline.fit` operating strictly on $t < t_0$. Estimates baseline log-latency $\hat{\mu}_i$, class offsets $\hat{\beta}_{\text{cls}}$, and baseline factor loading $\hat{\lambda}_i$. Excludes all exposed canary hosts from control estimation.
- [x] **Nested SVD Fleet & Rack Factor Estimation:** Contemporaneous shared fleet shock $\hat{f}_t$ and rack nuisance shifts $\hat{\gamma}_{j, t}$ estimated on unexposed baseline controls with identifiable sum-to-zero / rank-1 SVD constraints. Carries control estimation error into residual covariance matrix: $\Sigma_{\text{ctrl}} = \frac{1}{|C_t|} \hat{\sigma}_{\text{meas}}^2 I + \dots$.
- [x] **Identifiable Coverage Rules:** Enforces strict coverage minimums (default $\ge 4$ baseline controls per rack, $\ge 1$ control per class). If a rack has zero eligible controls, release effect inference is blocked: `infer_release_cause` returns `eligible=False` and reason `INSUFFICIENT_RACK_CONTROLS`.
- [x] **Temporal Invariance Verification:** Proved mathematically and via unit tests that arriving future records cannot revise earlier residual slices.
- [x] **Unit & Regression Verification:** 13 unit tests in `tests/test_slicing.py` passed in 0.52s.

---

## 14. Prompt 7 Exit Gate Checklist & Verification (Kalman Engine & Unified Streaming Detectors)

- [x] **Generic Numerically Stable Kalman Engine:** Implemented in `canary_ssm/detectors.py`. Features Cholesky solves (`scipy.linalg.cho_factor` / `cho_solve`), Joseph stabilized covariance updates ($P = (I - KH)P(I - KH)^T + KRK^T$), subset observation masking, correlated observation noise, and exact grid-gap matrix power propagation ($F^\Delta, Q^\Delta$).
- [x] **Scalar Fixture Match:** Verified exact match to textbook scalar fixture:
  - Prior $m_0=0, P_0=1, \rho=1, q=0.1, R=0.5$:
  - Step 1 ($y_1=1$): $m_1 = 0.6875, P_1 = 0.34375$.
  - Step 2 ($y_2=1$): $m_2 = 126/151 \approx 0.834437, P_2 = 71/302 \approx 0.235099$.
- [x] **Multivariate Agreement:** Checked against statsmodels / independent linear Gaussian reference.
- [x] **Unified Streaming Detector Suite:**
  - `OnlineEWMA`: Exponentially weighted moving average with recursive variance.
  - `OnlineCUSUM`: Page's recursive cumulative sum with reference slack $k = \kappa / 2$.
  - `OnlineTopKCUSUM`: Aggregate of top-$K$ host CUSUM accumulators for sparse effect detection.
  - `IndependentHostSSM`: Independent host state-space filter ($s_i = [a_i, e_i]^T$).
  - `CompletePoolingSSM`: Complete fleet-wide pooling filter ($s = [a_{\text{pool}}, e_1, \dots, e_n]^T$).
- [x] **Unit & Regression Verification:** 15 unit tests in `tests/test_kalman_and_detectors.py` passed in 0.44s.

---

## 15. Prompt 8 Exit Gate Checklist & Verification (Hierarchical SSM & Deterministic Helmert Basis)

- [x] **Deterministic Within-Class Orthonormal Helmert Basis:** Implemented `deterministic_helmert_basis(m)` in `canary_ssm/hierarchy.py`. Verifies $B^T B = I_{m-1}$ and $C^T B = \mathbf{0}^T$ analytically. Singleton classes handled deterministically with zero dimension ($m=1 \implies B \in \mathbb{R}^{1 \times 0}$).
- [x] **Full Hierarchical State-Space Formulation:**
  - Class incidence $C \in \mathbb{R}^{n \times G}$, orthonormal Helmert basis $B = \text{blockdiag}(B_1, \dots, B_G) \in \mathbb{R}^{n \times (n-G)}$.
  - State: $s = [\theta, b, e]^T$, where $\theta \in \mathbb{R}^G$ is class effect, $b \in \mathbb{R}^{n-G}$ is within-class host deviation, and $e \in \mathbb{R}^n$ is host AR(1) noise.
  - Transition: $F = \text{blockdiag}(\rho_g I_G, \rho_h I_{n-G}, \Phi_e)$.
  - Observation: $H = [C, B, I_n]$, effect extraction: $L = [C, B, \mathbf{0}]$.
  - Implied host effect: $a = L s = C \theta + B b$.
- [x] **Posterior Covariance & Pooling Strength Sensitivity:** Computes full posterior covariance $\Sigma_a = L P L^T$ and pooling parameter $\tau = \sigma_h / \sigma_g$. Proved that $\tau \to 0$ recovers Complete Pooling, while $\tau \to \infty$ recovers Independent Host SSM.
- [x] **Four Concrete Comparative Scenarios:** Evaluated Class-Wide Harm, Sparse Host Harm, Rack Shock Nuisance, and Rollout Recovery across Hierarchical, Independent, and Pooled filters. Saved 3,120 diagnostic rows to `results/table_pooling_diagnostic_traces.csv`.
- [x] **Unit & Regression Verification:** 10 unit tests in `tests/test_hierarchy_pooling.py` passed in 0.58s.

---

## 16. Prompt 9 Exit Gate Checklist & Verification (Memory Growth, GC Dynamics & Multi-Channel Policy)

- [x] **Target Definition & Declared Boundaries:** Implemented in `canary_ssm/memory.py` (`MemoryTargetDefinition`). Declares operational boundary $\kappa_M = 0.50$ MB/min and persistence window $H = 15$ ticks. Includes mandatory epistemic disclaimer: statistical evidence of abnormal resource growth consistent with a leak profile, NOT formal verification of a source code bug.
- [x] **Realistic Healthy Telemetry Extensions:** Extended `HealthyTelemetryConfig` and `HealthyTelemetryEngine` in `canary_ssm/world/telemetry.py` with periodic GC allocation/sweep cycles (`gc_period_ticks=15, gc_amplitude_mb=300.0`) and allocator post-boot warmup ramps (`warmup_ticks=15, warmup_amplitude_mb=400.0`).
- [x] **Operational Memory Capacity Guardrail:** Implemented in `canary_ssm/memory.py` (`OperationalMemoryGuardrail`). Tracks physical headroom ($M_{\text{cap}} - M_{it}$), capacity utilization ($M_{it} / M_{\text{cap}}$), and Time-To-Exhaustion ($\text{TTE} = \text{Headroom} / \max(\hat{g}, 0.01)$). Classifies into `NORMAL`, `PRESSURE` (util $\ge 70\%$ or headroom $\le 2048$ MB or TTE $\le 120$ min), and `CRITICAL_OOM_RISK` (util $\ge 85\%$ or headroom $\le 1024$ MB or TTE $\le 30$ min).
- [x] **Causal Memory Growth Residual & MA(1) Differencing Engine:** Implemented `OnlineMemoryGrowthDetector`. Computes control-adjusted increments $r_{M, it} = \Delta M_{it} - \Delta \bar{M}_{\text{ctrl}, t}$. State-space formulation absorbs differencing-induced MA(1) error ($s_i = [\text{drift}_i, \xi_{it}, \xi_{i, t-1}]^T, H = [1, 1, -1]$) using Joseph stabilized covariance updates and prior reset at $t_0$.
- [x] **Five Operationally Verified Fixtures:** Evaluated in `canary_ssm/studies/study_memory_channel.py` and saved to `results/table_memory_fixtures_evidence.csv` (1,680 rows):
  1. *GC Cycles:* Periodic sawtooth (+300 MB every 15 min); synchronous sweeps cancelled by controls -> `false_alarm: False`, `NORMAL`.
  2. *Demand Changes:* Fleet workload surge (+35% demand); cancelled by control baseline -> `false_alarm: False`, `NORMAL`.
  3. *Allocator Warmup:* Post-boot heap expansion (+400 MB) plateauing before $t_0$ -> `false_alarm: False`, `NORMAL`.
  4. *Process Restarts:* Server reboot at tick 50; memory drop resets drift and persistence counters -> `false_alarm: False`, `restart_handled_cleanly: True`.
  5. *Persistent Growth:* Canary memory leak (+1.5 MB/min); detected at tick 54 (`detection_delay: 24` min, `final_canary0_rate: 1.19` MB/min, `persistent_leak_flagged: True`, `alert_triggered: True`, `MEMORY_GROWTH_LEAK`).
- [x] **Multi-Channel Alert Policy & Joint Null Calibration:** Implemented `MultiChannelAlertPolicy` and `calibrate_multichannel_policy` in `canary_ssm/decision.py`. Evaluates Latency Channel, Memory Channel, and Physical Guardrail into 5 mutually exclusive attributions:
  - `CRITICAL_CAPACITY_LIMIT`: Physical OOM headroom breach.
  - `COMPOUND_REGRESSION`: Both latency and persistent memory regression breached.
  - `LATENCY_REGRESSION`: Latency regression only.
  - `MEMORY_GROWTH_LEAK`: Persistent memory regression only.
  - `NORMAL`: Clean deployment.
  Joint thresholds calibrated on 80 empirical fleet null runs via Bonferroni-Šidák split ($\alpha_{\text{channel}} = 1 - \sqrt{1 - \alpha}$), satisfying total FWER budget $\le 0.05$ (`achieved_far: 0.025`).
- [x] **Unit & Regression Verification:** 9 unit tests in `tests/test_memory_channel.py` passed in 0.69s; full fast regression suite (`203 passed, 1 skipped in 111.05s`) with zero regressions.

---

## 17. Prompt 10 Exit Gate Checklist & Verification (Independent Detector Service, UI Integration & Operational Safeguards)

- [x] **Independent DetectorService decoupled from Ground Truth:** Implemented in `canary_ssm/detector_service.py`. Enforces strict anti-leakage firewall: zero imports of `simulate.Truth`, `metrics`, or truth files (verified via AST inspection in `test_firewall_rule_no_truth_imports`).
- [x] **Causal Streaming Observation Ingestion:** Consumes only incoming `TelemetryRecord` objects where `arrival_time <= decision_tick`. Resolves duplicates, masks missing records, logs stale arrivals without revising past alerts. Pre-period parameters fit on $t < t_0$ only.
- [x] **Comprehensive SSM & Multi-Channel Evidence:**
  - Latency Effect Estimates with 95% Credible Intervals: $\hat{a}_i \pm 1.96 \cdot \text{SE}(a_i)$.
  - Class-Wide Effects: $\hat{\theta}_g \pm \text{SE}(\theta_g)$.
  - Host Exceedance Risk: $\Pr(a_i > \log(1.02))$.
  - Monotone Max-Host Risk Release Score: $\max_i \log \Pr(a_i > \log(1.02))$ (explicitly documented as max-host risk, never labeled joint probability).
  - Memory Drift Rate: $\hat{g}_i \pm \text{SE}(g_i)$ MB/min with persistence counter ($H \ge 15$ min).
  - Headroom MB, Time-To-Exhaustion (TTE min), and Operational Capacity Guardrails (`NORMAL`, `PRESSURE`, `CRITICAL_OOM_RISK`).
  - Identifiable Models and Frozen Policies: `MODEL-HSSM-V1`, `POLICY-SIDAK-0.05`.
  - Data Eligibility Status & Reason Codes.
- [x] **Separation of Infrastructure Advisory from Canary Release Advisory:**
  - Evaluates infrastructure degradation ($\hat{\gamma}_j$) strictly from unexposed baseline controls.
  - Infrastructure Advisories: `NORMAL`, `RACK_DEGRADED: {racks}`, `FLEET_OUTAGE`.
  - Release Advisories: `RUNNING`, `PAUSE RECOMMENDED`, `ROLLED BACK`, `INSUFFICIENT DATA / CONFLICT`.
  - Identifiability Guard: When same-rack controls are missing (confounded pure-canary rack), blocks attribution and sets `INSUFFICIENT DATA / CONFLICT`.
- [x] **Dashboard UI Enhancements (5-Page Single Page Application):**
  - Monitoring View: Added 2 dedicated Canvas charts: (1) Latency Effect Estimate ($\hat{a}_i$) with 95% Credible Interval and harm boundary $\kappa = \log(1.02)$; (2) Memory Growth Drift Rate ($\hat{g}_i$) with $\kappa_M = 0.50$ MB/min boundary.
  - First Alert Marker: Rendered vertical dashed line (`Alert (T=...)`) on all timeline charts.
  - Server Drill-Down Modal: Clicking any server tile opens an interactive detail modal displaying contemporaneous telemetry, effect estimates, credible intervals, host risk, drift rates, headroom, TTE, and guardrail status.
  - Explicit Manual Rollback Confirmation: Rollback Release button triggers confirmation modal (`#rollback-confirm-modal`) with option to toggle process restart / heap reset.
  - Opt-In Automated Rollback (Operational Demo Mode): Checkbox in control bar enables automated rollback on persistent alarm; records intervention tick, censors detector timeline, and preserves benchmark comparison traces.
  - Disconnection Fallback: Toggling SSM off maintains telemetry visibility while reporting `DETECTOR NOT CONNECTED` / `DETECTOR UNAVAILABLE`.
- [x] **Unit & Integration Verification:**
  - 8 new unit tests in `tests/test_detector_service.py` passed in 0.75s.
  - 13 API and integration tests in `tests/test_api.py` passed in 2.11s.
  - 5 acceptance scenario tests in `tests/test_acceptance_scenarios.py` passed in 5.28s.
  - Full fast regression test suite: **213 passed, 1 skipped, 1 deselected in 90.54s** with zero failures or regressions.

---

## 18. Prompt 11 Exit Gate Checklist & Verification (Rigorous Statistical Benchmark & Final Evaluation Suite)

- [x] **Strict Seed Separation & Split Hierarchy:**
  - Implemented in `canary_ssm/evaluation/splits.py`.
  - Disjoint seed blocks with 1,000,000 stride: `smoke` (10,000), `dev` (1,000,000), `calibration` (2,000,000), `null_validation` (3,000,000), `test` (4,000,000).
  - Explicit split collision checker (`verify_seed_separation()`) preventing data leakage.
- [x] **Frozen Model & Policy Artifact Hashing:**
  - Declared `FrozenModelParameters` frozen before calibration:
    - SHA-256: `9c3c83ba2a4ff173e1dd2fb80f3056f4c20c5edb0f8654108640ae0a90cdbe2a`.
    - Persisted to `results/frozen_model_v1.json`.
- [x] **Order-Statistic Threshold Calibration & Policy Replay:**
  - Implemented in `canary_ssm/evaluation/calibration.py`.
  - Simple first-crossing detectors calibrated on full-horizon release maxima with strict `>` threshold rule: $k = \lceil (B+1)(1-\alpha) \rceil$.
  - Multi-channel combined policy calibrated by full episode replay.
- [x] **Exact Beta-Binomial Validity Gate on Independent Null Validation:**
  - Evaluated on $n=100$ unseen validation nulls with Clopper-Pearson 95% confidence intervals and exact Beta-Binomial p-value test.
  - All detectors passed the validity gate at the 5% false alert budget:
    - `hier_ssm`: FAR = 1.0% (95% CI: [0.03%, 5.45%]), p = 0.2116 -> **PASS**
    - `pooled_ssm`: FAR = 1.0% (95% CI: [0.03%, 5.45%]), p = 0.2116 -> **PASS**
    - `indep_ssm`: FAR = 3.0% (95% CI: [0.62%, 8.52%]), p = 0.7209 -> **PASS**
    - `topk_cusum`: FAR = 0.0% (95% CI: [0.00%, 3.62%]), p = 0.0594 -> **PASS**
    - `adj_cusum`: FAR = 2.0% (95% CI: [0.24%, 7.04%]), p = 0.4448 -> **PASS**
    - `adj_ewma`: FAR = 2.0% (95% CI: [0.24%, 7.04%]), p = 0.4448 -> **PASS**
    - `multi_channel`: FAR = 0.0% (95% CI: [0.00%, 3.62%]), p = 0.0594 -> **PASS**
- [x] **Empirical Demonstration: Host Risk Cutoff Fails Fleet-Wide Control:**
  - Implemented `evaluate_host_risk_vs_fleet_control()`.
  - An uncalibrated host risk cutoff (e.g. $\Pr > 0.50$) inflates fleet-wide false alerts to **13.0%** across 4 hosts over 45 ticks (and 70%–99% for cutoffs $\le 0.30$), proving that order-statistic calibration on release maxima is mathematically required.
- [x] **13 Comprehensive Evaluation Scenarios:**
  - Evaluated across 6 detector methods:
    1. S1: Sparse release harm (single canary)
    2. S2: Class-wide release harm (canaries in class 0)
    3. S3: Broad release harm (all canaries)
    4. S4: Memory-only regression (leak without latency change)
    5. S5: CPU regression (high CPU without latency change)
    6. S6: Error rate regression (elevated errors without latency change)
    7. S7: Healthy GC allocation/sweep cycles (benign sawtooth)
    8. S8: Fleet-wide load spike (fleet demand surge)
    9. S9: Infrastructure rack latency fault (common rack delay)
    10. S10: Simultaneous rack + release fault (compound shock)
    11. S11: Missingness (15% missing observations)
    12. S12: Late arrivals (stale telemetry)
    13. S13: Rollout harm followed by recovery / rollback
- [x] **Separate Event Labels & No Boundary Conflation:**
  - Implemented `MultiMetricEventTruth` with independent physical thresholds:
    - Persistent Latency Harm: $\Delta \log \text{lat} > \log(1.02)$ for $\ge 15$ ticks.
    - Persistent Memory Growth: $\Delta M / \Delta t \ge 0.50$ MB/min for $\ge 15$ ticks.
    - CPU Saturation: $\text{CPU} \ge 0.85$ for $\ge 15$ ticks.
    - Error Regression: $\text{ErrorRate} \ge 0.05$ for $\ge 15$ ticks.
  - Zero conflation or silent reuse of $\log(1.02)$ for memory or other telemetry.
- [x] **Evaluator-Only Truth Firewall:**
  - Detector code never imports `simulate.Truth` or evaluator modules.
- [x] **Full Evaluation Metrics & Negative Result Retention:**
  - Detection by Horizon: 15, 30, 45, 60 minutes.
  - Misses, Premature Alarms, Capped Time-to-Success, Median Delay.
  - Localization: Top-3 Recall = 0.939, Recall = 0.911, Precision = 0.481, F1 = 0.594, FDP = 0.519.
  - Attribution Confusion Matrix: 130 rollouts evaluated, 2 abstentions (1.5%) due to missing rack controls.
  - Paired Bootstrap Contrasts: Resampling rollout pairs (never timestamps). Hierarchical SSM statistically outperforms Top-K CUSUM ($p=0.000$), Adj CUSUM ($p=0.000$), and Adj EWMA ($p=0.000$). Negative results retained: under broad homogeneous harm (S3), complete pooling performs identically or slightly faster (mean $\Delta = 2.34$, 95% CI: [-2.13, 6.67], $p=0.313$).
- [x] **Versioned Output Artifacts & Saved CSV Tables:**
  - `results/table_null_validation.csv`
  - `results/table_evaluation_scenarios.csv`
  - `results/table_attribution_confusion.csv`
  - `results/table_localization_performance.csv`
  - `results/table_paired_bootstrap_contrasts.csv`
  - `results/VIRTUAL_DATA_CENTER_EVALUATION_REPORT.md`
- [x] **Unit & Full Test Suite Verification:**
  - 9 dedicated evaluation tests in `tests/test_rigorous_evaluation.py` passed in 17.39s.
  - Full fast regression test suite: **222 passed, 1 skipped, 1 deselected in 138.81s** with zero regressions.

---

## 19. Prompt 12 Exit Gate Checklist & Verification (Versioned Scenarios, One-Command Demo, Scaling & Limitations)

- [x] **Declarative Versioned Scenarios:**
  - Implemented in `canary_ssm/scenarios/` (`models.py`, `catalog.py`, `runner.py`, `replay.py`).
  - 8 canonical versioned scenarios:
    1. `clean_release`: Clean rollout on 4 balanced canaries with zero regressions (`quiet`).
    2. `release_leak_two_canaries`: Memory leak (+35 MB/min) on exactly two canaries (`success`, `PAUSE RECOMMENDED`).
    3. `rack_slowdown`: Network latency fault on rack-0 (`quiet`, `RACK_DEGRADED: rack-0`, release `RUNNING`).
    4. `fleet_demand_spike`: Fleet traffic surge (+28%) absorbed by baseline controls across racks (`quiet`, `RUNNING`).
    5. `cpu_regression`: Compute saturation regression (+0.40 CPU) without latency/memory harm (`miss`, honest limitation).
    6. `healthy_gc`: Cyclic garbage collection sweeps (+300 MB every 15 ticks) cancelled by controls (`quiet`, `RUNNING`).
    7. `combined_rack_fault_release_leak`: Simultaneous rack latency on rack-0 AND memory leak on canaries (`success`, dual advisories alerted).
    8. `missing_controls`: Missing observation records on rack-0 controls (`quiet`, `INSUFFICIENT DATA / CONFLICT` abstention).
  - Parameter Independence: Topology, seed, warmup, deployment, workload, and injection are set completely independently of detector parameters.
  - Declarative manifests saved to `configs/scenarios/*.json`.
- [x] **Separate Persistence of Five Distinct Artifacts:**
  - Each scenario execution persists into `results/scenarios/<name>/`:
    1. `manifest.json`: Scenario spec, topology, seed, warmup, deployment, workload, and injection.
    2. `telemetry.csv`: Full per-tick, per-server observation table (15 fields).
    3. `model_policy.json`: Model ID (`MODEL-HSSM-V1`), Policy ID (`POLICY-SIDAK-0.05`), thresholds, and priors.
    4. `recommendations.csv`: Streaming detector outputs, CIs, drift rates, guardrails, and advisories per tick.
    5. `evaluator_results.json`: Evaluator truth comparison, classification, delay, and localization.
- [x] **One-Command Local Demo Startup & Replay Workflow:**
  - Added CLI `python -m canary_ssm.cli demo` and standalone script `scripts/demo.py` (with PowerShell runner `scripts/run_demo.ps1`):
    - `python scripts/demo.py` -> Runs canonical demo scenario (`release_leak_two_canaries`).
    - `python scripts/demo.py --list` -> Lists all 8 versioned scenarios.
    - `python scripts/demo.py --all` -> Executes and persists all 8 scenarios.
    - `python scripts/demo.py --scenario <name>` -> Runs specified scenario.
    - `python scripts/demo.py --replay <path>` -> Replays and verifies bit-for-bit parity.
    - `python scripts/demo.py --serve` -> Starts interactive web dashboard preset to demo.
- [x] **Computational Fleet Scaling Benchmark:**
  - Implemented in `canary_ssm/evaluation/benchmarking.py`.
  - Benchmarked across 12, 24, 48, 96, 192, and 384 hosts:
    - 12 hosts: 0.97 ms/tick, 146.5 MB RAM, 1111 ticks/s
    - 24 hosts: 1.47 ms/tick, 151.2 MB RAM, 714 ticks/s
    - 48 hosts: 2.65 ms/tick, 160.5 MB RAM, 392 ticks/s
    - 96 hosts: 5.15 ms/tick, 179.6 MB RAM, 205 ticks/s
    - 192 hosts: 10.94 ms/tick, 217.3 MB RAM, 77.5 ticks/s
    - 384 hosts: 36.27 ms/tick, 293.8 MB RAM, 26.2 ticks/s
  - Hardware profile recorded (`AMD64 Family 25 Model 116 Stepping 1, AuthenticAMD`, physical cores, logical CPUs, total RAM GB, OS, Python version).
  - Saved to `results/runtime_scaling.csv`.
- [x] **Publication Plots Generated Strictly From Saved Tables:**
  - Implemented in `canary_ssm/evaluation/plotting.py` (`canary-ssm plot`):
    1. `results/plots/runtime_memory_scaling.png`: Scaling curves for latency and RAM vs fleet size.
    2. `results/plots/null_validation_audit.png`: Achieved FAR vs 5% target with Clopper-Pearson CIs.
    3. `results/plots/paired_bootstrap_contrasts.png`: Forest plot of Capped Time differences vs Hierarchical SSM.
    4. `results/plots/scenario_timeline_leak.png`: 3-panel time series of memory, drift score, and latency contrast with 95% CIs.
- [x] **Honest Technical Limitations Documented:**
  - Written in `docs/LIMITATIONS.md`: Pre-period stationarity, mandatory concurrent same-rack controls, host-risk cutoff inflation, decoupled multi-channel modeling, persistence window delay trade-off, low volume uncertainty, pure CPU saturation miss, discrete simulation vs datacenter physics.
- [x] **Preserved Original Benchmark Configs:**
  - Original benchmark configs (`configs/smoke.yaml`, `configs/plain_main.yaml`, `configs/audit.yaml`, `configs/stress.yaml`, `configs/grids_v2.yaml`) preserved unmodified alongside `configs/scenarios/`.
- [x] **Exit Gate Verified:**
  - `test_exit_gate_reproducibility` verifies that a fresh local run reproduces telemetry and detector outputs with exact parity (`reproduced=True`, `telemetry_matches=True`, `advisories_match=True`) across all 8 scenarios.
  - Dedicated reproducibility test suite (`tests/test_scenario_reproducibility.py`): 7 passed in 6.75s.
  - Full fast regression test suite: **229 passed, 1 skipped, 1 deselected in 104.39s** with zero regressions.





