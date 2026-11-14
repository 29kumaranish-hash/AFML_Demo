"""Independent Detector Service for Virtual Data Center (Prompt 10).

Strict Anti-Leakage, Epistemic Honesty, and Identifiability Guarantees:
1. Complete Firewall: This service NEVER imports simulation or evaluator truth stores or metrics.
2. Causality & Streaming Parity: Observations are consumed tick-by-tick using ONLY records with
   arrival_time <= decision_tick. Future observation revisions cannot modify past alerts.
3. Latency Channel: Online Hierarchical State-Space Model with deterministic Helmert contrast basis,
   proper prior reset at deployment onset (t0), Cholesky solves, and Joseph-stabilized updates.
   Extracts host effects (a_i +/- 1.96 * SE), class effects (theta_g +/- 1.96 * SE),
   host harm risk Pr(a_i > log(1.02)), and release score max_i log Pr.
4. Memory Channel: Online differencing MA(1) Kalman growth detector with persistence gating
   and operational capacity guardrails (headroom MB, Time-To-Exhaustion min).
5. Identifiability Guard: If concurrent same-rack baseline controls are absent or insufficient,
   strictly blocks release cause inference and emits "INSUFFICIENT DATA / CONFLICT".
6. Infrastructure vs Release Separation: Rack/fleet nuisances are estimated from same-rack
   baseline controls, producing separate Infrastructure Advisories (NORMAL, RACK_DEGRADED, FLEET_OUTAGE)
   without confounding release effects.
7. Frozen Policy: Uses frozen calibrated thresholds (e.g. POLICY-SIDAK-0.05) rather than oracle truth
   or simulation behavior profiles.
8. Operational Safety: Manual rollback requires explicit confirmation. Automatic simulated rollback
   is opt-in only, preserving benchmark traces and declaring censoring upon intervention.
"""
from __future__ import annotations

import collections
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve
from scipy.stats import norm

from canary_ssm.decision import MultiChannelAlertPolicy
from canary_ssm.hierarchy import EffectParams, class_structure
from canary_ssm.kalman import symmetrize
from canary_ssm.memory import (
    MemoryStepResult,
    MemoryTargetDefinition,
    OnlineMemoryGrowthDetector,
    OperationalMemoryGuardrail,
)
from canary_ssm.schema import KAPPA
from canary_ssm.telemetry.records import TelemetryRecord
from canary_ssm.telemetry.slicing import (
    PrePeriodBaseline,
    SlicingProtocol,
    StreamSliceBuilder,
    TimeSlice,
    fit_preperiod_baseline,
    infer_release_cause,
)


@dataclass(frozen=True)
class FrozenPolicy:
    """Frozen decision policy calibrated on healthy null data (FWER <= 0.05)."""
    policy_id: str = "POLICY-SIDAK-0.05"
    model_id: str = "MODEL-HSSM-V1"
    alpha: float = 0.05
    threshold_lat: float = -0.693147        # log(0.50): max-host risk > 50%
    threshold_mem: float = -0.693147        # log(0.50): max-host drift exceedance > 50%
    require_persistence: bool = True
    persistence_window: int = 10            # Consecutive ticks above kappa_M
    kappa_lat: float = float(KAPPA)         # log(1.02) ~ 0.0198026
    kappa_mem: float = 0.50                 # 0.50 MB/min
    epistemic_disclaimer: str = (
        "Statistical evidence of abnormal metric shift relative to matched baseline controls. "
        "Observational evidence is consistent with a release regression, not proof of a code bug."
    )


@dataclass
class HostEvidence:
    """Host-level evidence breakdown for UI drill-down and audit."""
    server_id: str
    role: str                               # "CANARY" | "CONTROL"
    rack_id: str
    class_id: int
    # Latency effect estimates & uncertainty
    latency_effect: float | None = None
    latency_se: float | None = None
    latency_ci_lower: float | None = None
    latency_ci_upper: float | None = None
    host_risk: float | None = None          # Pr(a_i > kappa_lat)
    host_log_score: float | None = None     # log Pr(a_i > kappa_lat)
    # Memory growth & capacity guardrail
    memory_drift_mb_per_min: float | None = None
    memory_drift_se: float | None = None
    headroom_mb: float | None = None
    tte_min: float | None = None
    guardrail_status: str = "NORMAL"        # "NORMAL" | "PRESSURE" | "CRITICAL_OOM_RISK"
    is_persistent_mem: bool = False
    # Contemporaneous telemetry observations
    latency_ms: float = 0.0
    memory_mb: float = 0.0
    cpu_fraction: float = 0.0
    error_count: int = 0
    request_count: int = 0
    observed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "server_id": self.server_id,
            "role": self.role,
            "rack_id": self.rack_id,
            "class_id": self.class_id,
            "latency_effect": round(self.latency_effect, 5) if self.latency_effect is not None else None,
            "latency_se": round(self.latency_se, 5) if self.latency_se is not None else None,
            "latency_ci_lower": round(self.latency_ci_lower, 5) if self.latency_ci_lower is not None else None,
            "latency_ci_upper": round(self.latency_ci_upper, 5) if self.latency_ci_upper is not None else None,
            "host_risk": round(self.host_risk, 4) if self.host_risk is not None else None,
            "host_log_score": round(self.host_log_score, 4) if self.host_log_score is not None else None,
            "memory_drift_mb_per_min": round(self.memory_drift_mb_per_min, 3) if self.memory_drift_mb_per_min is not None else None,
            "memory_drift_se": round(self.memory_drift_se, 3) if self.memory_drift_se is not None else None,
            "headroom_mb": round(self.headroom_mb, 1) if self.headroom_mb is not None else None,
            "tte_min": round(self.tte_min, 1) if self.tte_min is not None and np.isfinite(self.tte_min) else None,
            "guardrail_status": self.guardrail_status,
            "is_persistent_mem": self.is_persistent_mem,
            "latency_ms": round(self.latency_ms, 2),
            "memory_mb": round(self.memory_mb, 1),
            "cpu_fraction": round(self.cpu_fraction, 4),
            "error_count": self.error_count,
            "request_count": self.request_count,
            "observed": self.observed,
        }


@dataclass
class ClassEvidence:
    """Class-level effect breakdown under hierarchical model."""
    class_id: int
    theta: float
    theta_se: float
    ci_lower: float
    ci_upper: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "theta": round(self.theta, 5),
            "theta_se": round(self.theta_se, 5),
            "ci_lower": round(self.ci_lower, 5),
            "ci_upper": round(self.ci_upper, 5),
        }


@dataclass
class DetectorTickResult:
    """Full detector evaluation result at a single discrete decision timestamp."""
    tick: int
    t0: int | None
    eligible: bool
    is_unidentifiable: bool
    ineligible_reasons: list[str]
    attribution_blocked_reasons: list[str]
    infrastructure_advisory: str            # NORMAL | RACK_DEGRADED: {racks} | FLEET_OUTAGE
    canary_release_advisory: str            # RUNNING | PAUSE RECOMMENDED | CRITICAL_OOM_RISK | INSUFFICIENT DATA / CONFLICT | ROLLED BACK
    attribution: str                        # NORMAL | LATENCY_REGRESSION | MEMORY_GROWTH_LEAK | COMPOUND_REGRESSION | CRITICAL_CAPACITY_LIMIT | INSUFFICIENT_DATA | ROLLED_BACK
    advisory_reason: str
    epistemic_disclaimer: str
    # Scores & Thresholds
    release_score: float | None
    score_lat: float | None
    score_mem: float | None
    threshold_lat: float
    threshold_mem: float
    first_alert_tick: int | None
    # Breakdowns
    hosts: dict[str, HostEvidence]
    class_effects: list[ClassEvidence]
    # Matched series averages for plotting
    canary_latency_mean: float
    control_latency_mean: float
    canary_memory_mean: float
    control_memory_mean: float
    canary_cpu_mean: float
    control_cpu_mean: float
    canary_errors_sum: int
    control_errors_sum: int
    canary_effect_mean: float
    canary_effect_ci_lower: float
    canary_effect_ci_upper: float
    # Operational intervention metadata
    auto_rollback_triggered: bool = False
    intervention_tick: int | None = None
    censored: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "t0": self.t0,
            "eligible": self.eligible,
            "is_unidentifiable": self.is_unidentifiable,
            "ineligible_reasons": self.ineligible_reasons,
            "attribution_blocked_reasons": self.attribution_blocked_reasons,
            "infrastructure_advisory": self.infrastructure_advisory,
            "canary_release_advisory": self.canary_release_advisory,
            "attribution": self.attribution,
            "advisory_reason": self.advisory_reason,
            "epistemic_disclaimer": self.epistemic_disclaimer,
            "release_score": round(self.release_score, 4) if self.release_score is not None else None,
            "score_lat": round(self.score_lat, 4) if self.score_lat is not None else None,
            "score_mem": round(self.score_mem, 4) if self.score_mem is not None else None,
            "threshold_lat": self.threshold_lat,
            "threshold_mem": self.threshold_mem,
            "first_alert_tick": self.first_alert_tick,
            "hosts": {k: v.to_dict() for k, v in self.hosts.items()},
            "class_effects": [c.to_dict() for c in self.class_effects],
            "canary_latency_mean": round(self.canary_latency_mean, 2),
            "control_latency_mean": round(self.control_latency_mean, 2),
            "canary_memory_mean": round(self.canary_memory_mean, 1),
            "control_memory_mean": round(self.control_memory_mean, 1),
            "canary_cpu_mean": round(self.canary_cpu_mean, 4),
            "control_cpu_mean": round(self.control_cpu_mean, 4),
            "canary_errors_sum": self.canary_errors_sum,
            "control_errors_sum": self.control_errors_sum,
            "canary_effect_mean": round(self.canary_effect_mean, 5),
            "canary_effect_ci_lower": round(self.canary_effect_ci_lower, 5),
            "canary_effect_ci_upper": round(self.canary_effect_ci_upper, 5),
            "auto_rollback_triggered": self.auto_rollback_triggered,
            "intervention_tick": self.intervention_tick,
            "censored": self.censored,
        }


class OnlineHierarchicalLatencyFilter:
    """Streaming Hierarchical Latency Kalman Filter with Joseph updates and Helmert basis."""

    def __init__(
        self,
        canary_ids: list[str],
        canary_classes: list[int],
        params: EffectParams | None = None,
        phi_e: float = 0.85,
        q_e: float = 0.015,
        v_meas: float = 0.005,
        kappa: float = float(KAPPA),
    ) -> None:
        self.canary_ids = list(canary_ids)
        self.canary_classes = np.asarray(canary_classes, int)
        self.nc = len(self.canary_ids)
        self.p = params or EffectParams(kind="hier", rho_g=0.95, rho_h=0.95, sd_g=0.005, sd_h_ratio=1.0, prior_sd=0.03)
        self.phi_e = float(phi_e)
        self.q_e = float(q_e)
        self.v_meas = float(v_meas)
        self.kappa = float(kappa)

        # Build class structure matrices C and Helmert basis B
        self.C, self.B = class_structure(self.canary_classes)
        self.G = self.C.shape[1]
        self.d = self.C.shape[1] + self.B.shape[1]
        self.L = np.hstack([self.C, self.B]) if self.B.shape[1] > 0 else self.C

        # Dimension: effect states d + AR noise states nc
        self.dim = self.d + self.nc

        # Transition matrix F
        rho_eff = np.r_[np.full(self.G, self.p.rho_g), np.full(self.B.shape[1], self.p.rho_h)]
        self.F = block_diag(np.diag(rho_eff), np.diag(np.full(self.nc, self.phi_e)))

        # Process noise Q
        sd_eff = np.r_[np.full(self.G, self.p.sd_g), np.full(self.B.shape[1], self.p.sd_g * self.p.sd_h_ratio)]
        self.Q = block_diag(np.diag(sd_eff ** 2), np.diag(np.full(self.nc, self.q_e ** 2)))

        # Observation matrix H: y_adj = L * s_eff + e_t + nu_t
        self.H = np.hstack([self.L, np.eye(self.nc)])

        # Effect extraction matrix: a_hat = L * s_eff
        self.L_full = np.hstack([self.L, np.zeros((self.nc, self.nc))])

        # State vector and covariance
        self.m = np.zeros(self.dim, dtype=float)
        # Proper prior: zero mean, prior_sd^2 on effects, stationary variance on AR noise
        var_ar = self.q_e ** 2 / max(1e-6, 1.0 - self.phi_e ** 2)
        P_eff = np.eye(self.d) * (self.p.prior_sd ** 2)
        P_ar = np.eye(self.nc) * var_ar
        self.P = symmetrize(block_diag(P_eff, P_ar))

    def reset_prior(self, pre_ar_state: np.ndarray | None = None, pre_ar_cov: np.ndarray | None = None) -> None:
        """Reset effect prior to zero with proper variance while carrying AR noise state forward."""
        self.m[:self.d] = 0.0
        if pre_ar_state is not None:
            self.m[self.d:] = pre_ar_state
        self.P[:self.d, :self.d] = np.eye(self.d) * (self.p.prior_sd ** 2)
        self.P[:self.d, self.d:] = 0.0
        self.P[self.d:, :self.d] = 0.0
        if pre_ar_cov is not None:
            self.P[self.d:, self.d:] = pre_ar_cov
        self.P = symmetrize(self.P)

    def step(
        self,
        r_latency: np.ndarray | None = None,
        obs_mask: np.ndarray | None = None,
        cov_U: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Causal step update.

        Returns:
            (a_hat, a_se, class_thetas, class_thetas_se, log_release_score)
        """
        # Step 1: Prediction
        self.m = self.F @ self.m
        self.P = symmetrize(self.F @ self.P @ self.F.T + self.Q)

        # Step 2: Measurement update if observations are available
        has_obs = (r_latency is not None and obs_mask is not None and np.any(obs_mask))
        if has_obs:
            idx = np.flatnonzero(obs_mask)
            y_obs = r_latency[idx]
            H_obs = self.H[idx, :]

            # Measurement covariance V_obs
            V_base = np.eye(self.nc) * (self.v_meas ** 2)
            if cov_U is not None and np.all(np.isfinite(cov_U)) and cov_U.shape == (self.nc, self.nc):
                V_total = V_base + cov_U
            else:
                V_total = V_base
            V_obs = V_total[np.ix_(idx, idx)]

            # Innovation & Joseph update
            v = y_obs - H_obs @ self.m
            PHt = self.P @ H_obs.T
            S = symmetrize(H_obs @ PHt + V_obs)

            try:
                cf = cho_factor(S, lower=True, check_finite=False)
                K = cho_solve(cf, PHt.T, check_finite=False).T
                self.m = self.m + K @ v
                IKH = np.eye(self.dim) - K @ H_obs
                self.P = symmetrize(IKH @ self.P @ IKH.T + K @ V_obs @ K.T)
            except Exception:
                # Numerical safeguard: fallback to pseudo-inverse if Cholesky fails
                K = PHt @ np.linalg.pinv(S)
                self.m = self.m + K @ v
                IKH = np.eye(self.dim) - K @ H_obs
                self.P = symmetrize(IKH @ self.P @ IKH.T + K @ V_obs @ K.T)

        # Extract posterior host effect estimates
        a_hat = self.L_full @ self.m
        cov_a = self.L @ self.P[:self.d, :self.d] @ self.L.T
        var_a = np.maximum(np.diag(cov_a), 1e-12)
        a_se = np.sqrt(var_a)

        # Extract class effect estimates
        class_thetas = self.m[:self.G].copy()
        class_thetas_se = np.sqrt(np.maximum(np.diag(self.P[:self.G, :self.G]), 1e-12))

        # Compute host harm risks Pr(a_i > kappa_lat) and log max-host release score
        z = (a_hat - self.kappa) / a_se
        probs = norm.sf(-z)  # sf(-z) = Phi(z)
        log_probs = norm.logsf(-z)
        log_release_score = float(np.max(log_probs)) if len(log_probs) else float("-inf")

        return a_hat, a_se, class_thetas, class_thetas_se, log_release_score


class DetectorService:
    """Decoupled Detector Service consuming streaming TelemetryRecords.

    Maintains:
    - Pre-period baseline fitting (strictly on pre-exposure ticks).
    - StreamSliceBuilder for causal time-slicing and nested SVD adjustments.
    - Online Hierarchical SSM for latency regressions.
    - Online differencing MA(1) growth detector for memory regressions.
    - Operational memory capacity guardrail (headroom, TTE).
    - Bonferroni-Šidák multi-channel alert policy with frozen thresholds.
    - Separate Infrastructure Advisory engine (same-rack baseline controls).
    - Complete audit and timeline history for UI visualization and replay.
    """

    def __init__(
        self,
        expected_servers: list[str],
        server_metadata: dict[str, dict[str, Any]],
        frozen_policy: FrozenPolicy | None = None,
        slicing_protocol: SlicingProtocol | None = None,
        effect_params: EffectParams | None = None,
        memory_target: MemoryTargetDefinition | None = None,
        ssm_connected: bool = True,
        enable_auto_rollback: bool = False,
    ) -> None:
        self.expected_servers = list(expected_servers)
        self.server_metadata = dict(server_metadata)
        self.frozen_policy = frozen_policy or FrozenPolicy()
        self.protocol = slicing_protocol or SlicingProtocol(min_controls_per_rack=3, min_controls_per_class=3)
        self.effect_params = effect_params or EffectParams(kind="hier", rho_g=0.95, rho_h=0.95, sd_g=0.005, sd_h_ratio=1.0)
        self.memory_target = memory_target or MemoryTargetDefinition()

        self.ssm_connected = bool(ssm_connected)
        self.enable_auto_rollback = bool(enable_auto_rollback)

        # Policy & Slice Builder
        self.policy = MultiChannelAlertPolicy(
            threshold_lat=self.frozen_policy.threshold_lat,
            threshold_mem=self.frozen_policy.threshold_mem,
            require_persistence=self.frozen_policy.require_persistence,
            alpha=self.frozen_policy.alpha,
        )
        self.slice_builder = StreamSliceBuilder(
            expected_servers=self.expected_servers,
            server_metadata=self.server_metadata,
            protocol=self.protocol,
        )

        # Deployment & Canary State
        self.active_release_id: str | None = None
        self.active_canary_ids: list[str] = []
        self.t0: int | None = None
        self.is_rolled_back: bool = False
        self.rollback_tick: int | None = None

        # Detectors (initialized upon deployment)
        self.lat_filter: OnlineHierarchicalLatencyFilter | None = None
        self.mem_detector: OnlineMemoryGrowthDetector | None = None
        self.guardrail = OperationalMemoryGuardrail(self.memory_target)

        # Streaming Record Ingestion Buffer
        self.raw_records: list[TelemetryRecord] = []
        self.latest_telemetry_by_server: dict[str, TelemetryRecord] = {}

        # First alert tracking
        self.first_alert_tick: int | None = None

        # Timeline Results History
        self.history: list[DetectorTickResult] = []

        # Operational intervention tracking
        self.auto_rollback_triggered: bool = False
        self.intervention_tick: int | None = None

    # -----------------------------------------------------------------
    # Configuration & Deployment Hooks
    # -----------------------------------------------------------------
    def set_ssm_connected(self, connected: bool) -> bool:
        """Toggle connection to SSM detector endpoints."""
        self.ssm_connected = bool(connected)
        return self.ssm_connected

    def set_auto_rollback(self, enabled: bool) -> bool:
        """Toggle opt-in automated simulated rollback (operational demo mode)."""
        self.enable_auto_rollback = bool(enabled)
        return self.enable_auto_rollback

    def set_deployment(self, release_id: str, canary_ids: list[str], t0: int) -> None:
        """Notify detector service of an active canary deployment at tick t0.

        Strict Causality:
        Fits PrePeriodBaseline using ONLY records received before t0 (event_time < t0).
        """
        self.active_release_id = release_id
        self.active_canary_ids = list(canary_ids)
        self.t0 = int(t0)
        self.is_rolled_back = False
        self.rollback_tick = None
        self.first_alert_tick = None
        self.auto_rollback_triggered = False
        self.intervention_tick = None

        # Fit pre-period baseline if not already fitted
        pre_records = [r for r in self.raw_records if r.event_time < self.t0 and r.arrival_time <= self.t0]
        if pre_records:
            baseline = fit_preperiod_baseline(
                records=pre_records,
                t0=self.t0,
                expected_servers=self.expected_servers,
                protocol=self.protocol,
            )
            self.slice_builder.baseline = baseline

        # Initialize Latency & Memory Detectors for active canaries
        canary_classes = [self.server_metadata[cid].get("class_id", 0) for cid in self.active_canary_ids]
        canary_caps = [float(self.server_metadata[cid].get("memory_capacity_mb", 16384.0)) for cid in self.active_canary_ids]
        ctrl_ids = [s for s in self.expected_servers if s not in self.active_canary_ids]

        self.memory_target = MemoryTargetDefinition(
            persistence_window_ticks=self.frozen_policy.persistence_window,
            growth_boundary_mb_per_min=self.frozen_policy.kappa_mem,
        )
        self.lat_filter = OnlineHierarchicalLatencyFilter(
            canary_ids=self.active_canary_ids,
            canary_classes=canary_classes,
            params=self.effect_params,
            kappa=self.frozen_policy.kappa_lat,
        )
        self.mem_detector = OnlineMemoryGrowthDetector(
            t0=self.t0,
            canary_ids=self.active_canary_ids,
            control_ids=ctrl_ids,
            canary_capacities_mb=canary_caps,
            target_definition=self.memory_target,
        )

    def set_rollback(self, tick: int) -> None:
        """Record rollback event. Freezes further canary release alerts."""
        self.is_rolled_back = True
        self.rollback_tick = int(tick)
        self.active_release_id = None
        self.active_canary_ids.clear()

    # -----------------------------------------------------------------
    # Telemetry Ingestion & Tick Evaluation
    # -----------------------------------------------------------------
    def ingest(self, records: list[TelemetryRecord]) -> None:
        """Ingest new TelemetryRecords arriving from the world or network."""
        for r in records:
            self.raw_records.append(r)
            self.latest_telemetry_by_server[r.server_id] = r
        self.slice_builder.ingest(records)

    def step(self, decision_tick: int) -> DetectorTickResult:
        """Process one discrete decision timestamp causally.

        Guarantees:
        - Only records with arrival_time <= decision_tick are admitted.
        - Physical rack nuisances are estimated from baseline controls.
        - Latency effect extraction isolates release effect with uncertainty intervals.
        - Memory drift and capacity guardrails are evaluated.
        - If rack controls are absent, release cause inference is strictly prohibited.
        """
        # Contemporaneous telemetry records
        contemp_recs = {
            r.server_id: r for r in self.raw_records
            if r.event_time == decision_tick and r.arrival_time <= decision_tick
        }

        # Step 0: Ensure baseline exists for slice builder if we have enough pre-period records
        if self.slice_builder.baseline is None:
            pre_records = [r for r in self.raw_records if r.arrival_time <= decision_tick]
            if len(pre_records) >= len(self.expected_servers):
                t0_fit = decision_tick + 1 if not self.active_canary_ids else (self.t0 or (decision_tick + 1))
                try:
                    self.slice_builder.baseline = fit_preperiod_baseline(
                        records=pre_records,
                        t0=t0_fit,
                        expected_servers=self.expected_servers,
                        protocol=self.protocol,
                    )
                except Exception:
                    pass

        # If baseline is still None or no active canary, evaluate infrastructure and return null result
        if self.slice_builder.baseline is None or not self.active_canary_ids:
            infra_adv, infra_reason = self._evaluate_infrastructure_advisory(None, contemp_recs)
            release_adv = "ROLLED BACK" if self.is_rolled_back else "RUNNING"
            attr = "ROLLED_BACK" if self.is_rolled_back else "NORMAL"
            res = self._build_null_result(
                tick=decision_tick,
                infra_adv=infra_adv,
                infra_reason=infra_reason,
                release_adv=release_adv,
                attribution=attr,
                contemp_recs=contemp_recs,
                time_slice=None,
            )
            self.history.append(res)
            return res

        # Step 1: Form contemporaneous TimeSlice
        time_slice = self.slice_builder.build_slice(decision_tick)

        # Step 2: Separate Infrastructure Advisory (From same-rack baseline controls)
        infra_adv, infra_reason = self._evaluate_infrastructure_advisory(time_slice, contemp_recs)

        # Step 3: Handle No Active Canary / Post-Rollback State
        if not self.active_canary_ids:
            release_adv = "ROLLED BACK" if self.is_rolled_back else "RUNNING"
            attr = "ROLLED_BACK" if self.is_rolled_back else "NORMAL"
            res = self._build_null_result(
                tick=decision_tick,
                infra_adv=infra_adv,
                infra_reason=infra_reason,
                release_adv=release_adv,
                attribution=attr,
                contemp_recs=contemp_recs,
                time_slice=time_slice,
            )
            self.history.append(res)
            return res

        # Step 4: Identifiability Guard & Control Coverage Check
        can_infer, ident_reason = infer_release_cause(time_slice)
        if not can_infer or time_slice.is_unidentifiable:
            res = self._build_unidentifiable_result(
                tick=decision_tick,
                infra_adv=infra_adv,
                ident_reason=ident_reason,
                contemp_recs=contemp_recs,
                time_slice=time_slice,
            )
            self.history.append(res)
            return res

        # Step 5: Latency Channel Update
        nc = len(self.active_canary_ids)
        r_lat_canaries = np.zeros(nc, dtype=float)
        mask_lat_canaries = np.zeros(nc, dtype=bool)
        for i, cid in enumerate(self.active_canary_ids):
            if cid in time_slice.canary_ids:
                s_idx = time_slice.canary_ids.index(cid)
                r_lat_canaries[i] = time_slice.r_latency[s_idx]
                mask_lat_canaries[i] = time_slice.mask_latency[s_idx]

        cov_canaries = np.zeros((nc, nc), dtype=float)
        if hasattr(time_slice, "cov_U") and time_slice.cov_U is not None and time_slice.cov_U.size > 0:
            for i, cid_i in enumerate(self.active_canary_ids):
                if cid_i in time_slice.canary_ids:
                    idx_i = time_slice.canary_ids.index(cid_i)
                    for j, cid_j in enumerate(self.active_canary_ids):
                        if cid_j in time_slice.canary_ids:
                            idx_j = time_slice.canary_ids.index(cid_j)
                            cov_canaries[i, j] = time_slice.cov_U[idx_i, idx_j]

        a_hat, a_se, class_thetas, class_thetas_se, score_lat = self.lat_filter.step(
            r_latency=r_lat_canaries,
            obs_mask=mask_lat_canaries,
            cov_U=cov_canaries,
        )

        # Step 6: Memory Channel Update
        ctrl_recs = {
            s: contemp_recs[s] for s in self.expected_servers
            if s not in self.active_canary_ids and s in contemp_recs
        }
        canary_recs = {
            s: contemp_recs[s] for s in self.active_canary_ids if s in contemp_recs
        }
        mem_res: MemoryStepResult = self.mem_detector.step(
            tick=decision_tick,
            canary_records=canary_recs,
            control_records=ctrl_recs,
            eligible=time_slice.eligible,
        )
        score_mem = float(mem_res.score) if np.isfinite(mem_res.score) else float("-inf")

        # Step 7: Multi-Channel Policy Decision & Attribution
        policy_eval = self.policy.evaluate_step(
            score_lat=score_lat,
            score_mem=score_mem,
            guardrail_status=mem_res.guardrail_status,
            is_persistent_mem=mem_res.is_persistent_leak,
            eligible=time_slice.eligible,
        )

        canary_release_adv = policy_eval["advisory"]
        attribution = policy_eval["attribution"]

        # Track first alert timestamp
        if policy_eval["is_alert"] and self.first_alert_tick is None:
            self.first_alert_tick = decision_tick

        # Formulate epistemic evidence text without claiming causal certainty
        advisory_reason = self._build_epistemic_reason(
            policy_eval=policy_eval,
            a_hat=a_hat,
            a_se=a_se,
            mem_res=mem_res,
            infra_reason=infra_reason,
        )

        # Check automated simulated rollback (Opt-in operational mode)
        auto_rollback_now = False
        if self.enable_auto_rollback and policy_eval["is_alert"] and not self.auto_rollback_triggered:
            self.auto_rollback_triggered = True
            self.intervention_tick = decision_tick
            auto_rollback_now = True

        # Build HostEvidence and ClassEvidence records
        hosts_evidence = self._build_host_evidence(
            contemp_recs=contemp_recs,
            a_hat=a_hat,
            a_se=a_se,
            mem_res=mem_res,
        )
        class_evidence = [
            ClassEvidence(
                class_id=int(self.lat_filter.canary_classes[g] if g < len(self.lat_filter.canary_classes) else g),
                theta=float(class_thetas[g]),
                theta_se=float(class_thetas_se[g]),
                ci_lower=float(class_thetas[g] - 1.96 * class_thetas_se[g]),
                ci_upper=float(class_thetas[g] + 1.96 * class_thetas_se[g]),
            )
            for g in range(len(class_thetas))
        ]

        # Aggregate series stats for charting
        canary_obs = [contemp_recs[s] for s in self.active_canary_ids if s in contemp_recs and contemp_recs[s].observed]
        ctrl_obs = [contemp_recs[s] for s in self.expected_servers if s not in self.active_canary_ids and s in contemp_recs and contemp_recs[s].observed]

        mean_c_eff = float(np.mean(a_hat)) if len(a_hat) else 0.0
        mean_c_se = float(np.mean(a_se)) if len(a_se) else 0.0

        tick_result = DetectorTickResult(
            tick=decision_tick,
            t0=self.t0,
            eligible=time_slice.eligible,
            is_unidentifiable=False,
            ineligible_reasons=list(time_slice.ineligible_reasons),
            attribution_blocked_reasons=[],
            infrastructure_advisory=infra_adv,
            canary_release_advisory=canary_release_adv,
            attribution=attribution,
            advisory_reason=advisory_reason,
            epistemic_disclaimer=self.frozen_policy.epistemic_disclaimer,
            release_score=float(max(score_lat, score_mem)),
            score_lat=score_lat,
            score_mem=score_mem,
            threshold_lat=self.frozen_policy.threshold_lat,
            threshold_mem=self.frozen_policy.threshold_mem,
            first_alert_tick=self.first_alert_tick,
            hosts=hosts_evidence,
            class_effects=class_evidence,
            canary_latency_mean=float(np.mean([r.latency_ms for r in canary_obs])) if canary_obs else 0.0,
            control_latency_mean=float(np.mean([r.latency_ms for r in ctrl_obs])) if ctrl_obs else 0.0,
            canary_memory_mean=float(np.mean([r.memory_mb for r in canary_obs])) if canary_obs else 0.0,
            control_memory_mean=float(np.mean([r.memory_mb for r in ctrl_obs])) if ctrl_obs else 0.0,
            canary_cpu_mean=float(np.mean([r.cpu_fraction for r in canary_obs])) if canary_obs else 0.0,
            control_cpu_mean=float(np.mean([r.cpu_fraction for r in ctrl_obs])) if ctrl_obs else 0.0,
            canary_errors_sum=int(sum(r.error_count for r in canary_obs)),
            control_errors_sum=int(sum(r.error_count for r in ctrl_obs)),
            canary_effect_mean=mean_c_eff,
            canary_effect_ci_lower=mean_c_eff - 1.96 * mean_c_se,
            canary_effect_ci_upper=mean_c_eff + 1.96 * mean_c_se,
            auto_rollback_triggered=self.auto_rollback_triggered,
            intervention_tick=self.intervention_tick,
            censored=self.auto_rollback_triggered,
        )

        self.history.append(tick_result)
        return tick_result

    # -----------------------------------------------------------------
    # Advisory Helpers & Epistemic Honesty
    # -----------------------------------------------------------------
    def _evaluate_infrastructure_advisory(
        self,
        time_slice: TimeSlice | None,
        contemp_recs: dict[str, TelemetryRecord],
    ) -> tuple[str, str]:
        """Separately evaluate physical rack and fleet infrastructure nuisances."""
        degraded_racks: list[str] = []
        racks_seen = set()

        for sid, rec in contemp_recs.items():
            if sid not in self.active_canary_ids and rec.observed:
                racks_seen.add(rec.rack_id)

        # Check estimated rack nuisances from TimeSlice (gamma_hat_j) if available
        if time_slice is not None and time_slice.gamma_hat:
            for rk, gamma_val in time_slice.gamma_hat.items():
                if gamma_val > 0.12:  # > +13% log-latency shift on baseline controls
                    degraded_racks.append(rk)
        else:
            # Direct contrast from contemporaneous control observations
            ctrl_recs = [r for sid, r in contemp_recs.items() if sid not in self.active_canary_ids and r.observed]
            if ctrl_recs:
                fleet_mean_log = float(np.mean([math.log(max(1.0, r.latency_ms)) for r in ctrl_recs]))
                for rk in racks_seen:
                    rk_ctrls = [r for r in ctrl_recs if r.rack_id == rk]
                    if rk_ctrls:
                        rk_mean_log = float(np.mean([math.log(max(1.0, r.latency_ms)) for r in rk_ctrls]))
                        if rk_mean_log - fleet_mean_log > 0.12:
                            degraded_racks.append(rk)

        if len(degraded_racks) >= len(racks_seen) and len(racks_seen) > 1:
            return "FLEET_OUTAGE", f"Fleet-wide degradation detected across all {len(degraded_racks)} racks."
        elif degraded_racks:
            return f"RACK_DEGRADED: {', '.join(degraded_racks)}", f"Infrastructure slowdown detected on {', '.join(degraded_racks)}."
        else:
            return "NORMAL", "Infrastructure telemetry within nominal bounds."

    def _build_epistemic_reason(
        self,
        policy_eval: dict[str, Any],
        a_hat: np.ndarray,
        a_se: np.ndarray,
        mem_res: MemoryStepResult,
        infra_reason: str,
    ) -> str:
        """Construct user-facing explanation preserving epistemic honesty."""
        attr = policy_eval["attribution"]
        if attr == "NORMAL":
            return f"Canary telemetry healthy and matched to concurrent controls. | {infra_reason}"

        details: list[str] = []
        if policy_eval["lat_breach"]:
            max_idx = int(np.argmax(a_hat))
            eff_pct = (np.exp(a_hat[max_idx]) - 1.0) * 100.0
            ci_lo = (np.exp(a_hat[max_idx] - 1.96 * a_se[max_idx]) - 1.0) * 100.0
            ci_hi = (np.exp(a_hat[max_idx] + 1.96 * a_se[max_idx]) - 1.0) * 100.0
            details.append(
                f"Elevated latency contrast on {self.active_canary_ids[max_idx]} (+{eff_pct:.1f}%, 95% CI: [{ci_lo:.1f}%, {ci_hi:.1f}%])"
            )

        if policy_eval["mem_breach"]:
            max_g_idx = int(np.argmax(mem_res.growth_rate_estimates))
            g_rate = mem_res.growth_rate_estimates[max_g_idx]
            details.append(
                f"Canary process memory elevated with persistent growth on {self.active_canary_ids[max_g_idx]} (+{g_rate:.2f} MB/min over controls, >=15 min)"
            )

        if policy_eval["guardrail_status"] == "CRITICAL_OOM_RISK":
            details.append("Critical capacity headroom limit reached (<1024 MB or TTE < 30 min)")

        explanation = "; ".join(details)
        return (
            f"STATISTICAL PAUSE ADVISORY [{attr}]: {explanation}. "
            f"Consistent with release regression; observational evidence, not proof of code bug. | {infra_reason}"
        )

    def _build_host_evidence(
        self,
        contemp_recs: dict[str, TelemetryRecord],
        a_hat: np.ndarray,
        a_se: np.ndarray,
        mem_res: MemoryStepResult,
    ) -> dict[str, HostEvidence]:
        """Compile host-level evidence dictionary."""
        res: dict[str, HostEvidence] = {}
        for sid in self.expected_servers:
            meta = self.server_metadata.get(sid, {})
            rk = meta.get("rack_id", "unknown")
            cls_id = meta.get("class_id", 0)
            rec = contemp_recs.get(sid)

            is_canary = sid in self.active_canary_ids
            if is_canary:
                idx = self.active_canary_ids.index(sid)
                eff = float(a_hat[idx])
                se = float(a_se[idx])
                z = (eff - self.frozen_policy.kappa_lat) / se
                risk = float(norm.sf(-z))
                score = float(norm.logsf(-z))
                g_rate = float(mem_res.growth_rate_estimates[idx]) if idx < len(mem_res.growth_rate_estimates) else None
                g_se = float(mem_res.growth_rate_se[idx]) if idx < len(mem_res.growth_rate_se) else None
                headroom = float(mem_res.host_headroom_mb[idx]) if idx < len(mem_res.host_headroom_mb) else None
                tte = float(mem_res.host_tte_min[idx]) if idx < len(mem_res.host_tte_min) else None
                guard = mem_res.guardrail_status
                persist = mem_res.is_persistent_leak
            else:
                eff = se = risk = score = g_rate = g_se = headroom = tte = None
                guard = "NORMAL"
                persist = False

            res[sid] = HostEvidence(
                server_id=sid,
                role="CANARY" if is_canary else "CONTROL",
                rack_id=rk,
                class_id=cls_id,
                latency_effect=eff,
                latency_se=se,
                latency_ci_lower=eff - 1.96 * se if eff is not None and se is not None else None,
                latency_ci_upper=eff + 1.96 * se if eff is not None and se is not None else None,
                host_risk=risk,
                host_log_score=score,
                memory_drift_mb_per_min=g_rate,
                memory_drift_se=g_se,
                headroom_mb=headroom,
                tte_min=tte,
                guardrail_status=guard,
                is_persistent_mem=persist,
                latency_ms=rec.latency_ms if rec else 0.0,
                memory_mb=rec.memory_mb if rec else 0.0,
                cpu_fraction=rec.cpu_fraction if rec else 0.0,
                error_count=rec.error_count if rec else 0,
                request_count=rec.request_count if rec else 0,
                observed=rec.observed if rec else False,
            )
        return res

    def _build_null_result(
        self,
        tick: int,
        infra_adv: str,
        infra_reason: str,
        release_adv: str,
        attribution: str,
        contemp_recs: dict[str, TelemetryRecord],
        time_slice: TimeSlice | None = None,
    ) -> DetectorTickResult:
        """Build null result when no canary is actively deployed or release is rolled back."""
        hosts = {}
        for sid in self.expected_servers:
            rec = contemp_recs.get(sid)
            meta = self.server_metadata.get(sid, {})
            hosts[sid] = HostEvidence(
                server_id=sid,
                role="CONTROL",
                rack_id=meta.get("rack_id", "unknown"),
                class_id=meta.get("class_id", 0),
                latency_ms=rec.latency_ms if rec else 0.0,
                memory_mb=rec.memory_mb if rec else 0.0,
                cpu_fraction=rec.cpu_fraction if rec else 0.0,
                error_count=rec.error_count if rec else 0,
                request_count=rec.request_count if rec else 0,
                observed=rec.observed if rec else False,
            )
        return DetectorTickResult(
            tick=tick,
            t0=self.t0,
            eligible=True,
            is_unidentifiable=False,
            ineligible_reasons=[],
            attribution_blocked_reasons=[],
            infrastructure_advisory=infra_adv,
            canary_release_advisory=release_adv,
            attribution=attribution,
            advisory_reason=f"No active canary release deployed. | {infra_reason}",
            epistemic_disclaimer=self.frozen_policy.epistemic_disclaimer,
            release_score=0.0,
            score_lat=None,
            score_mem=None,
            threshold_lat=self.frozen_policy.threshold_lat,
            threshold_mem=self.frozen_policy.threshold_mem,
            first_alert_tick=None,
            hosts=hosts,
            class_effects=[],
            canary_latency_mean=0.0,
            control_latency_mean=float(np.mean([r.latency_ms for r in contemp_recs.values()])) if contemp_recs else 0.0,
            canary_memory_mean=0.0,
            control_memory_mean=float(np.mean([r.memory_mb for r in contemp_recs.values()])) if contemp_recs else 0.0,
            canary_cpu_mean=0.0,
            control_cpu_mean=float(np.mean([r.cpu_fraction for r in contemp_recs.values()])) if contemp_recs else 0.0,
            canary_errors_sum=0,
            control_errors_sum=int(sum(r.error_count for r in contemp_recs.values())),
            canary_effect_mean=0.0,
            canary_effect_ci_lower=0.0,
            canary_effect_ci_upper=0.0,
        )

    def _build_unidentifiable_result(
        self,
        tick: int,
        infra_adv: str,
        ident_reason: str,
        contemp_recs: dict[str, TelemetryRecord],
        time_slice: TimeSlice,
    ) -> DetectorTickResult:
        """Build unidentifiable result when rack or class control coverage is violated."""
        hosts = {}
        for sid in self.expected_servers:
            rec = contemp_recs.get(sid)
            meta = self.server_metadata.get(sid, {})
            hosts[sid] = HostEvidence(
                server_id=sid,
                role="CANARY" if sid in self.active_canary_ids else "CONTROL",
                rack_id=meta.get("rack_id", "unknown"),
                class_id=meta.get("class_id", 0),
                latency_ms=rec.latency_ms if rec else 0.0,
                memory_mb=rec.memory_mb if rec else 0.0,
                cpu_fraction=rec.cpu_fraction if rec else 0.0,
                error_count=rec.error_count if rec else 0,
                request_count=rec.request_count if rec else 0,
                observed=rec.observed if rec else False,
            )
        return DetectorTickResult(
            tick=tick,
            t0=self.t0,
            eligible=False,
            is_unidentifiable=True,
            ineligible_reasons=list(time_slice.ineligible_reasons),
            attribution_blocked_reasons=list(time_slice.attribution_blocked_reasons),
            infrastructure_advisory=infra_adv,
            canary_release_advisory="INSUFFICIENT DATA / CONFLICT",
            attribution="INSUFFICIENT_DATA",
            advisory_reason=f"IDENTIFIABILITY_GUARD: Confounded allocation - {ident_reason}. Release cause attribution is strictly prohibited without concurrent same-rack controls.",
            epistemic_disclaimer=self.frozen_policy.epistemic_disclaimer,
            release_score=None,
            score_lat=None,
            score_mem=None,
            threshold_lat=self.frozen_policy.threshold_lat,
            threshold_mem=self.frozen_policy.threshold_mem,
            first_alert_tick=self.first_alert_tick,
            hosts=hosts,
            class_effects=[],
            canary_latency_mean=0.0,
            control_latency_mean=float(np.mean([r.latency_ms for r in contemp_recs.values()])) if contemp_recs else 0.0,
            canary_memory_mean=0.0,
            control_memory_mean=float(np.mean([r.memory_mb for r in contemp_recs.values()])) if contemp_recs else 0.0,
            canary_cpu_mean=0.0,
            control_cpu_mean=float(np.mean([r.cpu_fraction for r in contemp_recs.values()])) if contemp_recs else 0.0,
            canary_errors_sum=0,
            control_errors_sum=int(sum(r.error_count for r in contemp_recs.values())),
            canary_effect_mean=0.0,
            canary_effect_ci_lower=0.0,
            canary_effect_ci_upper=0.0,
        )

    # -----------------------------------------------------------------
    # Query API (Used by FastAPI Routes)
    # -----------------------------------------------------------------
    def get_status(self) -> dict[str, Any]:
        """Return current detector state, advisories, models, and eligibility."""
        if not self.ssm_connected:
            latest = self.history[-1] if self.history else None
            return {
                "ssm_connected": False,
                "model_id": self.frozen_policy.model_id,
                "policy_id": self.frozen_policy.policy_id,
                "tick": latest.tick if latest else 0,
                "canary_release_advisory": "DETECTOR NOT CONNECTED",
                "infrastructure_advisory": latest.infrastructure_advisory if latest else "NORMAL",
                "attribution": "INSUFFICIENT_DATA",
                "release_score": None,
                "threshold_lat": self.frozen_policy.threshold_lat,
                "threshold_mem": self.frozen_policy.threshold_mem,
                "first_alert_tick": None,
                "advisory_reason": "DETECTOR NOT CONNECTED: Hierarchical SSM is offline or pending connection. Telemetry stream remains visible.",
                "data_eligibility": {
                    "eligible": False,
                    "is_unidentifiable": True,
                    "reason": "DETECTOR NOT CONNECTED",
                },
                "epistemic_disclaimer": self.frozen_policy.epistemic_disclaimer,
                "auto_rollback_enabled": self.enable_auto_rollback,
                "auto_rollback_triggered": self.auto_rollback_triggered,
                "intervention_tick": self.intervention_tick,
            }

        latest = self.history[-1] if self.history else None
        if not latest:
            return {
                "ssm_connected": True,
                "model_id": self.frozen_policy.model_id,
                "policy_id": self.frozen_policy.policy_id,
                "tick": 0,
                "canary_release_advisory": "RUNNING",
                "infrastructure_advisory": "NORMAL",
                "attribution": "NORMAL",
                "release_score": 0.0,
                "threshold_lat": self.frozen_policy.threshold_lat,
                "threshold_mem": self.frozen_policy.threshold_mem,
                "first_alert_tick": None,
                "advisory_reason": "Awaiting initial simulation telemetry slice.",
                "data_eligibility": {
                    "eligible": True,
                    "is_unidentifiable": False,
                },
                "epistemic_disclaimer": self.frozen_policy.epistemic_disclaimer,
                "auto_rollback_enabled": self.enable_auto_rollback,
                "auto_rollback_triggered": self.auto_rollback_triggered,
                "intervention_tick": self.intervention_tick,
            }

        # Check post-rollback state
        if self.is_rolled_back and not self.active_canary_ids:
            return {
                "ssm_connected": True,
                "model_id": self.frozen_policy.model_id,
                "policy_id": self.frozen_policy.policy_id,
                "tick": latest.tick,
                "t0": self.t0,
                "canary_release_advisory": "ROLLED BACK",
                "infrastructure_advisory": latest.infrastructure_advisory,
                "attribution": "ROLLED_BACK",
                "release_score": 0.0,
                "score_lat": None,
                "score_mem": None,
                "threshold_lat": self.frozen_policy.threshold_lat,
                "threshold_mem": self.frozen_policy.threshold_mem,
                "first_alert_tick": self.first_alert_tick,
                "advisory_reason": f"Release was rolled back at tick {self.rollback_tick}. | {latest.infrastructure_advisory}",
                "data_eligibility": {
                    "eligible": True,
                    "is_unidentifiable": False,
                },
                "epistemic_disclaimer": self.frozen_policy.epistemic_disclaimer,
                "auto_rollback_enabled": self.enable_auto_rollback,
                "auto_rollback_triggered": self.auto_rollback_triggered,
                "intervention_tick": self.intervention_tick,
            }

        return {
            "ssm_connected": True,
            "model_id": self.frozen_policy.model_id,
            "policy_id": self.frozen_policy.policy_id,
            "tick": latest.tick,
            "t0": latest.t0,
            "canary_release_advisory": latest.canary_release_advisory,
            "infrastructure_advisory": latest.infrastructure_advisory,
            "attribution": latest.attribution,
            "release_score": latest.release_score,
            "score_lat": latest.score_lat,
            "score_mem": latest.score_mem,
            "threshold_lat": latest.threshold_lat,
            "threshold_mem": latest.threshold_mem,
            "first_alert_tick": latest.first_alert_tick,
            "advisory_reason": latest.advisory_reason,
            "data_eligibility": {
                "eligible": latest.eligible,
                "is_unidentifiable": latest.is_unidentifiable,
                "ineligible_reasons": latest.ineligible_reasons,
                "attribution_blocked_reasons": latest.attribution_blocked_reasons,
                "active_canaries": len(self.active_canary_ids),
                "min_controls_per_rack": self.protocol.min_controls_per_rack,
                "min_controls_per_class": self.protocol.min_controls_per_class,
            },
            "epistemic_disclaimer": latest.epistemic_disclaimer,
            "auto_rollback_enabled": self.enable_auto_rollback,
            "auto_rollback_triggered": self.auto_rollback_triggered,
            "intervention_tick": self.intervention_tick,
        }

    def get_evidence(self) -> dict[str, Any]:
        """Return server-level and class-level evidence breakdowns."""
        latest = self.history[-1] if self.history else None
        if not latest:
            return {"servers": {}, "class_effects": []}

        return {
            "tick": latest.tick,
            "ssm_connected": self.ssm_connected,
            "servers": {k: v.to_dict() for k, v in latest.hosts.items()},
            "class_effects": [c.to_dict() for c in latest.class_effects],
        }

    def get_timeline(self, limit: int = 120) -> list[dict[str, Any]]:
        """Return historical detector timeline records for UI plotting."""
        entries = self.history[-limit:]
        return [e.to_dict() for e in entries]
