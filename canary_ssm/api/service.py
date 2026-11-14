"""Simulator Service for Virtual Data Center.

Coordinates:
- VirtualWorld simulation clock, event queue, and telemetry generation.
- Release registry and deployment lifecycle.
- Dual Advisory Engine:
    * Infrastructure Advisory (NORMAL, RACK_DEGRADED, FLEET_OUTAGE)
    * Canary Release Advisory (RUNNING, PAUSE RECOMMENDED, ROLLED BACK, INSUFFICIENT DATA)
- Multi-metric telemetry buffering and hierarchical evidence aggregation.
"""
from __future__ import annotations

import collections
import math
import threading
import time
from typing import Any

import numpy as np

from canary_ssm.deployments import (
    AllocationValidationResult,
    BehaviorProfile,
    BehaviorProfileType,
    DeploymentManager,
    Release,
    ReleaseArtifact,
    generate_default_allocation,
    validate_canary_allocation,
)
from canary_ssm.detector_service import DetectorService, DetectorTickResult
from canary_ssm.faults.models import FaultCategory, FaultDefinition, FaultScope, FaultType
from canary_ssm.telemetry import TelemetryRecord
from canary_ssm.world import VirtualWorld, create_default_fleet
from canary_ssm.world.models import Fleet, Server


class SimulatorService:
    """Singleton-style service managing active simulation world, deployments, and detectors."""

    def __init__(
        self,
        n_racks: int = 4,
        servers_per_rack: int = 6,
        seed: int = 42,
        ssm_connected: bool = False,
    ) -> None:
        self.lock = threading.RLock()
        self.n_racks = n_racks
        self.servers_per_rack = servers_per_rack
        self.initial_seed = seed
        self.ssm_connected = bool(ssm_connected)

        # Initialize world and deployment manager
        self.world: VirtualWorld = VirtualWorld.create_default(
            n_racks=n_racks,
            servers_per_rack=servers_per_rack,
            seed=seed,
            start_paused=True,
        )
        self.deployment_mgr: DeploymentManager = DeploymentManager()

        # Telemetry ring buffer: up to 5000 recent observations
        self.telemetry_history: collections.deque[TelemetryRecord] = collections.deque(maxlen=5000)
        self.tick_metrics_summary: list[dict[str, Any]] = []

        # Background runner thread state
        self.is_auto_stepping = False
        self._stop_event = threading.Event()
        self._runner_thread: threading.Thread | None = None
        self.tick_speed_seconds = 1.0  # Seconds per simulation tick when auto-running

        # Deploy state cache
        self.active_release_id: str | None = None
        self.active_canary_ids: list[str] = []
        self.last_rollback_status: dict[str, Any] | None = None

        # Initialize decoupled DetectorService
        server_meta = {
            srv.server_id: {
                "server_id": srv.server_id,
                "rack_id": srv.rack_id,
                "class_id": srv.class_id,
                "memory_capacity_mb": srv.memory_capacity_mb,
                "base_latency_ms": srv.base_latency_ms,
            }
            for srv in self.world.fleet.list_servers()
        }
        self.detector_service = DetectorService(
            expected_servers=list(server_meta.keys()),
            server_metadata=server_meta,
            ssm_connected=self.ssm_connected,
        )

        # Pre-seed default pre-period baseline (e.g. warm-up ticks)
        # Pre-generate 30 ticks of healthy baseline telemetry
        self.step(30)

    # -----------------------------------------------------------------
    # World & Clock Controls
    # -----------------------------------------------------------------
    def start(self) -> dict[str, Any]:
        with self.lock:
            self.world.start()
            if not self.is_auto_stepping:
                self.is_auto_stepping = True
                self._stop_event.clear()
                self._runner_thread = threading.Thread(target=self._run_loop, daemon=True)
                self._runner_thread.start()
            return self.get_status()

    def pause(self) -> dict[str, Any]:
        with self.lock:
            self.world.pause()
            self.is_auto_stepping = False
            self._stop_event.set()
            return self.get_status()

    def step(self, n_ticks: int = 1) -> dict[str, Any]:
        with self.lock:
            for _ in range(max(1, n_ticks)):
                # Keep detector service deployment in sync if canaries were modified directly
                if self.active_canary_ids and set(self.active_canary_ids) != set(self.detector_service.active_canary_ids):
                    self.detector_service.set_deployment(
                        release_id=self.active_release_id or "forced-deployment",
                        canary_ids=list(self.active_canary_ids),
                        t0=self.world.clock.tick,
                    )
                elif not self.active_canary_ids and self.detector_service.active_canary_ids:
                    self.detector_service.set_rollback(tick=self.world.clock.tick)

                res = self.world.step(1)
                for record in res.telemetry:
                    self.telemetry_history.append(record)

                # Feed telemetry to decoupled detector service
                self.detector_service.ingest(res.telemetry)
                det_res = self.detector_service.step(res.tick)

                # Check opt-in automatic rollback (operational demo mode)
                if (
                    self.detector_service.enable_auto_rollback
                    and det_res.auto_rollback_triggered
                    and self.active_release_id is not None
                    and not self.detector_service.is_rolled_back
                ):
                    self.rollback_canary(
                        release_id=self.active_release_id,
                        restart_process=True,
                    )

                # Record aggregated tick summary
                self._record_tick_summary(res.tick, det_res)
            return self.get_status()

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        with self.lock:
            self.pause()
            new_seed = seed if seed is not None else self.initial_seed
            self.world = VirtualWorld.create_default(
                n_racks=self.n_racks,
                servers_per_rack=self.servers_per_rack,
                seed=new_seed,
                start_paused=True,
            )
            self.deployment_mgr = DeploymentManager()
            self.telemetry_history.clear()
            self.tick_metrics_summary.clear()
            self.active_release_id = None
            self.active_canary_ids.clear()
            self.last_rollback_status = None

            server_meta = {
                srv.server_id: {
                    "server_id": srv.server_id,
                    "rack_id": srv.rack_id,
                    "class_id": srv.class_id,
                    "memory_capacity_mb": srv.memory_capacity_mb,
                    "base_latency_ms": srv.base_latency_ms,
                }
                for srv in self.world.fleet.list_servers()
            }
            self.detector_service = DetectorService(
                expected_servers=list(server_meta.keys()),
                server_metadata=server_meta,
                ssm_connected=self.ssm_connected,
            )

            # Re-warm with 30 baseline ticks
            self.step(30)
            return self.get_status()

    def set_speed(self, speed_seconds: float) -> dict[str, Any]:
        with self.lock:
            self.tick_speed_seconds = max(0.05, min(10.0, float(speed_seconds)))
            return {"tick_speed_seconds": self.tick_speed_seconds}

    def _run_loop(self) -> None:
        """Background thread executing discrete ticks when auto-stepping."""
        while not self._stop_event.is_set():
            time.sleep(self.tick_speed_seconds)
            if self._stop_event.is_set():
                break
            try:
                self.step(1)
            except Exception as e:
                print(f"[SimulatorService] Auto-step error: {e}")
                self.pause()
                break

    # -----------------------------------------------------------------
    # Status & Server Inspection
    # -----------------------------------------------------------------
    def get_status(self) -> dict[str, Any]:
        with self.lock:
            adv = self.evaluate_advisories()
            return {
                "tick": self.world.clock.tick,
                "is_paused": self.world.clock.is_paused,
                "is_auto_stepping": self.is_auto_stepping,
                "tick_speed_seconds": self.tick_speed_seconds,
                "fleet_summary": self.world.fleet.summary(),
                "active_release_id": self.active_release_id,
                "canary_servers": list(self.active_canary_ids),
                "infrastructure_advisory": adv["infrastructure_advisory"],
                "canary_release_advisory": adv["canary_release_advisory"],
                "advisory_reason": adv["advisory_reason"],
                "ssm_connected": self.ssm_connected,
                "model_id": adv.get("model_id"),
                "policy_id": adv.get("policy_id"),
                "release_score": adv.get("release_score"),
                "alert_threshold": adv.get("alert_threshold"),
                "first_alert_tick": adv.get("first_alert_tick"),
                "data_eligibility": adv.get("data_eligibility"),
                "auto_rollback_enabled": adv.get("auto_rollback_enabled"),
                "auto_rollback_triggered": adv.get("auto_rollback_triggered"),
                "intervention_tick": adv.get("intervention_tick"),
                "data_freshness_tick": self.world.clock.tick,
            }

    def toggle_detector(self, connected: bool | None = None) -> bool:
        with self.lock:
            if connected is not None:
                self.ssm_connected = bool(connected)
            else:
                self.ssm_connected = not self.ssm_connected
            self.detector_service.set_ssm_connected(self.ssm_connected)
            return self.ssm_connected

    def configure_hazards(
        self,
        enable_stochastic: bool,
        server_outage_hazard_per_min: float | None = None,
        rack_latency_hazard_per_min: float | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            cfg = self.world.fault_engine.hazard_cfg
            cfg.enable_stochastic = bool(enable_stochastic)
            if server_outage_hazard_per_min is not None:
                cfg.server_outage_hazard_per_min = float(server_outage_hazard_per_min)
            if rack_latency_hazard_per_min is not None:
                cfg.rack_network_latency_hazard_per_min = float(rack_latency_hazard_per_min)
            if seed is not None:
                self.world.fault_engine.seed = int(seed)
                self.world.fault_engine.rng_fault = np.random.default_rng(int(seed))
            return {
                "enable_stochastic": cfg.enable_stochastic,
                "server_outage_hazard_per_min": cfg.server_outage_hazard_per_min,
                "rack_network_latency_hazard_per_min": cfg.rack_network_latency_hazard_per_min,
                "seed": self.world.fault_engine.seed,
            }

    def get_command_history(self) -> list[dict[str, Any]]:
        with self.lock:
            return list(self.deployment_mgr.deployment_history)

    def export_run(self) -> dict[str, Any]:
        with self.lock:
            return {
                "run_id": self.world.run_id,
                "current_tick": self.world.clock.tick,
                "fleet_summary": self.world.fleet.summary(),
                "telemetry_history": [t.to_dict() for t in self.telemetry_history],
                "timeline_summary": list(self.tick_metrics_summary),
                "command_history": self.get_command_history(),
                "fault_truth": self.world.fault_engine.truth_store.to_dict(),
            }

    def import_run(self, bundle: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.pause()
            records_raw = bundle.get("telemetry_history", [])
            timeline_raw = bundle.get("timeline_summary", [])
            self.tick_metrics_summary = list(timeline_raw)
            self.telemetry_history.clear()
            for r in records_raw:
                self.telemetry_history.append(TelemetryRecord.from_dict(r))
            return {
                "status": "LOADED",
                "total_records": len(self.telemetry_history),
                "total_ticks": len(self.tick_metrics_summary),
            }

    def get_servers(self) -> list[dict[str, Any]]:
        with self.lock:
            return [srv.to_dict() for srv in self.world.fleet.list_servers()]

    def get_racks(self) -> list[dict[str, Any]]:
        with self.lock:
            res = []
            for rack in self.world.fleet.list_racks():
                res.append({
                    "rack_id": rack.rack_id,
                    "switch_id": rack.switch_id,
                    "online": rack.online,
                    "server_count": len(rack.servers),
                    "servers": [s.to_dict() for s in rack.servers.values()],
                })
            return res

    # -----------------------------------------------------------------
    # Releases & Deployments
    # -----------------------------------------------------------------
    def list_releases(self) -> list[dict[str, Any]]:
        with self.lock:
            return [r.to_dict() for r in self.deployment_mgr.registry.list_releases()]

    def create_release(
        self,
        release_id: str,
        version_tag: str,
        behavior_profile: dict[str, Any],
        artifact_filename: str | None = None,
        artifact_content: str | bytes | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            rel = self.deployment_mgr.create_release(
                release_id=release_id,
                version_tag=version_tag,
                behavior_profile=behavior_profile,
                artifact_filename=artifact_filename,
                artifact_content=artifact_content,
                created_at_tick=self.world.clock.tick,
                metadata=metadata,
            )
            return rel.to_dict()

    def validate_allocation(self, canary_server_ids: list[str]) -> dict[str, Any]:
        with self.lock:
            val = self.deployment_mgr.validate_allocation(self.world.fleet, canary_server_ids)
            return val.to_dict()

    def get_default_allocation(self) -> dict[str, Any]:
        with self.lock:
            canaries = generate_default_allocation(self.world.fleet)
            val = self.deployment_mgr.validate_allocation(self.world.fleet, canaries)
            return {
                "recommended_canaries": canaries,
                "validation": val.to_dict(),
            }

    def deploy_canary(
        self,
        release_id: str,
        canary_server_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            res = self.deployment_mgr.deploy_canary(
                world=self.world,
                release_id=release_id,
                canary_ids=canary_server_ids,
            )
            self.active_release_id = release_id
            self.active_canary_ids = list(res["canary_servers"])
            self.detector_service.set_deployment(
                release_id=release_id,
                canary_ids=self.active_canary_ids,
                t0=self.world.clock.tick,
            )
            return res

    def rollback_canary(
        self,
        release_id: str,
        restart_process: bool = True,
    ) -> dict[str, Any]:
        with self.lock:
            res = self.deployment_mgr.rollback_canary(
                world=self.world,
                release_id=release_id,
                restart_process=restart_process,
            )
            self.last_rollback_status = res
            self.detector_service.set_rollback(tick=self.world.clock.tick)
            if self.active_release_id == release_id:
                self.active_release_id = None
                self.active_canary_ids.clear()
            return res

    # -----------------------------------------------------------------
    # Fault Lab (Developer View)
    # -----------------------------------------------------------------
    def inject_fault(
        self,
        fault_type: str,
        scope: str,
        target_id: str,
        severity: float,
        duration: int = 60,
        onset_tick: int | None = None,
        susceptible_class_id: int | None = None,
        target_version: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            current_tick = self.world.clock.tick
            onset = current_tick if onset_tick is None else onset_tick
            fid = f"manual-{fault_type.lower()}-{target_id}-{onset}"

            # Resolve fault_type and scope case-insensitively
            try:
                ft = FaultType(fault_type.lower())
            except ValueError:
                ft = FaultType[fault_type.upper()]

            try:
                sc = FaultScope(scope.lower())
            except ValueError:
                sc = FaultScope[scope.upper()]

            fdef = FaultDefinition(
                fault_id=fid,
                fault_type=ft,
                scope=sc,
                target_id=target_id,
                onset_tick=onset,
                duration=duration,
                severity=float(severity),
                susceptible_class_id=susceptible_class_id,
                target_version=target_version,
                parameters=dict(parameters or {}),
            )
            self.world.inject_fault(fdef)
            return fdef.to_dict()

    def get_fault_truth(self) -> dict[str, Any]:
        """Developer Fault Lab View: Inspect ground truth timeline."""
        with self.lock:
            return self.world.fault_engine.truth_store.to_dict()

    # -----------------------------------------------------------------
    # Monitoring, Telemetry & Dual Advisory Engine
    # -----------------------------------------------------------------
    def get_latest_telemetry(self) -> list[dict[str, Any]]:
        with self.lock:
            return [t.to_dict() for t in self.world.get_latest_telemetry()]

    def get_metrics_timeline(self, limit: int = 120) -> list[dict[str, Any]]:
        with self.lock:
            return list(self.tick_metrics_summary[-limit:])

    def _record_tick_summary(self, tick: int, det_res: Any | None = None) -> None:
        """Compute aggregated multi-metric stats for the tick and store in timeline."""
        latest = self.world.get_latest_telemetry()
        if not latest:
            return

        canary_recs = [r for r in latest if r.server_id in self.active_canary_ids or r.exposure]
        ctrl_recs = [r for r in latest if r.server_id not in self.active_canary_ids and not r.exposure]

        def stats(recs: list[TelemetryRecord]) -> dict[str, float]:
            if not recs:
                return {"latency_ms": 0.0, "memory_mb": 0.0, "cpu_fraction": 0.0, "errors": 0.0, "requests": 0.0}
            return {
                "latency_ms": float(np.mean([r.latency_ms for r in recs])),
                "memory_mb": float(np.mean([r.memory_mb for r in recs])),
                "cpu_fraction": float(np.mean([r.cpu_fraction for r in recs])),
                "errors": float(np.sum([r.error_count for r in recs])),
                "requests": float(np.sum([r.request_count for r in recs])),
            }

        adv = self.evaluate_advisories()

        summary_entry = {
            "tick": tick,
            "canary": stats(canary_recs),
            "control": stats(ctrl_recs),
            "canary_count": len(canary_recs),
            "control_count": len(ctrl_recs),
            "infrastructure_advisory": adv["infrastructure_advisory"],
            "canary_release_advisory": adv["canary_release_advisory"],
            "release_score": adv["release_score"],
            "threshold": adv["alert_threshold"],
            "model_id": adv.get("model_id"),
            "policy_id": adv.get("policy_id"),
            "advisory_reason": adv.get("advisory_reason"),
            "first_alert_tick": adv.get("first_alert_tick"),
            "canary_effect_mean": det_res.canary_effect_mean if det_res else 0.0,
            "canary_effect_ci_lower": det_res.canary_effect_ci_lower if det_res else 0.0,
            "canary_effect_ci_upper": det_res.canary_effect_ci_upper if det_res else 0.0,
            "auto_rollback_triggered": det_res.auto_rollback_triggered if det_res else False,
            "intervention_tick": det_res.intervention_tick if det_res else None,
            "censored": det_res.censored if det_res else False,
        }
        self.tick_metrics_summary.append(summary_entry)

    def evaluate_advisories(self) -> dict[str, Any]:
        """Compute Dual Advisory States via independent DetectorService without accessing hidden truth."""
        det_status = self.detector_service.get_status()
        latest_res = self.detector_service.history[-1] if self.detector_service.history else None

        channel_evidence: dict[str, Any] = {}
        if latest_res:
            mem_delta = round(latest_res.canary_memory_mean - latest_res.control_memory_mean, 1)
            channel_evidence = {
                "score_lat": latest_res.score_lat,
                "score_mem": latest_res.score_mem,
                "canary_effect_mean": round(latest_res.canary_effect_mean, 5),
                "canary_effect_ci": [round(latest_res.canary_effect_ci_lower, 5), round(latest_res.canary_effect_ci_upper, 5)],
                "canary_count": len(self.active_canary_ids),
                "control_count": len(self.detector_service.expected_servers) - len(self.active_canary_ids),
                "auto_rollback_triggered": latest_res.auto_rollback_triggered,
                "intervention_tick": latest_res.intervention_tick,
                "memory_growth_delta_mb": mem_delta,
            }

        return {
            "infrastructure_advisory": det_status["infrastructure_advisory"],
            "canary_release_advisory": det_status["canary_release_advisory"],
            "release_score": det_status["release_score"],
            "alert_threshold": det_status["threshold_lat"],
            "advisory_reason": det_status["advisory_reason"],
            "model_id": det_status["model_id"],
            "policy_id": det_status["policy_id"],
            "data_eligibility": det_status["data_eligibility"],
            "first_alert_tick": det_status["first_alert_tick"],
            "epistemic_disclaimer": det_status["epistemic_disclaimer"],
            "ssm_connected": det_status["ssm_connected"],
            "auto_rollback_enabled": det_status["auto_rollback_enabled"],
            "auto_rollback_triggered": det_status["auto_rollback_triggered"],
            "intervention_tick": det_status["intervention_tick"],
            "channel_evidence": channel_evidence,
        }

    def get_detector_status(self) -> dict[str, Any]:
        with self.lock:
            return self.detector_service.get_status()

    def get_detector_evidence(self) -> dict[str, Any]:
        with self.lock:
            return self.detector_service.get_evidence()

    def get_detector_timeline(self, limit: int = 120) -> list[dict[str, Any]]:
        with self.lock:
            return self.detector_service.get_timeline(limit=limit)

    def set_auto_rollback(self, enabled: bool) -> bool:
        with self.lock:
            return self.detector_service.set_auto_rollback(enabled)
