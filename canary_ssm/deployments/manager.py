"""Deployment Manager: Release Registry and Deployment Lifecycle Coordinator.

Handles:
- Creating versioned releases with unexecuted source artifacts and behavior profiles.
- Validating canary allocations with identifiability checks.
- Deploying releases to validated canary hosts.
- Rollback with explicit process restart / memory clearing semantics.
- Idempotent command execution via command_id tracking.
"""
from __future__ import annotations

import copy
import uuid
from typing import Any

from canary_ssm.faults.models import FaultCategory, FaultDefinition, FaultScope, FaultType
from canary_ssm.world import VirtualWorld
from canary_ssm.world.models import Fleet, Server
from .allocation import AllocationValidationResult, generate_default_allocation, validate_canary_allocation
from .models import (
    BehaviorProfile,
    BehaviorProfileType,
    Release,
    ReleaseArtifact,
    ReleaseRegistry,
)


class DeploymentManager:
    """Manages the full lifecycle of versioned releases and canary deployments."""

    def __init__(self, registry: ReleaseRegistry | None = None) -> None:
        self.registry = registry or ReleaseRegistry()
        self.executed_commands: dict[str, dict[str, Any]] = {}
        self.deployment_history: list[dict[str, Any]] = []

    def create_release(
        self,
        release_id: str,
        version_tag: str,
        behavior_profile: BehaviorProfile | dict[str, Any],
        artifact_filename: str | None = None,
        artifact_content: str | bytes | None = None,
        created_at_tick: int = 0,
        metadata: dict[str, Any] | None = None,
        command_id: str | None = None,
    ) -> Release:
        """Create and register a versioned release bundle.

        Strict Contract:
            Uploaded artifact contents are hashed and labeled unexecuted.
            The simulation behavior is dictated exclusively by behavior_profile.
        """
        cid = command_id or f"cmd-create-{release_id}"
        if cid in self.executed_commands:
            return self.registry.get(release_id)

        # Parse behavior profile
        if isinstance(behavior_profile, dict):
            prof = BehaviorProfile.from_dict(behavior_profile)
        else:
            prof = behavior_profile

        # Build artifact metadata if content provided
        art = None
        if artifact_filename and artifact_content is not None:
            art = ReleaseArtifact.create_from_content(
                filename=artifact_filename,
                content=artifact_content,
            )

        release = Release(
            release_id=release_id,
            version_tag=version_tag,
            behavior_profile=prof,
            artifact=art,
            created_at_tick=created_at_tick,
            metadata=dict(metadata or {}),
        )
        self.registry.register(release)

        record = {
            "command_id": cid,
            "action": "CREATE_RELEASE",
            "release_id": release_id,
            "version_tag": version_tag,
            "profile": prof.profile_type.value,
            "artifact_hash": art.checksum_sha256 if art else None,
            "tick": created_at_tick,
        }
        self.executed_commands[cid] = record
        self.deployment_history.append(record)
        return release

    def validate_allocation(self, fleet: Fleet, canary_ids: list[str]) -> AllocationValidationResult:
        """Validate candidate canary allocation against confounding and identifiability rules."""
        return validate_canary_allocation(fleet, canary_ids)

    def deploy_canary(
        self,
        world: VirtualWorld,
        release_id: str,
        canary_ids: list[str] | None = None,
        command_id: str | None = None,
    ) -> dict[str, Any]:
        """Deploy a versioned candidate release to canary servers.

        Actions:
            1. Enforces idempotency via command_id.
            2. Validates canary allocation for identifiability.
            3. Updates canary servers to installed_version = release.version_tag.
            4. If behavior profile specifies regressions, registers matching FaultDefinition.
        """
        cid = command_id or f"cmd-deploy-{release_id}-{world.clock.tick}"
        if cid in self.executed_commands:
            return dict(self.executed_commands[cid])

        release = self.registry.get(release_id)
        active_canaries = canary_ids if canary_ids is not None else generate_default_allocation(world.fleet)

        # Validate allocation
        val = self.validate_allocation(world.fleet, active_canaries)
        if not val.is_valid:
            raise ValueError(f"Canary allocation rejected: {'; '.join(val.issues)}")

        current_tick = world.clock.tick

        # Deploy software version to canary servers
        for sid in active_canaries:
            world.set_server_version(sid, release.version_tag)

        # Wire behavior profile into world's fault engine if regression declared
        prof = release.behavior_profile
        injected_fault_id = None

        if prof.profile_type != BehaviorProfileType.CLEAN:
            fault_type_map = {
                BehaviorProfileType.MEMORY_LEAK: FaultType.PROCESS_MEMORY_LEAK,
                BehaviorProfileType.CPU_REGRESSION: FaultType.CPU_SATURATION,
                BehaviorProfileType.ERROR_REGRESSION: FaultType.ERROR_REGRESSION,
                BehaviorProfileType.LATENCY_REGRESSION: FaultType.RELEASE_LATENCY_REGRESSION,
            }
            ft = fault_type_map[prof.profile_type]
            injected_fault_id = f"fault-{release.release_id}-{prof.profile_type.value.lower()}"

            fault_def = FaultDefinition(
                fault_id=injected_fault_id,
                fault_type=ft,
                scope=FaultScope.SERVER,
                target_id=release.release_id,
                onset_tick=current_tick,
                duration=prof.duration,
                severity=prof.severity,
                target_version=release.version_tag,
                susceptible_class_id=prof.susceptible_class_id,
                parameters={"target_server_ids": list(active_canaries), **prof.parameters},
            )
            world.inject_fault(fault_def)

        outcome = {
            "command_id": cid,
            "action": "DEPLOY_CANARY",
            "release_id": release_id,
            "version_tag": release.version_tag,
            "canary_servers": list(active_canaries),
            "control_servers": list(val.control_server_ids),
            "tick": current_tick,
            "injected_fault_id": injected_fault_id,
            "status": "DEPLOYED",
        }
        self.executed_commands[cid] = outcome
        self.deployment_history.append(outcome)
        return outcome

    def rollback_canary(
        self,
        world: VirtualWorld,
        release_id: str,
        baseline_version: str = "v1.0.0-baseline",
        restart_process: bool = True,
        command_id: str | None = None,
    ) -> dict[str, Any]:
        """Roll back canary servers running a candidate release.

        Semantics:
            - Reverts installed_version to baseline_version.
            - If restart_process == True:
                Models a full process restart: server memory resets to baseline,
                clearing accumulated memory leaks.
            - If restart_process == False:
                Models hot version tag revert without restarting process:
                leak progression stops, but current resident memory is NOT cleared.
                (Reflects principle: Memory does not disappear solely because a version label changes).
        """
        cid = command_id or f"cmd-rollback-{release_id}-{world.clock.tick}"
        if cid in self.executed_commands:
            return dict(self.executed_commands[cid])

        release = self.registry.get(release_id)
        current_tick = world.clock.tick
        rolled_back_servers: list[str] = []

        for srv in world.fleet.list_servers():
            if srv.installed_version == release.version_tag:
                rolled_back_servers.append(srv.server_id)
                world.set_server_version(srv.server_id, baseline_version)

                if restart_process:
                    # Clean process restart resets process memory to nominal baseline
                    srv.current_state.memory_mb = srv.base_memory_mb
                    # Clear internal leak memory in fault engine
                    world.fault_engine._server_memory_leaks_mb[srv.server_id] = 0.0

        outcome = {
            "command_id": cid,
            "action": "ROLLBACK_CANARY",
            "release_id": release_id,
            "version_tag": release.version_tag,
            "baseline_version": baseline_version,
            "rolled_back_servers": rolled_back_servers,
            "restart_process": restart_process,
            "memory_cleared": bool(restart_process),
            "tick": current_tick,
            "status": "ROLLED_BACK",
        }
        self.executed_commands[cid] = outcome
        self.deployment_history.append(outcome)
        return outcome
