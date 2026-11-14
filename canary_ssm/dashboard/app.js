/**
 * Virtual Data Center 5-Page Dashboard Controller
 * Vanilla ES6 JavaScript (Zero external dependencies)
 */

const API_BASE = "";
let currentTab = "world";
let pollInterval = null;
let activeCanarySet = new Set();
let cachedTimeline = [];
let isConnected = true;
let isReplaying = false;
let replayTimer = null;
let currentReplayIndex = 0;

// Initialize Dashboard
document.addEventListener("DOMContentLoaded", () => {
  setupNavigation();
  setupWorldControls();
  setupReleasesPage();
  setupFaultLabPage();
  setupEvaluationPage();

  // Initial refresh and periodic polling (every 1s)
  refreshAll();
  pollInterval = setInterval(refreshAll, 1000);
});

// ---------------------------------------------------------------------
// Toast Notifications
// ---------------------------------------------------------------------
function showToast(message, type = "info") {
  const container = document.getElementById("toast-container");
  if (!container) return;

  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.textContent = message;

  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transition = "opacity 0.3s ease";
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

// ---------------------------------------------------------------------
// Navigation & Tab Switching
// ---------------------------------------------------------------------
function setupNavigation() {
  const tabs = document.querySelectorAll(".tab-btn");
  tabs.forEach(btn => {
    btn.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      btn.classList.add("active");
      currentTab = btn.dataset.tab;

      document.querySelectorAll(".page-view").forEach(p => p.classList.remove("active"));
      const targetPage = document.getElementById(`page-${currentTab}`);
      if (targetPage) targetPage.classList.add("active");

      refreshAll();
    });
  });
}

// ---------------------------------------------------------------------
// API Helpers with Connection Health Tracking
// ---------------------------------------------------------------------
function updateConnectionStatus(connected) {
  isConnected = connected;
  const el = document.getElementById("connection-status");
  if (!el) return;

  if (connected) {
    el.textContent = "● CONNECTED";
    el.style.backgroundColor = "#22c55e";
    el.style.color = "#0f172a";
  } else {
    el.textContent = "● RECONNECTING...";
    el.style.backgroundColor = "#eab308";
    el.style.color = "#0f172a";
  }
}

async function apiGet(endpoint) {
  try {
    const res = await fetch(`${API_BASE}${endpoint}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    updateConnectionStatus(true);
    return await res.json();
  } catch (err) {
    console.error(`GET ${endpoint} failed:`, err);
    updateConnectionStatus(false);
    return null;
  }
}

async function apiPost(endpoint, body) {
  try {
    const res = await fetch(`${API_BASE}${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || `HTTP ${res.status}`);
    }
    updateConnectionStatus(true);
    return data;
  } catch (err) {
    console.error(`POST ${endpoint} failed:`, err);
    updateConnectionStatus(false);
    showToast(`Error: ${err.message}`, "error");
    return null;
  }
}

// ---------------------------------------------------------------------
// Main Poll Loop
// ---------------------------------------------------------------------
async function refreshAll() {
  const status = await apiGet("/api/world/status");
  if (!status) return;

  updateControlBar(status);

  if (currentTab === "world") {
    await renderWorldRacks();
  } else if (currentTab === "monitoring") {
    await renderMonitoringView();
  } else if (currentTab === "releases") {
    await Promise.all([renderReleasesList(), renderCommandHistory()]);
  } else if (currentTab === "faults") {
    await renderFaultTruth();
  } else if (currentTab === "evaluation") {
    await renderEvaluationView();
  }
}

function updateControlBar(status) {
  const tickEl = document.getElementById("clock-tick");
  const pauseStatusEl = document.getElementById("clock-status");
  const freshnessEl = document.getElementById("data-freshness");

  if (tickEl) tickEl.textContent = `T = ${status.tick} min`;
  if (pauseStatusEl) {
    pauseStatusEl.textContent = status.is_paused ? "PAUSED" : "RUNNING";
    pauseStatusEl.className = status.is_paused ? "badge badge-warning" : "badge badge-canary";
  }
  if (freshnessEl) {
    freshnessEl.textContent = `Freshness: T=${status.data_freshness_tick ?? status.tick} (Live)`;
  }

  // Update Advisories
  const infraEl = document.getElementById("advisory-infra");
  const releaseEl = document.getElementById("advisory-release");

  if (infraEl) {
    infraEl.textContent = status.infrastructure_advisory;
    infraEl.className = "advisory-pill " +
      (status.infrastructure_advisory === "NORMAL" ? "pill-normal" : "pill-danger");
  }

  if (releaseEl) {
    releaseEl.textContent = status.canary_release_advisory;
    let cls = "pill-normal";
    if (status.canary_release_advisory === "DETECTOR NOT CONNECTED") cls = "pill-neutral";
    else if (status.canary_release_advisory === "PAUSE RECOMMENDED") cls = "pill-danger";
    else if (status.canary_release_advisory === "ROLLED BACK") cls = "pill-neutral";
    else if (status.canary_release_advisory.includes("INSUFFICIENT")) cls = "pill-warning";
    releaseEl.className = `advisory-pill ${cls}`;
  }

  // Sync detector toggle checkbox
  const cbDetector = document.getElementById("cb-connect-detector");
  if (cbDetector && typeof status.ssm_connected === "boolean") {
    cbDetector.checked = status.ssm_connected;
  }

  // Sync auto-rollback checkbox
  const cbAutoRollback = document.getElementById("cb-auto-rollback");
  if (cbAutoRollback && typeof status.auto_rollback_enabled === "boolean") {
    cbAutoRollback.checked = status.auto_rollback_enabled;
  }

  // Active canaries cache
  activeCanarySet = new Set(status.canary_servers || []);
}

// ---------------------------------------------------------------------
// Page 1: World View
// ---------------------------------------------------------------------
function setupWorldControls() {
  document.getElementById("btn-play")?.addEventListener("click", async () => {
    const res = await apiPost("/api/world/control", { action: "start" });
    if (res) showToast("Simulation started", "info");
  });

  document.getElementById("btn-pause")?.addEventListener("click", async () => {
    const res = await apiPost("/api/world/control", { action: "pause" });
    if (res) showToast("Simulation paused", "info");
  });

  document.getElementById("btn-step-1")?.addEventListener("click", async () => {
    const res = await apiPost("/api/world/control", { action: "step", ticks: 1 });
    if (res) showToast(`Stepped +1 tick (T=${res.tick})`, "info");
  });

  document.getElementById("btn-step-5")?.addEventListener("click", async () => {
    const res = await apiPost("/api/world/control", { action: "step", ticks: 5 });
    if (res) showToast(`Stepped +5 ticks (T=${res.tick})`, "info");
  });

  document.getElementById("btn-reset")?.addEventListener("click", async () => {
    if (confirm("Reset simulation to initial seed and baseline?")) {
      const res = await apiPost("/api/world/control", { action: "reset" });
      if (res) showToast("Simulation reset to initial seed", "info");
    }
  });

  const speedSelect = document.getElementById("select-speed");
  speedSelect?.addEventListener("change", async (e) => {
    const res = await apiPost("/api/world/control", { action: "speed", speed_seconds: parseFloat(e.target.value) });
    if (res) showToast(`Simulation speed set to ${res.tick_speed_seconds}s/min`, "info");
  });

  // Toggle SSM Detector
  document.getElementById("cb-connect-detector")?.addEventListener("change", async (e) => {
    const res = await apiPost("/api/detector/toggle", { connected: e.target.checked });
    if (res) {
      showToast(res.ssm_connected ? "Hierarchical SSM Detector Connected" : "SSM Detector Disconnected", "info");
      refreshAll();
    }
  });

  // Toggle Auto-Rollback (Operational Demo Mode)
  document.getElementById("cb-auto-rollback")?.addEventListener("change", async (e) => {
    const res = await apiPost("/api/detector/auto-rollback", { enabled: e.target.checked });
    if (res) {
      showToast(res.auto_rollback_enabled ? "Autonomous Rollback ENABLED" : "Autonomous Rollback DISABLED", "info");
      refreshAll();
    }
  });
}

async function renderWorldRacks() {
  const racks = await apiGet("/api/world/racks");
  const container = document.getElementById("racks-container");
  if (!racks || !container) return;

  container.innerHTML = "";

  racks.forEach(rack => {
    const rackCard = document.createElement("div");
    rackCard.className = "rack-card";

    const header = document.createElement("div");
    header.className = "rack-header";
    header.innerHTML = `
      <div>
        <strong>${rack.rack_id.toUpperCase()}</strong>
        <span style="font-size: 0.75rem; color: var(--text-secondary); margin-left: 0.5rem;">[${rack.switch_id}]</span>
      </div>
      <div>
        <span class="status-dot ${rack.online ? 'dot-online' : 'dot-offline'}"></span>
        <span style="font-size: 0.75rem; color: var(--text-secondary);">${rack.online ? 'Online' : 'Degraded'}</span>
      </div>
    `;
    rackCard.appendChild(header);

    const serverGrid = document.createElement("div");
    serverGrid.className = "server-grid";

    rack.servers.forEach(srv => {
      const tile = document.createElement("div");
      const isCanary = activeCanarySet.has(srv.server_id);
      const isOutage = !srv.online || (srv.current_state && !srv.current_state.observed);

      tile.className = `server-tile ${isCanary ? 'canary-active' : ''} ${isOutage ? 'outage' : ''}`;

      const st = srv.current_state || {};
      const cpuPct = Math.round((st.cpu_fraction || 0) * 100);
      const memMb = Math.round(st.memory_mb || srv.base_memory_mb);
      const memMax = Math.round(srv.memory_capacity_mb);
      const memPct = Math.min(100, Math.round((memMb / memMax) * 100));
      const latMs = (st.latency_ms || srv.base_latency_ms).toFixed(1);
      const errCount = st.error_count || 0;

      tile.innerHTML = `
        <div class="tile-top">
          <span class="server-id-label">${srv.server_id}</span>
          <div>
            <span class="badge badge-class">Cls ${srv.class_id}</span>
            <span class="badge ${isCanary ? 'badge-canary' : 'badge-version'}">${isCanary ? 'CANARY' : srv.installed_version}</span>
            <span class="status-dot ${srv.online ? 'dot-online' : 'dot-offline'}"></span>
          </div>
        </div>
        <div class="tile-metrics">
          <div>
            <span>CPU: ${cpuPct}%</span>
            <div class="metric-bar"><div class="metric-fill" style="width: ${cpuPct}%; background-color: ${cpuPct > 80 ? 'var(--accent-red)' : 'var(--accent-blue)'}"></div></div>
          </div>
          <div>
            <span>RAM: ${memMb}MB</span>
            <div class="metric-bar"><div class="metric-fill" style="width: ${memPct}%; background-color: ${memPct > 70 ? 'var(--accent-orange)' : 'var(--accent-green)'}"></div></div>
          </div>
          <div>
            <span>Lat: ${latMs}ms ${errCount > 0 ? `<b style="color:var(--accent-red)">(${errCount}e)</b>` : ''}</span>
          </div>
        </div>
      `;

      tile.addEventListener("click", () => showServerModal(srv));
      serverGrid.appendChild(tile);
    });

    rackCard.appendChild(serverGrid);
    container.appendChild(rackCard);
  });
}

async function showServerModal(srv) {
  const modal = document.getElementById("server-modal");
  const title = document.getElementById("modal-server-title");
  const body = document.getElementById("modal-server-body");
  if (!modal || !body) return;

  const isCanary = activeCanarySet.has(srv.server_id);
  const st = srv.current_state || {};
  const evidenceRes = await apiGet("/api/detector/evidence");
  const serverEvidence = (evidenceRes && evidenceRes.servers) ? evidenceRes.servers[srv.server_id] : null;

  if (title) {
    title.innerHTML = `<span>Server Inspection: <strong>${srv.server_id}</strong></span> <span class="badge ${isCanary ? 'badge-canary' : 'badge-version'}">${isCanary ? 'CANARY HOST' : 'BASELINE CONTROL'}</span>`;
  }

  let detectorSection = "";
  if (isCanary && serverEvidence) {
    const latEff = serverEvidence.latency_effect !== null ? `${(serverEvidence.latency_effect * 100).toFixed(2)}%` : "Pending";
    const ciStr = (serverEvidence.latency_ci_lower !== null && serverEvidence.latency_ci_upper !== null)
      ? `[${(serverEvidence.latency_ci_lower * 100).toFixed(2)}%, ${(serverEvidence.latency_ci_upper * 100).toFixed(2)}%]`
      : "Pending";
    const hostRisk = serverEvidence.host_risk !== null ? `${(serverEvidence.host_risk * 100).toFixed(1)}%` : "N/A";
    const memRate = serverEvidence.memory_drift_mb_per_min !== null ? `${serverEvidence.memory_drift_mb_per_min.toFixed(2)} MB/min` : "N/A";
    const headroom = serverEvidence.headroom_mb !== null ? `${serverEvidence.headroom_mb.toFixed(0)} MB` : "N/A";
    const tte = serverEvidence.tte_min !== null ? `${serverEvidence.tte_min.toFixed(1)} min` : "Infinite";

    detectorSection = `
      <div style="background: var(--bg-primary); padding: 1rem; border-radius: 6px; margin-top: 1rem; border: 1px solid var(--accent-purple);">
        <h4 style="color: var(--accent-purple); margin-bottom: 0.5rem; font-size: 0.9rem;">Hierarchical SSM & Memory Channel Evidence</h4>
        <div class="grid-2" style="font-size: 0.82rem; gap: 0.5rem;">
          <div>Latency Effect (a_hat): <strong>${latEff}</strong></div>
          <div>95% Credible Interval: <strong>${ciStr}</strong></div>
          <div>Harm Risk Pr(a_i > log(1.02)): <strong style="color: ${(serverEvidence.host_risk || 0) > 0.5 ? 'var(--accent-red)' : 'var(--text-primary)'}">${hostRisk}</strong></div>
          <div>Guardrail Status: <span class="badge ${serverEvidence.guardrail_status === 'NORMAL' ? 'badge-version' : 'badge-warning'}">${serverEvidence.guardrail_status}</span></div>
          <div>Memory Drift Rate: <strong>${memRate}</strong></div>
          <div>Capacity Headroom: <strong>${headroom}</strong></div>
          <div>Time-To-Exhaustion (TTE): <strong>${tte}</strong></div>
          <div>Persistence (>=10 min): <strong>${serverEvidence.is_persistent_mem ? 'YES (Confirmed)' : 'NO (Accumulating)'}</strong></div>
        </div>
      </div>
    `;
  } else {
    detectorSection = `
      <div style="background: var(--bg-primary); padding: 1rem; border-radius: 6px; margin-top: 1rem; border: 1px solid var(--accent-blue);">
        <h4 style="color: var(--accent-blue); margin-bottom: 0.5rem; font-size: 0.9rem;">Baseline Control Alignment</h4>
        <p style="font-size: 0.8rem; color: var(--text-secondary);">
          This instance serves as an unexposed concurrent control for Rack ${srv.rack_id} and Class ${srv.class_id}.
          Used for nested SVD fleet factor and rack nuisance removal with uncertainty propagation.
        </p>
      </div>
    `;
  }

  body.innerHTML = `
    <div class="grid-2" style="font-size: 0.85rem; margin-bottom: 0.5rem; gap: 0.5rem;">
      <div><strong>Rack ID:</strong> ${srv.rack_id}</div>
      <div><strong>Class:</strong> Class ${srv.class_id} (${srv.cpu_cores} Cores, ${srv.memory_capacity_mb}MB RAM)</div>
      <div><strong>Installed Version:</strong> <code>${srv.installed_version}</code></div>
      <div><strong>Online State:</strong> <span class="status-dot ${srv.online ? 'dot-online' : 'dot-offline'}"></span> ${srv.online ? 'Online' : 'Offline'}</div>
      <div><strong>Measured Latency:</strong> ${(st.latency_ms || srv.base_latency_ms).toFixed(2)} ms</div>
      <div><strong>Process Memory:</strong> ${(st.memory_mb || srv.base_memory_mb).toFixed(1)} MB (${Math.round(((st.memory_mb || srv.base_memory_mb) / srv.memory_capacity_mb) * 100)}%)</div>
      <div><strong>CPU Utilization:</strong> ${Math.round((st.cpu_fraction || 0) * 100)}%</div>
      <div><strong>Traffic & Errors:</strong> ${st.request_count || 0} req/min &bull; ${st.error_count || 0} errors</div>
    </div>
    ${detectorSection}
  `;

  modal.style.display = "flex";
}

// Modal event listeners
document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("modal-close-btn")?.addEventListener("click", () => {
    const modal = document.getElementById("server-modal");
    if (modal) modal.style.display = "none";
  });
  document.getElementById("server-modal")?.addEventListener("click", (e) => {
    if (e.target.id === "server-modal") e.target.style.display = "none";
  });
  document.getElementById("rollback-modal-close")?.addEventListener("click", () => {
    const modal = document.getElementById("rollback-confirm-modal");
    if (modal) modal.style.display = "none";
  });
  document.getElementById("rollback-modal-cancel")?.addEventListener("click", () => {
    const modal = document.getElementById("rollback-confirm-modal");
    if (modal) modal.style.display = "none";
  });
  document.getElementById("rollback-confirm-modal")?.addEventListener("click", (e) => {
    if (e.target.id === "rollback-confirm-modal") e.target.style.display = "none";
  });
});

// ---------------------------------------------------------------------
// Page 2: Releases & Deployments View
// ---------------------------------------------------------------------
function setupReleasesPage() {
  // Sample manifest auto-fill buttons
  document.querySelectorAll(".btn-sample-manifest").forEach(btn => {
    btn.addEventListener("click", () => {
      const p = btn.dataset.profile;
      fillSampleRelease(p);
      showToast(`Loaded ${p} sample manifest`, "info");
    });
  });

  // Allocation matrix builder
  buildAllocationMatrix();

  // Validate allocation button
  document.getElementById("btn-validate-alloc")?.addEventListener("click", validateSelectedAllocation);

  // Recommended default allocation button
  document.getElementById("btn-default-alloc")?.addEventListener("click", async () => {
    const res = await apiGet("/api/releases/default-allocation");
    if (res && res.recommended_canaries) {
      document.querySelectorAll(".alloc-checkbox").forEach(cb => {
        cb.checked = res.recommended_canaries.includes(cb.value);
      });
      validateSelectedAllocation();
      showToast("Applied recommended 1-per-rack balanced allocation", "info");
    }
  });

  // File Upload SHA-256 calculation (Unexecuted artifact)
  const fileUpload = document.getElementById("rel-artifact-upload");
  fileUpload?.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const fileInput = document.getElementById("rel-artifact-file");
    if (fileInput) fileInput.value = file.name;

    const hashDisplay = document.getElementById("artifact-hash-display");
    if (hashDisplay) hashDisplay.textContent = "Computing SHA-256 hash...";

    try {
      const arrayBuffer = await file.arrayBuffer();
      const hashBuffer = await crypto.subtle.digest("SHA-256", arrayBuffer);
      const hashArray = Array.from(new Uint8Array(hashBuffer));
      const hashHex = hashArray.map(b => b.toString(16).padStart(2, "0")).join("");

      if (hashDisplay) {
        hashDisplay.textContent = `SHA-256: ${hashHex} (${(file.size / 1024).toFixed(1)} KB - UNEXECUTED)`;
      }
      showToast(`Artifact hash calculated: ${hashHex.substring(0, 12)}...`, "info");
    } catch (err) {
      if (hashDisplay) hashDisplay.textContent = "Failed to compute checksum.";
    }
  });

  // Create Release Form
  document.getElementById("form-create-release")?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const releaseId = document.getElementById("rel-id").value.trim();
    const versionTag = document.getElementById("rel-version").value.trim();
    const pType = document.getElementById("rel-profile-type").value;
    const severity = parseFloat(document.getElementById("rel-severity").value);
    const duration = parseInt(document.getElementById("rel-duration").value, 10);
    const classIdVal = document.getElementById("rel-class-filter").value;
    const susceptibleClass = classIdVal === "all" ? null : parseInt(classIdVal, 10);

    const artifactName = document.getElementById("rel-artifact-file").value.trim() || "service_code.py";
    const artifactContent = `# Source build artifact for ${versionTag}\nVERSION = '${versionTag}'\n`;

    const body = {
      release_id: releaseId,
      version_tag: versionTag,
      behavior_profile: {
        profile_type: pType,
        severity: severity,
        duration: duration,
        susceptible_class_id: susceptibleClass,
      },
      artifact_filename: artifactName,
      artifact_content: artifactContent,
      metadata: { author: "dashboard-operator" },
    };

    const res = await apiPost("/api/releases", body);
    if (res) {
      showToast(`Release ${res.release_id} (${res.version_tag}) registered!`, "success");
      renderReleasesList();
      renderCommandHistory();
    }
  });

  // Deploy Canary
  document.getElementById("btn-deploy-canary")?.addEventListener("click", async () => {
    const releaseSelect = document.getElementById("select-deploy-release");
    const releaseId = releaseSelect ? releaseSelect.value : "";
    if (!releaseId) {
      showToast("Please select a release to deploy.", "error");
      return;
    }

    const selectedCanaries = getSelectedCanaries();
    if (selectedCanaries.length === 0) {
      showToast("Please select at least one canary host.", "error");
      return;
    }

    const res = await apiPost("/api/releases/deploy", {
      release_id: releaseId,
      canary_server_ids: selectedCanaries,
    });
    if (res) {
      showToast(`Canary release deployed to ${res.canary_servers.length} servers!`, "success");
      refreshAll();
    }
  });

  // Rollback Release with explicit confirmation modal
  document.getElementById("btn-rollback-canary")?.addEventListener("click", () => {
    const releaseSelect = document.getElementById("select-deploy-release");
    const releaseId = releaseSelect ? releaseSelect.value : "";
    if (!releaseId) {
      showToast("Please select a release to rollback.", "error");
      return;
    }

    const modal = document.getElementById("rollback-confirm-modal");
    const textEl = document.getElementById("rollback-modal-text");
    if (modal && textEl) {
      textEl.textContent = `Are you sure you want to rollback release "${releaseId}"? This will revert all ${activeCanarySet.size} canary servers to baseline version v1.0.0.`;
      modal.style.display = "flex";
    }
  });

  // Explicit confirmation button in modal
  document.getElementById("rollback-modal-confirm")?.addEventListener("click", async () => {
    const modal = document.getElementById("rollback-confirm-modal");
    if (modal) modal.style.display = "none";

    const releaseSelect = document.getElementById("select-deploy-release");
    const releaseId = releaseSelect ? releaseSelect.value : "";
    const restartProcess = document.getElementById("cb-restart-process").checked;
    const res = await apiPost("/api/releases/rollback", {
      release_id: releaseId,
      restart_process: restartProcess,
    });
    if (res) {
      showToast(`Rollback complete for ${res.rolled_back_servers.length} servers (Heap Cleared: ${res.memory_cleared})`, "success");
      refreshAll();
    }
  });
}

function fillSampleRelease(profile) {
  const pUpper = profile.toUpperCase();
  const idEl = document.getElementById("rel-id");
  const verEl = document.getElementById("rel-version");
  const typeEl = document.getElementById("rel-profile-type");
  const sevEl = document.getElementById("rel-severity");
  const durEl = document.getElementById("rel-duration");

  if (pUpper === "CLEAN") {
    idEl.value = "rel-v1.1-clean";
    verEl.value = "v1.1.0-clean";
    typeEl.value = "CLEAN";
    sevEl.value = "0.0";
    durEl.value = "120";
  } else if (pUpper === "MEMORY_LEAK") {
    idEl.value = "rel-v1.2-leak";
    verEl.value = "v1.2.0-leak";
    typeEl.value = "MEMORY_LEAK";
    sevEl.value = "35.0";
    durEl.value = "120";
  } else if (pUpper === "LATENCY_REGRESSION") {
    idEl.value = "rel-v1.3-latency";
    verEl.value = "v1.3.0-lat";
    typeEl.value = "LATENCY_REGRESSION";
    sevEl.value = "0.18";
    durEl.value = "120";
  } else if (pUpper === "CPU_REGRESSION") {
    idEl.value = "rel-v1.4-cpu";
    verEl.value = "v1.4.0-cpu";
    typeEl.value = "CPU_REGRESSION";
    sevEl.value = "0.35";
    durEl.value = "120";
  } else if (pUpper === "ERROR_REGRESSION") {
    idEl.value = "rel-v1.5-err";
    verEl.value = "v1.5.0-err";
    typeEl.value = "ERROR_REGRESSION";
    sevEl.value = "0.04";
    durEl.value = "120";
  }
}

function buildAllocationMatrix() {
  const container = document.getElementById("alloc-matrix-container");
  if (!container) return;

  container.innerHTML = "";
  for (let r = 0; r < 4; r++) {
    const rackDiv = document.createElement("div");
    rackDiv.style.marginBottom = "0.75rem";
    rackDiv.innerHTML = `<strong>RACK-${r}:</strong> `;

    for (let s = 0; s < 6; s++) {
      const sid = `srv-r${r}-s${s}`;
      const lbl = document.createElement("label");
      lbl.style.display = "inline-flex";
      lbl.style.alignItems = "center";
      lbl.style.marginRight = "0.75rem";
      lbl.style.fontSize = "0.8rem";

      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.className = "alloc-checkbox";
      cb.value = sid;
      cb.style.marginRight = "0.25rem";
      cb.addEventListener("change", validateSelectedAllocation);

      lbl.appendChild(cb);
      lbl.appendChild(document.createTextNode(sid));
      rackDiv.appendChild(lbl);
    }
    container.appendChild(rackDiv);
  }
}

function getSelectedCanaries() {
  const cbs = document.querySelectorAll(".alloc-checkbox:checked");
  return Array.from(cbs).map(cb => cb.value);
}

async function validateSelectedAllocation() {
  const canaries = getSelectedCanaries();
  const alertBox = document.getElementById("alloc-validation-alert");
  if (!alertBox) return;

  if (canaries.length === 0) {
    alertBox.className = "alert-box alert-warning";
    alertBox.textContent = "No canaries selected. Select at least 1 canary to validate identifiability.";
    return;
  }

  const res = await apiPost("/api/releases/validate", { canary_server_ids: canaries });
  if (!res) return;

  if (res.is_valid) {
    alertBox.className = "alert-box alert-success";
    alertBox.innerHTML = `
      <strong>Valid Allocation:</strong> ${res.total_canaries} canaries, ${res.total_controls} controls across all racks & classes.
      ${res.warnings.length > 0 ? `<br><em>Warnings: ${res.warnings.join("; ")}</em>` : ""}
    `;
  } else {
    alertBox.className = "alert-box alert-danger";
    alertBox.innerHTML = `
      <strong>IDENTIFIABILITY REJECTION:</strong> ${res.issues.join("<br>")}
    `;
  }
}

async function renderReleasesList() {
  const releases = await apiGet("/api/releases");
  const tableBody = document.getElementById("releases-table-body");
  const selectDeploy = document.getElementById("select-deploy-release");
  if (!releases || !tableBody) return;

  tableBody.innerHTML = "";
  if (selectDeploy) selectDeploy.innerHTML = "";

  releases.forEach(r => {
    const row = document.createElement("tr");
    const p = r.behavior_profile || {};
    const art = r.artifact || {};

    row.innerHTML = `
      <td><strong>${r.release_id}</strong></td>
      <td><span class="badge badge-version">${r.version_tag}</span></td>
      <td><span class="badge badge-canary">${p.profile_type}</span> (sev=${p.severity})</td>
      <td>
        ${art.artifact_filename ? `<code>${art.artifact_filename}</code> <span class="badge badge-warning">UNEXECUTED</span>` : 'None'}
      </td>
      <td>${r.created_at_tick} min</td>
    `;
    tableBody.appendChild(row);

    if (selectDeploy) {
      const opt = document.createElement("option");
      opt.value = r.release_id;
      opt.textContent = `${r.release_id} (${r.version_tag} - ${p.profile_type})`;
      selectDeploy.appendChild(opt);
    }
  });
}

async function renderCommandHistory() {
  const history = await apiGet("/api/commands/history");
  const tableBody = document.getElementById("command-history-table-body");
  if (!history || !tableBody) return;

  tableBody.innerHTML = "";
  if (history.length === 0) {
    tableBody.innerHTML = `<tr><td colspan="4" style="text-align:center; color:var(--text-muted);">No commands recorded yet.</td></tr>`;
    return;
  }

  // Show latest commands first
  history.slice().reverse().forEach(cmd => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td><code>${cmd.command_id}</code></td>
      <td><strong>${cmd.action}</strong></td>
      <td>T=${cmd.tick}</td>
      <td><span class="badge ${cmd.status === 'DEPLOYED' || cmd.action === 'CREATE_RELEASE' ? 'badge-canary' : 'badge-warning'}">${cmd.status || 'OK'}</span></td>
    `;
    tableBody.appendChild(row);
  });
}

// ---------------------------------------------------------------------
// Page 3: Fault Lab View (Developer View)
// ---------------------------------------------------------------------
function setupFaultLabPage() {
  // Manual Fault Injector
  document.getElementById("form-inject-fault")?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const fType = document.getElementById("fault-type-select").value;
    const scope = document.getElementById("fault-scope-select").value;
    const targetId = document.getElementById("fault-target-input").value.trim();
    const severity = parseFloat(document.getElementById("fault-severity-input").value);
    const duration = parseInt(document.getElementById("fault-duration-input").value, 10);

    const body = {
      fault_type: fType,
      scope: scope,
      target_id: targetId,
      severity: severity,
      duration: duration,
    };

    const res = await apiPost("/api/faults/inject", body);
    if (res) {
      showToast(`Fault injected: ${res.fault_id} on ${res.target_id}`, "error");
      renderFaultTruth();
    }
  });

  // Seeded Stochastic Hazards Form
  document.getElementById("form-hazards-config")?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const enable = document.getElementById("cb-enable-stochastic").checked;
    const serverRate = parseFloat(document.getElementById("hazard-server-rate").value);
    const rackRate = parseFloat(document.getElementById("hazard-rack-rate").value);
    const seed = parseInt(document.getElementById("hazard-seed-input").value, 10);

    const res = await apiPost("/api/faults/hazards", {
      enable_stochastic: enable,
      server_outage_hazard_per_min: serverRate,
      rack_latency_hazard_per_min: rackRate,
      seed: seed,
    });
    if (res) {
      showToast(`Stochastic hazards ${res.enable_stochastic ? 'ENABLED' : 'DISABLED'} (seed=${res.seed})`, "info");
    }
  });
}

async function renderFaultTruth() {
  const truth = await apiGet("/api/faults/truth");
  const container = document.getElementById("fault-truth-table-body");
  if (!truth || !container) return;

  container.innerHTML = "";
  const faults = truth.injected_faults || truth.registered_faults || [];

  if (faults.length === 0) {
    container.innerHTML = `<tr><td colspan="6" style="text-align:center; color:var(--text-muted);">No injected faults recorded.</td></tr>`;
    return;
  }

  faults.forEach(f => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td><code>${f.fault_id}</code></td>
      <td><span class="badge ${f.category === 'release' || f.category === 'RELEASE' ? 'badge-canary' : 'badge-warning'}">${f.fault_type}</span></td>
      <td>${f.scope}: <b>${f.target_id}</b></td>
      <td>T=${f.onset_tick} to T=${f.onset_tick + f.duration} (${f.duration}m)</td>
      <td>${f.severity}</td>
      <td>${f.is_active ? '<b style="color:var(--accent-red)">ACTIVE</b>' : '<span style="color:var(--text-muted)">Scheduled / Expired</span>'}</td>
    `;
    container.appendChild(row);
  });
}

// ---------------------------------------------------------------------
// Page 4: Monitoring View
// ---------------------------------------------------------------------
async function renderMonitoringView() {
  const [adv, timeline, detTimeline] = await Promise.all([
    apiGet("/api/monitoring/advisory"),
    apiGet("/api/monitoring/timeline?limit=60"),
    apiGet("/api/detector/timeline?limit=60"),
  ]);

  if (!adv || !timeline) return;
  cachedTimeline = timeline;

  // Update Model / Policy Badges
  const modelBadge = document.getElementById("mon-model-badge");
  const policyBadge = document.getElementById("mon-policy-badge");
  const eligBadge = document.getElementById("mon-eligibility-badge");
  const intervBadge = document.getElementById("mon-intervention-badge");

  if (modelBadge) modelBadge.textContent = adv.model_id || "MODEL-HSSM-V1";
  if (policyBadge) policyBadge.textContent = adv.policy_id || "POLICY-SIDAK-0.05";
  if (eligBadge) {
    const elig = adv.data_eligibility || {};
    if (elig.eligible) {
      eligBadge.textContent = `Data Eligible (${elig.active_canaries || activeCanarySet.size} canaries)`;
      eligBadge.className = "badge badge-canary";
    } else {
      eligBadge.textContent = elig.is_unidentifiable ? "UNIDENTIFIABLE: Confounded/Absent controls" : "Ineligible Data";
      eligBadge.className = "badge badge-warning";
    }
  }
  if (intervBadge) {
    if (adv.auto_rollback_triggered) {
      intervBadge.style.display = "inline-block";
      intervBadge.textContent = `AUTONOMOUS INTERVENTION (T=${adv.intervention_tick})`;
    } else {
      intervBadge.style.display = "none";
    }
  }

  // Update Advisory Card
  const scoreEl = document.getElementById("mon-score-val");
  const threshEl = document.getElementById("mon-threshold-val");
  const alertTickEl = document.getElementById("mon-alert-tick-val");

  if (scoreEl) {
    if (adv.ssm_connected && adv.release_score !== null && adv.release_score !== undefined) {
      scoreEl.textContent = adv.release_score.toFixed(3);
    } else {
      scoreEl.textContent = "DETECTOR NOT CONNECTED";
    }
  }

  if (threshEl) {
    const tVal = adv.alert_threshold !== null && adv.alert_threshold !== undefined ? adv.alert_threshold : -0.693;
    threshEl.textContent = `Threshold: ${tVal.toFixed(3)}`;
  }

  if (alertTickEl) {
    alertTickEl.textContent = adv.first_alert_tick !== null && adv.first_alert_tick !== undefined
      ? `First Alert: T = ${adv.first_alert_tick} min`
      : "First Alert: None";
  }

  // Dual Advisory Status Boxes
  const infraStatusEl = document.getElementById("mon-infra-status");
  const infraReasonEl = document.getElementById("mon-infra-reason");
  const relStatusEl = document.getElementById("mon-release-status");
  const relReasonEl = document.getElementById("mon-release-reason");
  const disclaimerEl = document.getElementById("mon-epistemic-disclaimer");

  if (infraStatusEl) {
    infraStatusEl.textContent = adv.infrastructure_advisory;
    infraStatusEl.style.color = (adv.infrastructure_advisory === "NORMAL") ? "var(--accent-green)" : "var(--accent-red)";
  }
  if (infraReasonEl) {
    const parts = (adv.advisory_reason || "").split("|");
    infraReasonEl.textContent = parts.length > 1 ? parts[1].trim() : "Infrastructure telemetry nominal.";
  }

  if (relStatusEl) {
    relStatusEl.textContent = adv.canary_release_advisory;
    let col = "var(--accent-green)";
    if (adv.canary_release_advisory === "DETECTOR NOT CONNECTED") col = "var(--text-secondary)";
    else if (adv.canary_release_advisory === "PAUSE RECOMMENDED") col = "var(--accent-red)";
    else if (adv.canary_release_advisory.includes("INSUFFICIENT")) col = "var(--accent-yellow)";
    else if (adv.canary_release_advisory === "ROLLED BACK") col = "var(--accent-blue)";
    relStatusEl.style.color = col;
  }
  if (relReasonEl) {
    const parts = (adv.advisory_reason || "").split("|");
    relReasonEl.textContent = parts[0].trim();
  }

  if (disclaimerEl && adv.epistemic_disclaimer) {
    disclaimerEl.textContent = adv.epistemic_disclaimer;
  }

  const firstAlertTick = adv.first_alert_tick;

  // Render HTML5 Canvas Charts
  renderChart("chart-latency", timeline, "latency_ms", "Latency (ms)", 80, 160, null, false, firstAlertTick);
  renderChart("chart-memory", timeline, "memory_mb", "Memory (MB)", 1500, 17000, [11468, 14745, 16384], false, firstAlertTick);
  renderChart("chart-cpu", timeline, "cpu_fraction", "CPU Fraction", 0.0, 1.0, null, true, firstAlertTick);
  renderChart("chart-errors", timeline, "errors", "Error Count", 0, 50, null, false, firstAlertTick);

  // Dedicated Detector Charts
  renderEffectChart("chart-effect", detTimeline || [], firstAlertTick);
  renderMemoryDriftChart("chart-memdrift", detTimeline || [], firstAlertTick);
}

function renderChart(canvasId, timeline, metricKey, label, yMin, yMax, thresholds = null, isPct = false, firstAlertTick = null) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width = canvas.parentElement.clientWidth;
  const h = canvas.height = 200;

  ctx.clearRect(0, 0, w, h);

  if (timeline.length < 2) {
    ctx.fillStyle = "#64748b";
    ctx.font = "12px sans-serif";
    ctx.fillText("Awaiting simulation telemetry series...", 20, h / 2);
    return;
  }

  const padLeft = 45;
  const padBottom = 25;
  const padTop = 15;
  const padRight = 15;
  const plotW = w - padLeft - padRight;
  const plotH = h - padTop - padBottom;

  let minVal = yMin;
  let maxVal = yMax;
  timeline.forEach(d => {
    const cVal = d.canary ? d.canary[metricKey] : 0;
    const kVal = d.control ? d.control[metricKey] : 0;
    if (cVal > maxVal) maxVal = cVal * 1.1;
    if (kVal > maxVal) maxVal = kVal * 1.1;
  });

  const getY = (val) => padTop + plotH - ((val - minVal) / Math.max(1e-4, maxVal - minVal)) * plotH;
  const getX = (idx) => padLeft + (idx / (timeline.length - 1)) * plotW;

  // Draw Grid lines
  ctx.strokeStyle = "#334155";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 4; i++) {
    const yVal = minVal + (i / 4) * (maxVal - minVal);
    const y = getY(yVal);
    ctx.moveTo(padLeft, y);
    ctx.lineTo(w - padRight, y);

    ctx.fillStyle = "#64748b";
    ctx.font = "10px monospace";
    const labelStr = isPct ? `${Math.round(yVal * 100)}%` : Math.round(yVal);
    ctx.fillText(labelStr, 5, y + 3);
  }
  ctx.stroke();

  // Draw Threshold dashed lines if provided
  if (thresholds) {
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = "#ef4444";
    thresholds.forEach(tVal => {
      if (tVal >= minVal && tVal <= maxVal) {
        const y = getY(tVal);
        ctx.beginPath();
        ctx.moveTo(padLeft, y);
        ctx.lineTo(w - padRight, y);
        ctx.stroke();
      }
    });
    ctx.setLineDash([]);
  }

  // Draw Control Series (Blue)
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 2;
  ctx.beginPath();
  timeline.forEach((d, i) => {
    const val = d.control ? d.control[metricKey] : minVal;
    const x = getX(i);
    const y = getY(val);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Draw Canary Series (Purple)
  ctx.strokeStyle = "#a855f7";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  timeline.forEach((d, i) => {
    const val = d.canary ? d.canary[metricKey] : minVal;
    const x = getX(i);
    const y = getY(val);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // First Alert Marker vertical line
  if (firstAlertTick !== null && firstAlertTick !== undefined) {
    const alertIdx = timeline.findIndex(d => d.tick === firstAlertTick);
    if (alertIdx >= 0) {
      const xAlert = getX(alertIdx);
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = "#f97316";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(xAlert, padTop);
      ctx.lineTo(xAlert, padTop + plotH);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#f97316";
      ctx.font = "10px sans-serif";
      ctx.fillText(`Alert (T=${firstAlertTick})`, Math.max(padLeft, xAlert - 25), padTop + 20);
    }
  }

  // Legend
  ctx.fillStyle = "#38bdf8";
  ctx.fillRect(w - 140, 8, 10, 10);
  ctx.fillStyle = "#94a3b8";
  ctx.font = "11px sans-serif";
  ctx.fillText("Control", w - 125, 17);

  ctx.fillStyle = "#a855f7";
  ctx.fillRect(w - 75, 8, 10, 10);
  ctx.fillStyle = "#94a3b8";
  ctx.fillText("Canary", w - 60, 17);
}

function renderEffectChart(canvasId, detTimeline, firstAlertTick = null) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width = canvas.parentElement.clientWidth;
  const h = canvas.height = 200;

  ctx.clearRect(0, 0, w, h);

  if (!detTimeline || detTimeline.length < 2) {
    ctx.fillStyle = "#64748b";
    ctx.font = "12px sans-serif";
    ctx.fillText("Awaiting SSM effect estimate series...", 20, h / 2);
    return;
  }

  const padLeft = 50;
  const padBottom = 25;
  const padTop = 15;
  const padRight = 15;
  const plotW = w - padLeft - padRight;
  const plotH = h - padTop - padBottom;

  let minVal = -0.05;
  let maxVal = 0.25;
  detTimeline.forEach(d => {
    if (d.canary_effect_ci_lower < minVal) minVal = d.canary_effect_ci_lower - 0.02;
    if (d.canary_effect_ci_upper > maxVal) maxVal = d.canary_effect_ci_upper + 0.02;
  });

  const getY = (val) => padTop + plotH - ((val - minVal) / Math.max(1e-4, maxVal - minVal)) * plotH;
  const getX = (idx) => padLeft + (idx / (detTimeline.length - 1)) * plotW;

  // Grid lines
  ctx.strokeStyle = "#334155";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 4; i++) {
    const yVal = minVal + (i / 4) * (maxVal - minVal);
    const y = getY(yVal);
    ctx.moveTo(padLeft, y);
    ctx.lineTo(w - padRight, y);
    ctx.fillStyle = "#64748b";
    ctx.font = "10px monospace";
    ctx.fillText(`${(yVal * 100).toFixed(1)}%`, 5, y + 3);
  }
  ctx.stroke();

  // Baseline 0 line
  const y0 = getY(0.0);
  ctx.strokeStyle = "#64748b";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(padLeft, y0);
  ctx.lineTo(w - padRight, y0);
  ctx.stroke();

  // Harm Threshold line kappa = log(1.02) ~ 0.0198
  const kappa = 0.0198;
  if (kappa >= minVal && kappa <= maxVal) {
    const yKappa = getY(kappa);
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = "#ef4444";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padLeft, yKappa);
    ctx.lineTo(w - padRight, yKappa);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#ef4444";
    ctx.font = "10px sans-serif";
    ctx.fillText("Harm Boundary kappa=log(1.02)", padLeft + 5, yKappa - 4);
  }

  // Draw 95% Credible Interval Shaded Band
  ctx.fillStyle = "rgba(168, 85, 247, 0.2)";
  ctx.beginPath();
  detTimeline.forEach((d, i) => {
    const x = getX(i);
    const yLo = getY(d.canary_effect_ci_lower);
    if (i === 0) ctx.moveTo(x, yLo);
    else ctx.lineTo(x, yLo);
  });
  for (let i = detTimeline.length - 1; i >= 0; i--) {
    const x = getX(i);
    const yHi = getY(detTimeline[i].canary_effect_ci_upper);
    ctx.lineTo(x, yHi);
  }
  ctx.closePath();
  ctx.fill();

  // Draw Canary Effect Mean line (Purple)
  ctx.strokeStyle = "#a855f7";
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  detTimeline.forEach((d, i) => {
    const x = getX(i);
    const y = getY(d.canary_effect_mean);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // First Alert Marker
  if (firstAlertTick !== null && firstAlertTick !== undefined) {
    const alertIdx = detTimeline.findIndex(d => d.tick === firstAlertTick);
    if (alertIdx >= 0) {
      const xAlert = getX(alertIdx);
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = "#f97316";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(xAlert, padTop);
      ctx.lineTo(xAlert, padTop + plotH);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#f97316";
      ctx.font = "10px sans-serif";
      ctx.fillText(`Alert (T=${firstAlertTick})`, Math.max(padLeft, xAlert - 25), padTop + 22);
    }
  }

  // Legend
  ctx.fillStyle = "#a855f7";
  ctx.fillRect(w - 180, 8, 10, 10);
  ctx.fillStyle = "#94a3b8";
  ctx.font = "11px sans-serif";
  ctx.fillText("Effect (a_hat) +/- 95% CI", w - 165, 17);
}

function renderMemoryDriftChart(canvasId, detTimeline, firstAlertTick = null) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width = canvas.parentElement.clientWidth;
  const h = canvas.height = 200;

  ctx.clearRect(0, 0, w, h);

  if (!detTimeline || detTimeline.length < 2) {
    ctx.fillStyle = "#64748b";
    ctx.font = "12px sans-serif";
    ctx.fillText("Awaiting memory growth series...", 20, h / 2);
    return;
  }

  const padLeft = 45;
  const padBottom = 25;
  const padTop = 15;
  const padRight = 15;
  const plotW = w - padLeft - padRight;
  const plotH = h - padTop - padBottom;

  let minVal = -2.0;
  let maxVal = 10.0;
  detTimeline.forEach(d => {
    const hosts = d.hosts || {};
    Object.values(hosts).forEach(h => {
      if (h.memory_drift_mb_per_min !== null && h.memory_drift_mb_per_min > maxVal) {
        maxVal = h.memory_drift_mb_per_min * 1.15;
      }
    });
  });

  const getY = (val) => padTop + plotH - ((val - minVal) / Math.max(1e-4, maxVal - minVal)) * plotH;
  const getX = (idx) => padLeft + (idx / (detTimeline.length - 1)) * plotW;

  // Grid lines
  ctx.strokeStyle = "#334155";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 4; i++) {
    const yVal = minVal + (i / 4) * (maxVal - minVal);
    const y = getY(yVal);
    ctx.moveTo(padLeft, y);
    ctx.lineTo(w - padRight, y);
    ctx.fillStyle = "#64748b";
    ctx.font = "10px monospace";
    ctx.fillText(`${yVal.toFixed(1)}M/m`, 5, y + 3);
  }
  ctx.stroke();

  // Zero line
  const y0 = getY(0.0);
  ctx.strokeStyle = "#64748b";
  ctx.beginPath();
  ctx.moveTo(padLeft, y0);
  ctx.lineTo(w - padRight, y0);
  ctx.stroke();

  // Memory Boundary line kappa_M = 0.50 MB/min
  const kappaM = 0.50;
  if (kappaM >= minVal && kappaM <= maxVal) {
    const yKappa = getY(kappaM);
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = "#ef4444";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padLeft, yKappa);
    ctx.lineTo(w - padRight, yKappa);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#ef4444";
    ctx.font = "10px sans-serif";
    ctx.fillText("Boundary kappa_M = 0.50 MB/min", padLeft + 5, yKappa - 4);
  }

  // Draw Drift Rate Lines for active canaries
  ctx.strokeStyle = "#f59e0b"; // amber
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  detTimeline.forEach((d, i) => {
    let meanDrift = 0.0;
    let count = 0;
    Object.values(d.hosts || {}).forEach(h => {
      if (h.role === "CANARY" && h.memory_drift_mb_per_min !== null) {
        meanDrift += h.memory_drift_mb_per_min;
        count++;
      }
    });
    const avg = count > 0 ? meanDrift / count : 0.0;
    const x = getX(i);
    const y = getY(avg);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // First Alert Marker
  if (firstAlertTick !== null && firstAlertTick !== undefined) {
    const alertIdx = detTimeline.findIndex(d => d.tick === firstAlertTick);
    if (alertIdx >= 0) {
      const xAlert = getX(alertIdx);
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = "#f97316";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(xAlert, padTop);
      ctx.lineTo(xAlert, padTop + plotH);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#f97316";
      ctx.font = "10px sans-serif";
      ctx.fillText(`Alert (T=${firstAlertTick})`, Math.max(padLeft, xAlert - 25), padTop + 22);
    }
  }

  // Legend
  ctx.fillStyle = "#f59e0b";
  ctx.fillRect(w - 180, 8, 10, 10);
  ctx.fillStyle = "#94a3b8";
  ctx.font = "11px sans-serif";
  ctx.fillText("Canary Memory Drift Rate", w - 165, 17);
}

// ---------------------------------------------------------------------
// Page 5: Evaluation & Run Replay View
// ---------------------------------------------------------------------
function setupEvaluationPage() {
  // Replay scrubber slider setup
  const scrubber = document.getElementById("replay-scrubber");
  const tickLabel = document.getElementById("replay-tick-label");

  scrubber?.addEventListener("input", (e) => {
    currentReplayIndex = parseInt(e.target.value, 10);
    updateReplayTickDisplay();
  });

  document.getElementById("btn-replay-play")?.addEventListener("click", () => {
    if (isReplaying) return;
    isReplaying = true;
    showToast("Replay started", "info");
    replayTimer = setInterval(() => {
      if (!cachedTimeline || currentReplayIndex >= cachedTimeline.length - 1) {
        clearInterval(replayTimer);
        isReplaying = false;
        showToast("Replay reached end of timeline", "info");
        return;
      }
      currentReplayIndex++;
      if (scrubber) scrubber.value = currentReplayIndex;
      updateReplayTickDisplay();
    }, 500);
  });

  document.getElementById("btn-replay-pause")?.addEventListener("click", () => {
    isReplaying = false;
    if (replayTimer) clearInterval(replayTimer);
    showToast("Replay paused", "info");
  });

  document.getElementById("btn-replay-step")?.addEventListener("click", () => {
    if (cachedTimeline && currentReplayIndex < cachedTimeline.length - 1) {
      currentReplayIndex++;
      if (scrubber) scrubber.value = currentReplayIndex;
      updateReplayTickDisplay();
    }
  });

  // Export Full Run
  document.getElementById("btn-export-run")?.addEventListener("click", async () => {
    const bundle = await apiGet("/api/runs/export");
    if (bundle) {
      downloadFile(JSON.stringify(bundle, null, 2), "canary_run_bundle.json", "application/json");
      showToast("Full run bundle exported!", "success");
    }
  });

  // Import Saved Run
  const importInput = document.getElementById("input-import-run");
  importInput?.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    try {
      const text = await file.text();
      const bundle = JSON.parse(text);
      const res = await apiPost("/api/runs/import", { bundle: bundle });
      if (res && res.status === "LOADED") {
        showToast(`Imported run with ${res.total_ticks} ticks!`, "success");
        refreshAll();
      }
    } catch (err) {
      showToast(`Failed to parse run bundle: ${err.message}`, "error");
    }
  });

  // Download Telemetry CSV
  document.getElementById("btn-download-csv")?.addEventListener("click", () => {
    if (cachedTimeline.length === 0) {
      showToast("No telemetry records to export yet.", "error");
      return;
    }
    let csv = "tick,canary_latency_ms,control_latency_ms,canary_memory_mb,control_memory_mb,canary_cpu,control_cpu,canary_errors,infra_advisory,release_advisory\n";
    cachedTimeline.forEach(d => {
      csv += `${d.tick},${d.canary.latency_ms},${d.control.latency_ms},${d.canary.memory_mb},${d.control.memory_mb},${d.canary.cpu_fraction},${d.control.cpu_fraction},${d.canary.errors},"${d.infrastructure_advisory}","${d.canary_release_advisory}"\n`;
    });
    downloadFile(csv, "canary_telemetry_history.csv", "text/csv");
    showToast("Telemetry CSV exported!", "success");
  });

  // Download Evaluation JSON
  document.getElementById("btn-download-json")?.addEventListener("click", async () => {
    const summary = await apiGet("/api/evaluation/summary");
    if (summary) {
      downloadFile(JSON.stringify(summary, null, 2), "canary_evaluation_summary.json", "application/json");
      showToast("Evaluation summary exported!", "success");
    }
  });
}

function updateReplayTickDisplay() {
  const scrubber = document.getElementById("replay-scrubber");
  const tickLabel = document.getElementById("replay-tick-label");
  if (!cachedTimeline || cachedTimeline.length === 0) return;

  const entry = cachedTimeline[currentReplayIndex] || cachedTimeline[0];
  if (tickLabel) tickLabel.textContent = `Tick: ${entry.tick}`;

  // Slice timeline up to current scrubber tick and re-render charts
  const replaySlice = cachedTimeline.slice(0, currentReplayIndex + 1);
  if (replaySlice.length >= 2) {
    renderChart("chart-latency", replaySlice, "latency_ms", "Latency (ms)", 80, 160);
    renderChart("chart-memory", replaySlice, "memory_mb", "Memory (MB)", 1500, 17000, [11468, 14745, 16384]);
    renderChart("chart-cpu", replaySlice, "cpu_fraction", "CPU Fraction", 0.0, 1.0, null, true);
    renderChart("chart-errors", replaySlice, "errors", "Error Count", 0, 50);
  }
}

async function renderEvaluationView() {
  const summary = await apiGet("/api/evaluation/summary");
  if (!summary) return;

  // Sync scrubber bounds
  const scrubber = document.getElementById("replay-scrubber");
  if (scrubber && cachedTimeline && cachedTimeline.length > 0) {
    scrubber.max = cachedTimeline.length - 1;
  }

  const trueFaultEl = document.getElementById("eval-true-fault");
  const detectorPauseEl = document.getElementById("eval-detector-pause");
  const verdictEl = document.getElementById("eval-verdict");

  if (trueFaultEl) trueFaultEl.textContent = summary.has_true_release_fault ? "YES (True Fault Active)" : "NO (Clean/Null)";
  if (detectorPauseEl) {
    if (summary.canary_release_advisory === "DETECTOR NOT CONNECTED") {
      detectorPauseEl.textContent = "DETECTOR NOT CONNECTED";
    } else {
      detectorPauseEl.textContent = summary.detector_recommends_pause ? "PAUSE RECOMMENDED" : "RUNNING";
    }
  }

  if (verdictEl) {
    if (summary.canary_release_advisory === "DETECTOR NOT CONNECTED") {
      verdictEl.textContent = "DETECTOR NOT CONNECTED (Awaiting SSM)";
      verdictEl.className = "badge badge-class";
    } else if (summary.has_true_release_fault && summary.detector_recommends_pause) {
      verdictEl.textContent = "CORRECT PAUSE (True Positive)";
      verdictEl.className = "badge badge-canary";
    } else if (!summary.has_true_release_fault && !summary.detector_recommends_pause) {
      verdictEl.textContent = "CORRECT RELEASE (True Negative)";
      verdictEl.className = "badge badge-version";
    } else if (!summary.has_true_release_fault && summary.detector_recommends_pause) {
      verdictEl.textContent = "FALSE ALARM (Type I Error)";
      verdictEl.className = "badge badge-warning";
    } else {
      verdictEl.textContent = "ACCUMULATING EVIDENCE (Evaluating)";
      verdictEl.className = "badge badge-class";
    }
  }
}

function downloadFile(content, filename, type) {
  const blob = new Blob([content], { type: type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
