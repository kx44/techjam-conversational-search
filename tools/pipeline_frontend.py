from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator import local_evaluator as ev
from starter import agent as agent_mod
from starter.agent import Agent


MAX_TURNS = ev.MAX_TURNS
TOP_K = ev.TOP_K


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Shopping Copilot Pipeline</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: light-dark(#f4f6f8, #101214);
      --surface: light-dark(#ffffff, #181b1f);
      --soft: light-dark(#eef2f6, #21262d);
      --softer: light-dark(#f8fafc, #15181c);
      --text: light-dark(#17202a, #edf2f7);
      --muted: light-dark(#657386, #a6b0be);
      --border: light-dark(#d7dee8, #343b46);
      --blue: light-dark(#1d65d8, #76adff);
      --green: light-dark(#187247, #68d391);
      --orange: light-dark(#a35d00, #f6b15f);
      --purple: light-dark(#6c48b8, #bda7ff);
      --red: light-dark(#b3261e, #ff8a82);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-width: 320px;
      background: var(--bg);
      color: var(--text);
      font-size: 14px;
    }

    button, select {
      font: inherit;
      color: var(--text);
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 7px;
      min-height: 38px;
    }

    button {
      padding: 0 13px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 7px;
    }

    button.primary, .pill.active {
      background: var(--blue);
      color: white;
      border-color: transparent;
    }

    button:disabled { opacity: .5; cursor: not-allowed; }
    select { width: 100%; padding: 0 36px 0 11px; }
    input[type="checkbox"] { width: 18px; height: 18px; accent-color: var(--blue); }

    .runner-note {
      color: var(--muted);
      font-size: 13px;
      min-height: 18px;
    }

    .runner-note.active { color: var(--orange); }

    .shell {
      width: min(1480px, 100%);
      margin: 0 auto;
      padding: 18px;
      display: grid;
      grid-template-columns: 300px minmax(420px, 1fr) 360px;
      gap: 16px;
      align-items: start;
    }

    .sidebar, .workbench, .recommendations {
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    .panel {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
    }

    .brand {
      padding: 18px;
      min-height: 112px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      gap: 14px;
      background:
        linear-gradient(135deg, color-mix(in srgb, var(--blue) 14%, transparent), transparent 42%),
        var(--surface);
    }

    h1, h2, h3, p { margin: 0; }
    h1 { font-size: 22px; font-weight: 600; letter-spacing: 0; }
    h2 {
      padding: 13px 15px;
      font-size: 14px;
      font-weight: 600;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }
    h3 { font-size: 13px; font-weight: 600; }

    .subtle { color: var(--muted); }
    .body { padding: 14px; }

    .score-strip {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }

    .score, .outcome, .evidence-tile {
      background: var(--soft);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px;
    }

    .score span, .outcome span, .evidence-tile span {
      display: block;
      color: var(--muted);
      font-size: 12px;
    }

    .score strong, .outcome strong {
      display: block;
      margin-top: 3px;
      font-size: 19px;
      font-weight: 600;
      font-variant-numeric: tabular-nums;
    }

    .field {
      display: grid;
      gap: 6px;
      margin-bottom: 12px;
    }

    .field label, .switch span {
      color: var(--muted);
      font-size: 13px;
    }

    .switch {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 9px 0;
    }

    .actions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }

    .actions .primary { grid-column: 1 / -1; }

    .outcomes {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 25px;
      padding: 0 8px;
      border-radius: 999px;
      background: var(--soft);
      color: var(--muted);
      border: 1px solid var(--border);
      font-size: 12px;
      white-space: nowrap;
    }

    .badge.good { color: var(--green); border-color: color-mix(in srgb, var(--green) 45%, var(--border)); }
    .badge.warn { color: var(--orange); border-color: color-mix(in srgb, var(--orange) 45%, var(--border)); }
    .badge.hot { color: var(--purple); border-color: color-mix(in srgb, var(--purple) 45%, var(--border)); }

    .turns {
      display: flex;
      flex-direction: column;
      gap: 7px;
    }

    .turn-button {
      width: 100%;
      min-height: auto;
      padding: 9px 10px;
      justify-content: space-between;
      background: var(--softer);
    }

    .turn-button.active {
      border-color: var(--blue);
      background: color-mix(in srgb, var(--blue) 11%, var(--surface));
    }

    .conversation {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }

    .bubble {
      min-height: 132px;
      padding: 13px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: var(--soft);
    }

    .bubble.customer { border-left: 4px solid var(--green); }
    .bubble.agent { border-left: 4px solid var(--blue); }
    .speaker {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 9px;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .04em;
    }

    .pipeline {
      display: grid;
      gap: 9px;
    }

    .stage {
      display: grid;
      grid-template-columns: 34px 1fr auto;
      gap: 11px;
      align-items: center;
      padding: 11px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--softer);
    }

    .stage-number {
      width: 28px;
      height: 28px;
      display: grid;
      place-items: center;
      border-radius: 50%;
      background: var(--soft);
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }

    .stage.active {
      border-color: var(--blue);
      background: color-mix(in srgb, var(--blue) 9%, var(--surface));
    }

    .stage.good { border-color: var(--green); }
    .stage.warn { border-color: var(--orange); }

    .stage-title { display: flex; align-items: center; gap: 8px; margin-bottom: 3px; }
    .stage p { color: var(--muted); font-size: 13px; overflow-wrap: anywhere; }

    .evidence-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }

    .evidence-tile strong {
      display: block;
      margin-top: 5px;
      font-size: 18px;
      font-weight: 600;
      font-variant-numeric: tabular-nums;
    }

    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }

    .chip {
      padding: 4px 8px;
      border-radius: 999px;
      background: color-mix(in srgb, var(--blue) 10%, var(--surface));
      border: 1px solid color-mix(in srgb, var(--blue) 25%, var(--border));
      font-size: 12px;
    }

    .candidate-toolbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 10px;
      color: var(--muted);
    }

    .candidate-list {
      display: flex;
      flex-direction: column;
      gap: 7px;
    }

    .candidate {
      display: grid;
      grid-template-columns: 32px 1fr;
      gap: 10px;
      padding: 10px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--softer);
    }

    .candidate.target {
      border-color: var(--green);
      background: color-mix(in srgb, var(--green) 11%, var(--surface));
    }

    .rank {
      width: 27px;
      height: 27px;
      border-radius: 50%;
      display: grid;
      place-items: center;
      background: var(--surface);
      border: 1px solid var(--border);
      font-variant-numeric: tabular-nums;
      font-size: 13px;
    }

    .asin, .mono {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      overflow-wrap: anywhere;
    }

    .asin { font-size: 12px; }
    .summary {
      color: var(--muted);
      font-size: 12px;
      margin-top: 3px;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }

    .empty, .error {
      padding: 16px;
      color: var(--muted);
      border: 1px dashed var(--border);
      border-radius: 8px;
      background: var(--soft);
    }

    .error { color: var(--red); }

    .detail-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }

    .kv {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }

    .kv th, .kv td {
      padding: 7px 0;
      border-bottom: 1px solid var(--border);
      text-align: left;
      vertical-align: top;
    }

    .kv th { color: var(--muted); font-weight: 500; width: 42%; }
    .section-note { color: var(--muted); font-size: 13px; }
    .sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0,0,0,0); }

    @media (max-width: 1180px) {
      .shell { grid-template-columns: 280px 1fr; }
      .recommendations { grid-column: 1 / -1; }
    }

    @media (max-width: 820px) {
      .shell { grid-template-columns: 1fr; padding: 12px; }
      .conversation, .detail-grid, .evidence-grid { grid-template-columns: 1fr; }
      .actions { grid-template-columns: 1fr; }
      .stage { grid-template-columns: 30px 1fr; }
      .stage .badge { justify-self: start; grid-column: 2; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside class="sidebar">
      <section class="panel brand">
        <h1>Shopping Copilot Pipeline</h1>
        <div class="score-strip">
          <div class="score"><span>Hit rate</span><strong id="scoreHit">-</strong></div>
          <div class="score"><span>Score</span><strong id="scoreTech">-</strong></div>
        </div>
      </section>
      <section class="panel">
        <h2>Run A Session <span id="statusBadge" class="badge">ready</span></h2>
        <div class="body">
          <div class="field">
            <label for="scenarioFilter">Scenario</label>
            <select id="scenarioFilter">
              <option value="all">All scenarios</option>
            </select>
          </div>
          <div class="field">
            <label for="sessionSelect">Public session</label>
            <select id="sessionSelect"></select>
          </div>
          <label class="switch" for="qwenToggle">
            <span>Use local Qwen reranker</span>
            <input id="qwenToggle" type="checkbox">
          </label>
          <div class="actions">
            <button id="startBtn" class="primary" type="button">Start session</button>
            <button id="stepBtn" type="button" disabled>Next turn</button>
            <button id="runBtn" type="button" disabled>Run full session</button>
          </div>
          <div id="runnerNote" class="runner-note">Ready.</div>
        </div>
      </section>
      <section class="panel">
        <h2>Outcome <span id="scenarioBadge" class="badge">idle</span></h2>
        <div class="body">
          <div class="outcomes">
            <div class="outcome"><span>Turn</span><strong id="turnMetric">0 / 10</strong></div>
            <div class="outcome"><span>Best rank</span><strong id="rankMetric">-</strong></div>
            <div class="outcome"><span>Hit</span><strong id="hitMetric">no</strong></div>
            <div class="outcome"><span>Qwen</span><strong id="qwenMetric">off</strong></div>
          </div>
        </div>
      </section>
      <section class="panel">
        <h2>Turns</h2>
        <div id="turnRail" class="body turns">
          <div class="empty">No turns yet.</div>
        </div>
      </section>
    </aside>

    <main class="workbench">
      <section class="panel">
        <h2>Conversation <span id="selectedTurnBadge" class="badge">not started</span></h2>
        <div class="body">
          <div id="conversation" class="conversation">
            <div class="empty">Start a session to see the shopper and agent side by side.</div>
          </div>
        </div>
      </section>

      <section class="panel">
        <h2>What Happened Inside</h2>
        <div class="body">
          <div id="pipeline" class="pipeline">
            <div class="empty">The pipeline will fill in after the first turn.</div>
          </div>
        </div>
      </section>

      <section class="panel">
        <h2>Evidence</h2>
        <div class="body">
          <div id="evidenceGrid" class="evidence-grid"></div>
          <div id="terms" class="chips" style="margin-top:12px"></div>
          <div class="detail-grid" style="margin-top:14px">
            <table class="kv"><tbody id="stateTable"></tbody></table>
            <table class="kv"><tbody id="retrievalTable"></tbody></table>
          </div>
        </div>
      </section>
    </main>

    <aside class="recommendations">
      <section class="panel">
        <h2>Recommendations <span id="targetBadge" class="badge">target hidden</span></h2>
        <div class="body">
          <div class="candidate-toolbar">
            <span id="candidateHint">Top 10 returned to evaluator</span>
            <span id="candidateCount" class="mono">0</span>
          </div>
          <div id="candidates" class="candidate-list">
            <div class="empty">No recommendations yet.</div>
          </div>
        </div>
      </section>
    </aside>
  </div>

  <script>
    const $ = (id) => document.getElementById(id);
    let current = null;
    let busy = false;
    let allSessions = [];
    let selectedTurnIndex = -1;

    function esc(value) {
      return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
      }[ch]));
    }

    async function api(path, body) {
      const options = body ? {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body),
      } : {};
      const response = await fetch(path, options);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Request failed");
      return payload;
    }

    function fmt(value, digits = 3) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
      return Number(value).toFixed(digits).replace(/0+$/, "").replace(/\.$/, "");
    }

    function setRunnerNote(text, active = false) {
      $("runnerNote").textContent = text;
      $("runnerNote").className = active ? "runner-note active" : "runner-note";
    }

    function setBusy(value, note = "") {
      busy = value;
      $("startBtn").disabled = value;
      $("stepBtn").disabled = value || !current || current.done;
      $("runBtn").disabled = value || !current || current.done;
      $("statusBadge").textContent = value ? "running" : (current?.done ? "done" : "ready");
      $("statusBadge").className = value ? "badge warn" : (current?.done ? "badge good" : "badge");
      setRunnerNote(note || (value ? "Working..." : "Ready."), value);
    }

    function selectedTurn() {
      const turns = current?.turns || [];
      if (!turns.length) return null;
      const index = selectedTurnIndex >= 0 ? selectedTurnIndex : turns.length - 1;
      return turns[Math.min(index, turns.length - 1)];
    }

    function renderConversation(turn) {
      if (!turn) {
        $("conversation").innerHTML = `<div class="empty">Start a session to see the shopper and agent side by side.</div>`;
        $("selectedTurnBadge").textContent = "not started";
        return;
      }
      $("selectedTurnBadge").textContent = `turn ${turn.turn}`;
      $("conversation").innerHTML = `
        <article class="bubble customer">
          <div class="speaker"><span>shopper</span><span class="badge">input</span></div>
          <p>${esc(turn.user_message)}</p>
        </article>
        <article class="bubble agent">
          <div class="speaker"><span>agent</span><span class="badge hot">${esc(turn.ask_attribute || "no question")}</span></div>
          <p>${esc(turn.agent_message || "")}</p>
        </article>
      `;
    }

    function stageClass(status) {
      if (status === "accepted" || status === "good") return "good";
      if (status === "skipped" || status === "fallback" || status === "timeout") return "warn";
      return "active";
    }

    function renderPipeline(trace) {
      if (!trace) {
        $("pipeline").innerHTML = `<div class="empty">The pipeline will fill in after the first turn.</div>`;
        return;
      }
      const s = trace.stages || {};
      const data = [
        ["Understand", "Detect intent", s.intent?.summary, s.intent?.status],
        ["Remember", "Update session state", s.state?.summary, "active"],
        ["Search", "Raw, stemmed and dense routes", `${s.raw_bm25?.summary || ""} ${s.stem_bm25?.summary || ""} ${s.dense?.summary || ""}`.trim(), s.dense?.status],
        ["Merge", "Reciprocal rank fusion", s.fusion?.summary, "active"],
        ["Rerank", "Catalog scoring plus optional Qwen", `${s.rerank?.summary || ""} ${s.qwen?.summary || ""}`.trim(), s.qwen?.status],
        ["Reply", "Ask and recommend", s.response?.summary, "good"],
      ];
      $("pipeline").innerHTML = data.map(([title, subtitle, text, status], index) => {
        const cls = stageClass(status);
        const label = status === "accepted" ? "Qwen" : (status === "timeout" || status === "fallback" ? "fallback" : "done");
        return `
          <article class="stage ${cls}">
            <div class="stage-number">${index + 1}</div>
            <div>
              <div class="stage-title"><h3>${esc(title)}</h3><span class="subtle">${esc(subtitle)}</span></div>
              <p>${esc(text || "-")}</p>
            </div>
            <span class="badge ${cls === "good" ? "good" : cls === "warn" ? "warn" : ""}">${esc(label)}</span>
          </article>
        `;
      }).join("");
    }

    function renderState(trace) {
      const state = trace?.state || {};
      $("stateTable").innerHTML = `
        <tr><th>Memory terms</th><td class="mono">${esc(state.term_count ?? 0)}</td></tr>
        <tr><th>Phrase clues</th><td class="mono">${esc(state.phrase_count ?? 0)}</td></tr>
        <tr><th>Budget</th><td>${esc(state.budget ?? "-")}</td></tr>
        <tr><th>Next question</th><td>${esc(state.last_ask ?? "-")}</td></tr>
        <tr><th>Question decay</th><td>${esc(state.suppression || "-")}</td></tr>
      `;
      $("terms").innerHTML = (state.top_terms || []).length
        ? (state.top_terms || []).map(t => `<span class="chip">${esc(t)}</span>`).join("")
        : `<span class="section-note">No query terms yet.</span>`;
    }

    function renderRetrieval(trace) {
      const r = trace?.retrieval || {};
      $("retrievalTable").innerHTML = `
        <tr><th>Raw BM25</th><td class="mono">${esc(r.raw_count ?? 0)}</td></tr>
        <tr><th>Stemmed BM25</th><td class="mono">${esc(r.stem_count ?? 0)}</td></tr>
        <tr><th>Dense BGE</th><td class="mono">${esc(r.dense_count ?? 0)}</td></tr>
        <tr><th>Fused</th><td class="mono">${esc(r.fused_count ?? 0)}</td></tr>
        <tr><th>Cache</th><td>${r.cache_hit ? "hit" : "miss"}</td></tr>
        <tr><th>Elapsed</th><td class="mono">${esc(trace?.elapsed_ms ?? "-")} ms</td></tr>
      `;
      $("evidenceGrid").innerHTML = `
        <div class="evidence-tile"><span>Raw BM25</span><strong>${esc(r.raw_count ?? 0)}</strong></div>
        <div class="evidence-tile"><span>Stemmed BM25</span><strong>${esc(r.stem_count ?? 0)}</strong></div>
        <div class="evidence-tile"><span>Dense BGE</span><strong>${esc(r.dense_count ?? 0)}</strong></div>
      `;
    }

    function renderCandidates(items) {
      $("candidateCount").textContent = String(items?.length || 0);
      if (!items?.length) {
        $("candidates").innerHTML = `<div class="empty">No recommendations yet.</div>`;
        return;
      }
      $("candidates").innerHTML = items.map((item, index) => `
        <div class="candidate ${item.is_target ? "target" : ""}">
            <div class="rank">${index + 1}</div>
            <div>
            <div class="asin">${esc(item.parent_asin)} ${item.is_target ? '<span class="badge good">target</span>' : ""}</div>
            <div class="summary">${esc(item.summary)}</div>
          </div>
        </div>
      `).join("");
    }

    function renderTurnRail(turns) {
      if (!turns?.length) {
        $("turnRail").innerHTML = `<div class="empty">No turns yet.</div>`;
        return;
      }
      $("turnRail").innerHTML = turns.map((turn, index) => `
        <button type="button" class="turn-button ${index === selectedTurnIndex ? "active" : ""}" data-turn-index="${index}">
          <span>Turn ${turn.turn}</span>
          <span class="badge">${esc(turn.ask_attribute || "none")}</span>
        </button>
      `).join("");
      document.querySelectorAll("[data-turn-index]").forEach((button) => {
        button.addEventListener("click", () => {
          selectedTurnIndex = Number(button.dataset.turnIndex);
          render(current);
        });
      });
    }

    function render(payload) {
      current = payload;
      if (!payload) {
        $("scenarioBadge").textContent = "idle";
        $("scenarioBadge").className = "badge";
        $("turnMetric").textContent = "0 / 10";
        $("rankMetric").textContent = "-";
        $("hitMetric").textContent = "no";
        $("qwenMetric").textContent = $("qwenToggle").checked ? "on" : "off";
        $("targetBadge").textContent = "target hidden";
        $("targetBadge").className = "badge";
        $("candidateHint").textContent = "Top 10 returned to evaluator";
        $("candidateCount").textContent = "0";
        renderTurnRail([]);
        renderConversation(null);
        renderPipeline(null);
        renderState(null);
        renderRetrieval(null);
        renderCandidates(null);
        setBusy(false, "Ready.");
        return;
      }
      const turns = payload.turns || [];
      if (turns.length && (selectedTurnIndex < 0 || selectedTurnIndex >= turns.length)) {
        selectedTurnIndex = turns.length - 1;
      }
      const turn = selectedTurn();
      const trace = turn?.trace;
      $("scenarioBadge").textContent = payload.scenario || "idle";
      $("scenarioBadge").className = payload.scenario ? "badge hot" : "badge";
      $("turnMetric").textContent = `${payload.turn || 0} / 10`;
      $("rankMetric").textContent = payload.best_rank || "-";
      $("hitMetric").textContent = payload.hit ? "yes" : "no";
      $("qwenMetric").textContent = payload.qwen ? "on" : "off";
      $("targetBadge").textContent = payload.hit ? payload.target : "target hidden";
      $("targetBadge").className = payload.hit ? "badge good" : "badge";
      $("candidateHint").textContent = payload.hit ? `Target found on turn ${payload.first_hit_turn}` : "Top 10 returned to evaluator";
      renderTurnRail(turns);
      renderConversation(turn);
      renderPipeline(trace);
      renderState(trace);
      renderRetrieval(trace);
      renderCandidates(turn?.recommendations);
      setBusy(false, payload.done ? "Session complete." : "Ready for next turn.");
    }

    function fillSessionSelect() {
      const scenario = $("scenarioFilter").value;
      const filtered = allSessions.filter(s => scenario === "all" || s.scenario_type === scenario);
      $("sessionSelect").innerHTML = filtered.map(s =>
        `<option value="${esc(s.sample_id)}">${esc(s.sample_id)} · ${esc(s.scenario_type)}</option>`
      ).join("");
    }

    async function loadOptions() {
      try {
        const payload = await api("/api/options");
        allSessions = payload.sessions || [];
        const scenarios = [...new Set(allSessions.map(s => s.scenario_type))].sort();
        $("scenarioFilter").innerHTML = `<option value="all">All scenarios</option>` + scenarios.map(name =>
          `<option value="${esc(name)}">${esc(name)}</option>`
        ).join("");
        fillSessionSelect();
        const metrics = payload.metrics || {};
        $("scoreHit").textContent = metrics.hit_rate_at_10 !== undefined ? fmt(metrics.hit_rate_at_10, 3) : "-";
        $("scoreTech").textContent = metrics.recommended_technical_score !== undefined ? fmt(metrics.recommended_technical_score, 3) : "-";
        $("statusBadge").textContent = "ready";
      } catch (error) {
        $("conversation").innerHTML = `<div class="error">${esc(error.message)}</div>`;
        $("statusBadge").textContent = "error";
        $("statusBadge").className = "badge warn";
      }
    }

    async function startSession() {
      setBusy(true, "Warming search engine for this preview...");
      selectedTurnIndex = -1;
      $("conversation").innerHTML = `<div class="empty">Preparing this session...</div>`;
      $("pipeline").innerHTML = `<div class="empty">Loading the local retrieval and ranking engine.</div>`;
      try {
        const payload = await api("/api/start", {
          sample_id: $("sessionSelect").value,
          qwen: $("qwenToggle").checked,
        });
        render(payload);
      } catch (error) {
        $("conversation").innerHTML = `<div class="error">${esc(error.message)}</div>`;
        setBusy(false, "Start failed.");
      }
    }

    async function nextTurn() {
      if (!current || busy || current.done) return;
      setBusy(true, "Running the next shopper turn...");
      try {
        render(await api("/api/turn", {id: current.id}));
      } catch (error) {
        $("conversation").innerHTML = `<div class="error">${esc(error.message)}</div>`;
        setBusy(false, "Turn failed.");
      }
    }

    async function runToHit() {
      if (!current || busy || current.done) return;
      setBusy(true, "Running until hit or max turns...");
      try {
        render(await api("/api/run", {id: current.id}));
      } catch (error) {
        $("conversation").innerHTML = `<div class="error">${esc(error.message)}</div>`;
        setBusy(false, "Run failed.");
      }
    }

    $("scenarioFilter").addEventListener("change", fillSessionSelect);
    $("startBtn").addEventListener("click", startSession);
    $("stepBtn").addEventListener("click", nextTurn);
    $("runBtn").addEventListener("click", runToHit);
    render(null);
    loadOptions();
  </script>
</body>
</html>
"""


STORE_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TikTechToh Shop</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: light-dark(#f5f3ef, #121314);
      --surface: light-dark(#ffffff, #1a1c1f);
      --soft: light-dark(#f0f4f3, #202428);
      --ink: light-dark(#18211f, #f1f5f2);
      --muted: light-dark(#64706d, #aab4b0);
      --border: light-dark(#d8ded9, #353b3a);
      --accent: light-dark(#0f6b5f, #5fc4b4);
      --accent-ink: light-dark(#ffffff, #071412);
      --blue: light-dark(#275da8, #83aff3);
      --rose: light-dark(#a94357, #f2a0ad);
      --gold: light-dark(#8a650f, #e5c25a);
      --shadow: light-dark(0 18px 45px rgba(28, 35, 33, .09), 0 18px 45px rgba(0, 0, 0, .28));
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-width: 320px;
      background: var(--bg);
      color: var(--ink);
      font-size: 15px;
    }

    button, input {
      font: inherit;
    }

    button {
      min-height: 40px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--surface);
      color: var(--ink);
      cursor: pointer;
    }

    button:disabled {
      cursor: wait;
      opacity: .62;
    }

    .topbar {
      position: sticky;
      top: 0;
      z-index: 10;
      display: flex;
      align-items: center;
      justify-content: flex-start;
      padding: 14px clamp(16px, 3vw, 34px);
      background: color-mix(in srgb, var(--surface) 92%, transparent);
      border-bottom: 1px solid var(--border);
      backdrop-filter: blur(14px);
    }

    .brand {
      font-size: 19px;
      font-weight: 700;
      letter-spacing: 0;
      white-space: nowrap;
    }

    .page {
      width: min(1440px, 100%);
      margin: 0 auto;
      padding: 22px clamp(14px, 3vw, 34px) 40px;
    }

    .search-area {
      padding: 8px 0 20px;
    }

    .search-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 12px;
      width: min(980px, 100%);
    }

    .search-form {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
    }

    .search-input {
      min-height: 52px;
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--soft);
      color: var(--ink);
      padding: 0 14px;
      outline: none;
    }

    .search-input:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 18%, transparent);
    }

    .search-button {
      min-width: 112px;
      border-color: var(--accent);
      background: var(--accent);
      color: var(--accent-ink);
      font-weight: 700;
      padding: 0 18px;
    }

    .results-panel {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
    }

    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 15px 16px;
      border-bottom: 1px solid var(--border);
    }

    .panel-head h2 {
      margin: 0;
      font-size: 16px;
      letter-spacing: 0;
    }

    .results-panel {
      min-height: 560px;
      overflow: hidden;
    }

    .result-meta {
      color: var(--muted);
      font-size: 14px;
      white-space: nowrap;
    }

    .product-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
      gap: 14px;
      padding: 16px;
    }

    .product-card {
      display: grid;
      grid-template-rows: 176px auto;
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      background: var(--surface);
      min-width: 0;
    }

    .product-art {
      display: grid;
      place-items: center;
      padding: 18px;
      background:
        linear-gradient(135deg,
          color-mix(in srgb, var(--accent) 16%, var(--surface)),
          color-mix(in srgb, var(--rose) 10%, var(--soft)) 55%,
          color-mix(in srgb, var(--gold) 14%, var(--surface)));
      border-bottom: 1px solid var(--border);
      text-align: center;
      color: var(--ink);
    }

    .product-art span {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 46px;
      padding: 9px 12px;
      background: color-mix(in srgb, var(--surface) 78%, transparent);
      border: 1px solid color-mix(in srgb, var(--border) 74%, transparent);
      border-radius: 8px;
      font-weight: 700;
      overflow-wrap: anywhere;
    }

    .product-photo {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }

    .product-body {
      display: grid;
      gap: 9px;
      padding: 13px;
    }

    .product-title {
      min-height: 43px;
      margin: 0;
      font-size: 15px;
      font-weight: 700;
      line-height: 1.35;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }

    .store-line, .category-line, .feature-line {
      color: var(--muted);
      font-size: 13px;
      overflow-wrap: anywhere;
    }

    .price-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
    }

    .price {
      font-size: 18px;
      font-weight: 800;
      font-variant-numeric: tabular-nums;
    }

    .rating {
      color: var(--gold);
      font-size: 13px;
      white-space: nowrap;
      font-variant-numeric: tabular-nums;
    }

    .product-actions {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      margin-top: 3px;
    }

    .quick-view {
      background: var(--ink);
      color: var(--surface);
      border-color: var(--ink);
      font-weight: 700;
    }

    .save-button {
      padding: 0 10px;
      font-size: 13px;
      font-weight: 700;
    }

    .save-button.saved {
      border-color: var(--rose);
      color: var(--rose);
      background: color-mix(in srgb, var(--rose) 10%, var(--surface));
    }

    .empty-results, .loading-results {
      padding: 34px 18px;
      color: var(--muted);
      text-align: center;
    }

    .empty-results strong {
      display: block;
      margin-bottom: 7px;
      color: var(--ink);
      font-size: 18px;
    }

    .skeleton-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
      gap: 14px;
      padding: 16px;
    }

    .skeleton {
      height: 310px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background:
        linear-gradient(90deg,
          color-mix(in srgb, var(--soft) 84%, transparent),
          color-mix(in srgb, var(--surface) 88%, transparent),
          color-mix(in srgb, var(--soft) 84%, transparent));
      background-size: 220% 100%;
      animation: shimmer 1.35s linear infinite;
    }

    @keyframes shimmer {
      to { background-position: -220% 0; }
    }

    @media (prefers-reduced-motion: reduce) {
      .skeleton { animation: none; }
    }

    .toast {
      display: none;
      margin: 0 16px 16px;
      padding: 10px 12px;
      border-radius: 8px;
      background: color-mix(in srgb, var(--rose) 10%, var(--surface));
      color: var(--rose);
      border: 1px solid color-mix(in srgb, var(--rose) 30%, var(--border));
    }

    .toast.show { display: block; }

    dialog {
      width: min(680px, calc(100vw - 26px));
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--surface);
      color: var(--ink);
      padding: 0;
    }

    dialog::backdrop {
      background: rgba(0, 0, 0, .45);
    }

    .modal-head {
      display: flex;
      align-items: start;
      justify-content: space-between;
      gap: 12px;
      padding: 16px;
      border-bottom: 1px solid var(--border);
    }

    .modal-head h3 {
      margin: 0;
      font-size: 18px;
      line-height: 1.3;
    }

    .modal-body {
      display: grid;
      gap: 14px;
      padding: 16px;
    }

    .modal-list {
      margin: 0;
      padding-left: 20px;
      color: var(--muted);
    }

    .close-modal {
      width: 38px;
      padding: 0;
      font-size: 20px;
    }

    .sr-only {
      position: absolute;
      width: 1px;
      height: 1px;
      padding: 0;
      margin: -1px;
      overflow: hidden;
      clip: rect(0, 0, 0, 0);
      white-space: nowrap;
      border: 0;
    }

    @media (max-width: 620px) {
      .search-form { grid-template-columns: 1fr; }
      .search-button { width: 100%; }
      .panel-head { align-items: start; flex-direction: column; }
      .result-meta { white-space: normal; }
      .product-grid, .skeleton-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="brand">TikTechToh Shop</div>
  </header>

  <main class="page">
    <section class="search-area">
      <div class="search-card">
        <form id="searchForm" class="search-form">
          <label for="promptInput" class="sr-only">Shopping request</label>
          <input id="promptInput" class="search-input" autocomplete="off" placeholder="silver necklace for a birthday gift under $40">
          <button id="searchBtn" class="search-button" type="submit">Search</button>
        </form>
      </div>
    </section>

    <section class="results-panel">
      <div class="panel-head">
        <h2>Matching products</h2>
        <span id="resultMeta" class="result-meta">No products yet</span>
      </div>
      <div id="toast" class="toast" role="alert"></div>
      <div id="products" class="product-grid" aria-live="polite">
        <div class="empty-results">
          <strong>Search for something to buy</strong>
          <span>Your top matching products will appear here.</span>
        </div>
      </div>
    </section>
  </main>

  <dialog id="productModal">
    <div class="modal-head">
      <h3 id="modalTitle"></h3>
      <button id="closeModal" class="close-modal" type="button" aria-label="Close">x</button>
    </div>
    <div id="modalBody" class="modal-body"></div>
  </dialog>

  <script>
    const $ = (id) => document.getElementById(id);
    const state = {
      sessionId: null,
      history: [],
      products: [],
      saved: new Set(),
      loading: false,
      lastMessage: "",
    };

    function esc(value) {
      return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
      }[ch]));
    }

    async function api(path, body) {
      const response = await fetch(path, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Request failed");
      return payload;
    }

    function showToast(message) {
      $("toast").textContent = message;
      $("toast").className = "toast show";
    }

    function clearToast() {
      $("toast").textContent = "";
      $("toast").className = "toast";
    }

    function setLoading(value) {
      state.loading = value;
      $("searchBtn").disabled = value;
      $("promptInput").disabled = value;
      if (value) {
        $("resultMeta").textContent = "Finding matches...";
        $("products").className = "skeleton-grid";
        $("products").innerHTML = Array.from({length: 6}, () => `<div class="skeleton"></div>`).join("");
      }
    }

    function money(product) {
      if (product.price_label) return product.price_label;
      return "Price unavailable";
    }

    function rating(product) {
      if (!product.rating_label) return "No rating";
      return product.rating_label;
    }

    function artLabel(product) {
      return product.store || "Top match";
    }

    function renderProducts() {
      $("products").className = "product-grid";
      $("resultMeta").textContent = state.products.length
        ? `${state.products.length} recommendations`
        : "No products yet";
      if (!state.products.length) {
        $("products").innerHTML = `
          <div class="empty-results">
            <strong>No matches yet</strong>
            <span>Try a style, material, size, budget, or occasion.</span>
          </div>
        `;
        return;
      }
      $("products").innerHTML = state.products.map((product, index) => `
        <article class="product-card">
          <div class="product-art">
            ${product.image_url ? `<img class="product-photo" src="${esc(product.image_url)}" alt="">` : `<span>${esc(artLabel(product))}</span>`}
          </div>
          <div class="product-body">
            <p class="product-title">${esc(product.title)}</p>
            <div class="store-line">${esc(product.store || "Marketplace seller")}</div>
            <div class="price-row">
              <span class="price">${esc(money(product))}</span>
              <span class="rating">${esc(rating(product))}</span>
            </div>
            <div class="feature-line">${esc(product.feature || "Details available in quick view.")}</div>
            <div class="product-actions">
              <button class="quick-view" type="button" data-action="view" data-index="${index}">Quick view</button>
              <button class="save-button ${state.saved.has(product.id) ? "saved" : ""}" type="button" data-action="save" data-index="${index}">${state.saved.has(product.id) ? "Saved" : "Save"}</button>
            </div>
          </div>
        </article>
      `).join("");
    }

    function render(payload = null) {
      if (payload) {
        state.sessionId = payload.id;
        state.history = payload.history || [];
        state.products = payload.products || [];
      }
      renderProducts();
    }

    async function submitSearch(message) {
      const text = (message || $("promptInput").value).trim();
      if (!text) {
        showToast("Type what you want to shop for first.");
        return;
      }
      clearToast();
      state.lastMessage = text;
      state.history = [...state.history, {role: "user", text}];
      $("promptInput").value = "";
      setLoading(true);
      try {
        const payload = await api("/api/search", {
          id: state.sessionId,
          message: text,
        });
        render(payload);
      } catch (error) {
        showToast(error.message);
        state.history = state.history.filter((item, index) => index !== state.history.length - 1);
      } finally {
        setLoading(false);
      }
    }

    function openProduct(index) {
      const product = state.products[index];
      if (!product) return;
      $("modalTitle").textContent = product.title;
      const features = (product.features || []).map((item) => `<li>${esc(item)}</li>`).join("");
      $("modalBody").innerHTML = `
        <div class="price-row">
          <span class="price">${esc(money(product))}</span>
          <span class="rating">${esc(rating(product))}</span>
        </div>
        <div class="store-line">${esc(product.store || "Marketplace seller")}</div>
        ${features ? `<ul class="modal-list">${features}</ul>` : `<div class="feature-line">No extra product details are available.</div>`}
      `;
      if ($("productModal").showModal) {
        $("productModal").showModal();
      } else {
        alert(product.title);
      }
    }

    $("searchForm").addEventListener("submit", (event) => {
      event.preventDefault();
      submitSearch();
    });

    $("products").addEventListener("click", (event) => {
      const button = event.target.closest("button[data-action]");
      if (!button) return;
      const index = Number(button.dataset.index);
      const product = state.products[index];
      if (!product) return;
      if (button.dataset.action === "view") {
        openProduct(index);
      } else {
        if (state.saved.has(product.id)) state.saved.delete(product.id);
        else state.saved.add(product.id);
        renderProducts();
      }
    });

    $("closeModal").addEventListener("click", () => $("productModal").close());
    $("productModal").addEventListener("click", (event) => {
      if (event.target === $("productModal")) $("productModal").close();
    });

    render();
  </script>
</body>
</html>
"""


def _short(value: object, limit: int = 170) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _flatten(value: object) -> list[str]:
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "", [])]
    return [str(value)] if value not in (None, "") else []


def _number(value: object) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _product_image(product: dict) -> str | None:
    images = product.get("images")
    candidates: list[object] = []
    if isinstance(images, dict):
        candidates.extend(images.values())
    elif isinstance(images, list):
        candidates.extend(images)
    for item in candidates:
        if isinstance(item, str) and item.startswith(("http://", "https://")):
            return item
        if isinstance(item, dict):
            for key in ("large", "hi_res", "thumb", "main", "url"):
                url = item.get(key)
                if isinstance(url, str) and url.startswith(("http://", "https://")):
                    return url
    return None


def _product_payload(parent_asin: str, product: dict) -> dict:
    categories = [str(value) for value in product.get("categories") or []]
    category = " / ".join(categories[-3:]) if categories else ""
    department = " ".join(categories[-2:]) if len(categories) >= 2 else (categories[-1] if categories else "")
    price = _number(product.get("price"))
    rating = _number(product.get("average_rating"))
    reviews = product.get("rating_number")
    features = [_short(item, 180) for item in _flatten(product.get("features"))[:5]]
    if not features:
        features = [_short(item, 180) for item in _flatten(product.get("description"))[:3]]
    return {
        "id": parent_asin,
        "title": _short(product.get("title") or "Untitled product", 145),
        "store": _short(product.get("store") or "", 70),
        "category": _short(category, 120),
        "department": _short(department, 54),
        "price": price,
        "price_label": f"${price:.2f}" if price is not None else "",
        "rating": rating,
        "rating_label": f"{rating:.1f} stars" if rating is not None else "",
        "reviews": reviews,
        "review_label": f"{reviews:,} reviews" if isinstance(reviews, int) else "",
        "feature": _short(features[0], 130) if features else "",
        "features": features,
        "image_url": _product_image(product),
    }


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "for", "from", "i", "in", "is",
    "it", "me", "my", "of", "on", "or", "please", "show", "the", "to", "under",
    "with", "want", "need", "looking",
    "dad", "female", "girlfriend", "husband", "ladies", "lady", "male", "man",
    "men", "mens", "mom", "mother", "woman", "women", "womens", "wife",
}
TOKEN_RE = re.compile(r"[a-z0-9]+")
BUDGET_RE = re.compile(r"(?:under|below|less than|max|maximum|budget|<=|up to)?\s*\$?\s*(\d+(?:\.\d+)?)", re.I)
DEPARTMENT_RULES = {
    "men": {
        "patterns": (r"\bmen'?s?\b", r"\bman\b", r"\bmale\b", r"\bfather\b", r"\bdad\b", r"\bboyfriend\b", r"\bhusband\b"),
        "include": {"men", "mens", "men's", "man", "male", "father", "dad", "boyfriend", "husband"},
        "allow": (" men ", " mens ", " men's ", " male "),
        "block": (" women ", " womens ", " women's ", " woman ", " female ", " girls ", " girl ", " ladies ", " boys ", " boy "),
    },
    "women": {
        "patterns": (r"\bwomen'?s?\b", r"\bwoman\b", r"\bfemale\b", r"\blad(?:y|ies)\b", r"\bgirlfriend\b", r"\bwife\b", r"\bmom\b", r"\bmother\b"),
        "include": {"women", "womens", "women's", "woman", "female", "ladies", "lady", "girlfriend", "wife", "mom", "mother"},
        "allow": (" women ", " womens ", " women's ", " woman ", " female ", " ladies ", " lady "),
        "block": (" men ", " mens ", " men's ", " man ", " male ", " boys ", " boy ", " girls ", " girl "),
    },
}
SLEEVE_RULES = {
    "long_sleeve": {
        "patterns": (r"\blong[-\s]+sleeve(?:d|s)?\b",),
        "allow": (" long sleeve ", " long sleeves ", " long-sleeve ", " long-sleeved ", " longsleeve ", " longsleeved "),
        "block": (" short sleeve ", " short sleeves ", " short-sleeve ", " short-sleeved ", " shortsleeve ", " shortsleeved ", " sleeveless "),
    },
    "short_sleeve": {
        "patterns": (r"\bshort[-\s]+sleeve(?:d|s)?\b",),
        "allow": (" short sleeve ", " short sleeves ", " short-sleeve ", " short-sleeved ", " shortsleeve ", " shortsleeved "),
        "block": (" long sleeve ", " long sleeves ", " long-sleeve ", " long-sleeved ", " longsleeve ", " longsleeved ", " sleeveless "),
    },
    "sleeveless": {
        "patterns": (r"\bsleeveless\b",),
        "allow": (" sleeveless ",),
        "block": (" long sleeve ", " long sleeves ", " long-sleeve ", " long-sleeved ", " longsleeve ", " longsleeved ", " short sleeve ", " short sleeves ", " short-sleeve ", " short-sleeved ", " shortsleeve ", " shortsleeved "),
    },
}
COLOR_RULES = {
    "black": {"patterns": (r"\bblack\b",), "allow": (" black ",)},
    "white": {"patterns": (r"\bwhite\b",), "allow": (" white ",)},
    "blue": {"patterns": (r"\bblue\b", r"\bnavy\b"), "allow": (" blue ", " navy ")},
    "red": {"patterns": (r"\bred\b", r"\bmaroon\b", r"\bburgundy\b"), "allow": (" red ", " maroon ", " burgundy ")},
    "pink": {"patterns": (r"\bpink\b",), "allow": (" pink ",)},
    "green": {"patterns": (r"\bgreen\b",), "allow": (" green ",)},
    "brown": {"patterns": (r"\bbrown\b",), "allow": (" brown ",)},
    "gray": {"patterns": (r"\bgr[ae]y\b",), "allow": (" gray ", " grey ")},
    "purple": {"patterns": (r"\bpurple\b",), "allow": (" purple ",)},
    "yellow": {"patterns": (r"\byellow\b",), "allow": (" yellow ",)},
    "orange": {"patterns": (r"\borange\b",), "allow": (" orange ",)},
    "gold": {"patterns": (r"\bgold(?:en)?\b",), "allow": (" gold ", " golden ")},
    "silver": {"patterns": (r"\bsilver\b",), "allow": (" silver ",)},
    "beige": {"patterns": (r"\bbeige\b", r"\btan\b", r"\bkhaki\b"), "allow": (" beige ", " tan ", " khaki ")},
}
MATERIAL_RULES = {
    "cotton": {"patterns": (r"\bcotton\b",), "allow": (" cotton ",)},
    "polyester": {"patterns": (r"\bpolyester\b",), "allow": (" polyester ",)},
    "nylon": {"patterns": (r"\bnylon\b",), "allow": (" nylon ",)},
    "leather": {"patterns": (r"\bleather\b", r"\bpu leather\b", r"\bfaux leather\b"), "allow": (" leather ", " pu leather ", " faux leather ")},
    "wool": {"patterns": (r"\bwool\b", r"\bmerino\b"), "allow": (" wool ", " merino ")},
    "spandex": {"patterns": (r"\bspandex\b", r"\belastane\b", r"\blycra\b"), "allow": (" spandex ", " elastane ", " lycra ")},
    "silk": {"patterns": (r"\bsilk\b",), "allow": (" silk ",)},
    "rayon": {"patterns": (r"\brayon\b", r"\bviscose\b"), "allow": (" rayon ", " viscose ")},
    "denim": {"patterns": (r"\bdenim\b",), "allow": (" denim ",)},
    "linen": {"patterns": (r"\blinen\b",), "allow": (" linen ",)},
    "fleece": {"patterns": (r"\bfleece\b",), "allow": (" fleece ",)},
    "alloy": {"patterns": (r"\balloy\b",), "allow": (" alloy ",)},
    "stainless_steel": {"patterns": (r"\bstainless[-\s]+steel\b",), "allow": (" stainless steel ", " stainless-steel ")},
    "sterling_silver": {"patterns": (r"\bsterling[-\s]+silver\b", r"\bs925\b", r"\b925[-\s]+silver\b"), "allow": (" sterling silver ", " sterling-silver ", " s925 ", " 925 silver ")},
    "gold_plated": {"patterns": (r"\bgold[-\s]+plated\b", r"\b14k[-\s]+gold\b", r"\b18k[-\s]+gold\b"), "allow": (" gold plated ", " gold-plated ", " 14k gold ", " 18k gold ")},
}
PRODUCT_TYPE_RULES = {
    "shirt": {
        "patterns": (r"\bshirt(s)?\b", r"\bt[-\s]?shirt(s)?\b", r"\btee(s)?\b", r"\bpolo(s)?\b", r"\bhenley(s)?\b", r"\bblouse(s)?\b", r"\btop(s)?\b"),
        "allow": (" shirt ", " shirts ", " t shirt ", " t-shirt ", " tshirt ", " tee ", " tees ", " polo ", " polos ", " henley ", " henleys ", " blouse ", " blouses ", " top ", " tops "),
    },
    "shoes": {
        "patterns": (r"\bshoe(s)?\b", r"\bfootwear\b"),
        "allow": (" shoe ", " shoes ", " footwear "),
        "block": (" shoe care ", " cleaner ", " conditioner ", " polish ", " spray ", " insole ", " insoles ", " lace ", " laces ", " insert ", " inserts "),
    },
    "sneakers": {
        "patterns": (r"\bsneaker(s)?\b", r"\brunning[-\s]+shoe(s)?\b"),
        "allow": (" sneaker ", " sneakers ", " running shoe ", " running shoes "),
        "block": (" shoe care ", " cleaner ", " conditioner ", " polish ", " spray ", " insole ", " insoles ", " lace ", " laces ", " insert ", " inserts "),
    },
    "boots": {
        "patterns": (r"\bboot(s)?\b",),
        "allow": (" boot ", " boots "),
        "block": (" shoe care ", " boot care ", " cleaner ", " conditioner ", " polish ", " spray ", " insole ", " insoles ", " lace ", " laces ", " insert ", " inserts "),
    },
    "sandals": {
        "patterns": (r"\bsandal(s)?\b",),
        "allow": (" sandal ", " sandals "),
    },
    "necklace": {
        "patterns": (r"\bnecklace(s)?\b", r"\bpendant(s)?\b"),
        "allow": (" necklace ", " necklaces ", " pendant ", " pendants "),
        "block": (" organizer ", " holder ", " display ", " stand ", " box "),
    },
    "earrings": {
        "patterns": (r"\bearring(s)?\b", r"\bhoop(s)?\b"),
        "allow": (" earring ", " earrings ", " hoop ", " hoops "),
        "block": (" organizer ", " holder ", " display ", " stand ", " box "),
    },
    "ring": {
        "patterns": (r"\bring(s)?\b",),
        "allow": (" ring ", " rings "),
    },
    "bracelet": {
        "patterns": (r"\bbracelet(s)?\b", r"\bbangle(s)?\b"),
        "allow": (" bracelet ", " bracelets ", " bangle ", " bangles "),
    },
    "dress": {
        "patterns": (r"\bdress(es)?\b",),
        "allow": (" dress ", " dresses "),
    },
    "pants": {
        "patterns": (r"\bpant(s)?\b", r"\btrouser(s)?\b", r"\bjean(s)?\b", r"\blegging(s)?\b"),
        "allow": (" pant ", " pants ", " trouser ", " trousers ", " jean ", " jeans ", " legging ", " leggings "),
    },
    "jacket": {
        "patterns": (r"\bjacket(s)?\b", r"\bcoat(s)?\b", r"\bhoodie(s)?\b", r"\bsweatshirt(s)?\b"),
        "allow": (" jacket ", " jackets ", " coat ", " coats ", " hoodie ", " hoodies ", " sweatshirt ", " sweatshirts "),
    },
    "socks": {
        "patterns": (r"\bsock(s)?\b",),
        "allow": (" sock ", " socks "),
    },
}


def _tokens(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if len(token) > 1 and token not in STOPWORDS]


def _norm_match(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.lower()))


def _padded(text: str) -> str:
    return f" {re.sub(r'[^a-z0-9$.-]+', ' ', text.lower())} "


def _price_budget(text: str) -> float | None:
    lowered = text.lower()
    if not any(marker in lowered for marker in ("$", "under", "below", "less than", "max", "budget", "up to", "<=")):
        return None
    matches = [float(value) for value in BUDGET_RE.findall(text)]
    return max(matches) if matches else None


def _is_negated(text: str, start: int) -> bool:
    prefix = text[max(0, start - 44):start]
    return bool(re.search(r"\b(?:not|no|without|avoid|exclude|except|don't|do not)(?:\s+\w+){0,3}\s*$", prefix))


def _slot_matches(text: str, rules: dict[str, dict]) -> tuple[list[str], list[str]]:
    positives: list[str] = []
    exclusions: list[str] = []
    for name, rule in rules.items():
        for pattern in rule["patterns"]:
            match = re.search(pattern, text)
            if not match:
                continue
            target = exclusions if _is_negated(text, match.start()) else positives
            if name not in target:
                target.append(name)
            break
    return positives, exclusions


def _query_constraints(query: str) -> dict:
    lowered = query.lower()
    departments, exclude_departments = _slot_matches(lowered, DEPARTMENT_RULES)
    sleeves, exclude_sleeves = _slot_matches(lowered, SLEEVE_RULES)
    colors, exclude_colors = _slot_matches(lowered, COLOR_RULES)
    materials, exclude_materials = _slot_matches(lowered, MATERIAL_RULES)
    product_types, exclude_product_types = _slot_matches(lowered, PRODUCT_TYPE_RULES)
    return {
        "department": departments[-1] if departments else None,
        "sleeve": sleeves[-1] if sleeves else None,
        "colors": colors[:2],
        "materials": materials[:2],
        "product_type": product_types[-1] if product_types else None,
        "exclude_departments": exclude_departments,
        "exclude_sleeves": exclude_sleeves,
        "exclude_colors": exclude_colors,
        "exclude_materials": exclude_materials,
        "exclude_product_types": exclude_product_types,
        "budget": _price_budget(query),
    }


def _empty_constraints() -> dict:
    return {
        "department": None,
        "sleeve": None,
        "colors": [],
        "materials": [],
        "product_type": None,
        "exclude_departments": [],
        "exclude_sleeves": [],
        "exclude_colors": [],
        "exclude_materials": [],
        "exclude_product_types": [],
        "budget": None,
    }


def _merge_unique(values: list[str], additions: list[str]) -> list[str]:
    merged = list(values)
    for value in additions:
        if value not in merged:
            merged.append(value)
    return merged


def _merge_constraints(current: dict, update: dict) -> dict:
    merged = {
        key: list(value) if isinstance(value, list) else value
        for key, value in current.items()
    }
    scalar_pairs = (
        ("department", "exclude_departments"),
        ("sleeve", "exclude_sleeves"),
        ("product_type", "exclude_product_types"),
    )
    for include_key, exclude_key in scalar_pairs:
        if update.get(include_key):
            merged[include_key] = update[include_key]
            merged[exclude_key] = [value for value in merged[exclude_key] if value != update[include_key]]
        for value in update.get(exclude_key, []):
            if merged.get(include_key) == value:
                merged[include_key] = None
        merged[exclude_key] = _merge_unique(merged[exclude_key], update.get(exclude_key, []))

    for include_key, exclude_key in (("colors", "exclude_colors"), ("materials", "exclude_materials")):
        if update.get(include_key):
            merged[include_key] = list(update[include_key])
            merged[exclude_key] = [value for value in merged[exclude_key] if value not in update[include_key]]
        exclusions = update.get(exclude_key, [])
        if exclusions:
            merged[include_key] = [value for value in merged[include_key] if value not in exclusions]
            merged[exclude_key] = _merge_unique(merged[exclude_key], exclusions)

    if update.get("budget") is not None:
        merged["budget"] = update["budget"]
    return merged


def _constraint_terms(constraints: dict) -> list[str]:
    terms: list[str] = []
    if constraints["department"]:
        terms.append("men" if constraints["department"] == "men" else "women")
    if constraints["sleeve"]:
        terms.append(constraints["sleeve"].replace("_", " "))
    terms.extend(color.replace("_", " ") for color in constraints["colors"])
    terms.extend(material.replace("_", " ") for material in constraints["materials"])
    if constraints["product_type"]:
        terms.append(constraints["product_type"].replace("_", " "))
    if constraints["budget"] is not None:
        terms.append(f"under {constraints['budget']}")
    for color in constraints["exclude_colors"]:
        terms.append(f"not {color.replace('_', ' ')}")
    for material in constraints["exclude_materials"]:
        terms.append(f"not {material.replace('_', ' ')}")
    for sleeve in constraints["exclude_sleeves"]:
        terms.append(f"not {sleeve.replace('_', ' ')}")
    return terms


def _contains_any(text: str, values: tuple[str, ...]) -> bool:
    return any(value in text for value in values)


def _rule_match(primary: str, fallback: str, rule: dict) -> bool:
    if _contains_any(primary, rule["allow"]):
        return not _contains_any(primary, rule.get("block", ()))
    return _contains_any(fallback, rule["allow"]) and not _contains_any(fallback, rule.get("block", ()))


class StorefrontIndex:
    """Fast, offline-only search for the shopper-facing demo page."""

    def __init__(self, products: dict[str, dict]) -> None:
        self.products = products
        self.docs: dict[str, dict] = {}
        self.postings: dict[str, list[tuple[str, float]]] = {}
        self.constraint_sets: dict[str, set[str]] = {
            **{f"department:{name}": set() for name in DEPARTMENT_RULES},
            **{f"sleeve:{name}": set() for name in SLEEVE_RULES},
            **{f"color:{name}": set() for name in COLOR_RULES},
            **{f"material:{name}": set() for name in MATERIAL_RULES},
            **{f"type:{name}": set() for name in PRODUCT_TYPE_RULES},
        }
        for pid, product in products.items():
            title = _short(product.get("title") or "", 500)
            categories = " ".join(str(value) for value in product.get("categories") or [])
            features = " ".join(_flatten(product.get("features"))[:8])
            details = " ".join(_flatten(product.get("details"))[:10])
            description = " ".join(_flatten(product.get("description"))[:4])
            store = str(product.get("store") or "")
            combined = " ".join([title, categories, features, details, description, store]).lower()
            detail_department = ""
            if isinstance(product.get("details"), dict):
                detail_department = " ".join(
                    str(value)
                    for key, value in product["details"].items()
                    if "department" in str(key).lower() and value not in (None, "", [])
                )
            audience = _padded(" ".join([title, categories, detail_department]))
            core = _padded(" ".join([title, categories]))
            garment = _padded(" ".join([title, categories, features]))
            attributes = _padded(" ".join([title, categories, features, details]))
            product_type_text = _padded(" ".join([title, categories]))
            padded = _padded(combined)
            self.docs[pid] = {
                "text": combined,
                "padded": padded,
                "audience": audience,
                "core": core,
                "garment": garment,
                "attributes": attributes,
                "product_type": product_type_text,
                "title": title.lower(),
                "title_norm": _norm_match(title),
                "price": _number(product.get("price")),
                "rating": _number(product.get("average_rating")) or 0.0,
                "reviews": int(product.get("rating_number") or 0),
            }
            self._add_constraint_memberships(pid)
            weighted_fields = (
                (title, 7.0),
                (categories, 4.5),
                (store, 2.0),
                (features, 2.5),
                (details, 1.5),
                (description, 1.2),
            )
            weights: dict[str, float] = {}
            for field_text, weight in weighted_fields:
                for token in _tokens(field_text):
                    weights[token] = weights.get(token, 0.0) + weight
            for token, weight in weights.items():
                self.postings.setdefault(token, []).append((pid, weight))

    def search(self, query: str, top_k: int = TOP_K, constraints: dict | None = None) -> list[str]:
        query = re.sub(r"\s+", " ", query).strip()
        exact_id = query.upper()
        if exact_id in self.products:
            return [exact_id] + [pid for pid in self._popular(top_k) if pid != exact_id][: top_k - 1]
        if constraints is None:
            constraints = _query_constraints(query)
        allowed_pool = self._constraint_pool(constraints)
        terms = _tokens(query)
        if not terms:
            return self._popular(top_k, allowed_pool)
        scores: dict[str, float] = {}
        for term in terms:
            for pid, weight in self.postings.get(term, []):
                if allowed_pool is not None and pid not in allowed_pool:
                    continue
                scores[pid] = scores.get(pid, 0.0) + weight
        if not scores:
            return self._popular(top_k, allowed_pool)

        budget = constraints["budget"]
        phrases = [" ".join(pair) for pair in zip(terms, terms[1:])]
        qnorm = _norm_match(query)
        for pid in list(scores):
            doc = self.docs[pid]
            text = doc["text"]
            padded = doc["padded"]
            title = doc["title"]
            title_norm = doc["title_norm"]
            phrase_hits = sum(1 for phrase in phrases if phrase in text)
            popularity = math.log1p(doc["reviews"]) * 0.08 + doc["rating"] * 0.18
            scores[pid] += phrase_hits * 5.0 + popularity
            title_hits = sum(1 for term in terms if term in title)
            scores[pid] += title_hits * 8.0
            if qnorm and qnorm == title_norm:
                scores[pid] += 1000.0
            elif qnorm and (qnorm in title_norm or title_norm in qnorm):
                scores[pid] += 350.0
            elif terms and all(term in title for term in terms):
                scores[pid] += 120.0
            department = constraints["department"]
            if department:
                rule = DEPARTMENT_RULES[department]
                audience = doc["audience"]
                has_allowed = any(value in audience for value in rule["allow"])
                has_blocked = any(value in audience for value in rule["block"])
                if has_allowed:
                    scores[pid] += 60.0
                if has_blocked:
                    scores[pid] -= 120.0
            sleeve = constraints["sleeve"]
            if sleeve:
                rule = SLEEVE_RULES[sleeve]
                core = doc["core"]
                garment = doc["garment"]
                has_allowed = any(value in core for value in rule["allow"]) or any(value in garment for value in rule["allow"])
                has_blocked = any(value in core for value in rule["block"])
                if has_allowed:
                    scores[pid] += 70.0
                if has_blocked:
                    scores[pid] -= 120.0
            product_type = constraints["product_type"]
            if product_type:
                if pid in self.constraint_sets[f"type:{product_type}"]:
                    scores[pid] += 90.0
                else:
                    scores[pid] -= 80.0
            for color in constraints["colors"]:
                if pid in self.constraint_sets[f"color:{color}"]:
                    scores[pid] += 55.0
            for material in constraints["materials"]:
                if pid in self.constraint_sets[f"material:{material}"]:
                    scores[pid] += 50.0
            for color in constraints["exclude_colors"]:
                if pid in self.constraint_sets[f"color:{color}"]:
                    scores[pid] -= 90.0
            for material in constraints["exclude_materials"]:
                if pid in self.constraint_sets[f"material:{material}"]:
                    scores[pid] -= 80.0
            price = doc["price"]
            if budget is not None and price is not None:
                if price <= budget:
                    scores[pid] += 4.0
                else:
                    scores[pid] -= min(6.0, (price - budget) / max(budget, 1.0) * 4.0)
        search_depth = min(len(scores), max(top_k * 8, 80))
        return heapq.nlargest(search_depth, scores, key=lambda pid: (scores[pid], pid))[:top_k]

    def _add_constraint_memberships(self, pid: str) -> None:
        doc = self.docs[pid]
        audience = doc["audience"]
        garment = doc["garment"]
        attributes = doc["attributes"]
        product_type_text = doc["product_type"]
        for name, rule in DEPARTMENT_RULES.items():
            if _contains_any(audience, rule["allow"]) and not _contains_any(audience, rule["block"]):
                self.constraint_sets[f"department:{name}"].add(pid)
        for name, rule in SLEEVE_RULES.items():
            if _rule_match(doc["core"], garment, rule):
                self.constraint_sets[f"sleeve:{name}"].add(pid)
        for name, rule in COLOR_RULES.items():
            if _contains_any(attributes, rule["allow"]):
                self.constraint_sets[f"color:{name}"].add(pid)
        for name, rule in MATERIAL_RULES.items():
            if _contains_any(attributes, rule["allow"]):
                self.constraint_sets[f"material:{name}"].add(pid)
        for name, rule in PRODUCT_TYPE_RULES.items():
            if _contains_any(product_type_text, rule["allow"]) and not _contains_any(product_type_text, rule.get("block", ())):
                self.constraint_sets[f"type:{name}"].add(pid)

    def _constraint_pool(self, constraints: dict) -> set[str] | None:
        pool: set[str] | None = None
        primary_keys: list[str] = []
        if constraints["department"]:
            primary_keys.append(f"department:{constraints['department']}")
        if constraints["product_type"]:
            primary_keys.append(f"type:{constraints['product_type']}")
        if constraints["sleeve"]:
            primary_keys.append(f"sleeve:{constraints['sleeve']}")
        for key in primary_keys:
            candidates = self.constraint_sets[key]
            narrowed = candidates if pool is None else pool & candidates
            if narrowed:
                pool = set(narrowed)
        for color in constraints["colors"]:
            candidates = self.constraint_sets[f"color:{color}"]
            narrowed = candidates if pool is None else pool & candidates
            if len(narrowed) >= TOP_K:
                pool = set(narrowed)
        for material in constraints["materials"]:
            candidates = self.constraint_sets[f"material:{material}"]
            narrowed = candidates if pool is None else pool & candidates
            if len(narrowed) >= TOP_K:
                pool = set(narrowed)
        excluded: set[str] = set()
        for department in constraints["exclude_departments"]:
            excluded |= self.constraint_sets[f"department:{department}"]
        for sleeve in constraints["exclude_sleeves"]:
            excluded |= self.constraint_sets[f"sleeve:{sleeve}"]
        for product_type in constraints["exclude_product_types"]:
            excluded |= self.constraint_sets[f"type:{product_type}"]
        for color in constraints["exclude_colors"]:
            excluded |= self.constraint_sets[f"color:{color}"]
        for material in constraints["exclude_materials"]:
            excluded |= self.constraint_sets[f"material:{material}"]
        if excluded:
            pool = (set(self.docs) if pool is None else pool) - excluded
        return pool

    def _satisfies(self, pid: str, constraints: dict) -> bool:
        doc = self.docs[pid]
        department = constraints["department"]
        if department:
            rule = DEPARTMENT_RULES[department]
            audience = doc["audience"]
            if not _contains_any(audience, rule["allow"]):
                return False
            if _contains_any(audience, rule["block"]):
                return False
        sleeve = constraints["sleeve"]
        if sleeve:
            rule = SLEEVE_RULES[sleeve]
            core = doc["core"]
            garment = doc["garment"]
            if not _rule_match(core, garment, rule):
                return False
            if _contains_any(core, rule["block"]):
                return False
        product_type = constraints["product_type"]
        if product_type and pid not in self.constraint_sets[f"type:{product_type}"]:
            return False
        for department in constraints["exclude_departments"]:
            if pid in self.constraint_sets[f"department:{department}"]:
                return False
        for sleeve in constraints["exclude_sleeves"]:
            if pid in self.constraint_sets[f"sleeve:{sleeve}"]:
                return False
        for product_type in constraints["exclude_product_types"]:
            if pid in self.constraint_sets[f"type:{product_type}"]:
                return False
        for color in constraints["exclude_colors"]:
            if pid in self.constraint_sets[f"color:{color}"]:
                return False
        for material in constraints["exclude_materials"]:
            if pid in self.constraint_sets[f"material:{material}"]:
                return False
        return True

    def _popular(self, top_k: int, pool: set[str] | None = None) -> list[str]:
        candidates = pool if pool is not None else self.docs.keys()
        return heapq.nlargest(
            top_k,
            candidates,
            key=lambda pid: (self.docs[pid]["rating"], self.docs[pid]["reviews"], pid),
        )


def _now_ms(start: float) -> int:
    return int(round((time.perf_counter() - start) * 1000))


class TraceAgent(Agent):
    def __init__(self, catalog_path: str | Path) -> None:
        self.use_qwen = False
        self._last_qwen_trace: dict = {}
        super().__init__(catalog_path)

    def _qwen_command(self) -> list[str] | None:
        if not self.use_qwen:
            return None
        return super()._qwen_command()

    def _run_qwen(self, prompt: str) -> str | None:
        command = self._qwen_command()
        trace = self._last_qwen_trace
        trace.update({
            "enabled": self.use_qwen,
            "model": agent_mod.QWEN_RERANK_MODEL,
            "timeout": agent_mod._env_float("QWEN_RERANK_TIMEOUT", agent_mod.QWEN_RERANK_TIMEOUT),
        })
        if not command:
            trace["status"] = "skipped"
            trace["summary"] = "No local Qwen command/model available."
            return None
        uses_placeholder = any("{prompt}" in part for part in command)
        if uses_placeholder:
            command = [part.replace("{prompt}", prompt) for part in command]
            stdin = None
        else:
            stdin = prompt
        started = time.perf_counter()
        trace["attempted"] = True
        try:
            completed = subprocess.run(
                command,
                input=stdin,
                capture_output=True,
                text=True,
                timeout=trace["timeout"],
                check=False,
            )
        except subprocess.TimeoutExpired:
            self._qwen_failed = True
            trace["status"] = "timeout"
            trace["elapsed_ms"] = _now_ms(started)
            trace["summary"] = f"Timed out after {trace['timeout']}s; used algorithmic order."
            return None
        except (OSError, subprocess.SubprocessError):
            self._qwen_failed = True
            trace["status"] = "fallback"
            trace["elapsed_ms"] = _now_ms(started)
            trace["summary"] = "Local Qwen call failed; used algorithmic order."
            return None
        trace["elapsed_ms"] = _now_ms(started)
        if completed.returncode != 0 or not completed.stdout.strip():
            self._qwen_failed = True
            trace["status"] = "fallback"
            trace["summary"] = "Qwen returned no usable output."
            return None
        trace["raw_chars"] = len(completed.stdout)
        return completed.stdout

    def _qwen_rerank(self, state: dict, ranked: list[str], top_k: int) -> list[str]:
        self._last_qwen_trace = {
            "enabled": self.use_qwen,
            "min_turn": agent_mod.QWEN_MIN_TURN,
            "pool_size": min(len(ranked), max(top_k, agent_mod.QWEN_RERANK_DEPTH)),
        }
        if len(ranked) < 2:
            self._last_qwen_trace.update({"status": "skipped", "summary": "Not enough candidates."})
            return ranked[:top_k]
        if not self.use_qwen:
            self._last_qwen_trace.update({"status": "skipped", "summary": "Qwen toggle is off."})
            return ranked[:top_k]
        if state.get("turn", 0) < agent_mod.QWEN_MIN_TURN:
            self._last_qwen_trace.update({
                "status": "skipped",
                "summary": f"Skipped until turn {agent_mod.QWEN_MIN_TURN}.",
            })
            return ranked[:top_k]

        pool = ranked[:max(top_k, agent_mod.QWEN_RERANK_DEPTH)]
        output = self._run_qwen(self._qwen_prompt(state, pool, top_k))
        if not output:
            return ranked[:top_k]
        ordered = self._parse_qwen_order(output, pool)
        if len(ordered) < top_k:
            self._last_qwen_trace.update({
                "status": "fallback",
                "summary": "Qwen output did not contain a full valid order.",
            })
            return ranked[:top_k]
        seen = set(ordered)
        reranked = ordered + [pid for pid in pool if pid not in seen]
        self._last_qwen_trace.update({
            "status": "accepted",
            "summary": f"Accepted Qwen order for {len(ordered[:top_k])} candidates.",
        })
        return (reranked + ranked[len(pool):])[:top_k]

    def _preview(self, ids: list[str], target: str | None = None, limit: int = 10) -> list[dict]:
        return [
            {
                "parent_asin": pid,
                "summary": _short(self._summary.get(pid, ""), 260),
                "is_target": pid == target,
            }
            for pid in ids[:limit]
        ]

    def traced_respond(self, session_id: str, user_message: str, turn: int, top_k: int, target: str) -> tuple[dict, dict]:
        started = time.perf_counter()
        state = self._sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")

        before = len(state["plain"])
        was_decline = self._declined(user_message) if state.get("last_ask") else False
        was_override = self._override(user_message) if agent_mod.USE_OVERRIDE else False
        self._accumulate(state, user_message)
        retired = False
        if state["last_ask"] and len(state["plain"]) == before:
            state["retired"].add(state["last_ask"])
            retired = True
        state["size"] = len(state["plain"])
        state["turn"] = turn

        plain = self._query(state["plain"])
        stems = self._query(state["stems"])
        qwen_gate = self.use_qwen and agent_mod.USE_QWEN_RERANK and turn >= agent_mod.QWEN_MIN_TURN
        key = (tuple(plain), top_k, qwen_gate)

        raw_ranking: list[str] = []
        stemmed_ranking: list[str] = []
        dense_ranking: list[str] = []
        fused: list[tuple[str, float]] = []
        cache_hit = key in self._cache
        if cache_hit:
            ranked = self._cache[key]
            self._last_qwen_trace = {"status": "skipped", "summary": "Used cached ranking."}
        else:
            raw_ranking = self._search("products", plain, agent_mod.RETRIEVE)
            stemmed_ranking = self._search("products_stem", stems, agent_mod.RETRIEVE)
            dense_ranking = (
                self._dense_ranking(" ".join(state["text"]), agent_mod.DENSE_LIMIT)
                if agent_mod.USE_DENSE else []
            )
            rankings = [
                (raw_ranking, 1.0),
                (stemmed_ranking, 1.0),
                (dense_ranking, agent_mod.DENSE_WEIGHT),
            ]
            if agent_mod.USE_EXPANSION:
                rankings.append((
                    self._search(
                        "products_stem",
                        stems + self._expand(stemmed_ranking[: agent_mod.FEEDBACK_DOCS], stems),
                        agent_mod.RETRIEVE,
                    ),
                    1.0,
                ))
            fused = self._fuse([ranking for ranking in rankings if ranking[0]], max(top_k, agent_mod.RERANK_DEPTH))
            ranked = self._rerank(state, fused, top_k)
            self._cache[key] = ranked

        response = self._reply(state, ranked)
        recommendations = [item["parent_asin"] for item in response.get("recommendations", [])]
        qwen_trace = dict(self._last_qwen_trace or {})
        dense_status = "active" if dense_ranking else ("skipped" if self._index is None else "active")
        trace = {
            "elapsed_ms": _now_ms(started),
            "state": {
                "term_count": len(state["plain"]),
                "phrase_count": len(state["phrases"]),
                "budget": state.get("budget"),
                "last_ask": state.get("last_ask"),
                "top_terms": plain[:18],
                "suppression": ", ".join(
                    f"{name}:{value:.2f}" for name, value in sorted(state.get("suppress", {}).items())
                ),
            },
            "retrieval": {
                "raw_count": len(raw_ranking),
                "stem_count": len(stemmed_ranking),
                "dense_count": len(dense_ranking),
                "fused_count": len(fused),
                "cache_hit": cache_hit,
            },
            "stages": {
                "message": {"summary": _short(user_message, 120), "status": "active"},
                "intent": {
                    "summary": f"decline={was_decline}, override={was_override}, retired={retired}",
                    "status": "active",
                },
                "state": {"summary": f"{len(plain)} terms, {len(state['phrases'])} phrases.", "status": "active"},
                "raw_bm25": {"summary": f"{len(raw_ranking)} candidates from raw tokens.", "status": "active"},
                "stem_bm25": {"summary": f"{len(stemmed_ranking)} candidates from Porter stems.", "status": "active"},
                "dense": {
                    "summary": f"{len(dense_ranking)} semantic candidates." if dense_ranking else "No dense index/model loaded.",
                    "status": dense_status,
                },
                "retrieval": {"summary": "Hybrid candidate generation complete.", "status": "active"},
                "fusion": {"summary": f"{len(fused)} candidates after reciprocal rank fusion.", "status": "active"},
                "rerank": {"summary": f"Algorithmic top {min(top_k, len(recommendations))} selected.", "status": "active"},
                "qwen": qwen_trace or {"summary": "Qwen not reached.", "status": "skipped"},
                "response": {
                    "summary": f"Asked {response.get('ask_attribute')}; returned {len(recommendations)} products.",
                    "status": "good",
                },
            },
            "previews": {
                "raw": self._preview(raw_ranking, target, 5),
                "stemmed": self._preview(stemmed_ranking, target, 5),
                "dense": self._preview(dense_ranking, target, 5),
                "fused": self._preview([pid for pid, _ in fused], target, 5),
            },
        }
        return response, trace


class VisualSession:
    def __init__(self, sample: dict, agent: TraceAgent, categories: dict[str, list[str]], products: dict[str, dict], qwen: bool) -> None:
        self.id = f"viz_{uuid.uuid4().hex[:12]}"
        self.sample = sample
        self.agent = agent
        self.qwen = qwen
        self.session_id = f"frontend_{uuid.uuid4().hex}"
        self.target = str(sample["ground_truth"]["parent_asin"])
        self.card, self.behavior = ev.materialize_hidden_fields(sample, products)
        self.effective_sample = {**sample, "intent_card": self.card, "behavior": self.behavior}
        self.disclosed: set[str] = set()
        self.boundary_used = False
        self.override_applied = sample["scenario_type"] != "intent_override"
        self.user_message = ev.initial_message(
            self.effective_sample,
            ev.coarse_category(categories.get(self.target, [])),
            self.disclosed,
        )
        self.turn = 0
        self.done = False
        self.hit = False
        self.best_rank: int | None = None
        self.first_hit_turn: int | None = None
        self.turns: list[dict] = []
        self.agent.use_qwen = qwen
        self.agent.reset(self.session_id, sample["user_profile"])

    def step(self) -> None:
        if self.done:
            return
        self.turn += 1
        self.agent.use_qwen = self.qwen
        response, trace = self.agent.traced_respond(self.session_id, self.user_message, self.turn, TOP_K, self.target)
        ranked = ev.normalize_recommendations(response.get("recommendations"), SERVER.catalog_ids)
        if self.override_applied and self.target in ranked:
            self.hit = True
            self.done = True
            self.best_rank = ranked.index(self.target) + 1
            self.first_hit_turn = self.turn
        recommendations = self.agent._preview(ranked, self.target, TOP_K)
        self.turns.append({
            "turn": self.turn,
            "user_message": self.user_message,
            "agent_message": response.get("message", ""),
            "ask_attribute": response.get("ask_attribute"),
            "recommendations": recommendations,
            "trace": trace,
        })
        if self.done or self.turn >= MAX_TURNS:
            self.done = True
            return
        override = self.effective_sample.get("behavior", {}).get("override") or {}
        if not self.override_applied and self.turn + 1 == int(override.get("turn", 3)):
            self.override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                self.disclosed.add(new_value)
            self.user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
        else:
            self.user_message, self.boundary_used = ev.customer_reply(
                self.effective_sample,
                response.get("ask_attribute"),
                self.disclosed,
                self.boundary_used,
            )

    def payload(self) -> dict:
        return {
            "id": self.id,
            "sample_id": self.sample["sample_id"],
            "scenario": self.sample["scenario_type"],
            "qwen": self.qwen,
            "target": self.target,
            "turn": self.turn,
            "done": self.done,
            "hit": self.hit,
            "best_rank": self.best_rank,
            "first_hit_turn": self.first_hit_turn,
            "turns": self.turns,
        }


class ShoppingSession:
    def __init__(self, index: StorefrontIndex, products: dict[str, dict]) -> None:
        self.id = f"shop_{uuid.uuid4().hex[:12]}"
        self.index = index
        self.products = products
        self.turn = 0
        self.history: list[dict] = []
        self.query_parts: list[str] = []
        self.constraints = _empty_constraints()
        self.query_text = ""

    def submit(self, message: str) -> dict:
        message = re.sub(r"\s+", " ", message).strip()
        if not message:
            raise ValueError("Please enter a shopping request.")
        if self.turn >= MAX_TURNS:
            self.turn = 0
            self.history = []
            self.query_parts = []
            self.constraints = _empty_constraints()
            self.query_text = ""
        self.turn += 1
        if re.search(r"\b(start over|new search|reset)\b", message, re.I):
            message = re.sub(r"\b(start over|new search|reset)\b", "", message, flags=re.I).strip()
            self.history = []
            self.query_parts = []
            self.constraints = _empty_constraints()
        parsed = _query_constraints(message)
        self.constraints = _merge_constraints(self.constraints, parsed)
        if message:
            self.query_parts.append(message)
        self.query_text = " ".join([*self.query_parts, *_constraint_terms(self.constraints)]).strip()
        if not self.query_text:
            raise ValueError("Please enter a shopping request.")
        ranked = self.index.search(self.query_text, TOP_K, self.constraints)
        products = [_product_payload(pid, self.products.get(pid, {})) for pid in ranked]
        reply = f"Showing {len(products)} matching products."
        self.history.append({"role": "user", "text": message})
        self.history.append({"role": "assistant", "text": reply})
        return {
            "id": self.id,
            "turn": self.turn,
            "message": reply,
            "history": self.history[-12:],
            "products": products,
        }


class PipelineServer:
    def __init__(self, catalog: Path, dataset: Path) -> None:
        if not catalog.exists():
            raise FileNotFoundError(f"Missing catalog: {catalog}")
        if not dataset.exists():
            raise FileNotFoundError(f"Missing public sessions: {dataset}")
        self.catalog = catalog
        self.samples = ev.load_jsonl(dataset)
        self.catalog_ids, self.categories, self.products = ev.catalog_index(catalog)
        self.store_index = StorefrontIndex(self.products)
        self.agent: TraceAgent | None = None
        self.sessions: dict[str, VisualSession] = {}
        self.shopping_sessions: dict[str, ShoppingSession] = {}

    def get_agent(self) -> TraceAgent:
        if self.agent is None:
            self.agent = TraceAgent(self.catalog)
        return self.agent

    def options(self) -> dict:
        metrics = {}
        result_path = ROOT / "results.json"
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
                metrics = {
                    key: result[key]
                    for key in ("hit_rate_at_10", "mrr", "mttc", "efficiency", "recommended_technical_score")
                    if key in result
                }
            except (OSError, json.JSONDecodeError):
                metrics = {}
        return {
            "metrics": metrics,
            "sessions": [
                {"sample_id": sample["sample_id"], "scenario_type": sample["scenario_type"]}
                for sample in self.samples
            ]
        }

    def start(self, sample_id: str, qwen: bool) -> dict:
        sample = next((item for item in self.samples if item["sample_id"] == sample_id), None)
        if sample is None:
            raise ValueError(f"Unknown sample id: {sample_id}")
        session = VisualSession(sample, self.get_agent(), self.categories, self.products, qwen)
        self.sessions[session.id] = session
        return session.payload()

    def search(self, sid: str, message: str, qwen: bool = False) -> dict:
        session = self.shopping_sessions.get(sid)
        if session is None:
            session = ShoppingSession(self.store_index, self.products)
            self.shopping_sessions[session.id] = session
        return session.submit(message)

    def get(self, sid: str) -> VisualSession:
        session = self.sessions.get(sid)
        if session is None:
            raise ValueError("Session expired or unknown.")
        return session


SERVER: PipelineServer


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, format: str, *args) -> None:
        return

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.close_connection = True
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def _body(self) -> dict:
        size = int(self.headers.get("Content-Length") or 0)
        if size <= 0:
            return {}
        return json.loads(self.rfile.read(size).decode("utf-8"))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(200, STORE_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/options":
            self._json(200, SERVER.options())
            return
        self._json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        try:
            parsed = urlparse(self.path)
            payload = self._body()
            if parsed.path == "/api/start":
                self._json(200, SERVER.start(str(payload.get("sample_id")), bool(payload.get("qwen"))))
                return
            if parsed.path == "/api/search":
                self._json(200, SERVER.search(
                    str(payload.get("id") or ""),
                    str(payload.get("message") or ""),
                    bool(payload.get("qwen", False)),
                ))
                return
            if parsed.path == "/api/turn":
                session = SERVER.get(str(payload.get("id")))
                session.step()
                self._json(200, session.payload())
                return
            if parsed.path == "/api/run":
                session = SERVER.get(str(payload.get("id")))
                while not session.done:
                    session.step()
                self._json(200, session.payload())
                return
            self._json(404, {"error": "Not found"})
        except Exception as exc:
            self._json(400, {"error": str(exc)})


def main() -> None:
    global SERVER
    parser = argparse.ArgumentParser(description="Local frontend for the shopping copilot pipeline")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    SERVER = PipelineServer(Path(args.catalog), Path(args.dataset))
    httpd = HTTPServer((args.host, args.port), Handler)
    print(f"Pipeline frontend running at http://{args.host}:{args.port}/", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
