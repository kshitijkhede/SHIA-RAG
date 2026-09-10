/**
 * SHIA-RAG 2.0 Web Application (ChatGPT & Claude AI Interface)
 * ============================================================
 * Client logic for PDF ingestion, verifiable query retrieval,
 * dynamic reasoning inspection, citation navigation, and forest visualization.
 */

class ShiaRagApp {
  constructor() {
    this.tokenBudget = 2048;
    this.activeDoc = null;
    this.activeDocId = null;
    this.currentForestDocId = null;
    this.documents = [];
    this.messages = [];
    this.forestData = null;
    this.isQuerying = false;

    this.initElements();
    this.initEventListeners();
    this.fetchForestStats();
    this.fetchSampleQueries();
    this.fetchDocuments();
    this.checkLlmStatus();
    this.checkUrlQueryParams();
  }

  checkUrlQueryParams() {
    const params = new URLSearchParams(window.location.search);
    const q = params.get("q");
    if (q) {
      this.chatInput.value = q;
      setTimeout(() => this.handleSendQuery(), 300);
    }
    const view = params.get("view");
    const docParam = params.get("doc_id");
    if (view === "forest") {
      setTimeout(() => this.openForestModal(docParam || null), 300);
    }
    const node = params.get("node");
    if (node) {
      setTimeout(() => this.openInspector(node), 300);
    }
    const demo = params.get("demo");
    if (demo === "multiturn") {
      setTimeout(() => {
        this.chatInput.value = "what is operating system";
        this.handleSendQuery();
        setTimeout(() => {
          this.chatInput.value = "can you give me more content";
          this.handleSendQuery();
        }, 3200);
      }, 500);
    }
  }

  initElements() {
    // Top Nav & Sidebar
    this.sidebar = document.getElementById("sidebar");
    this.toggleSidebarBtn = document.getElementById("toggleSidebarBtn");
    this.newChatBtn = document.getElementById("newChatBtn");
    this.topNavDocName = document.getElementById("topNavDocName");
    this.activeDocSelect = document.getElementById("activeDocSelect");
    this.tokenBudgetSelect = document.getElementById("tokenBudgetSelect");
    this.navForestBtn = document.getElementById("navForestBtn");
    this.auditInvariantsBtn = document.getElementById("auditInvariantsBtn");

    // Document & Preset Actions
    this.pdfFileInput = document.getElementById("pdfFileInput");
    this.sidebarUploadTrigger = document.getElementById("sidebarUploadTrigger");
    this.documentList = document.getElementById("documentList");
    this.freshTreeUploadToggle = document.getElementById("freshTreeUploadToggle");
    this.heroFreshTreeToggle = document.getElementById("heroFreshTreeToggle");
    this.loadSampleKbBtn = document.getElementById("loadSampleKbBtn");
    this.resetForestBtn = document.getElementById("resetForestBtn");
    this.openForestExplorerBtn = document.getElementById("openForestExplorerBtn");

    // Chat Containers
    this.chatContainer = document.getElementById("chatContainer");
    this.welcomeScreen = document.getElementById("welcomeScreen");
    this.messagesStream = document.getElementById("messagesStream");
    this.chatInput = document.getElementById("chatInput");
    this.sendMessageBtn = document.getElementById("sendMessageBtn");
    this.composerAttachBtn = document.getElementById("composerAttachBtn");
    this.starterPromptsGrid = document.getElementById("starterPromptsGrid");

    // Dropzones & Ingestion Banner
    this.heroDropzone = document.getElementById("heroDropzone");
    this.heroUploadBtn = document.getElementById("heroUploadBtn");
    this.ingestionBanner = document.getElementById("ingestionBanner");
    this.bannerProgressFill = document.getElementById("bannerProgressFill");
    this.ingestionTitle = document.getElementById("ingestionTitle");
    this.ingestionSub = document.getElementById("ingestionSub");

    // Forest Stats Elements
    this.statTotalNodes = document.getElementById("statTotalNodes");
    this.statTotalEdges = document.getElementById("statTotalEdges");
    this.statMaxDepth = document.getElementById("statMaxDepth");
    this.statCrosslinks = document.getElementById("statCrosslinks");
    this.sidebarInvariantBadge = document.getElementById("sidebarInvariantBadge");

    // Inspector Drawer
    this.nodeInspectorDrawer = document.getElementById("nodeInspectorDrawer");
    this.closeInspectorBtn = document.getElementById("closeInspectorBtn");
    this.inspectorNodeType = document.getElementById("inspectorNodeType");
    this.inspectorNodeName = document.getElementById("inspectorNodeName");
    this.inspectorNodeId = document.getElementById("inspectorNodeId");
    this.inspectorDepth = document.getElementById("inspectorDepth");
    this.inspectorConfidence = document.getElementById("inspectorConfidence");
    this.inspectorAbstraction = document.getElementById("inspectorAbstraction");
    this.inspectorTokens = document.getElementById("inspectorTokens");
    this.inspectorDefinition = document.getElementById("inspectorDefinition");
    this.inspectorParent = document.getElementById("inspectorParent");
    this.inspectorChildren = document.getElementById("inspectorChildren");
    this.inspectorCrosslinks = document.getElementById("inspectorCrosslinks");
    this.inspectorEvidence = document.getElementById("inspectorEvidence");

    // Forest Modal
    this.forestModalOverlay = document.getElementById("forestModalOverlay");
    this.forestDocTabs = document.getElementById("forestDocTabs");
    this.forestTreeSummaryBanner = document.getElementById("forestTreeSummaryBanner");
    this.forestGraphContainer = document.getElementById("forestGraphContainer");
    this.closeForestModalBtn = document.getElementById("closeForestModalBtn");
    this.modalCloseActionBtn = document.getElementById("modalCloseActionBtn");
    this.modalAuditBtn = document.getElementById("modalAuditBtn");
    this.forestSearchInput = document.getElementById("forestSearchInput");

    // Cloud LLM Acceleration Modal Elements
    this.llmConfigBtn = document.getElementById("llmConfigBtn");
    this.llmEngineBadge = document.getElementById("llmEngineBadge");
    this.llmModalOverlay = document.getElementById("llmModalOverlay");
    this.closeLlmModalBtn = document.getElementById("closeLlmModalBtn");
    this.cancelLlmBtn = document.getElementById("cancelLlmBtn");
    this.saveLlmBtn = document.getElementById("saveLlmBtn");
    this.llmActiveStatusText = document.getElementById("llmActiveStatusText");
    this.llmProviderSelect = document.getElementById("llmProviderSelect");
    this.llmApiKeyInput = document.getElementById("llmApiKeyInput");

    // Multi-location LLM Triggers & Badges
    this.sidebarEngineCard = document.getElementById("sidebarEngineCard");
    this.sidebarEngineTypeBadge = document.getElementById("sidebarEngineTypeBadge");
    this.sidebarEngineName = document.getElementById("sidebarEngineName");
    this.sidebarEngineStatusDetail = document.getElementById("sidebarEngineStatusDetail");
    this.heroLlmBanner = document.getElementById("heroLlmBanner");
    this.heroLlmBtn = document.getElementById("heroLlmBtn");
    this.heroLlmEngineDesc = document.getElementById("heroLlmEngineDesc");
    this.composerEngineIndicator = document.getElementById("composerEngineIndicator");
    this.composerEngineBadge = document.getElementById("composerEngineBadge");

    // Toast Container
    this.toastContainer = document.getElementById("toastContainer");
  }

  initEventListeners() {
    // Sidebar toggle (mobile)
    if (this.toggleSidebarBtn) {
      this.toggleSidebarBtn.addEventListener("click", () => {
        this.sidebar.classList.toggle("open");
      });
    }

    // New chat action
    this.newChatBtn.addEventListener("click", () => this.startNewChat());
    document.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        this.startNewChat();
      }
    });

    // Token budget change
    this.tokenBudgetSelect.addEventListener("change", (e) => {
      this.tokenBudget = parseInt(e.target.value, 10);
      this.showToast(`Token budget set to ${this.tokenBudget} tokens`, "info");
    });

    // Preset Knowledge Base actions
    this.loadSampleKbBtn.addEventListener("click", () => this.preloadSampleKB());
    this.resetForestBtn.addEventListener("click", () => this.resetKnowledgeForest());

    // Upload mode toggles sync
    if (this.freshTreeUploadToggle && this.heroFreshTreeToggle) {
      this.freshTreeUploadToggle.addEventListener("change", () => {
        this.heroFreshTreeToggle.checked = this.freshTreeUploadToggle.checked;
      });
      this.heroFreshTreeToggle.addEventListener("change", () => {
        this.freshTreeUploadToggle.checked = this.heroFreshTreeToggle.checked;
      });
    }

    // Active document scope selector
    if (this.activeDocSelect) {
      this.activeDocSelect.addEventListener("change", (e) => {
        const val = e.target.value;
        this.activeDocId = val || null;
        const docObj = this.documents.find((d) => d.doc_id === val);
        this.activeDoc = docObj ? docObj.filename : null;
        this.renderDocumentList();
        this.updateScopeIndicators();
        this.fetchSampleQueries();
        this.showToast(this.activeDoc ? `Retrieval scoped to: '${this.activeDoc}'` : "Retrieval scoped to: Entire Forest", "info");
      });
    }

    // File Upload Input & Change Listeners
    if (this.pdfFileInput) {
      this.pdfFileInput.addEventListener("click", () => {
        this.pdfFileInput.value = "";
      });
      this.pdfFileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
          this.uploadPdfFile(e.target.files[0]);
        }
      });
    }

    // Secondary Click Handlers (Safe Fallback)
    if (this.sidebarUploadTrigger) {
      this.sidebarUploadTrigger.addEventListener("click", (e) => {
        if (e.target !== this.pdfFileInput) {
          this.pdfFileInput.click();
        }
      });
    }

    if (this.heroDropzone) {
      this.heroDropzone.addEventListener("click", (e) => {
        // Prevent opening file chooser if clicking on checkbox or label
        if (e.target.closest(".hero-mode-opt")) return;
        this.pdfFileInput.click();
      });
    }

    // Window-level Drag & Drop Handling (Prevents Chrome from navigating away from page)
    ["dragenter", "dragover"].forEach((eventName) => {
      window.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (this.heroDropzone) this.heroDropzone.classList.add("dragover");
      }, false);
    });

    ["dragleave"].forEach((eventName) => {
      window.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.clientX === 0 || e.clientY === 0 || e.target === document.documentElement || e.target === document.body) {
          if (this.heroDropzone) this.heroDropzone.classList.remove("dragover");
        }
      }, false);
    });

    window.addEventListener("drop", (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (this.heroDropzone) this.heroDropzone.classList.remove("dragover");
      const files = e.dataTransfer ? e.dataTransfer.files : null;
      if (files && files.length > 0) {
        const file = files[0];
        if (file.name.toLowerCase().endsWith(".pdf") || file.type === "application/pdf") {
          this.uploadPdfFile(file);
        } else {
          this.showToast("Please drop a valid PDF document (*.pdf)", "error");
        }
      }
    }, false);

    // Starter Prompt Clicks
    this.starterPromptsGrid.addEventListener("click", (e) => {
      const card = e.target.closest(".prompt-card");
      if (card && card.dataset.query) {
        this.chatInput.value = card.dataset.query;
        this.handleSendQuery();
      }
    });

    // Composer Input & Send
    this.sendMessageBtn.addEventListener("click", () => this.handleSendQuery());
    this.chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        this.handleSendQuery();
      }
    });

    // Auto-resize composer textarea
    this.chatInput.addEventListener("input", () => {
      this.chatInput.style.height = "auto";
      this.chatInput.style.height = Math.min(this.chatInput.scrollHeight, 180) + "px";
    });

    // Forest Modal & Invariants
    this.navForestBtn.addEventListener("click", () => this.openForestModal());
    this.openForestExplorerBtn.addEventListener("click", () => this.openForestModal());
    this.closeForestModalBtn.addEventListener("click", () => this.closeForestModal());
    this.modalCloseActionBtn.addEventListener("click", () => this.closeForestModal());
    this.modalAuditBtn.addEventListener("click", () => this.auditInvariants());
    this.auditInvariantsBtn.addEventListener("click", () => this.auditInvariants());

    this.forestSearchInput.addEventListener("input", (e) => {
      this.filterForestGraph(e.target.value.trim().toLowerCase());
    });

    // Cloud LLM Acceleration Modal Triggers
    const triggerModal = () => this.openLlmModal();
    if (this.llmConfigBtn) this.llmConfigBtn.addEventListener("click", triggerModal);
    if (this.sidebarEngineCard) this.sidebarEngineCard.addEventListener("click", triggerModal);
    if (this.heroLlmBtn) this.heroLlmBtn.addEventListener("click", triggerModal);
    if (this.composerEngineIndicator) this.composerEngineIndicator.addEventListener("click", triggerModal);

    if (this.closeLlmModalBtn) {
      this.closeLlmModalBtn.addEventListener("click", () => this.closeLlmModal());
    }
    if (this.cancelLlmBtn) {
      this.cancelLlmBtn.addEventListener("click", () => this.closeLlmModal());
    }
    if (this.saveLlmBtn) {
      this.saveLlmBtn.addEventListener("click", () => this.saveLlmConfig());
    }

    // Inspector Drawer
    this.closeInspectorBtn.addEventListener("click", () => this.closeInspector());
    document.addEventListener("click", (e) => {
      const citation = e.target.closest(".citation-pill");
      if (citation && citation.dataset.nid) {
        e.preventDefault();
        this.openInspector(citation.dataset.nid);
      }
      const chip = e.target.closest(".concept-chip");
      if (chip && chip.dataset.nid) {
        e.preventDefault();
        this.openInspector(chip.dataset.nid);
      }
    });
  }

  // ==========================================================================
  // Chat & Query Execution
  // ==========================================================================

  startNewChat() {
    this.messages = [];
    this.messagesStream.innerHTML = "";
    this.welcomeScreen.style.display = "flex";
    this.chatInput.value = "";
    this.chatInput.style.height = "auto";
    fetch("/api/reset_chat", { method: "POST" }).catch(() => {});
    this.showToast("Started a new conversation session.", "info");
  }

  async handleSendQuery() {
    const query = this.chatInput.value.trim();
    if (!query || this.isQuerying) return;

    // Switch to active chat stream
    this.welcomeScreen.style.display = "none";
    this.isQuerying = true;
    this.sendMessageBtn.disabled = true;
    this.chatInput.value = "";
    this.chatInput.style.height = "auto";

    // 1. Render User Message
    this.renderUserMessage(query);
    this.messages.push({ role: "user", content: query });

    // 2. Render Temporary Thinking Message
    const assistantMsgEl = this.createAssistantThinkingElement();
    this.messagesStream.appendChild(assistantMsgEl);
    this.scrollToBottom();

    try {
      const payload = {
        query: query,
        token_budget: this.tokenBudget,
        chat_history: this.messages.slice(-8),
      };
      if (this.activeDocId) {
        payload.doc_id = this.activeDocId;
      }

      const response = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Query execution failed");
      }

      const result = await response.json();
      this.messages.push({ role: "assistant", content: result.formatted_answer || result.answer });
      this.updateAssistantMessage(assistantMsgEl, result);
      this.fetchForestStats(); // Update stats if Thompson evolution smoothed priors
    } catch (err) {
      console.error(err);
      assistantMsgEl.innerHTML = `
        <div class="message-row assistant">
          <div class="message-avatar">AI</div>
          <div class="message-bubble">
            <div class="message-author">SHIA-RAG Verifier</div>
            <div class="answer-markdown" style="color: var(--accent-danger);">
              <strong>Error executing retrieval:</strong> ${err.message}
            </div>
          </div>
        </div>
      `;
    } finally {
      this.isQuerying = false;
      this.sendMessageBtn.disabled = false;
      this.scrollToBottom();
    }
  }

  renderUserMessage(text) {
    const row = document.createElement("div");
    row.className = "message-row user";
    row.innerHTML = `
      <div class="message-avatar">U</div>
      <div class="message-bubble">
        <div class="message-author">You</div>
        <div class="message-text">${this.escapeHtml(text)}</div>
      </div>
    `;
    this.messagesStream.appendChild(row);
  }

  createAssistantThinkingElement() {
    const container = document.createElement("div");
    container.className = "message-row assistant";
    container.innerHTML = `
      <div class="message-avatar">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
        </svg>
      </div>
      <div class="message-bubble">
        <div class="message-author">
          <span>SHIA-RAG 2.0 Verifier</span>
          <span class="author-tag">Dual-Tier Forest</span>
        </div>
        <div class="reasoning-accordion expanded">
          <div class="reasoning-header">
            <div class="reasoning-title-group">
              <div class="reasoning-icon">
                <div class="status-indicator live"></div>
              </div>
              <span class="reasoning-title">Thinking & Navigating Knowledge Forest...</span>
            </div>
          </div>
          <div class="reasoning-content" style="display:flex;">
            <div style="font-size:12px; color:var(--text-muted);">
              Analyzing query semantics, solving DC-Knapsack ancestor precedence, and verifying citation attribution...
            </div>
          </div>
        </div>
      </div>
    `;
    return container;
  }

  updateAssistantMessage(element, res) {
    const mode = res.routing_mode || "FACTUAL";
    const budgetPct = Math.min(100, Math.round((res.total_tokens / this.tokenBudget) * 100));
    const reward = (res.attribution_reward !== undefined) ? res.attribution_reward : 1.0;
    const rewardClass = (reward >= 0.7) ? "reward-high" : "";
    const traversedCount = res.evolution_update ? res.evolution_update.traversed_edge_count : 0;

    // Format concepts chips
    const conceptChips = (res.enriched_nodes || res.selected_nodes || []).map((n) => `
      <div class="concept-chip" data-nid="${n.node_id}" title="Click to inspect ${n.name}">
        <span>${n.name}</span>
        <span class="chip-depth">d=${n.depth}</span>
      </div>
    `).join("");

    // Format answer markdown with interactive citation pills
    const answerHtml = this.renderMarkdownWithCitations(res.formatted_answer || res.answer);

    element.innerHTML = `
      <div class="message-avatar">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
        </svg>
      </div>
      <div class="message-bubble">
        <div class="message-author">
          <span>SHIA-RAG 2.0 Verifier</span>
          <span class="author-tag">${mode} MODE</span>
        </div>

        <!-- Collapsible Reasoning Drawer -->
        <div class="reasoning-accordion">
          <div class="reasoning-header" onclick="this.parentElement.classList.toggle('expanded')">
            <div class="reasoning-title-group">
              <div class="reasoning-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 1 1 7.072 0l-.548.547A3.374 3.374 0 0 0 14 18.469V19a2 2 0 1 1-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"></path>
                </svg>
              </div>
              <span class="reasoning-title">Retrieval & Attribution Analysis</span>
              <span class="reasoning-mode-badge">${mode}</span>
            </div>
            <div class="reasoning-toggle-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16">
                <polyline points="6 9 12 15 18 9"></polyline>
              </svg>
            </div>
          </div>

          <div class="reasoning-content">
            <div class="reasoning-metrics-row">
              <div class="r-metric-card">
                <span class="r-metric-label">SRDR Route</span>
                <span class="r-metric-value">${mode}</span>
              </div>
              <div class="r-metric-card">
                <span class="r-metric-label">DC-Knapsack</span>
                <span class="r-metric-value">${res.total_tokens} / ${this.tokenBudget} tok</span>
                <div class="knapsack-progress-bar">
                  <div class="knapsack-progress-fill" style="width: ${budgetPct}%;"></div>
                </div>
              </div>
              <div class="r-metric-card">
                <span class="r-metric-label">Attribution R</span>
                <span class="r-metric-value ${rewardClass}">R = ${reward.toFixed(3)}</span>
              </div>
            </div>

            <div class="selected-concepts-tray">
              <div class="concepts-tray-title">Grounded Concepts Selected via Precedence DAG (${(res.selected_nodes || []).length}):</div>
              <div class="concepts-chips-wrap">
                ${conceptChips || "<span style='color:var(--text-muted);font-size:12px;'>No specific concepts selected</span>"}
              </div>
            </div>

            <div style="font-size:11.5px; color:var(--text-muted); display:flex; justify-content:space-between;">
              <span>Online Thompson Sampling: <strong>${traversedCount}</strong> edge posteriors updated</span>
              <span>Laplacian Smoothing: <strong>Active</strong></span>
            </div>
          </div>
        </div>

        <!-- Grounded Answer Body -->
        <div class="answer-markdown">
          ${answerHtml}
        </div>

        <!-- Action Toolbar -->
        <div class="message-actions">
          <button class="msg-action-btn copy-btn" title="Copy answer">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>
            <span>Copy</span>
          </button>
          <button class="msg-action-btn view-forest-action" title="View in forest">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="18" cy="5" r="3"></circle>
              <circle cx="6" cy="12" r="3"></circle>
              <circle cx="18" cy="19" r="3"></circle>
              <line x1="8.59" y1="13.51" x2="15.42" y2="17.49"></line>
              <line x1="15.41" y1="6.51" x2="8.59" y2="10.49"></line>
            </svg>
            <span>Forest</span>
          </button>
        </div>
      </div>
    `;

    // Bind copy and forest buttons
    const copyBtn = element.querySelector(".copy-btn");
    if (copyBtn) {
      copyBtn.addEventListener("click", () => {
        navigator.clipboard.writeText(res.formatted_answer || res.answer);
        this.showToast("Grounded answer copied to clipboard!", "success");
      });
    }

    const viewForestBtn = element.querySelector(".view-forest-action");
    if (viewForestBtn) {
      viewForestBtn.addEventListener("click", () => this.openForestModal());
    }
  }

  // ==========================================================================
  // File Upload Handling
  // ==========================================================================

  async uploadPdfFile(file) {
    if (!file) return;
    const isPdf = file.name.toLowerCase().endsWith(".pdf") || file.type === "application/pdf";
    if (!isPdf) {
      this.showToast("Please select a valid PDF file (*.pdf)", "error");
      return;
    }

    console.log("[SHIA-RAG] Initiating PDF upload:", file.name, "bytes:", file.size);

    // Show Ingestion Progress Overlay
    this.ingestionBanner.style.display = "flex";
    this.ingestionTitle.innerText = `Ingesting '${file.name}'...`;
    this.ingestionSub.innerText = "Extracting Tier 1 TextBlocks & reading order...";
    this.bannerProgressFill.style.width = "35%";

    const isFresh = this.heroFreshTreeToggle ? this.heroFreshTreeToggle.checked : (this.freshTreeUploadToggle ? this.freshTreeUploadToggle.checked : true);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("clear_existing", isFresh ? "true" : "false");
    formData.append("allow_cross_document", "false");

    try {
      setTimeout(() => {
        this.bannerProgressFill.style.width = "75%";
        this.ingestionSub.innerText = "Inducing Tier 2 Taxonomy Tree & Anchoring Evidence...";
      }, 500);

      const response = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Upload failed");
      }

      this.bannerProgressFill.style.width = "100%";
      const res = await response.json();
      const data = res.data;

      this.activeDoc = data.document;
      this.activeDocId = data.doc_id;
      this.showToast(`'${data.document}' ingested into its dedicated tree!`, "success");

      // Update documents list & forest stats
      await this.fetchDocuments();
      await this.fetchForestStats();
      await this.fetchSampleQueries();

      // Clear chat & welcome user to the newly ingested document
      this.startNewChat();
      this.welcomeScreen.style.display = "none";

      const introRow = document.createElement("div");
      introRow.className = "message-row assistant";
      introRow.innerHTML = `
        <div class="message-avatar">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
          </svg>
        </div>
        <div class="message-bubble">
          <div class="message-author">
            <span>SHIA-RAG 2.0 Ingestion Complete</span>
            <span class="author-tag">${data.document}</span>
          </div>
          <div class="answer-markdown">
            <p>I have ingested and processed <strong>${data.document}</strong> into the <strong>Dual-Tier Heterogeneous Knowledge Forest</strong>.</p>
            <ul>
              <li><strong>Total Pages:</strong> ${data.total_pages}</li>
              <li><strong>Text Blocks Extracted (Tier 1):</strong> ${data.blocks_ingested}</li>
              <li><strong>Concepts Induced (Tier 2):</strong> ${data.concepts_extracted}</li>
              <li><strong>Semantic Cross-Links Discovered:</strong> ${data.crosslinks_discovered}</li>
              <li><strong>Maximum Taxonomy Depth:</strong> ${data.forest_stats.max_depth} (within limit &le; 8)</li>
              <li><strong>Taxonomy Root:</strong> <code>${data.forest_stats.roots[0] || data.document}</code></li>
            </ul>
            <p>You can ask any factual or thematic question, or click <strong>Graph View</strong> in the top bar to inspect the induced hierarchy!</p>
          </div>
        </div>
      `;
      this.messagesStream.appendChild(introRow);
      this.scrollToBottom();
    } catch (err) {
      console.error(err);
      this.showToast(`Ingestion error: ${err.message}`, "error");
    } finally {
      setTimeout(() => {
        this.ingestionBanner.style.display = "none";
        this.bannerProgressFill.style.width = "0%";
      }, 700);
      this.pdfFileInput.value = "";
    }
  }

  // ==========================================================================
  // Preset Knowledge Base & Invariants
  // ==========================================================================

  async preloadSampleKB() {
    try {
      this.showToast("Loading Computer Science & AI sample ontology...", "info");
      const res = await fetch("/api/preload_sample", { method: "POST" });
      const data = await res.json();
      await this.fetchDocuments();
      if (this.documents.length > 0) {
        this.activeDocId = this.documents[0].doc_id;
        this.activeDoc = this.documents[0].filename;
      }
      await this.fetchForestStats();
      await this.fetchSampleQueries();
      this.showToast("Sample CS & AI Knowledge Forest loaded!", "success");
      this.startNewChat();
    } catch (e) {
      this.showToast("Failed to load sample ontology.", "error");
    }
  }

  async resetKnowledgeForest() {
    if (!confirm("Are you sure you want to reset the Knowledge Forest? All documents and trees will be cleared.")) {
      return;
    }
    try {
      const res = await fetch("/api/reset", { method: "POST" });
      const data = await res.json();
      this.activeDoc = null;
      this.activeDocId = null;
      this.currentForestDocId = null;
      await this.fetchDocuments();
      await this.fetchForestStats();
      this.startNewChat();
      this.showToast("Knowledge Forest has been reset to a clean empty slate.", "info");
    } catch (e) {
      this.showToast("Failed to reset forest.", "error");
    }
  }

  async auditInvariants() {
    try {
      const res = await fetch("/api/invariants");
      const data = await res.json();
      if (res.ok) {
        this.showToast(`Forest Invariant Check PASSED: ${data.invariants}`, "success");
      } else {
        this.showToast(`Invariant Warning: ${data.detail}`, "error");
      }
    } catch (e) {
      this.showToast("Error checking forest invariants.", "error");
    }
  }

  // ==========================================================================
  // Node Inspector Drawer
  // ==========================================================================

  async openInspector(nodeId) {
    try {
      const res = await fetch(`/api/node/${nodeId}`);
      if (!res.ok) {
        this.showToast(`Node '${nodeId}' details not found`, "error");
        return;
      }
      const node = await res.json();

      this.inspectorNodeType.innerText = node.node_type || "CONCEPT";
      this.inspectorNodeName.innerText = node.canonical_name;
      this.inspectorNodeId.innerText = node.node_id;
      this.inspectorDepth.innerText = node.depth;
      this.inspectorConfidence.innerText = node.confidence.toFixed(3);
      this.inspectorAbstraction.innerText = node.abstraction_level.toFixed(2);
      this.inspectorTokens.innerText = `${node.token_cost} tok`;
      this.inspectorDefinition.innerText = node.definition_text;

      // Parent & Children
      if (node.parent) {
        this.inspectorParent.innerHTML = `<a class="citation-pill" data-nid="${node.parent.id}">${node.parent.name} (${node.parent.id})</a>`;
      } else {
        this.inspectorParent.innerText = "None (Domain Taxonomy Root)";
      }

      if (node.children && node.children.length > 0) {
        this.inspectorChildren.innerHTML = node.children.map((c) => `
          <a class="citation-pill" data-nid="${c.id}">${c.name} [d=${c.depth}]</a>
        `).join(" ");
      } else {
        this.inspectorChildren.innerText = "None (Leaf proposition)";
      }

      // Crosslinks
      if (node.cross_links && node.cross_links.length > 0) {
        this.inspectorCrosslinks.innerHTML = node.cross_links.map((cl) => `
          <div class="crosslink-item">
            <span class="predicate">${cl.predicate}</span>
            <span class="target-name">
              <a class="citation-pill" data-nid="${cl.target_id || cl.source_id}">
                ${cl.target_name || cl.source_name}
              </a>
            </span>
          </div>
        `).join("");
      } else {
        this.inspectorCrosslinks.innerHTML = "<div style='color:var(--text-muted);font-size:12px;'>No cross-links associated.</div>";
      }

      // Evidence Text Blocks
      if (node.source_blocks && node.source_blocks.length > 0) {
        this.inspectorEvidence.innerHTML = node.source_blocks.map((b) => `
          <div class="evidence-card">
            <div class="evidence-header">
              <span>Page ${b.page_num} • Section ${b.section_path || "1.0"}</span>
              <span>Block ${b.block_id}</span>
            </div>
            <div class="evidence-text">"${this.escapeHtml(b.text_content)}"</div>
          </div>
        `).join("");
      } else if (node.evidence_texts && node.evidence_texts.length > 0) {
        this.inspectorEvidence.innerHTML = node.evidence_texts.map((t) => `
          <div class="evidence-card">
            <div class="evidence-text">"${this.escapeHtml(t)}"</div>
          </div>
        `).join("");
      } else {
        this.inspectorEvidence.innerHTML = "<div style='color:var(--text-muted);font-size:12px;'>No source blocks recorded.</div>";
      }

      this.nodeInspectorDrawer.classList.add("open");
    } catch (e) {
      console.error(e);
      this.showToast("Could not load node details", "error");
    }
  }

  closeInspector() {
    this.nodeInspectorDrawer.classList.remove("open");
  }

  // ==========================================================================
  // Knowledge Forest Visualizer Modal & Dedicated Tree Tabs
  // ==========================================================================

  async openForestModal(docId = null) {
    if (docId === "ALL" || docId === "all") {
      this.currentForestDocId = null;
    } else if (docId) {
      this.currentForestDocId = docId;
    } else if (!this.currentForestDocId && this.documents.length > 0) {
      this.currentForestDocId = this.documents[0].doc_id;
    }
    this.forestModalOverlay.style.display = "flex";
    await this.loadForestGraphData();
  }

  async loadForestGraphData() {
    this.forestGraphContainer.innerHTML = "<div style='color:var(--text-muted);padding:20px;'>Loading Knowledge Forest DAG...</div>";
    try {
      const url = this.currentForestDocId ? `/api/forest?doc_id=${encodeURIComponent(this.currentForestDocId)}` : "/api/forest";
      const res = await fetch(url);
      this.forestData = await res.json();
      this.renderDocTabs();
      this.renderForestGraph(this.forestData);
    } catch (e) {
      console.error(e);
      this.forestGraphContainer.innerHTML = "<div style='color:var(--accent-danger);padding:20px;'>Failed to load forest graph.</div>";
    }
  }

  closeForestModal() {
    this.forestModalOverlay.style.display = "none";
  }

  renderDocTabs() {
    if (!this.forestDocTabs) return;
    if (!this.documents || this.documents.length === 0) {
      this.forestDocTabs.innerHTML = "";
      return;
    }

    let tabsHtml = "";
    if (this.documents.length > 1) {
      const isAllActive = !this.currentForestDocId ? "active" : "";
      tabsHtml += `
        <button class="forest-doc-tab ${isAllActive}" onclick="app.switchForestDocTab(null)">
          <span>🌲 All Documents Forest</span>
          <span class="tab-badge">${this.forestData?.stats?.total_nodes || 0}</span>
        </button>
      `;
    }

    this.documents.forEach((d) => {
      const isActive = (this.currentForestDocId === d.doc_id) ? "active" : "";
      tabsHtml += `
        <button class="forest-doc-tab ${isActive}" onclick="app.switchForestDocTab('${d.doc_id}')">
          <span>📄 ${this.escapeHtml(d.filename)}</span>
          <span class="tab-badge">${d.concepts_count} nodes</span>
        </button>
      `;
    });

    this.forestDocTabs.innerHTML = tabsHtml;
  }

  async switchForestDocTab(docId) {
    this.currentForestDocId = docId;
    await this.loadForestGraphData();
  }

  renderForestGraph(data) {
    if (!data || !data.nodes || data.nodes.length === 0) {
      if (this.forestTreeSummaryBanner) this.forestTreeSummaryBanner.style.display = "none";
      this.forestGraphContainer.innerHTML = "<div style='color:var(--text-muted);padding:20px;'>Knowledge Forest is empty. Ingest a PDF or load the sample ontology!</div>";
      return;
    }

    // Render tree summary banner
    if (this.forestTreeSummaryBanner) {
      this.forestTreeSummaryBanner.style.display = "flex";
      if (this.currentForestDocId && data.doc_trees && data.doc_trees[this.currentForestDocId]) {
        const docInfo = data.doc_trees[this.currentForestDocId];
        this.forestTreeSummaryBanner.innerHTML = `
          <div class="tree-summary-left">
            <div>
              <div class="tree-summary-title">Dedicated Document Tree: <strong>${this.escapeHtml(docInfo.filename)}</strong></div>
              <div class="tree-summary-sub">Document Root: <code>${this.escapeHtml(docInfo.root_name)}</code> (${docInfo.root_id})</div>
            </div>
          </div>
          <div class="tree-summary-pills">
            <span class="summary-pill">Nodes: ${data.nodes.length}</span>
            <span class="summary-pill">Max Depth: ${docInfo.max_depth}</span>
            <span class="summary-pill">Edges: ${data.edges.length}</span>
          </div>
        `;
      } else {
        this.forestTreeSummaryBanner.innerHTML = `
          <div class="tree-summary-left">
            <div>
              <div class="tree-summary-title">Multi-Document Knowledge Forest</div>
              <div class="tree-summary-sub">${this.documents.length} Standalone Document Trees</div>
            </div>
          </div>
          <div class="tree-summary-pills">
            <span class="summary-pill">Total Nodes: ${data.nodes.length}</span>
            <span class="summary-pill">Total Edges: ${data.edges.length}</span>
          </div>
        `;
      }
    }

    // Group nodes by depth
    const levels = {};
    data.nodes.forEach((n) => {
      const d = n.depth || 0;
      if (!levels[d]) levels[d] = [];
      levels[d].push(n);
    });

    const maxDepth = Math.max(...Object.keys(levels).map(Number));
    let html = '<div class="forest-tree-view">';

    for (let d = 0; d <= maxDepth; d++) {
      const levelNodes = levels[d] || [];
      const label = (d === 0) ? "Level 0 • Document Taxonomy Root(s)" : `Level ${d} • Taxonomy Depth ${d}`;
      html += `
        <div class="tree-level" data-depth="${d}">
          <div class="tree-level-label">
            <span>${label}</span>
            <span>${levelNodes.length} nodes</span>
          </div>
          <div class="tree-nodes-row">
            ${levelNodes.map((n) => {
              const isRoot = (d === 0);
              const cardClass = isRoot ? "tree-node-card root-card" : "tree-node-card";
              const parentNode = data.nodes.find((p) => p.id === n.parent_id);
              const parentText = parentNode ? parentNode.name : "None";
              return `
                <div class="${cardClass}" onclick="app.openInspector('${n.id}')" data-name="${n.name.toLowerCase()}">
                  <div class="tree-node-header">
                    <span class="tree-node-type">${n.type || "CONCEPT"}</span>
                    <span class="tree-node-conf">C=${n.confidence}</span>
                  </div>
                  <div class="tree-node-title">${n.name}</div>
                  <div class="tree-node-parent">Parent: <span>${parentText}</span></div>
                  ${n.doc_name ? `<div style="font-size:10px;color:var(--text-muted);margin-top:2px;">Doc: ${this.escapeHtml(n.doc_name)}</div>` : ''}
                </div>
              `;
            }).join("")}
          </div>
        </div>
      `;
    }

    html += '</div>';
    this.forestGraphContainer.innerHTML = html;
  }

  filterForestGraph(searchTerm) {
    const cards = this.forestGraphContainer.querySelectorAll(".tree-node-card");
    cards.forEach((card) => {
      const name = card.dataset.name || "";
      if (!searchTerm || name.includes(searchTerm)) {
        card.style.display = "flex";
      } else {
        card.style.display = "none";
      }
    });
  }

  // ==========================================================================
  // Data Fetching & Sync
  // ==========================================================================

  async fetchForestStats() {
    try {
      const res = await fetch("/api/stats");
      const stats = await res.json();
      this.statTotalNodes.innerText = stats.total_nodes;
      this.statTotalEdges.innerText = stats.total_edges;
      this.statMaxDepth.innerText = stats.max_depth;
      this.statCrosslinks.innerText = stats.semantic_edges;
    } catch (e) {
      console.warn("Could not fetch forest statistics", e);
    }
  }

  async fetchDocuments() {
    try {
      const res = await fetch("/api/documents");
      this.documents = await res.json();
      if (this.documents.length > 0 && !this.activeDocId) {
        const latestDoc = this.documents[this.documents.length - 1];
        this.activeDocId = latestDoc.doc_id;
        this.activeDoc = latestDoc.filename;
      } else if (this.documents.length === 0) {
        this.activeDocId = null;
        this.activeDoc = null;
      }
      this.renderDocumentList();
      this.updateActiveDocSelect();
      this.updateScopeIndicators();
    } catch (e) {
      console.warn("Could not fetch documents list", e);
    }
  }

  updateScopeIndicators() {
    const composerPill = document.getElementById("composerScopeIndicator");
    const composerText = document.getElementById("composerScopeText");
    if (composerPill && composerText) {
      if (this.activeDoc) {
        composerText.innerHTML = `Active Document: <strong>${this.escapeHtml(this.activeDoc)}</strong>`;
        composerPill.style.display = "flex";
      } else {
        composerText.innerHTML = `Active Scope: <strong>Entire Knowledge Forest</strong>`;
        composerPill.style.display = "flex";
      }
    }
    const contextPill = document.getElementById("activeContextPill");
    if (contextPill && this.activeDoc) {
      contextPill.title = `Retrieval currently scoped to '${this.activeDoc}'`;
    }
  }

  renderDocumentList() {
    if (!this.documents || this.documents.length === 0) {
      this.documentList.innerHTML = `
        <div class="empty-docs-hint">
          No PDF uploaded yet.<br>Upload a file to induce its separate tree.
        </div>
      `;
      return;
    }

    this.documentList.innerHTML = this.documents.map((d) => {
      const isActive = (this.activeDocId === d.doc_id) || (!this.activeDocId && d.filename === this.activeDoc);
      return `
        <div class="doc-item ${isActive ? 'active' : ''}">
          <div class="doc-header-row" onclick="app.selectDoc('${d.doc_id}', '${this.escapeHtml(d.filename)}')">
            <div class="doc-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
              </svg>
            </div>
            <div class="doc-info">
              <div class="doc-name" title="${this.escapeHtml(d.filename)}">${this.escapeHtml(d.filename)}</div>
              <div class="doc-meta">${d.total_pages} pgs • ${d.concepts_count} concepts • depth ${d.max_depth}</div>
            </div>
          </div>
          <div class="doc-actions-row">
            <button class="doc-action-btn view-tree-btn" onclick="app.openForestModal('${d.doc_id}')" title="Inspect this document's separate tree">
              🌲 View Tree
            </button>
            <button class="doc-action-btn delete-doc-btn" onclick="app.deleteDocument('${d.doc_id}', '${this.escapeHtml(d.filename)}')" title="Delete this document and its tree">
              🗑️ Delete
            </button>
          </div>
        </div>
      `;
    }).join("");
  }

  updateActiveDocSelect() {
    if (!this.activeDocSelect) return;
    let html = `<option value="">All Documents (Forest)</option>`;
    this.documents.forEach((d) => {
      const sel = (d.doc_id === this.activeDocId) ? "selected" : "";
      html += `<option value="${d.doc_id}" ${sel}>${this.escapeHtml(d.filename)} (${d.concepts_count} concepts)</option>`;
    });
    this.activeDocSelect.innerHTML = html;
    if (this.activeDocId) {
      this.activeDocSelect.value = this.activeDocId;
    }
  }

  selectDoc(docId, filename) {
    this.activeDocId = docId;
    this.activeDoc = filename;
    this.renderDocumentList();
    this.updateActiveDocSelect();
    this.updateScopeIndicators();
    this.fetchSampleQueries();
    this.showToast(`Active document set to '${filename}'`, "info");
  }

  async deleteDocument(docId, filename) {
    if (!confirm(`Are you sure you want to delete '${filename}' and its separate knowledge tree?`)) return;
    try {
      const res = await fetch(`/api/document/${docId}`, { method: "DELETE" });
      if (!res.ok) throw new Error("Delete failed");
      this.showToast(`Deleted '${filename}' and its tree.`, "info");
      if (this.activeDocId === docId) {
        this.activeDocId = null;
        this.activeDoc = null;
      }
      if (this.currentForestDocId === docId) {
        this.currentForestDocId = null;
      }
      await this.fetchDocuments();
      await this.fetchForestStats();
      await this.fetchSampleQueries();
    } catch (e) {
      console.error(e);
      this.showToast("Failed to delete document", "error");
    }
  }

  async fetchSampleQueries() {
    try {
      const res = await fetch("/api/sample_queries");
      const queries = await res.json();
      if (queries && queries.length > 0) {
        this.starterPromptsGrid.innerHTML = queries.map((q) => {
          const modeClass = (q.mode || "factual").toLowerCase();
          return `
            <div class="prompt-card" data-query="${this.escapeHtml(q.query)}">
              <div class="prompt-mode-tag ${modeClass}">${q.title}</div>
              <div class="prompt-text">"${this.escapeHtml(q.query)}"</div>
            </div>
          `;
        }).join("");
      }
    } catch (e) {
      console.warn("Could not load sample queries", e);
    }
  }

  // ==========================================================================
  // Utilities
  // ==========================================================================

  renderMarkdownWithCitations(text) {
    if (!text) return "";

    let s = text;

    // 1. Strip any residual node IDs [KN-xxxx], [Node-xx], [Doc-xx], and numeric bracket citations
    s = s.replace(/\s*\[\s*(?:KN|Node|Chunk|Doc)[^\]]*\]/gi, "");
    s = s.replace(/\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\]/g, "");

    // 2. Convert LaTeX fractions \frac{a}{b} -> (a) / (b)
    let prev = "";
    while (s.includes("\\frac") && s !== prev) {
      prev = s;
      s = s.replace(/\\frac\{([^{}]+)\}\{([^{}]+)\}/g, "($1) / ($2)");
    }

    // 3. Convert common LaTeX math commands and symbols to clean Unicode
    const latexMap = {
      "\\neq": "≠",
      "\\ne": "≠",
      "\\leq": "≤",
      "\\le": "≤",
      "\\geq": "≥",
      "\\ge": "≥",
      "\\pm": "±",
      "\\mp": "∓",
      "\\times": "×",
      "\\cdot": "·",
      "\\div": "÷",
      "\\approx": "≈",
      "\\alpha": "α",
      "\\beta": "β",
      "\\gamma": "γ",
      "\\delta": "δ",
      "\\theta": "θ",
      "\\lambda": "λ",
      "\\pi": "π",
      "\\sigma": "σ",
      "\\Delta": "Δ",
      "\\infty": "∞",
      "\\in": "∈",
      "\\notin": "∉",
      "\\sqrt": "√",
      "^2": "²",
      "^{2}": "²",
      "^3": "³",
      "^{3}": "³",
      "^0": "⁰",
      "^1": "¹",
      "^n": "ⁿ",
      "^x": "ˣ",
      "_1": "₁",
      "_{1}": "₁",
      "_2": "₂",
      "_{2}": "₂",
      "_0": "₀",
      "_{0}": "₀",
    };
    for (const [k, v] of Object.entries(latexMap)) {
      s = s.split(k).join(v);
    }
    s = s.replace(/√\{([^}]+)\}/g, "√($1)");
    s = s.replace(/\\[,;! ]/g, " ");

    // 4. Strip dollar signs $...$ and standalone $
    s = s.replace(/\$([^\$]+)\$/g, "$1");
    s = s.replace(/\$/g, "");

    // 5. Clean punctuation spacing
    s = s.replace(/\s+,/g, ",");
    s = s.replace(/\s+\./g, ".");
    s = s.replace(/\s+;/g, ";");
    s = s.replace(/\s+:/g, ":");

    // 6. Format run-together headers into distinct markdown bullets if needed
    const headers = [
      "Standard Form:", "Roots:", "Relationship to Polynomials:", "Nature of Roots:",
      "Examples:", "Key Points:", "Definition:", "Formula:", "Method:", "Procedure:"
    ];
    for (const h of headers) {
      const reHeader = new RegExp(`(?:^|(?<=[.:;])\\s+|\\n\\s*)${h.replace(":", "\\:")}\\s*`, "g");
      s = s.replace(reHeader, `\n\n- **${h}** `);
    }

    // Format discriminant sub-bullets if run-together
    s = s.replace(/(?:^|(?<=[.:;])\s+|\n\s*)\*?\s*(Two distinct [^\n]*roots\b)/gi, "\n  - $1");
    s = s.replace(/(?:(?<=[.:;])\s+|\n\s*|\s*\*\s*)\*?\s*(Two equal [^\n]*roots\b)/gi, "\n  - $1");
    s = s.replace(/(?:(?<=[.:;])\s+|\n\s*|\s*[\*\-]\s*)\*?\s*(No real roots\b)/gi, "\n  - $1");

    let escaped = this.escapeHtml(s);

    // Convert horizontal rule
    escaped = escaped.replace(/^\s*---\s*$/gim, "<hr/>");

    // Convert headings
    escaped = escaped.replace(/^#### (.*$)/gim, "<h4>$1</h4>");
    escaped = escaped.replace(/^### (.*$)/gim, "<h3>$1</h3>");
    escaped = escaped.replace(/^## (.*$)/gim, "<h2>$1</h2>");

    // Convert blockquotes
    escaped = escaped.replace(/^\> (.*$)/gim, "<blockquote>$1</blockquote>");

    // Convert nested sub-bullets (indented lines with - or *)
    escaped = escaped.replace(/^(?:\t|\s{2,})[\*\-]\s+(.*$)/gim, '<li class="sub-bullet">$1</li>');

    // Convert unordered lists (both - and * prefixes)
    escaped = escaped.replace(/^[\*\-]\s+(.*$)/gim, "<li>$1</li>");

    // Convert bold, italics, and code (after list items so list markers do not trigger italics)
    escaped = escaped.replace(/\*\*(.*?)\*\*/gim, "<strong>$1</strong>");
    escaped = escaped.replace(/(?<=\s|^)\*([^*\n]+)\*(?=\s|$|[.,;:!?])/gim, "<em>$1</em>");
    escaped = escaped.replace(/`([^`]+)`/gim, "<code>$1</code>");

    // Wrap consecutive list items in <ul>...</ul>
    escaped = escaped.replace(/((?:<li(?: class="sub-bullet")?>.*?<\/li>\s*)+)/gs, "<ul>$1</ul>");

    // Convert double newlines to paragraphs
    escaped = escaped.replace(/\n\n+/g, "</p><p>");
    escaped = `<p>${escaped}</p>`;
    escaped = escaped.replace(/<p>\s*<\/p>/g, "");
    escaped = escaped.replace(/<p>\s*(<ul>.*?<\/ul>)\s*<\/p>/gs, "$1");
    escaped = escaped.replace(/<p>\s*(<h[2-4]>.*?<\/h[2-4]>)\s*<\/p>/gs, "$1");
    escaped = escaped.replace(/<p>\s*(<hr\/>)\s*<\/p>/gs, "$1");

    return escaped;
  }

  async checkLlmStatus() {
    try {
      const res = await fetch("/api/llm/status");
      if (res.ok) {
        const data = await res.json();
        if (this.llmActiveStatusText) {
          this.llmActiveStatusText.innerText = data.active_engine;
        }

        if (data.gemini_configured) {
          const modelName = data.active_model ? data.active_model.replace("models/", "") : "gemini-3.6-flash";
          const displayTitle = modelName.includes("3.6") ? "Gemini 3.6 Flash" : (modelName.includes("flash") ? "Gemini Flash" : modelName);
          if (this.llmEngineBadge) {
            this.llmEngineBadge.innerText = `⚡ ${displayTitle}`;
            this.llmEngineBadge.style.color = "#38bdf8";
          }
          if (this.sidebarEngineTypeBadge) {
            this.sidebarEngineTypeBadge.innerText = "Gemini";
            this.sidebarEngineTypeBadge.style.background = "rgba(56, 189, 248, 0.2)";
            this.sidebarEngineTypeBadge.style.color = "#38bdf8";
          }
          if (this.sidebarEngineName) this.sidebarEngineName.innerText = displayTitle;
          if (this.sidebarEngineStatusDetail) this.sidebarEngineStatusDetail.innerText = "Cloud Accelerated";
          if (this.heroLlmEngineDesc) this.heroLlmEngineDesc.innerHTML = `Active Engine: <strong style="color:#38bdf8;">Google ${displayTitle}</strong> (Cloud Accelerated with citation grounding).`;
          if (this.composerEngineBadge) {
            this.composerEngineBadge.innerText = displayTitle;
            this.composerEngineBadge.style.color = "#38bdf8";
          }
        } else if (data.openai_configured) {
          if (this.llmEngineBadge) {
            this.llmEngineBadge.innerText = "⚡ GPT-4o-mini";
            this.llmEngineBadge.style.color = "#10b981";
          }
          if (this.sidebarEngineTypeBadge) {
            this.sidebarEngineTypeBadge.innerText = "OpenAI";
            this.sidebarEngineTypeBadge.style.background = "rgba(16, 185, 129, 0.2)";
            this.sidebarEngineTypeBadge.style.color = "#10b981";
          }
          if (this.sidebarEngineName) this.sidebarEngineName.innerText = "OpenAI GPT-4o-mini";
          if (this.sidebarEngineStatusDetail) this.sidebarEngineStatusDetail.innerText = "Cloud Accelerated";
          if (this.heroLlmEngineDesc) this.heroLlmEngineDesc.innerHTML = 'Active Engine: <strong style="color:#10b981;">OpenAI GPT-4o-mini</strong> (Cloud Accelerated with citation grounding).';
          if (this.composerEngineBadge) {
            this.composerEngineBadge.innerText = "OpenAI GPT-4o-mini";
            this.composerEngineBadge.style.color = "#10b981";
          }
        } else {
          if (this.llmEngineBadge) {
            this.llmEngineBadge.innerText = "⚡ Cloud LLM";
            this.llmEngineBadge.style.color = "#fbbf24";
          }
          if (this.sidebarEngineTypeBadge) {
            this.sidebarEngineTypeBadge.innerText = "Local";
            this.sidebarEngineTypeBadge.style.background = "rgba(245, 158, 11, 0.2)";
            this.sidebarEngineTypeBadge.style.color = "#fbbf24";
          }
          if (this.sidebarEngineName) this.sidebarEngineName.innerText = "Local Synthesizer";
          if (this.sidebarEngineStatusDetail) this.sidebarEngineStatusDetail.innerText = "Zero cost offline";
          if (this.heroLlmEngineDesc) this.heroLlmEngineDesc.innerHTML = 'Active Engine: <strong>Local Semantic Synthesizer</strong> (Zero cost). Connect Gemini or OpenAI for frontier natural generation.';
          if (this.composerEngineBadge) {
            this.composerEngineBadge.innerText = "Local Synthesizer";
            this.composerEngineBadge.style.color = "#fbbf24";
          }
        }
      }
    } catch (e) {
      console.warn("Could not fetch LLM status:", e);
    }
  }

  openLlmModal() {
    this.checkLlmStatus();
    const errEl = document.getElementById("llmModalError");
    if (errEl) errEl.style.display = "none";
    if (this.llmModalOverlay) {
      this.llmModalOverlay.style.display = "flex";
      if (this.llmApiKeyInput) this.llmApiKeyInput.focus();
    }
  }

  closeLlmModal() {
    const errEl = document.getElementById("llmModalError");
    if (errEl) errEl.style.display = "none";
    if (this.llmModalOverlay) {
      this.llmModalOverlay.style.display = "none";
    }
  }

  async saveLlmConfig() {
    const provider = this.llmProviderSelect.value;
    const apiKey = this.llmApiKeyInput.value.trim();
    const errEl = document.getElementById("llmModalError");
    if (errEl) errEl.style.display = "none";

    if (!apiKey) {
      if (errEl) {
        errEl.style.display = "block";
        errEl.innerHTML = "⚠️ Please enter a valid API key.";
      } else {
        this.showToast("Please enter a valid API key.", "error");
      }
      return;
    }

    if (this.saveLlmBtn) {
      this.saveLlmBtn.disabled = true;
      this.saveLlmBtn.innerText = "Testing & Connecting...";
    }

    try {
      const res = await fetch("/api/llm/configure", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider, api_key: apiKey }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Failed to configure LLM");
      }

      const data = await res.json();
      this.showToast(data.message || "Connected to Cloud LLM!", "success");
      this.closeLlmModal();
      this.checkLlmStatus();
    } catch (e) {
      if (errEl) {
        errEl.style.display = "block";
        errEl.innerHTML = `⚠️ <strong>Connection Failed:</strong> ${e.message}`;
      }
      this.showToast(e.message, "error");
    } finally {
      if (this.saveLlmBtn) {
        this.saveLlmBtn.disabled = false;
        this.saveLlmBtn.innerText = "Save & Connect";
      }
    }
  }

  showToast(message, type = "info") {
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.innerText = message;
    this.toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateY(10px)";
      toast.style.transition = "all 0.3s ease";
      setTimeout(() => toast.remove(), 300);
    }, 3200);
  }

  scrollToBottom() {
    this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
  }

  escapeHtml(str) {
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
}

// Instantiate App on window.app
function initShiaRagApp() {
  if (!window.app) {
    window.app = new ShiaRagApp();
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initShiaRagApp);
} else {
  initShiaRagApp();
}
