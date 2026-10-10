(() => {
  "use strict";

  const root = document.documentElement;
  const saved = localStorage.getItem("segbench-theme");
  if (saved) root.dataset.theme = saved;
  document.querySelector("#theme-toggle")?.addEventListener("click", () => {
    const next = root.dataset.theme === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    localStorage.setItem("segbench-theme", next);
  });

  const payload = JSON.parse(document.querySelector("#public-data").textContent);
  const resultRows = payload.results;
  const statusRows = payload.tables;
  const statisticRows = payload.statistics;
  let activeTable = "A";
  let activeMetric = "dice";
  let activeGroup = "all";
  let sortAscending = false;
  const hullTables = new Set(["B1", "B_rolled", "B_plain", "D"]);

  const lowerIsBetter = new Set(["mdr", "fdr", "ohf", "ms_per_img"]);
  const htmlEscape = value => String(value).replace(
    /[&<>"']/g,
    character => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"})[character],
  );
  const fmt = (value, metric) => {
    if (value === null || value === "" || Number.isNaN(Number(value))) return "pending";
    const number = Number(value);
    if (["params_m", "ms_per_img"].includes(metric)) return number.toFixed(2);
    if (metric === "par") return number.toFixed(3);
    return (number * 100).toFixed(2);
  };
  const fmtWithStd = (row, metric) => {
    const mean = fmt(row[metric], metric);
    const spread = row[`${metric}_seed_std`];
    return spread === "" || spread === undefined ? mean : `${mean} ± ${fmt(spread, metric)}`;
  };
  const statusLabel = row => (
    row.status === "pending" && row.comment.includes("training — pending")
      ? "training — pending" : row.status
  );

  function filteredRows() {
    return resultRows.filter(row => row.table === activeTable && (
      activeGroup === "all" || row.group === activeGroup
    ));
  }

  function metricCell(row) {
    const wrapper = document.createElement("div");
    wrapper.className = "metric-cell";
    const value = document.createElement("span");
    value.textContent = fmtWithStd(row, activeMetric);
    wrapper.appendChild(value);
    if (["dice", "hr"].includes(activeMetric) && row.ci_lower !== "" && row.ci_upper !== "") {
      const interval = document.createElement("span");
      interval.className = "ci-label";
      interval.textContent = `[${fmt(row.ci_lower, activeMetric)}, ${fmt(row.ci_upper, activeMetric)}]`;
      interval.title = "Subject-cluster bootstrap 95% confidence interval";
      wrapper.appendChild(interval);
      const bar = document.createElement("span");
      bar.className = "ci-bar";
      const left = Math.max(0, Math.min(100, Number(row.ci_lower) * 100));
      const right = Math.max(left, Math.min(100, Number(row.ci_upper) * 100));
      bar.innerHTML = `<i style="left:${left}%;width:${right - left}%"></i>`;
      wrapper.appendChild(bar);
    }
    if (value.textContent === "pending") wrapper.classList.add("pending");
    return wrapper;
  }

  function renderGroupFilter() {
    const wrapper = document.querySelector("#group-filter-wrap");
    const select = document.querySelector("#group-filter");
    const groups = [...new Set(resultRows
      .filter(row => row.table === activeTable && row.group && row.group !== "overall")
      .map(row => row.group))].sort();
    wrapper.hidden = !hullTables.has(activeTable) || groups.length === 0;
    select.innerHTML = '<option value="all">All groups</option>';
    for (const group of groups) {
      const option = document.createElement("option");
      option.value = group; option.textContent = group; select.appendChild(option);
    }
    if (!groups.includes(activeGroup)) activeGroup = "all";
    select.value = activeGroup;
  }

  function renderTable() {
    const tableBody = document.querySelector("#result-body");
    const rows = filteredRows();
    tableBody.innerHTML = "";
    if (!rows.length) {
      const status = statusRows.find(row => row.table === activeTable);
      tableBody.innerHTML = `<tr><td colspan="8" class="pending">${status?.note || "Pending"}</td></tr>`;
      return;
    }
    rows.sort((a, b) => {
      const av = Number(a[activeMetric]); const bv = Number(b[activeMetric]);
      if (Number.isNaN(av) && Number.isNaN(bv)) return a.approach.localeCompare(b.approach);
      if (Number.isNaN(av)) return 1; if (Number.isNaN(bv)) return -1;
      const preferred = lowerIsBetter.has(activeMetric) ? av - bv : bv - av;
      return sortAscending ? -preferred : preferred;
    });
    for (const row of rows) {
      const tr = document.createElement("tr");
      const label = row.group && row.group !== "overall"
        ? `${row.approach} — ${row.group}` : row.approach;
      const cells = [
        label, row.family, statusLabel(row), null, row.seeds_completed || "—",
        fmtWithStd(row, "params_m"), fmtWithStd(row, "ms_per_img"),
        row.device || "—",
      ];
      cells.forEach((cell, index) => {
        const td = document.createElement("td");
        if (index === 3) td.appendChild(metricCell(row));
        else td.textContent = cell;
        if (index >= 3) td.className = "numeric";
        if (typeof cell === "string" && cell.includes("pending")) td.classList.add("pending");
        tr.appendChild(td);
      });
      tableBody.appendChild(tr);
    }
  }

  function renderSignificance() {
    const body = document.querySelector("#significance-body");
    const rows = statisticRows.filter(row => row.table === activeTable);
    body.innerHTML = "";
    if (!rows.length) {
      body.innerHTML = '<tr><td colspan="7" class="pending">No aggregate comparison published.</td></tr>';
      return;
    }
    for (const row of rows) {
      const interval = row.ci_lower === "" ? "pending" : `[${row.ci_lower}, ${row.ci_upper}]`;
      const values = [
        row.metric, row.comparison, row.estimate || "pending", interval,
        row.p_holm || "pending", row.significant || "pending", row.status,
      ];
      const tr = document.createElement("tr");
      values.forEach((value, index) => {
        const td = document.createElement("td");
        if (index === 5 && value !== "pending") {
          const marker = document.createElement("span");
          marker.className = `sig-marker ${value === "true" ? "significant" : "not-significant"}`;
          marker.textContent = value === "true" ? "● yes" : "○ no";
          marker.title = value === "true" ? "Significant after Holm correction" : "Not significant after Holm correction";
          td.appendChild(marker);
        } else {
          td.textContent = value;
        }
        if (value === "pending") td.className = "pending";
        tr.appendChild(td);
      });
      body.appendChild(tr);
    }
  }

  document.querySelectorAll(".tab").forEach(button => button.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(item => item.setAttribute("aria-selected", "false"));
    button.setAttribute("aria-selected", "true"); activeTable = button.dataset.table;
    activeGroup = "all";
    if (hullTables.has(activeTable) && !["mc", "hr", "par", "ohf", "ms_per_img"].includes(activeMetric)) {
      activeMetric = "hr"; document.querySelector("#metric").value = activeMetric;
    } else if (!hullTables.has(activeTable) && ["mc", "hr", "par", "ohf"].includes(activeMetric)) {
      activeMetric = "dice"; document.querySelector("#metric").value = activeMetric;
    }
    renderGroupFilter(); renderTable(); renderPareto(); renderSignificance();
  }));
  document.querySelector("#metric")?.addEventListener("change", event => {
    activeMetric = event.target.value; renderTable();
  });
  document.querySelector("#group-filter")?.addEventListener("change", event => {
    activeGroup = event.target.value; renderTable(); renderPareto();
  });
  document.querySelector("#sort-toggle")?.addEventListener("click", event => {
    sortAscending = !sortAscending;
    const label = sortAscending ? "Low to high" : "High to low";
    event.currentTarget.textContent = label;
    event.currentTarget.setAttribute("aria-label", label);
    renderTable();
  });

  function renderPareto() {
    const box = document.querySelector("#pareto");
    const rows = filteredRows().filter(row => (
      row.family !== "Reference" && row.dice !== "" && row.ms_per_img !== ""
    ));
    if (!rows.length) {
      box.innerHTML = '<p class="pending">Pareto points pending final aggregate results.</p>';
      return;
    }
    const width = 760; const height = 290; const pad = 48;
    const xs = rows.map(row => Number(row.ms_per_img));
    const ys = rows.map(row => Number(row.dice));
    const xmax = Math.max(...xs) || 1; const ymin = Math.min(...ys); const ymax = Math.max(...ys);
    const circles = rows.map(row => {
      const x = pad + Number(row.ms_per_img) / xmax * (width - pad * 2);
      const y = height - pad - (Number(row.dice) - ymin) / Math.max(ymax - ymin, .001) * (height - pad * 2);
      const label = `${htmlEscape(row.approach)}: Dice ${(Number(row.dice) * 100).toFixed(2)}, ${Number(row.ms_per_img).toFixed(2)}`;
      return `<g tabindex="0" aria-label="${label} milliseconds"><circle cx="${x}" cy="${y}" r="7" fill="var(--accent)"/><title>${label} ms</title></g>`;
    }).join("");
    box.innerHTML = `<svg role="img" aria-label="Dice versus runtime scatter plot" viewBox="0 0 ${width} ${height}"><line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="currentColor"/><line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" stroke="currentColor"/>${circles}<text x="${width / 2}" y="${height - 10}" text-anchor="middle" fill="currentColor">ms / image</text><text x="15" y="${height / 2}" transform="rotate(-90 15 ${height / 2})" text-anchor="middle" fill="currentColor">Dice</text></svg>`;
  }

  const examples = payload.synthetic;
  const selector = document.querySelector("#synthetic-select");
  const methodSelector = document.querySelector("#synthetic-method");
  const source = document.querySelector("#synthetic-source");
  const mask = document.querySelector("#synthetic-mask");
  const caption = document.querySelector("#synthetic-caption");
  const availability = document.querySelector("#synthetic-availability");
  function populateSyntheticMethods(row) {
    methodSelector.innerHTML = '<option value="ground_truth">Ground truth reference</option>';
    const predictions = new Map((row.predictions || []).map(item => [item.method, item]));
    availability.innerHTML = "";
    for (const method of row.methods || []) {
      const option = document.createElement("option");
      option.value = method.method;
      option.textContent = method.status === "complete"
        ? method.label : `${method.label} — ${method.status.replace("_", " ")}`;
      option.disabled = method.status !== "complete";
      methodSelector.appendChild(option);
      const item = document.createElement("li");
      item.className = `method-status ${method.status}`;
      const state = document.createElement("strong");
      state.textContent = `${method.label}: ${method.status.replace("_", " ")}`;
      item.appendChild(state);
      if (method.reason) item.append(` — ${method.reason}`);
      if (method.status === "complete" && !predictions.has(method.method)) {
        throw new Error(`complete synthetic method lacks prediction: ${method.method}`);
      }
      availability.appendChild(item);
    }
  }
  function showSynthetic() {
    if (!selector || !methodSelector) return;
    const row = examples.find(item => item.key === selector.value);
    if (!row) return;
    source.src = row.image;
    const prediction = (row.predictions || []).find(item => item.method === methodSelector.value);
    mask.src = prediction?.mask || row.mask;
    caption.textContent = prediction?.label || "Synthetic ground truth";
  }
  if (selector && methodSelector) {
    for (const item of examples) {
      const option = document.createElement("option");
      option.value = item.key; option.textContent = item.label; selector.appendChild(option);
    }
    selector.addEventListener("change", () => {
      const row = examples.find(item => item.key === selector.value);
      if (row) populateSyntheticMethods(row);
      showSynthetic();
    });
    methodSelector.addEventListener("change", showSynthetic);
    if (examples.length) populateSyntheticMethods(examples[0]);
    showSynthetic();
  }

  function renderFindings() {
    const list = document.querySelector("#findings-list");
    const complete = resultRows.filter(row => row.status === "complete");
    list.innerHTML = "";
    if (!complete.length) {
      list.innerHTML = '<li class="pending">Pending aggregate finalisation; no result claim is published.</li>';
      return;
    }
    const facts = [];
    const tableA = complete.filter(row => (
      row.table === "A" && row.family !== "Reference" && row.dice !== ""
    ));
    if (tableA.length) {
      const dice = tableA.map(row => Number(row.dice));
      facts.push(`Table A Dice spans ${(Math.min(...dice) * 100).toFixed(2)}–${(Math.max(...dice) * 100).toFixed(2)} across ${tableA.length} complete aggregate rows.`);
      const timed = tableA.filter(row => row.ms_per_img !== "").sort((a, b) => Number(a.ms_per_img) - Number(b.ms_per_img));
      if (timed.length) facts.push(`${timed[0].approach} has the lowest published Table A runtime (${Number(timed[0].ms_per_img).toFixed(2)} ms/image) and Dice ${(Number(timed[0].dice) * 100).toFixed(2)}.`);
    }
    const valueRows = complete.filter(row => row.table === "A_value" && row.dice !== "");
    if (valueRows.length) {
      const groups = new Set(valueRows.map(row => row.group));
      facts.push(`The examiner-value tab reports ${valueRows.length} method/group Dice cells across ${groups.size} published groups.`);
    }
    const hullRows = complete.filter(row => row.table === "B1" && row.hr !== "");
    if (hullRows.length) {
      const recall = hullRows.map(row => Number(row.hr));
      facts.push(`B1 hull recall spans ${(Math.min(...recall) * 100).toFixed(2)}–${(Math.max(...recall) * 100).toFixed(2)}; PAR remains visible beside every row.`);
    }
    const comparisons = statisticRows.filter(row => row.status === "complete" && row.estimate !== "");
    if (comparisons.length) {
      const row = comparisons[0];
      const interval = row.ci_lower === "" ? "CI pending" : `95% CI [${row.ci_lower}, ${row.ci_upper}]`;
      facts.push(`${row.comparison}: estimate ${row.estimate}, ${interval}; Holm p ${row.p_holm || "pending"}.`);
    }
    if (!facts.length) facts.push(`${complete.length} aggregate rows are complete; endpoint metrics remain pending.`);
    for (const fact of facts) {
      const li = document.createElement("li"); li.textContent = fact; list.appendChild(li);
    }
  }

  document.querySelector("#copy-cite")?.addEventListener("click", async event => {
    const text = document.querySelector("#bibtex").textContent;
    try {
      await navigator.clipboard.writeText(text);
      event.currentTarget.textContent = "Copied";
    } catch (_error) {
      event.currentTarget.textContent = "Select citation text to copy";
    }
  });

  renderGroupFilter(); renderTable(); renderPareto(); renderSignificance(); renderFindings();
})();
