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
  let activeTable = "A";
  let activeMetric = "dice";

  const fmt = (value, metric) => {
    if (value === null || value === "" || Number.isNaN(Number(value))) return "pending";
    const v = Number(value);
    if (["params_m", "ms_per_img"].includes(metric)) return v.toFixed(2);
    if (metric === "par") return v.toFixed(3);
    return (v * 100).toFixed(2);
  };

  function renderTable() {
    const tableBody = document.querySelector("#result-body");
    const rows = resultRows.filter(row => row.table === activeTable);
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
      return ["mdr", "fdr", "ohf", "ms_per_img"].includes(activeMetric) ? av - bv : bv - av;
    });
    for (const row of rows) {
      const tr = document.createElement("tr");
      const label = row.group && row.group !== "overall" ? `${row.approach} — ${row.group}` : row.approach;
      const cells = [label, row.family, row.status, fmt(row[activeMetric], activeMetric), row.seeds_completed || "—", fmt(row.params_m, "params_m"), fmt(row.ms_per_img, "ms_per_img"), row.device || "—"];
      cells.forEach((cell, index) => {
        const td = document.createElement("td"); td.textContent = cell;
        if (index >= 3) td.className = "numeric";
        if (cell === "pending") td.classList.add("pending");
        tr.appendChild(td);
      });
      tableBody.appendChild(tr);
    }
  }

  document.querySelectorAll(".tab").forEach(button => button.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(item => item.setAttribute("aria-selected", "false"));
    button.setAttribute("aria-selected", "true"); activeTable = button.dataset.table;
    if (activeTable === "B1" && !["mc", "hr", "par", "ohf", "ms_per_img"].includes(activeMetric)) {
      activeMetric = "hr"; document.querySelector("#metric").value = activeMetric;
    } else if (activeTable !== "B1" && ["mc", "hr", "par", "ohf"].includes(activeMetric)) {
      activeMetric = "dice"; document.querySelector("#metric").value = activeMetric;
    }
    renderTable(); renderPareto();
  }));
  document.querySelector("#metric")?.addEventListener("change", event => { activeMetric = event.target.value; renderTable(); });

  function renderPareto() {
    const box = document.querySelector("#pareto");
    const rows = resultRows.filter(row => row.table === activeTable && row.dice !== "" && row.ms_per_img !== "");
    if (!rows.length) { box.innerHTML = '<p class="pending">Pareto points pending final aggregate results.</p>'; return; }
    const width = 760, height = 290, pad = 48;
    const xs = rows.map(row => Number(row.ms_per_img)); const ys = rows.map(row => Number(row.dice));
    const xmax = Math.max(...xs) || 1, ymin = Math.min(...ys), ymax = Math.max(...ys);
    const circles = rows.map(row => {
      const x = pad + Number(row.ms_per_img) / xmax * (width - pad * 2);
      const y = height - pad - (Number(row.dice) - ymin) / Math.max(ymax - ymin, .001) * (height - pad * 2);
      return `<g><circle cx="${x}" cy="${y}" r="6" fill="var(--accent)"/><title>${row.approach}: Dice ${(Number(row.dice)*100).toFixed(2)}, ${Number(row.ms_per_img).toFixed(2)} ms</title></g>`;
    }).join("");
    box.innerHTML = `<svg role="img" aria-label="Dice versus runtime scatter plot" viewBox="0 0 ${width} ${height}"><line x1="${pad}" y1="${height-pad}" x2="${width-pad}" y2="${height-pad}" stroke="currentColor"/><line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height-pad}" stroke="currentColor"/>${circles}<text x="${width/2}" y="${height-10}" text-anchor="middle" fill="currentColor">ms / image</text><text x="15" y="${height/2}" transform="rotate(-90 15 ${height/2})" text-anchor="middle" fill="currentColor">Dice</text></svg>`;
  }

  const examples = payload.synthetic;
  const selector = document.querySelector("#synthetic-select");
  const source = document.querySelector("#synthetic-source");
  const mask = document.querySelector("#synthetic-mask");
  function showSynthetic() {
    const row = examples.find(item => item.key === selector.value);
    if (!row) return;
    source.src = row.image; mask.src = row.mask;
  }
  for (const item of examples) { const option = document.createElement("option"); option.value = item.key; option.textContent = item.label; selector.appendChild(option); }
  selector.addEventListener("change", showSynthetic); showSynthetic();

  document.querySelector("#copy-cite")?.addEventListener("click", async () => {
    const text = document.querySelector("#bibtex").textContent;
    await navigator.clipboard.writeText(text);
    document.querySelector("#copy-cite").textContent = "Copied";
  });
  renderTable(); renderPareto();
})();
