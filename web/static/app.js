/* ── State ──────────────────────────────────────────────────────────────── */
let selectedFile  = null;
let stats         = { total: 0, pass: 0, fail: 0 };

/* ── DOM Refs ────────────────────────────────────────────────────────────── */
const dropZone       = document.getElementById("dropZone");
const fileInput      = document.getElementById("fileInput");
const previewArea    = document.getElementById("previewArea");
const previewImg     = document.getElementById("previewImg");
const previewName    = document.getElementById("previewName");
const clearBtn       = document.getElementById("clearBtn");
const inspectBtn     = document.getElementById("inspectBtn");
const btnText        = inspectBtn.querySelector(".btn-text");
const btnSpinner     = document.getElementById("btnSpinner");
const resultPlaceholder = document.getElementById("resultPlaceholder");
const resultContent  = document.getElementById("resultContent");
const toast          = document.getElementById("toast");
const statusDot      = document.getElementById("statusDot");
const statusLabel    = document.getElementById("statusLabel");

/* ── Server Health Check ─────────────────────────────────────────────────── */
async function checkHealth() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    if (data.model_loaded) {
      statusDot.className   = "status-dot online";
      statusLabel.textContent = `模型就绪 · 记忆库 ${data.memory_bank_size.toLocaleString()} patches`;
    } else {
      statusDot.className   = "status-dot offline";
      statusLabel.textContent = "模型未加载，请先训练";
    }
  } catch {
    statusDot.className   = "status-dot offline";
    statusLabel.textContent = "服务离线";
  }
}
checkHealth();

/* ── Drag & Drop ─────────────────────────────────────────────────────────── */
dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) handleFileSelect(file);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFileSelect(fileInput.files[0]);
});

function handleFileSelect(file) {
  const allowed = ["image/png","image/jpeg","image/bmp","image/tiff"];
  if (!allowed.includes(file.type) && !file.name.match(/\.(png|jpg|jpeg|bmp|tiff)$/i)) {
    showToast("请上传图片文件 (PNG / JPG / BMP / TIFF)");
    return;
  }
  selectedFile = file;
  const url    = URL.createObjectURL(file);
  previewImg.src       = url;
  previewName.textContent = file.name;
  dropZone.style.display  = "none";
  previewArea.style.display = "block";
  inspectBtn.disabled  = false;

  // Reset result panel
  resultPlaceholder.style.display = "flex";
  resultContent.style.display     = "none";
}

/* ── Clear ───────────────────────────────────────────────────────────────── */
clearBtn.addEventListener("click", () => {
  selectedFile = null;
  fileInput.value = "";
  dropZone.style.display   = "block";
  previewArea.style.display = "none";
  inspectBtn.disabled = true;
  resultPlaceholder.style.display = "flex";
  resultContent.style.display     = "none";
});

/* ── Inspect ─────────────────────────────────────────────────────────────── */
inspectBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  // Loading state
  btnText.style.display    = "none";
  btnSpinner.style.display = "flex";
  inspectBtn.disabled = true;

  try {
    const formData = new FormData();
    formData.append("file", selectedFile);

    const res  = await fetch("/predict", { method: "POST", body: formData });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "服务器错误");
    }
    const data = await res.json();
    renderResult(data);

    // Update stats
    stats.total++;
    if (data.is_pass) stats.pass++;
    else              stats.fail++;
    updateStats();

  } catch (err) {
    showToast("检测失败: " + err.message);
  } finally {
    btnText.style.display    = "inline";
    btnSpinner.style.display = "none";
    inspectBtn.disabled = false;
  }
});

/* ── Render Result ───────────────────────────────────────────────────────── */
function renderResult(data) {
  const isPass = data.is_pass;

  // Verdict banner
  const banner = document.getElementById("verdictBanner");
  banner.className = "verdict-banner " + (isPass ? "pass" : "fail");

  document.getElementById("verdictIcon").textContent  = isPass ? "✅" : "❌";
  document.getElementById("verdictLabel").textContent = isPass ? "PASS" : "FAIL";
  document.getElementById("verdictSub").textContent   = isPass
    ? "产品符合质量标准，允许放行"
    : "检测到表面异常，需人工复判";

  // Confidence
  const conf = Math.max(0, Math.min(100, data.confidence));
  document.getElementById("confValue").textContent = conf.toFixed(1) + "%";
  document.getElementById("confBar").style.width   = conf + "%";
  document.getElementById("confBar").style.background = isPass
    ? "linear-gradient(90deg, #10b981, #34d399)"
    : "linear-gradient(90deg, #ef4444, #f97316)";

  // Score
  document.getElementById("scoreValue").textContent = data.score.toFixed(4);
  document.getElementById("threshValue").textContent = data.threshold.toFixed(4);

  // Images
  document.getElementById("origImg").src    = "data:image/png;base64," + data.original_img;
  document.getElementById("heatmapImg").src = "data:image/png;base64," + data.heatmap_img;

  // Show result
  resultPlaceholder.style.display = "none";
  resultContent.style.display     = "block";
}

/* ── Stats ───────────────────────────────────────────────────────────────── */
function updateStats() {
  document.getElementById("statTotal").textContent = stats.total;
  document.getElementById("statPass").textContent  = stats.pass;
  document.getElementById("statFail").textContent  = stats.fail;
  const yieldRate = stats.total > 0
    ? ((stats.pass / stats.total) * 100).toFixed(1) + "%"
    : "—";
  document.getElementById("statYield").textContent = yieldRate;
}

/* ── Toast ───────────────────────────────────────────────────────────────── */
let toastTimer;
function showToast(msg) {
  toast.textContent = msg;
  toast.style.display = "block";
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.style.display = "none"; }, 3500);
}
