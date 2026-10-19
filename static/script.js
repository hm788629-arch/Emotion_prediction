const EMO = {
  sadness:{e:"😢",c:"#5b8def"}, joy:{e:"😄",c:"#ffc23d"}, love:{e:"❤️",c:"#ff5d8f"},
  anger:{e:"😠",c:"#ff4d3d"}, fear:{e:"😨",c:"#a47bff"}, surprise:{e:"😲",c:"#2fd5c3"}
};
const SAMPLES = [
  "I feel so happy and excited today",
  "I miss her so much it hurts",
  "How dare you lie to my face",
  "I'm terrified of what's behind that door",
  "Wow, I never saw that coming",
  "You mean the world to me"
];
const $ = id => document.getElementById(id);
const text = $("text"), form = $("form"), go = $("go"), result = $("result");
const C = 339.3;

// chips
SAMPLES.forEach(s => {
  const b = document.createElement("button");
  b.type = "button"; b.className = "chip"; b.textContent = s;
  b.onclick = () => { text.value = s; text.dispatchEvent(new Event("input")); form.requestSubmit(); };
  $("chips").appendChild(b);
});
text.addEventListener("input", () => $("count").textContent = `${text.value.length} / 2000`);
text.addEventListener("keydown", e => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") form.requestSubmit(); });

// server status
fetch("/health").then(r => r.json()).then(d => {
  $("dot").className = "dot " + (d.model_loaded ? "ok" : "bad");
  $("status").textContent = d.model_loaded ? "Model ready" : "Server up, model not loaded";
}).catch(() => { $("dot").className = "dot bad"; $("status").textContent = "Server unreachable"; });

function countUp(el, to) {
  const t0 = performance.now();
  (function step(t) {
    const k = Math.min((t - t0) / 900, 1), v = to * (1 - Math.pow(1 - k, 3));
    el.textContent = v.toFixed(1) + "%";
    if (k < 1) requestAnimationFrame(step);
  })(t0);
}

function render(d) {
  const top = d.predicted_emotion, info = EMO[top];
  document.documentElement.style.setProperty("--accent", info.c);
  result.classList.remove("empty", "loading");
  $("error").innerHTML = "";

  const face = $("face");
  face.textContent = info.e; face.classList.remove("pop"); void face.offsetWidth; face.classList.add("pop");
  $("arc").style.strokeDashoffset = C * (1 - d.confidence);
  $("title").textContent = top;
  $("sub").textContent = d.confidence < 0.5
    ? "The model is unsure. This sentence may mix feelings."
    : `${(d.confidence * 100).toFixed(1)}% confident`;

  const rows = Object.entries(d.all_probabilities).sort((a, b) => b[1] - a[1]);
  $("bars").innerHTML = "";
  rows.forEach(([k, v], i) => {
    const r = document.createElement("div");
    r.className = "row" + (i === 0 ? " top" : "");
    r.innerHTML = `<span class="n"></span><div class="track"><i></i></div><span class="p">0%</span>`;
    r.querySelector(".n").textContent = `${EMO[k].e} ${k}`;
    $("bars").appendChild(r);
    requestAnimationFrame(() => setTimeout(() => {
      r.querySelector("i").style.width = (v * 100) + "%";
      countUp(r.querySelector(".p"), v * 100);
    }, 60 * i));
  });
}

function showError(msg) {
  result.classList.remove("loading");
  const box = document.createElement("div");
  box.className = "err"; box.setAttribute("role", "alert");
  box.textContent = msg;
  $("error").replaceChildren(box);
}

form.addEventListener("submit", async e => {
  e.preventDefault();
  const value = text.value.trim();
  if (!value) return;
  go.disabled = true; go.textContent = "Reading…";
  result.classList.add("loading");
  $("title").textContent = "Reading the room"; $("sub").textContent = "One moment.";
  try {
    const res = await fetch("/predict", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: value })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "The request was rejected.");
    render(data);
  } catch (err) {
    $("title").textContent = "No result"; $("sub").textContent = "Something went wrong.";
    showError(err.message || "Could not reach the server. Check your connection and try again.");
  } finally {
    go.disabled = false; go.textContent = "Read the mood";
  }
});

// empty-state bars
Object.keys(EMO).forEach(k => {
  const r = document.createElement("div"); r.className = "row";
  r.innerHTML = `<span class="n"></span><div class="track"><i></i></div><span class="p">–</span>`;
  r.querySelector(".n").textContent = `${EMO[k].e} ${k}`;
  $("bars").appendChild(r);
});
