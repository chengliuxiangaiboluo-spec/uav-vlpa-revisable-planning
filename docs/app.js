const protocols = {
  synthetic: {
    kicker: "PROTOCOL 01",
    title: "Synthetic multimodal evaluation",
    body: "Evaluates fusion, task decomposition, ablations, robustness, and controlled comparisons under the synthetic benchmark protocol.",
    points: [
      "Independent learned-method training seeds are retained.",
      "Results are not pooled with visual-grounding or GPS results."
    ]
  },
  grounding: {
    kicker: "PROTOCOL 02",
    title: "OSM visual-grounding evaluation",
    body: "Evaluates held-out map grounding and replanning under the strict OSM protocol. Reference coordinates remain evaluator-side only.",
    points: [
      "The frozen Molmo reference is a deterministic single-run local adaptation.",
      "Predicted coordinates, parser validity, and route outcomes are audited separately."
    ]
  },
  gps: {
    kicker: "PROTOCOL 03",
    title: "Fixed-input GPS trajectory check",
    body: "Uses fixed GPS trajectories as a descriptive transfer check. It does not establish natural multimodal interaction or closed-loop flight validation.",
    points: [
      "Trajectory-oriented measures are interpreted within this fixed-input protocol.",
      "Its values are not a replacement for synthetic or OSM-grounding evidence."
    ]
  }
};

const panel = document.querySelector("#protocol-panel");
const tabs = document.querySelectorAll(".protocol-tab");

function renderProtocol(name) {
  const data = protocols[name];
  panel.innerHTML = `
    <p class="protocol-kicker">${data.kicker}</p>
    <h3>${data.title}</h3>
    <p>${data.body}</p>
    <ul>${data.points.map((item) => `<li>${item}</li>`).join("")}</ul>
  `;
  tabs.forEach((tab) => {
    const selected = tab.dataset.protocol === name;
    tab.classList.toggle("active", selected);
    tab.setAttribute("aria-selected", String(selected));
  });
}

tabs.forEach((tab) => tab.addEventListener("click", () => renderProtocol(tab.dataset.protocol)));
