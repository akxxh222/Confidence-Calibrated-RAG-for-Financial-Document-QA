const form = document.querySelector("#research-form");
const questionInput = document.querySelector("#question");
const submitButton = document.querySelector("#submit-button");
const requestStatus = document.querySelector("#request-status");
const result = document.querySelector("#result");
const answerText = document.querySelector("#answer-text");
const answerTitle = document.querySelector("#answer-title");
const confidenceScore = document.querySelector("#confidence-score");
const retrievalScore = document.querySelector("#retrieval-score");
const spreadScore = document.querySelector("#spread-score");
const consistencyScore = document.querySelector("#consistency-score");
const sourceCount = document.querySelector("#source-count");
const resultBadge = document.querySelector("#result-badge");
const sourcesList = document.querySelector("#sources-list");
const systemStatus = document.querySelector("#system-status");
const statusDot = document.querySelector("#status-dot");

function percent(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? `${(numeric * 100).toFixed(1)}%` : "—";
}

function readableSection(section) {
  return String(section || "unlabeled")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function setLoading(loading) {
  submitButton.disabled = loading;
  submitButton.classList.toggle("loading", loading);
  questionInput.disabled = loading;
}

function renderSources(sources) {
  sourcesList.replaceChildren();
  sources.forEach((source) => {
    const item = document.createElement("li");
    item.className = "source-item";

    const title = document.createElement("div");
    title.className = "source-title";
    const section = document.createElement("span");
    section.textContent = readableSection(source.section);
    title.append(section);

    const meta = document.createElement("div");
    meta.className = "source-meta";
    const fields = [`Document ${source.document_id}`];
    if (source.page_ref) fields.push(`Page ${source.page_ref}`);
    fields.push(`${percent(source.similarity)} match`);
    fields.forEach((value) => {
      const field = document.createElement("span");
      field.textContent = value;
      meta.append(field);
    });

    item.append(title, meta);
    sourcesList.append(item);
  });
}

async function submitQuestion(question) {
  setLoading(true);
  result.hidden = true;
  requestStatus.classList.remove("error");
  requestStatus.textContent = "Searching filings and evaluating confidence…";

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const contentType = response.headers.get("content-type") || "";
    if (!contentType.includes("application/json")) {
      throw new Error(
        response.ok
          ? "The server returned an unexpected response."
          : `The server could not complete the request (HTTP ${response.status}).`,
      );
    }
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "The request could not be completed.");
    }

    answerText.textContent = payload.answer;
    confidenceScore.textContent = percent(payload.confidence_score);
    retrievalScore.textContent = percent(payload.retrieval_confidence);
    spreadScore.textContent = percent(payload.spread);
    consistencyScore.textContent = percent(payload.self_consistency_score);
    sourceCount.textContent = String(payload.sources.length);

    const labels = { answer: "Answer", hedge: "Hedge", refuse: "Refuse" };
    resultBadge.textContent = labels[payload.mode] || "SEC filings";
    answerTitle.textContent = payload.mode === "hedge"
      ? "Qualified answer"
      : payload.mode === "refuse" ? "Cannot answer" : "Answer";

    renderSources(payload.sources);
    result.hidden = false;
    requestStatus.textContent = "";
    result.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    requestStatus.classList.add("error");
    requestStatus.textContent = error.message || "The request could not be completed.";
  } finally {
    setLoading(false);
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (question) submitQuestion(question);
});

document.querySelectorAll(".example-question").forEach((button) => {
  button.addEventListener("click", () => {
    questionInput.value = button.textContent.trim();
    questionInput.focus();
  });
});

fetch("/health")
  .then((response) => {
    if (!response.ok) throw new Error("offline");
    return response.json();
  })
  .then(() => {
    systemStatus.textContent = "System online";
    statusDot.classList.add("online");
  })
  .catch(() => {
    systemStatus.textContent = "System unavailable";
  });
