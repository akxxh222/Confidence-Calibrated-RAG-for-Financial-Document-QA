const form = document.querySelector("#research-form");
const questionInput = document.querySelector("#question");
const submitButton = document.querySelector("#submit-button");
const requestStatus = document.querySelector("#request-status");
const result = document.querySelector("#result");
const answerText = document.querySelector("#answer-text");
const retrievalScore = document.querySelector("#retrieval-score");
const spreadScore = document.querySelector("#spread-score");
const sourceCount = document.querySelector("#source-count");
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
    const fields = [
      `Document ${source.document_id}`,
      source.page_ref ? `Page ${source.page_ref}` : "Page not supplied",
      `${percent(source.similarity)} match`,
    ];
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
  requestStatus.textContent = "Searching filings and preparing a grounded answer…";

  try {
    const response = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "The request could not be completed.");
    }

    answerText.textContent = payload.answer;
    retrievalScore.textContent = percent(payload.retrieval_confidence);
    spreadScore.textContent = percent(payload.spread);
    sourceCount.textContent = String(payload.sources.length);
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
  if (question) {
    submitQuestion(question);
  }
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
