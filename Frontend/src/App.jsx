import { useState } from "react";
import { planTrip, reviewPlan } from "./api.js";
import Header from "./components/Header.jsx";
import TripForm from "./components/TripForm.jsx";
import Loader from "./components/Loader.jsx";
import PlanCard from "./components/PlanCard.jsx";
import ApprovalPanel from "./components/ApprovalPanel.jsx";
import DetailsPanel from "./components/DetailsPanel.jsx";
import AgentChips from "./components/AgentChips.jsx";

// phase: "idle" | "loading" | "review" | "done" | "blocked"
export default function App() {
  const [phase, setPhase] = useState("idle");
  const [loadingText, setLoadingText] = useState("");
  const [query, setQuery] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  async function handleSubmit(message) {
    setQuery(message);
    setError("");
    setResult(null);
    setLoadingText("Our agents are planning your trip. This can take a minute...");
    setPhase("loading");

    try {
      const data = await planTrip(message);
      setResult(data);

      if (data.guardrail_allowed === false) {
        setPhase("blocked");
      } else if (data.requires_approval) {
        setPhase("review");
      } else {
        setPhase("done");
      }
    } catch (err) {
      setError(err.message);
      setPhase("idle");
    }
  }

  async function handleReview(approved, feedback) {
    setError("");
    setLoadingText(
      approved
        ? "Polishing your final plan..."
        : "Revising your plan with your feedback..."
    );
    setPhase("loading");

    try {
      const data = await reviewPlan(result.thread_id, approved, feedback);
      setResult(data);
      setPhase("done");
    } catch (err) {
      setError(err.message);
      setPhase("review");
    }
  }

  function reset() {
    setPhase("idle");
    setResult(null);
    setError("");
    setQuery("");
  }

  return (
    <div className="app">
      <Header onNew={phase === "idle" ? null : reset} />

      <main className="main">
        {phase === "idle" && (
          <TripForm onSubmit={handleSubmit} initialValue={query} />
        )}

        {error && <div className="error-box">{error}</div>}

        {phase === "loading" && <Loader text={loadingText} />}

        {phase === "blocked" && result && (
          <div className="card blocked">
            <h2>Request not supported</h2>
            <p>{result.answer || result.guardrail_reason}</p>
            <button className="btn primary" onClick={reset}>
              Ask something else
            </button>
          </div>
        )}

        {(phase === "review" || phase === "done") && result && (
          <>
            <div className="query-echo">
              <span className="label">Your request</span>
              <p>{query}</p>
            </div>

            <AgentChips
              agents={result.selected_agents}
              reasoning={result.supervisor_reasoning}
            />

            <PlanCard
              title={phase === "review" ? "Draft itinerary" : "Your travel plan"}
              markdown={result.answer}
            />

            {phase === "review" && (
              <ApprovalPanel
                message={result.approval_request}
                onApprove={() => handleReview(true, "")}
                onRevise={(feedback) => handleReview(false, feedback)}
              />
            )}

            <DetailsPanel result={result} />

            {phase === "done" && (
              <div className="center">
                <button className="btn primary" onClick={reset}>
                  Plan another trip
                </button>
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}
