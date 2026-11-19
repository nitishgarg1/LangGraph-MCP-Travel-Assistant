import { useState } from "react";

export default function ApprovalPanel({ message, onApprove, onRevise }) {
  const [feedback, setFeedback] = useState("");

  return (
    <section className="card approval">
      <h3>Review this draft</h3>
      <p className="muted">
        {message ||
          "Approve to get the final polished plan, or request changes."}
      </p>

      <textarea
        rows={3}
        value={feedback}
        onChange={(e) => setFeedback(e.target.value)}
        placeholder="Want changes? e.g. add a day trip, cheaper hotels, more food spots"
      />

      <div className="btn-row">
        <button className="btn primary" onClick={onApprove}>
          Approve
        </button>
        <button
          className="btn secondary"
          disabled={!feedback.trim()}
          onClick={() => onRevise(feedback.trim())}
        >
          Request changes
        </button>
      </div>
    </section>
  );
}
