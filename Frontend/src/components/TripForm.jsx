import { useState } from "react";

const EXAMPLES = [
  "Plan a 5-day trip from Delhi to Goa under ₹40,000",
  "Weekend in Paris for two, mid-range budget, love food and museums",
  "7 days in Japan in April, first time visitor",
];

export default function TripForm({ onSubmit, initialValue = "" }) {
  const [text, setText] = useState(initialValue);

  function submit(e) {
    e.preventDefault();
    const trimmed = text.trim();
    if (trimmed) onSubmit(trimmed);
  }

  return (
    <form className="card trip-form" onSubmit={submit}>
      <h2>Where to next?</h2>
      <p className="muted">
        Describe your trip: destination, dates, budget, and anything you like.
      </p>

      <textarea
        rows={4}
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="e.g. 5 days in Goa from Delhi, budget ₹40,000, beaches and seafood"
      />

      <div className="examples">
        {EXAMPLES.map((ex) => (
          <button
            type="button"
            key={ex}
            className="chip"
            onClick={() => setText(ex)}
          >
            {ex}
          </button>
        ))}
      </div>

      <button className="btn primary" type="submit" disabled={!text.trim()}>
        Plan my trip
      </button>
    </form>
  );
}
