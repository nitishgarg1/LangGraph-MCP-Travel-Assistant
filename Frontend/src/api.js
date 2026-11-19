
const API_BASE = (
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000"
).replace(/\/$/, "");

async function post(path, body) {
  const url = `${API_BASE}${path}`;

  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  let data = null;

  try {
    data = await res.json();
  } catch {
    // Response was not JSON
  }

  if (!res.ok || !data || !data.success) {
    throw new Error(
      data?.error ||
      `Request failed (${res.status}). Check API response.`
    );
  }

  return data;
}

// Start a new planning run. Pauses at human approval.
export function planTrip(message) {
  return post("/api/travel", {
    message,
    thread_id: null,
  });
}

// Resume the paused run: approve, or reject with feedback.
export function reviewPlan(threadId, approved, feedback = "") {
  return post("/api/travel/approve", {
    thread_id: threadId,
    approved,
    feedback,
  });
}
