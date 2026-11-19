async function post(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  let data = null;
  try {
    data = await res.json();
  } catch {
    // response was not JSON
  }

  if (!res.ok || !data || !data.success) {
    throw new Error(data?.error || `Request failed (${res.status})`);
  }
  return data;
}

// Start a new planning run. Pauses at human approval.
export function planTrip(message) {
  return post("/api/travel", { message, thread_id: null });
}

// Resume the paused run: approve, or reject with feedback.
export function reviewPlan(threadId, approved, feedback = "") {
  return post("/api/travel/approve", {
    thread_id: threadId,
    approved,
    feedback,
  });
}
