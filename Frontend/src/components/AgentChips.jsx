const LABELS = {
  flight_agent: "Flights",
  hotel_agent: "Hotels",
  weather_agent: "Weather",
  budget_agent: "Budget",
  itinerary_agent: "Itinerary",
};

export default function AgentChips({ agents = [], reasoning }) {
  if (!agents.length) return null;

  return (
    <div className="agents">
      <span className="label">Agents used</span>
      <div className="chip-row">
        {agents.map((a) => (
          <span key={a} className="chip static">
            {LABELS[a] || a}
          </span>
        ))}
      </div>
      {reasoning && <p className="muted small">{reasoning}</p>}
    </div>
  );
}
