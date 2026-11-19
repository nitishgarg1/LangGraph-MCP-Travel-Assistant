export default function Header({ onNew }) {
  return (
    <header className="header">
      <div className="brand">
        <span className="logo">✈</span>
        <div>
          <h1>TripMate AI</h1>
          <p>Multi-agent travel planner</p>
        </div>
      </div>
      {onNew && (
        <button className="btn ghost" onClick={onNew}>
          New trip
        </button>
      )}
    </header>
  );
}
