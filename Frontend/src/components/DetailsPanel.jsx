import ReactMarkdown from "react-markdown";

const SECTIONS = [
  ["flight_results", "Flights"],
  ["hotel_results", "Hotels"],
  ["weather_results", "Weather"],
  ["budget_results", "Budget"],
];

export default function DetailsPanel({ result }) {
  const available = SECTIONS.filter(([key]) => result[key]);
  if (!available.length) return null;

  return (
    <section className="details">
      <h3>Agent details</h3>
      {available.map(([key, label]) => (
        <details key={key} className="detail-item">
          <summary>{label}</summary>
          <div className="markdown">
            <ReactMarkdown>{String(result[key])}</ReactMarkdown>
          </div>
        </details>
      ))}
    </section>
  );
}
