import ReactMarkdown from "react-markdown";

export default function PlanCard({ title, markdown }) {
  return (
    <section className="card plan">
      <h2>{title}</h2>
      <div className="markdown">
        <ReactMarkdown>{markdown || "No content returned."}</ReactMarkdown>
      </div>
    </section>
  );
}
