export default function Loader({ text }) {
  return (
    <div className="card loader">
      <div className="spinner" />
      <p>{text}</p>
    </div>
  );
}
