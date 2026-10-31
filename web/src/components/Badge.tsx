export function Badge({ label, note }: { label: string; note: string }) {
  return (
    <div className="badge-row">
      <span className="badge">{label}</span>
      <span className="badge-note">{note}</span>
    </div>
  );
}
