export function EmptyState({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="state-block">
      <h3 style={{ marginTop: 0 }}>{title}</h3>
      <p>{children}</p>
    </div>
  );
}

export function ErrorState({
  title,
  children,
  onRetry,
}: {
  title: string;
  children: React.ReactNode;
  onRetry?: () => void;
}) {
  return (
    <div className="state-block error">
      <h3 style={{ marginTop: 0 }}>{title}</h3>
      <p>{children}</p>
      {onRetry && (
        <button className="btn btn-primary" onClick={onRetry} type="button">
          Retry
        </button>
      )}
    </div>
  );
}

export function LoadingState({ label }: { label: string }) {
  return (
    <div className="state-block" role="status" aria-live="polite">
      <p style={{ margin: 0 }}>{label}</p>
    </div>
  );
}
