
interface EmptyStateProps {
  message: string;
  action?: { label: string; onClick: () => void };
}

export function EmptyState({ message, action }: EmptyStateProps) {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 'var(--space-8)',
        gap: 12,
        color: 'var(--text-muted)',
        textAlign: 'center',
        height: '100%',
      }}
    >
      <span style={{ fontSize: 14 }}>{message}</span>
      {action && (
        <button
          onClick={action.onClick}
          style={{
            background: 'var(--accent)',
            color: '#0b0d12',
            border: 'none',
            borderRadius: 'var(--radius-s)',
            padding: '8px 16px',
            fontWeight: 500,
            cursor: 'pointer',
          }}
        >
          {action.label}
        </button>
      )}
    </div>
  );
}