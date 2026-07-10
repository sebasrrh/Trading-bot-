import { DeltaChip } from './delta-chip';

interface StatTileProps {
  label: string;
  value: string;
  delta?: number;
}

export function StatTile({ label, value, delta }: StatTileProps) {
  return (
    <div
      style={{
        background: 'var(--bg-surface-1)',
        border: '1px solid var(--border-hairline)',
        borderRadius: 'var(--radius-m)',
        padding: 'var(--space-5)',
        display: 'flex',
        flexDirection: 'column',
        gap: 4,
      }}
    >
      <span style={{ color: 'var(--text-secondary)', fontSize: 12, fontWeight: 500 }}>{label}</span>
      <span
        style={{
          color: 'var(--text-primary)',
          fontSize: 28,
          fontWeight: 650,
          lineHeight: '34px',
          fontVariantNumeric: 'tabular-nums',
        }}
      >
        {value}
      </span>
      {delta !== undefined && <DeltaChip value={delta} />}
    </div>
  );
}