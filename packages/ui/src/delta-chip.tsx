
interface DeltaChipProps {
  value: number;
  suffix?: string;
}

export function DeltaChip({ value, suffix = '%' }: DeltaChipProps) {
  const isGain = value >= 0;
  const arrow = isGain ? '\u25B2' : '\u25BC';
  return (
    <span
      style={{
        color: isGain ? 'var(--gain)' : 'var(--loss)',
        fontSize: 12,
        fontWeight: 600,
        fontVariantNumeric: 'tabular-nums',
        display: 'inline-flex',
        alignItems: 'center',
        gap: 2,
      }}
    >
      {arrow} {Math.abs(value).toFixed(1)}{suffix}
    </span>
  );
}