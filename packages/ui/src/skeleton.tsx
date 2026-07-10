
export function Skeleton({ width, height }: { width?: number | string; height?: number | string }) {
  return (
    <div
      style={{
        width: width ?? '100%',
        height: height ?? 20,
        background: 'var(--bg-surface-2)',
        borderRadius: 'var(--radius-s)',
        animation: 'shimmer 1.5s infinite',
      }}
    />
  );
}