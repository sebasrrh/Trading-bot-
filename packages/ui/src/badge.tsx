import type React from 'react';

interface BadgeProps {
  variant?: 'accent' | 'gain' | 'loss' | 'warn';
  children: React.ReactNode;
}

const variantStyles: Record<string, React.CSSProperties> = {
  accent: { background: 'var(--accent-soft)', color: 'var(--accent)' },
  gain: { background: 'var(--gain-soft)', color: 'var(--gain)' },
  loss: { background: 'var(--loss-soft)', color: 'var(--loss)' },
  warn: { background: 'rgba(245,184,61,0.12)', color: 'var(--warn)' },
};

export function Badge({ variant = 'accent', children }: BadgeProps) {
  return (
    <span
      style={{
        ...variantStyles[variant],
        display: 'inline-flex',
        alignItems: 'center',
        padding: '2px 8px',
        borderRadius: 'var(--radius-full)',
        fontSize: 11,
        fontWeight: 500,
        letterSpacing: '0.04em',
        textTransform: 'uppercase',
        lineHeight: '14px',
      }}
    >
      {children}
    </span>
  );
}