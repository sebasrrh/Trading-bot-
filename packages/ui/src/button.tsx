import type React from 'react';

interface ButtonProps {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
  size?: 'sm' | 'md';
  children: React.ReactNode;
  onClick?: () => void;
  disabled?: boolean;
}

const variantStyles: Record<string, React.CSSProperties> = {
  primary: { background: 'var(--accent)', color: '#0b0d12', border: 'none' },
  secondary: { background: 'var(--bg-surface-2)', color: 'var(--text-primary)', border: '1px solid var(--border-strong)' },
  ghost: { background: 'transparent', color: 'var(--text-secondary)', border: 'none' },
  danger: { background: 'var(--loss)', color: '#fff', border: 'none' },
};

const sizeStyles: Record<string, React.CSSProperties> = {
  sm: { height: 28, padding: '0 12px', fontSize: 12 },
  md: { height: 32, padding: '0 16px', fontSize: 13 },
};

export function Button({ variant = 'primary', size = 'md', children, onClick, disabled }: ButtonProps) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        ...variantStyles[variant],
        ...sizeStyles[size],
        borderRadius: 'var(--radius-s)',
        fontWeight: 500,
        cursor: disabled ? 'not-allowed' : 'pointer',
        opacity: disabled ? 0.5 : 1,
        display: 'inline-flex',
        alignItems: 'center',
        gap: 6,
        transition: 'background 120ms ease-out',
      }}
    >
      {children}
    </button>
  );
}