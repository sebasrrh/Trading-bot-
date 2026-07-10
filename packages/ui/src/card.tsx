import type React from 'react';

interface CardProps {
  children: React.ReactNode;
  className?: string;
}

export function Card({ children, className = '' }: CardProps) {
  return (
    <div
      className={className}
      style={{
        background: 'var(--bg-surface-1)',
        border: '1px solid var(--border-hairline)',
        borderRadius: 'var(--radius-m)',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {children}
    </div>
  );
}

export function CardHeader({ children }: { children: React.ReactNode }) {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: 'var(--space-3) var(--space-4)',
        borderBottom: '1px solid var(--border-hairline)',
        fontSize: '14px',
        fontWeight: 600,
        lineHeight: '20px',
      }}
    >
      {children}
    </div>
  );
}

export function CardBody({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ padding: 'var(--space-4)', flex: 1 }}>
      {children}
    </div>
  );
}