export function Btn({
  children,
  onClick,
  variant = 'primary',
  disabled = false,
  loading = false,
  className = '',
  type = 'button',
}: {
  children: React.ReactNode
  onClick?: () => void
  variant?: 'primary' | 'danger' | 'ghost'
  disabled?: boolean
  loading?: boolean
  className?: string
  type?: 'button' | 'submit'
}) {
  const base =
    'font-mono text-[12px] uppercase tracking-wider px-4 py-2 cursor-pointer transition-all border-0 outline-none disabled:cursor-not-allowed'

  const variants = {
    primary: `bg-terminal-green text-black hover:brightness-110 disabled:bg-text-dim disabled:text-text-muted ${loading ? 'animate-flicker' : ''}`,
    danger: 'bg-terminal-red text-black hover:brightness-110 disabled:bg-text-dim disabled:text-text-muted',
    ghost: 'bg-transparent border border-screen-border text-text-muted hover:border-terminal-green hover:text-terminal-green',
  }

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled || loading}
      className={`${base} ${variants[variant]} ${className}`}
    >
      {loading ? '...' : children}
    </button>
  )
}