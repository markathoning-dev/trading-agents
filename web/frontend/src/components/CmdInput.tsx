export function CmdInput({
  label,
  value,
  onChange,
  type = 'text',
  placeholder,
  className = '',
}: {
  label: string
  value: string | number
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void
  type?: 'text' | 'number'
  placeholder?: string
  className?: string
}) {
  return (
    <div className={`flex items-center gap-2 font-mono text-[13px] ${className}`}>
      <span className="text-terminal-green select-none">&gt;</span>
      <span className="text-text-muted">{label}:</span>
      <input
        type={type}
        value={value}
        onChange={onChange}
        placeholder={placeholder}
        className="bg-transparent border-0 border-b border-screen-border text-text-primary outline-none px-1 py-0.5 focus:border-terminal-green transition-colors"
        style={{ minWidth: `${String(value || placeholder || '').length + 2}ch`, width: 'auto' }}
      />
    </div>
  )
}