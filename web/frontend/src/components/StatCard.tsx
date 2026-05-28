import { Sparkline } from './Sparkline'

export function StatCard({
  label,
  value,
  change,
  sparkline,
}: {
  label: string
  value: string | number
  change?: { value: number; period?: string }
  sparkline?: number[]
}) {
  const isPositive = change && change.value >= 0
  const isNegative = change && change.value < 0

  return (
    <div className="bg-screen-elevated border border-screen-border p-4 rounded min-w-0">
      <div className="text-[10px] text-text-muted uppercase tracking-[0.2em] mb-1">
        {label}
      </div>
      <div className="text-xl font-bold tabular-nums text-terminal-green truncate">
        {value}
      </div>
      {change && (
        <div
          className={`text-xs tabular-nums mt-1 ${isPositive ? 'text-terminal-green' : isNegative ? 'text-terminal-red' : 'text-text-muted'}`}
        >
          {isPositive ? '\u25B2' : '\u25BC'} {change.value >= 0 ? '+' : ''}
          {change.value.toFixed(2)}%{change.period ? ` ${change.period}` : ''}
        </div>
      )}
      {sparkline && sparkline.length > 0 && (
        <div className="mt-2">
          <Sparkline data={sparkline} width={80} height={24} />
        </div>
      )}
    </div>
  )
}