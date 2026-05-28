import { useEffect, useState } from 'react'
import { get } from '../api/client'
import type { ModelCompareRow } from '../types'
import { TerminalTable } from '../components/TerminalTable'

export function ModelCompare() {
  const [rows, setRows] = useState<ModelCompareRow[]>([])

  useEffect(() => {
    get<ModelCompareRow[]>('/models/compare').then(setRows).catch(console.error)
  }, [])

  const maxReturn = Math.max(...rows.map((r) => r.avg_return), 0.01)

  return (
    <div className="py-4">
      <h1 className="text-sm text-terminal-green tracking-wider mb-6">MODEL COMPARISON</h1>

      <TerminalTable<ModelCompareRow>
        columns={[
          { key: 'model_name', label: 'Model' },
          {
            key: 'avg_return',
            label: 'Avg Return',
            colorize: true,
            render: (r) => `${(r.avg_return * 100).toFixed(2)}%`,
          },
          { key: 'avg_sharpe', label: 'Avg Sharpe', render: (r) => r.avg_sharpe.toFixed(2) },
          {
            key: 'avg_drawdown',
            label: 'Avg Drawdown',
            colorize: true,
            render: (r) => `${(r.avg_drawdown * 100).toFixed(2)}%`,
          },
          { key: 'count', label: 'Runs' },
        ]}
        rows={rows}
        sortable
        defaultSortKey="avg_return"
        defaultSortDir="desc"
        emptyMessage="No model comparison data yet. Run multiple backtests to compare."
      />

      {rows.length > 0 && (
        <div className="mt-8">
          <div className="text-[10px] text-text-muted uppercase tracking-[0.2em] mb-4">
            Return Distribution
          </div>
          <div className="space-y-2">
            {rows.map((row) => {
              const pct = Math.max(0, (row.avg_return / maxReturn) * 100)
              return (
                <div key={row.model_name} className="flex items-center gap-3 font-mono text-[12px]">
                  <span className="text-text-muted w-36 text-right truncate">{row.model_name}</span>
                  <div className="flex-1 h-5 bg-screen-elevated border border-screen-border rounded-sm overflow-hidden">
                    <div
                      className="h-full bg-terminal-green/30 transition-all"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <span className={`w-20 text-right tabular-nums ${row.avg_return >= 0 ? 'text-terminal-green' : 'text-terminal-red'}`}>
                    {row.avg_return >= 0 ? '+' : ''}{(row.avg_return * 100).toFixed(2)}%
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}