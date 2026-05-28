import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { get } from '../api/client'
import type { BacktestRun } from '../types'
import { StatCard } from '../components/StatCard'
import { TerminalTable } from '../components/TerminalTable'

export function Dashboard() {
  const [runs, setRuns] = useState<BacktestRun[]>([])

  useEffect(() => {
    get<BacktestRun[]>('/dashboard').then(setRuns).catch(console.error)
  }, [])

  const completed = runs.filter((r) => r.status === 'completed' && r.result)
  const winRate =
    completed.length > 0
      ? (completed.filter((r) => (r.result?.total_return ?? 0) > 0).length / completed.length) * 100
      : 0
  const avgSharpe =
    completed.length > 0
      ? completed.reduce((sum, r) => sum + (r.result?.sharpe_ratio ?? 0), 0) / completed.length
      : 0

  const modelCounts: Record<string, number> = {}
  completed.forEach((r) => {
    modelCounts[r.model_name] = (modelCounts[r.model_name] || 0) + 1
  })
  let bestModel = '\u2014'
  let bestSharpe = -Infinity
  Object.entries(modelCounts).forEach(([model]) => {
    const modelRuns = completed.filter((r) => r.model_name === model)
    const sharpe = modelRuns.reduce((s, r) => s + (r.result?.sharpe_ratio ?? 0), 0) / modelRuns.length
    if (sharpe > bestSharpe) {
      bestSharpe = sharpe
      bestModel = model
    }
  })

  const sparkline = completed.slice(-20).map((r) => (r.result?.total_return ?? 0) * 100)

  return (
    <div className="py-4">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-sm text-terminal-green tracking-wider">DASHBOARD</h1>
        <span className="text-[10px] text-text-dim">
          last refresh: {new Date().toLocaleTimeString()}
        </span>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <StatCard label="Total Runs" value={runs.length} sparkline={sparkline} />
        <StatCard
          label="Win Rate"
          value={`${winRate.toFixed(1)}%`}
          change={winRate > 0 ? { value: winRate } : undefined}
        />
        <StatCard label="Avg Sharpe" value={avgSharpe.toFixed(2)} />
        <StatCard label="Best Model" value={bestModel} />
      </div>

      <div className="mb-3">
        <h2 className="text-[11px] text-text-muted uppercase tracking-[0.2em] mb-3">Recent Backtest Runs</h2>
        <TerminalTable<BacktestRun>
          columns={[
            {
              key: 'id',
              label: 'ID',
              render: (run) => (
                <Link to={`/app/backtests/${run.id}`} className="text-terminal-cyan hover:underline">
                  #{run.id}
                </Link>
              ),
            },
            { key: 'model_name', label: 'Model' },
            {
              key: 'status',
              label: 'Status',
              render: (run) => (
                <span className="text-[11px] uppercase tracking-wider text-text-muted">
                  {run.status}
                </span>
              ),
            },
            {
              key: 'total_return',
              label: 'Return',
              render: (run) => {
                if (!run.result) return '\u2014'
                const pct = (run.result.total_return * 100).toFixed(2)
                const isPos = run.result.total_return >= 0
                return <span className={isPos ? 'text-terminal-green' : 'text-terminal-red'}>{pct}%</span>
              },
            },
            {
              key: 'sharpe_ratio',
              label: 'Sharpe',
              render: (run) =>
                run.result ? run.result.sharpe_ratio.toFixed(2) : '\u2014',
            },
            {
              key: 'created_at',
              label: 'Date',
              render: (run) =>
                run.created_at ? new Date(run.created_at).toLocaleDateString() : '\u2014',
            },
          ]}
          rows={runs.slice(0, 25)}
          emptyMessage="No backtest runs yet. Start one from Backtests > New Run."
        />
      </div>

      <Link
        to="/app/backtests"
        className="inline-block text-terminal-green text-[13px] mt-3 hover:underline"
      >
        &gt; view all backtests
      </Link>
    </div>
  )
}