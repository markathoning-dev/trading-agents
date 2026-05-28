import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { get } from '../api/client'
import type { BacktestRun } from '../types'
import { TerminalTable } from '../components/TerminalTable'
import { Btn } from '../components/Btn'

export function BacktestList() {
  const [runs, setRuns] = useState<BacktestRun[]>([])

  useEffect(() => {
    get<BacktestRun[]>('/backtests').then(setRuns).catch(console.error)
  }, [])

  return (
    <div className="py-4">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-sm text-terminal-green tracking-wider">BACKTEST RUNS</h1>
        <Link to="/app/backtests/new">
          <Btn>New Backtest</Btn>
        </Link>
      </div>

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
            key: 'data_source',
            label: 'Symbol',
            render: (r) => r.data_source.split(':')[1] ?? r.data_source,
          },
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
            colorize: true,
            render: (run) =>
              run.result ? `${(run.result.total_return * 100).toFixed(2)}%` : '\u2014',
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
              run.created_at ? new Date(run.created_at).toLocaleString() : '\u2014',
          },
        ]}
        rows={runs}
        sortable
        defaultSortKey="id"
        defaultSortDir="desc"
        emptyMessage="No backtest runs yet. Create one from New Run."
      />
    </div>
  )
}