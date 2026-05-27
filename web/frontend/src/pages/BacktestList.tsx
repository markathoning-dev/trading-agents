import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { get } from '../api/client'
import type { BacktestRun } from '../types'

export function BacktestList() {
  const [runs, setRuns] = useState<BacktestRun[]>([])

  useEffect(() => {
    get<BacktestRun[]>('/backtests').then(setRuns).catch(console.error)
  }, [])

  return (
    <>
      <h1>Backtest Runs</h1>
      <Link to="/app/backtests/new">New Backtest</Link>
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>Model</th>
            <th>Symbol</th>
            <th>Status</th>
            <th>Return</th>
            <th>Sharpe</th>
            <th>Date</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => (
            <tr key={run.id}>
              <td>
                <Link to={`/app/backtests/${run.id}`}>{run.id}</Link>
              </td>
              <td>{run.model_name}</td>
              <td>{run.data_source.split(':')[1] ?? run.data_source}</td>
              <td>{run.status}</td>
              <td>
                {run.result
                  ? `${(run.result.total_return * 100).toFixed(2)}%`
                  : '\u2014'}
              </td>
              <td>
                {run.result ? run.result.sharpe_ratio.toFixed(2) : '\u2014'}
              </td>
              <td>
                {run.created_at
                  ? new Date(run.created_at).toLocaleString()
                  : '\u2014'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  )
}