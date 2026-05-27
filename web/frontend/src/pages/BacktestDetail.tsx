import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { get } from '../api/client'
import type { BacktestDetail } from '../types'

export function BacktestDetailPage() {
  const { runId } = useParams<{ runId: string }>()
  const [data, setData] = useState<BacktestDetail | null>(null)

  useEffect(() => {
    if (runId) get<BacktestDetail>(`/backtests/${runId}`).then(setData).catch(console.error)
  }, [runId])

  if (!data) return <p>Loading...</p>

  return (
    <>
      <Link to="/app/backtests">&larr; Back</Link>
      <h1>
        Backtest #{data.id} &mdash; {data.model_name}
      </h1>
      {data.result ? (
        <ul>
          <li>Portfolio Value: ${data.result.final_portfolio_value.toFixed(2)}</li>
          <li>Return: {(data.result.total_return * 100).toFixed(2)}%</li>
          <li>Sharpe: {data.result.sharpe_ratio.toFixed(2)}</li>
          <li>Max Drawdown: {(data.result.max_drawdown * 100).toFixed(2)}%</li>
          <li>Steps: {data.result.num_steps}</li>
        </ul>
      ) : (
        <p>Status: {data.status}</p>
      )}
      {data.steps && data.steps.length > 0 && (
        <table>
          <thead>
            <tr>
              <th>Step</th>
              <th>Price</th>
              <th>Action</th>
              <th>Portfolio</th>
              <th>Reward</th>
            </tr>
          </thead>
          <tbody>
            {data.steps.map((s) => (
              <tr key={s.step}>
                <td>{s.step}</td>
                <td>${s.price.toFixed(2)}</td>
                <td>{s.action}</td>
                <td>${s.portfolio_value.toFixed(2)}</td>
                <td>{s.reward.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </>
  )
}