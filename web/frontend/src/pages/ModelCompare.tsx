import { useEffect, useState } from 'react'
import { get } from '../api/client'
import type { ModelCompareRow } from '../types'

export function ModelCompare() {
  const [rows, setRows] = useState<ModelCompareRow[]>([])

  useEffect(() => {
    get<ModelCompareRow[]>('/models/compare').then(setRows).catch(console.error)
  }, [])

  return (
    <>
      <h1>Model Comparison</h1>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Avg Return</th>
            <th>Avg Sharpe</th>
            <th>Avg Drawdown</th>
            <th>Runs</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.model_name}>
              <td>{row.model_name}</td>
              <td>{(row.avg_return * 100).toFixed(2)}%</td>
              <td>{row.avg_sharpe.toFixed(2)}</td>
              <td>{(row.avg_drawdown * 100).toFixed(2)}%</td>
              <td>{row.count}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  )
}