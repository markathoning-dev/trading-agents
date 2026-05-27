import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { postForm } from '../api/client'

export function NewBacktest() {
  const navigate = useNavigate()
  const [modelName, setModelName] = useState('openai/gpt-4o-mini')
  const [symbol, setSymbol] = useState('AAPL')
  const [maxSteps, setMaxSteps] = useState(50)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    const data = await postForm<{ run_id: number }>('/backtests/new', {
      model_name: modelName,
      symbol,
      max_steps: maxSteps,
    })
    navigate(`/app/backtests/${data.run_id}`)
  }

  return (
    <>
      <h1>Run New Backtest</h1>
      <form onSubmit={handleSubmit}>
        <label>
          Model:{' '}
          <input
            type="text"
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
          />
        </label>
        <br />
        <label>
          Symbol:{' '}
          <input
            type="text"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
          />
        </label>
        <br />
        <label>
          Max Steps:{' '}
          <input
            type="number"
            value={maxSteps}
            onChange={(e) => setMaxSteps(Number(e.target.value))}
          />
        </label>
        <br />
        <button type="submit">Run Backtest</button>
      </form>
    </>
  )
}