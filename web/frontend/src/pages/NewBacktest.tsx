import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { startBacktestWithDeck } from '../api/cards'
import { CmdInput } from '../components/CmdInput'
import { Btn } from '../components/Btn'

const POPULAR = ['AAPL', 'NVDA', 'TSLA', 'SPY', 'BTC-USD', 'MSFT', 'GOOGL']

export function NewBacktest() {
  const navigate = useNavigate()
  const [modelName, setModelName] = useState('openai/gpt-4o-mini')
  const [symbol, setSymbol] = useState('AAPL')
  const [maxSteps, setMaxSteps] = useState(50)
  const [deckId, setDeckId] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    try {
      const data = await startBacktestWithDeck(
        modelName,
        symbol,
        maxSteps,
        deckId || undefined,
      )
      navigate(`/app/backtests/${data.run_id}`)
    } catch (err) {
      console.error(err)
      setLoading(false)
    }
  }

  return (
    <div className="py-4">
      <h1 className="text-sm text-terminal-green tracking-wider mb-6">NEW BACKTEST</h1>

      <form onSubmit={handleSubmit} className="space-y-4 max-w-lg">
        <CmdInput
          label="model_name"
          value={modelName}
          onChange={(e) => setModelName(e.target.value)}
        />
        <CmdInput
          label="symbol"
          value={symbol}
          onChange={(e) => setSymbol(e.target.value.toUpperCase())}
        />
        <CmdInput
          label="max_steps"
          value={maxSteps}
          onChange={(e) => setMaxSteps(Number(e.target.value))}
          type="number"
        />
        <CmdInput
          label="deck"
          value={deckId}
          onChange={(e) => setDeckId(e.target.value)}
          placeholder="(optional)"
        />

        <div className="flex gap-3 pt-2">
          <Btn type="submit" loading={loading}>
            Run Backtest
          </Btn>
          <Btn variant="ghost" onClick={() => navigate('/app/backtests')}>
            Cancel
          </Btn>
        </div>
      </form>

      <div className="mt-8">
        <div className="text-[10px] text-text-muted uppercase tracking-[0.2em] mb-2">
          Recent Symbols
        </div>
        <div className="flex flex-wrap gap-2">
          {POPULAR.map((sym) => (
            <button
              key={sym}
              type="button"
              onClick={() => setSymbol(sym)}
              className={`font-mono text-[12px] px-3 py-1 border cursor-pointer transition-colors ${
                symbol === sym
                  ? 'border-terminal-green text-terminal-green bg-terminal-green/10'
                  : 'border-screen-border text-text-muted hover:border-screen-border-hi hover:text-text-primary'
              }`}
            >
              {sym}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}