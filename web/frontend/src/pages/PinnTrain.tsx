import { useEffect, useState } from 'react'
import { get } from '../api/client'
import type { PinnTrainForm } from '../types'
import { Btn } from '../components/Btn'

export function PinnTrain() {
  const [form, setForm] = useState<PinnTrainForm | null>(null)

  useEffect(() => {
    get<PinnTrainForm>('/pinn/train/form').then(setForm).catch(console.error)
  }, [])

  if (!form) {
    return (
      <div className="py-4">
        <p className="text-text-muted animate-flicker">Loading form...</p>
      </div>
    )
  }

  return (
    <div className="py-4">
      <h1 className="text-sm text-terminal-green tracking-wider mb-6">TRAIN MARKET PINN</h1>

      <form method="POST" action="/pinn/train" className="max-w-lg space-y-4">
        {form.fields.map((f) => (
          <div key={f.name} className="flex items-center gap-2 font-mono text-[13px]">
            <span className="text-terminal-green select-none">&gt;</span>
            <span className="text-text-muted shrink-0">{f.name}:</span>
            {f.type === 'select' ? (
              <select
                name={f.name}
                className="bg-transparent border-b border-screen-border text-text-primary outline-none px-1 py-0.5 focus:border-terminal-green transition-colors"
              >
                {f.options?.map((o) => (
                  <option key={o} value={o} className="bg-screen-bg">
                    {o}
                  </option>
                ))}
              </select>
            ) : (
              <input
                type={f.type}
                name={f.name}
                defaultValue={f.default}
                className="bg-transparent border-b border-screen-border text-text-primary outline-none px-1 py-0.5 focus:border-terminal-green transition-colors"
              />
            )}
          </div>
        ))}
        <Btn type="submit">Start Training</Btn>
      </form>
    </div>
  )
}