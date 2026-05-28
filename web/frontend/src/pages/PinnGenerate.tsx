import { useEffect, useState } from 'react'
import { get } from '../api/client'
import type { PinnGenerateForm } from '../types'
import { Btn } from '../components/Btn'

export function PinnGenerate() {
  const [form, setForm] = useState<PinnGenerateForm | null>(null)

  useEffect(() => {
    get<PinnGenerateForm>('/pinn/generate/form').then(setForm).catch(console.error)
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
      <h1 className="text-sm text-terminal-green tracking-wider mb-6">GENERATE SYNTHETIC DATA</h1>

      <form method="POST" action="/pinn/generate" className="max-w-lg space-y-4">
        <div className="flex items-center gap-2 font-mono text-[13px]">
          <span className="text-terminal-green select-none">&gt;</span>
          <span className="text-text-muted shrink-0">model:</span>
          <select
            name="model_id"
            className="bg-transparent border-b border-screen-border text-text-primary outline-none px-1 py-0.5 focus:border-terminal-green transition-colors"
          >
            {form.models.map((m) => (
              <option key={m.id} value={m.id} className="bg-screen-bg">
                {m.name} ({m.pde_type})
              </option>
            ))}
          </select>
        </div>
        {form.fields.map((f) => (
          <div key={f.name} className="flex items-center gap-2 font-mono text-[13px]">
            <span className="text-terminal-green select-none">&gt;</span>
            <span className="text-text-muted shrink-0">{f.name}:</span>
            <input
              type={f.type}
              name={f.name}
              defaultValue={f.default}
              className="bg-transparent border-b border-screen-border text-text-primary outline-none px-1 py-0.5 focus:border-terminal-green transition-colors"
            />
          </div>
        ))}
        <Btn type="submit">Generate</Btn>
      </form>
    </div>
  )
}