import { useEffect, useState } from 'react'
import { get } from '../api/client'
import type { PinnGenerateForm } from '../types'

export function PinnGenerate() {
  const [form, setForm] = useState<PinnGenerateForm | null>(null)

  useEffect(() => {
    get<PinnGenerateForm>('/pinn/generate/form').then(setForm).catch(console.error)
  }, [])

  if (!form) return <p>Loading...</p>

  return (
    <>
      <h1>Generate Synthetic Market Data</h1>
      <form method="POST" action="/pinn/generate">
        <label>
          Model:{' '}
          <select name="model_id">
            {form.models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name} ({m.pde_type})
              </option>
            ))}
          </select>
        </label>
        <br />
        {form.fields.map((f) => (
          <div key={f.name}>
            <label>
              {f.name}: <input type={f.type} name={f.name} defaultValue={f.default} />
            </label>
            <br />
          </div>
        ))}
        <button type="submit">Generate</button>
      </form>
    </>
  )
}