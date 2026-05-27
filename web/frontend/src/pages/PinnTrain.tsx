import { useEffect, useState } from 'react'
import { get } from '../api/client'
import type { PinnTrainForm } from '../types'

export function PinnTrain() {
  const [form, setForm] = useState<PinnTrainForm | null>(null)

  useEffect(() => {
    get<PinnTrainForm>('/pinn/train/form').then(setForm).catch(console.error)
  }, [])

  if (!form) return <p>Loading...</p>

  return (
    <>
      <h1>Train Market PINN</h1>
      <form method="POST" action="/pinn/train">
        {form.fields.map((f) => (
          <div key={f.name}>
            <label>
              {f.name}:{' '}
              {f.type === 'select' ? (
                <select name={f.name}>
                  {f.options?.map((o) => (
                    <option key={o} value={o}>
                      {o}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  type={f.type}
                  name={f.name}
                  defaultValue={f.default}
                />
              )}
            </label>
            <br />
          </div>
        ))}
        <button type="submit">Start Training</button>
      </form>
    </>
  )
}