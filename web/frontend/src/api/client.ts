const BASE = '/api'

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, {
    headers: { 'Accept': 'application/json', ...options?.headers },
    ...options,
  })
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`${res.status} ${res.statusText}: ${body}`)
  }
  return res.json()
}

export function get<T>(url: string): Promise<T> {
  return request<T>(url)
}

export function postForm<T>(url: string, data: Record<string, string | number>): Promise<T> {
  const body = new URLSearchParams()
  for (const [k, v] of Object.entries(data)) {
    body.append(k, String(v))
  }
  return request<T>(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body,
  })
}