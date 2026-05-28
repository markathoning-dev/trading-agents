import { ReactNode, useState } from 'react'
import { Sparkline } from './Sparkline'

interface Column<T> {
  key: string
  label: string
  align?: 'left' | 'right'
  render?: (row: T) => ReactNode
  sparkline?: (row: T) => number[] | undefined
  colorize?: boolean
}

interface TerminalTableProps<T> {
  columns: Column<T>[]
  rows: T[]
  onRowClick?: (row: T) => void
  expandable?: {
    render: (row: T) => ReactNode
  }
  emptyMessage?: string
  sortable?: boolean
  defaultSortKey?: string
  defaultSortDir?: 'asc' | 'desc'
}

export function TerminalTable<T>({
  columns,
  rows,
  onRowClick,
  expandable,
  emptyMessage = 'No data',
  sortable,
  defaultSortKey,
  defaultSortDir = 'desc',
}: TerminalTableProps<T>) {
  const [sortKey, setSortKey] = useState<string | undefined>(defaultSortKey)
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>(defaultSortDir)
  const [expanded, setExpanded] = useState<number | null>(null)

  const handleSort = (key: string) => {
    if (!sortable) return
    if (sortKey === key) {
      setSortDir(sortDir === 'asc' ? 'desc' : 'asc')
    } else {
      setSortKey(key)
      setSortDir('asc')
    }
  }

  const sorted = [...rows]
  if (sortable && sortKey) {
    sorted.sort((a, b) => {
      const aVal = (a as Record<string, unknown>)[sortKey]
      const bVal = (b as Record<string, unknown>)[sortKey]
      const aNum = typeof aVal === 'number' ? aVal : Number(aVal)
      const bNum = typeof bVal === 'number' ? bVal : Number(bVal)
      const cmp = !isNaN(aNum) && !isNaN(bNum) ? aNum - bNum : String(aVal).localeCompare(String(bVal))
      return sortDir === 'asc' ? cmp : -cmp
    })
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full font-mono text-[13px]">
        <thead>
          <tr className="border-b-2 border-screen-border">
            {columns.map((col) => (
              <th
                key={col.key}
                onClick={() => handleSort(col.key)}
                className={`text-left text-[11px] text-text-muted uppercase tracking-[0.2em] py-2.5 px-3 select-none ${
                  sortable ? 'cursor-pointer hover:text-text-primary' : ''
                } ${col.align === 'right' ? 'text-right' : ''}`}
              >
                {col.label}{' '}
                {sortable && sortKey === col.key && (
                  <span className="text-terminal-green">{sortDir === 'asc' ? '\u2191' : '\u2193'}</span>
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.length === 0 ? (
            <tr>
              <td colSpan={columns.length} className="py-8 text-center text-text-muted text-[13px]">
                {emptyMessage}
              </td>
            </tr>
          ) : (
            sorted.map((row, i) => (
              <>
                <tr
                  key={i}
                  onClick={() => {
                    if (expandable) setExpanded(expanded === i ? null : i)
                    onRowClick?.(row)
                  }}
                  className={`border-b border-[#111] transition-colors ${
                    onRowClick || expandable ? 'cursor-pointer hover:bg-[#0d0d0d]' : ''
                  } ${expanded === i ? 'bg-[#0c0c0c]' : ''}`}
                >
                  {columns.map((col) => {
                    const raw = (row as Record<string, unknown>)[col.key]
                    const sparklineData = col.sparkline?.(row)

                    if (col.render) {
                      return (
                        <td key={col.key} className={`py-2 px-3 ${col.align === 'right' ? 'text-right' : ''}`}>
                          {col.render(row)}
                        </td>
                      )
                    }

                    if (col.colorize && raw != null) {
                      const num = typeof raw === 'number' ? raw : parseFloat(String(raw))
                      const isPositive = !isNaN(num) && num >= 0
                      return (
                        <td
                          key={col.key}
                          className={`py-2 px-3 tabular-nums ${col.align === 'right' ? 'text-right' : ''} ${
                            isNaN(num) ? '' : isPositive ? 'text-terminal-green' : 'text-terminal-red'
                          }`}
                        >
                          {sparklineData && sparklineData.length > 0 && (
                            <Sparkline data={sparklineData} className="inline-block mr-2 align-middle" />
                          )}
                          {String(raw)}
                        </td>
                      )
                    }

                    return (
                      <td key={col.key} className={`py-2 px-3 tabular-nums ${col.align === 'right' ? 'text-right' : ''}`}>
                        {raw != null ? String(raw) : '\u2014'}
                      </td>
                    )
                  })}
                </tr>
                {expandable && expanded === i && (
                  <tr className="bg-[#0c0c0c] animate-slide-down">
                    <td colSpan={columns.length} className="px-4 py-3">
                      {expandable.render(row)}
                    </td>
                  </tr>
                )}
              </>
            ))
          )}
        </tbody>
      </table>
    </div>
  )
}