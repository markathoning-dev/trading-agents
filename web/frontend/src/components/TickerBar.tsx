interface TickerItem {
  symbol: string
  price: number
  change: number
}

export function TickerBar({ items }: { items?: TickerItem[] }) {
  if (!items || items.length === 0) return null

  const doubled = [...items, ...items]

  return (
    <div className="fixed top-10 left-0 right-0 z-40 h-7 bg-[#080808] border-b border-screen-border overflow-hidden font-mono">
      <div className="flex animate-marquee whitespace-nowrap">
        {doubled.map((item, i) => (
          <span key={i} className="inline-flex items-center gap-1 text-[10px] mr-8">
            <span className="text-text-muted">{item.symbol}</span>
            <span className="text-text-primary tabular-nums">{item.price.toFixed(2)}</span>
            <span className={`tabular-nums ${item.change >= 0 ? 'text-terminal-green' : 'text-terminal-red'}`}>
              {item.change >= 0 ? '\u25B2' : '\u25BC'}{Math.abs(item.change).toFixed(2)}
            </span>
          </span>
        ))}
      </div>
    </div>
  )
}

export type { TickerItem }