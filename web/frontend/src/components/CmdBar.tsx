import { useState, useEffect } from 'react'
import { NavLink } from 'react-router-dom'

function useClock() {
  const [time, setTime] = useState(new Date().toLocaleTimeString())
  useEffect(() => {
    const id = setInterval(() => setTime(new Date().toLocaleTimeString()), 1000)
    return () => clearInterval(id)
  }, [])
  return time
}

const links = [
  { to: '/app', label: 'Dashboard' },
  { to: '/app/backtests', label: 'Backtests' },
  { to: '/app/backtests/new', label: 'New Run' },
  { to: '/app/models/compare', label: 'Compare' },
  { to: '/app/cards', label: 'Cards' },
  { to: '/app/decks', label: 'Decks' },
]

export function CmdBar() {
  const time = useClock()

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 h-10 bg-screen-elevated border-b border-screen-border flex items-center px-4 gap-0 font-mono">
      <span className="text-text-dim text-xs tracking-wider mr-4 select-none whitespace-nowrap">
        <span className="text-terminal-green">&#9632;</span> Trading Agent
      </span>
      {links.map((l) => (
        <NavLink
          key={l.to}
          to={l.to}
          end={l.to === '/app'}
          className={({ isActive }) =>
            `text-xs tracking-wide px-3 py-2.5 cursor-pointer transition-colors border-b-2 ${
              isActive
                ? 'text-terminal-green border-terminal-green'
                : 'text-text-muted border-transparent hover:text-text-primary'
            }`
          }
        >
          {l.label}
        </NavLink>
      ))}
      <div className="flex-1" />
      <span className="text-[10px] text-text-dim tabular-nums">
        {time}
      </span>
    </nav>
  )
}