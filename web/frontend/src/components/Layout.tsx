import { NavLink } from 'react-router-dom'

const links = [
  { to: '/app', label: 'Dashboard' },
  { to: '/app/backtests', label: 'Backtests' },
  { to: '/app/backtests/new', label: 'New Run' },
  { to: '/app/models/compare', label: 'Compare Models' },
  { to: '/app/pinn/train', label: 'Train PINN' },
]

export function Layout({ children }: { children: React.ReactNode }) {
  return (
    <>
      <nav>
        {links.map((l) => (
          <NavLink key={l.to} to={l.to} end>
            {l.label}
          </NavLink>
        ))}
      </nav>
      <main>{children}</main>
    </>
  )
}