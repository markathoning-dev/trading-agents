import { CmdBar } from './CmdBar'

export function Layout({ children }: { children: React.ReactNode }) {
  return (
    <>
      <CmdBar />
      <main className="pt-10 px-4 pb-8 max-w-7xl mx-auto">{children}</main>
    </>
  )
}