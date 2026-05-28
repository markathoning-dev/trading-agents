export function StatusBadge({ status }: { status: string }) {
  const isRunning = status === 'running'
  const isCompleted = status === 'completed'
  const isFailed = status === 'failed'

  return (
    <span className="inline-flex items-center gap-1.5 text-[11px] uppercase tracking-wider">
      <span
        className={`w-1.5 h-1.5 rounded-full ${
          isRunning
            ? 'bg-terminal-green animate-pulse'
            : isCompleted
              ? 'bg-terminal-cyan'
              : isFailed
                ? 'bg-terminal-red'
                : 'bg-text-muted'
        }`}
      />
      <span className={isFailed ? 'text-terminal-red' : isCompleted ? 'text-terminal-cyan' : 'text-text-muted'}>
        {status}
      </span>
    </span>
  )
}