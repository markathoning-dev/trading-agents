export function Modal({
  open,
  onClose,
  title,
  children,
}: {
  open: boolean
  onClose: () => void
  title: string
  children: React.ReactNode
}) {
  if (!open) return null

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/80 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="bg-screen-elevated border border-[#222] rounded-lg max-w-lg w-[90%] mt-20 p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="text-terminal-green text-sm font-mono border-b border-screen-border pb-3 mb-4">
          {title}
        </div>
        {children}
      </div>
    </div>
  )
}