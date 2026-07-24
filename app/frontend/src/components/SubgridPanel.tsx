import type { GridPayload, SubgridInfo } from '../types'

interface Props {
  grid: GridPayload
  subgrid: SubgridInfo | null
  selection: string[]
  onAdd: (id: string) => void
  onRemove: (id: string) => void
}

export default function SubgridPanel({ grid, subgrid, selection, onAdd, onRemove }: Props) {
  const initials = new Set(grid.nodes.filter((n) => n.is_initial).map((n) => n.id))
  const names = new Map(grid.nodes.map((n) => [n.id, n.name]))

  return (
    <section>
      <h2 className="section-title">
        Subgrid <span className="count">{selection.length} nodes</span>
        {subgrid && <span className={`tier tier-${subgrid.tier}`}>{subgrid.tier}</span>}
      </h2>
      {subgrid && !subgrid.valid && <p className="invalid">⚠ {subgrid.reason}</p>}
      <ul className="node-list">
        {selection.map((id) => (
          <li key={id}>
            {names.get(id) ?? id}
            {initials.has(id) ? (
              <span className="lock" title="initial node — locked">🔒</span>
            ) : (
              <button
                className="icon-btn"
                onClick={() => onRemove(id)}
                aria-label={`remove ${names.get(id) ?? id}`}
                title="remove"
              >
                ×
              </button>
            )}
          </li>
        ))}
      </ul>
      <h2 className="section-title">Adjacent candidates</h2>
      <div className="chips">
        {(subgrid?.adjacent ?? []).map((id) => (
          <button
            key={id}
            className="chip"
            onClick={() => onAdd(id)}
            aria-label={`add ${names.get(id) ?? id}`}
          >
            {names.get(id) ?? id}
          </button>
        ))}
      </div>
    </section>
  )
}
