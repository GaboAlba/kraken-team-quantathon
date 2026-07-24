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
      <h2>Subgrid ({selection.length} nodes)</h2>
      {subgrid && !subgrid.valid && <p className="invalid">⚠ {subgrid.reason}</p>}
      <ul className="node-list">
        {selection.map((id) => (
          <li key={id}>
            {names.get(id) ?? id}
            {initials.has(id) ? (
              <span title="initial node — locked"> 🔒</span>
            ) : (
              <button onClick={() => onRemove(id)} title="remove">×</button>
            )}
          </li>
        ))}
      </ul>
      <h3>Adjacent candidates</h3>
      <ul className="node-list">
        {(subgrid?.adjacent ?? []).map((id) => (
          <li key={id}>
            {names.get(id) ?? id}
            <button onClick={() => onAdd(id)} title="add">+</button>
          </li>
        ))}
      </ul>
    </section>
  )
}
