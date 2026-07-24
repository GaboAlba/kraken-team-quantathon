import { useEffect, useState } from 'react'
import './App.css'
import { fetchGrid, fetchSubgrid } from './api'
import GridMap from './components/GridMap'
import SubgridPanel from './components/SubgridPanel'
import type { GridPayload, RunRecord, SubgridInfo } from './types'

export default function App() {
  const [grid, setGrid] = useState<GridPayload | null>(null)
  const [selection, setSelection] = useState<string[]>([])
  const [subgrid, setSubgrid] = useState<SubgridInfo | null>(null)
  const [run, setRun] = useState<RunRecord | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchGrid()
      .then((g) => {
        setGrid(g)
        setSelection(g.nodes.filter((n) => n.is_initial).map((n) => n.id))
      })
      .catch((e: Error) => setError(e.message))
  }, [])

  useEffect(() => {
    if (selection.length === 0) return
    fetchSubgrid(selection)
      .then(setSubgrid)
      .catch((e: Error) => setError(e.message))
  }, [selection])

  if (error) return <div className="banner error">{error}</div>
  if (!grid) return <div className="banner">Loading grid…</div>

  return (
    <div className="layout">
      <div className="left">
        <GridMap
          grid={grid}
          selection={new Set(selection)}
          subgridEdges={subgrid?.edges ?? []}
        />
      </div>
      <div className="right">
        <SubgridPanel
          grid={grid}
          subgrid={subgrid}
          selection={selection}
          onAdd={(id) => {
            setSelection((s) => [...s, id])
            setRun(null)
          }}
          onRemove={(id) => {
            setSelection((s) => s.filter((x) => x !== id))
            setRun(null)
          }}
        />
        {run && <p>run: {run.status}</p>}
      </div>
    </div>
  )
}
