import { useEffect, useState } from 'react'
import './App.css'
import { fetchGrid, fetchSubgrid } from './api'
import { useRun } from './hooks/useRun'
import GridMap from './components/GridMap'
import SimulationPanel from './components/SimulationPanel'
import SubgridPanel from './components/SubgridPanel'
import ResultsPanel from './components/ResultsPanel'
import InfoModal from './components/InfoModal'
import type { GridPayload, SubgridInfo } from './types'

export default function App() {
  const [grid, setGrid] = useState<GridPayload | null>(null)
  const [selection, setSelection] = useState<string[]>([])
  const [subgrid, setSubgrid] = useState<SubgridInfo | null>(null)
  const { run, start, cancel, error: runError, busy, reset } = useRun()
  const [error, setError] = useState<string | null>(null)
  const [showInfo, setShowInfo] = useState(false)

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
    <div className="shell">
      <header className="topbar">
        <span className="brand">
          Grid<span className="accent">·</span>Simulator
        </span>
        <span className="sub">Costa Rica transmission grid — fault-zone QUBO / QAOA</span>
        <span className="spacer" />
        <button className="info-btn" onClick={() => setShowInfo(true)}>info</button>
        <span className="telemetry">
          <b>{grid.nodes.length}</b> substations · <b>{grid.edges.length}</b> lines ·{' '}
          <b>{grid.plants.length}</b> plants · subgrid <b>{selection.length}</b>
        </span>
      </header>
      <div className="layout">
        <div className="left">
          <GridMap
            grid={grid}
            selection={new Set(selection)}
            subgridEdges={subgrid?.edges ?? []}
            partition={run?.results?.best_partition ?? null}
          />
        </div>
        <div className="right">
          <SubgridPanel
            grid={grid}
            subgrid={subgrid}
            selection={selection}
            onAdd={(id) => {
              setSelection((s) => [...s, id])
              reset()
            }}
            onRemove={(id) => {
              setSelection((s) => s.filter((x) => x !== id))
              reset()
            }}
          />
          <SimulationPanel
            disabled={!subgrid?.valid}
            busy={busy}
            run={run}
            error={runError}
            onRun={() => { void start(selection) }}
            onStop={() => { void cancel() }}
          />
          {run?.results && <ResultsPanel results={run.results} run={run} />}
        </div>
      </div>
      {showInfo && <InfoModal onClose={() => setShowInfo(false)} />}
    </div>
  )
}
