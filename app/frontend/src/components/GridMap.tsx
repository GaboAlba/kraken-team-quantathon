import { useEffect, useMemo } from 'react'
import { latLngBounds } from 'leaflet'
import { CircleMarker, MapContainer, Polyline, TileLayer, Tooltip, useMap } from 'react-leaflet'
import type { GridEdge, GridNode, GridPayload } from '../types'

const TECH_COLORS: Record<string, string> = {
  'Hidroeléctrico': '#1f77b4',
  'Geotérmico': '#d62728',
  'Eólico': '#2ca02c',
  Solar: '#ff7f0e',
  'Térmico': '#8c564b',
}

interface Props {
  grid: GridPayload
  selection: Set<string>
  subgridEdges: GridEdge[]
}

/** Smoothly refits the viewport to the active subgrid whenever it changes. */
function FitToSelection({ points }: { points: Array<[number, number]> }) {
  const map = useMap()
  useEffect(() => {
    if (points.length === 0) return
    // pad(0.2): frame ~20% more area around the subgrid (gentler zoom).
    map.flyToBounds(latLngBounds(points).pad(0.2), {
      padding: [70, 70],
      maxZoom: 10.5,
      duration: 0.9,
    })
  }, [map, points])
  return null
}

export default function GridMap({ grid, selection, subgridEdges }: Props) {
  const byId = useMemo(
    () => new Map<string, GridNode>(grid.nodes.map((n) => [n.id, n])),
    [grid.nodes],
  )
  const subgridPairs = new Set(subgridEdges.map((e) => `${e.u}|${e.v}`))

  // Stable key so the viewport refit only fires on real selection changes,
  // not on every poll-driven re-render.
  const selectionKey = Array.from(selection).sort().join(',')
  const selectedPoints = useMemo(
    () =>
      selectionKey
        .split(',')
        .filter(Boolean)
        .flatMap((id): Array<[number, number]> => {
          const n = byId.get(id)
          return n ? [[n.lat, n.lon]] : []
        }),
    [selectionKey, byId],
  )

  return (
    <MapContainer center={[9.9, -84.2]} zoom={8} zoomSnap={0.25} zoomDelta={0.5} className="map">
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
        opacity={0.6}
        className="base-tiles"
      />
      <FitToSelection points={selectedPoints} />
      {grid.edges.map((e) => {
        const a = byId.get(e.u)
        const b = byId.get(e.v)
        if (!a || !b) return null
        const inSub = subgridPairs.has(`${e.u}|${e.v}`) || subgridPairs.has(`${e.v}|${e.u}`)
        return (
          <Polyline
            key={`${e.u}-${e.v}`}
            positions={[[a.lat, a.lon], [b.lat, b.lon]]}
            pathOptions={{
              color: e.voltage === 230 ? '#d62728' : '#1f77b4',
              weight: inSub ? 5 : 2.2,
              opacity: inSub ? 0.95 : 0.55,
            }}
          />
        )
      })}
      {grid.plants.map((p) => {
        const sub = p.substation ? byId.get(p.substation) : undefined
        const color = (p.technology && TECH_COLORS[p.technology]) ?? '#7f7f7f'
        const active = p.substation !== null && selection.has(p.substation)
        return (
          <span key={p.name}>
            {sub && (
              <Polyline
                positions={[[p.lat, p.lon], [sub.lat, sub.lon]]}
                pathOptions={{
                  color,
                  dashArray: '4 4',
                  weight: active ? 1.5 : 1,
                  opacity: active ? 0.9 : 0.35,
                }}
              />
            )}
            <CircleMarker
              center={[p.lat, p.lon]}
              radius={active ? 4 + Math.sqrt(p.mw) : 2.5 + Math.sqrt(p.mw) * 0.6}
              pathOptions={{
                color: 'black',
                weight: active ? 0.7 : 0.4,
                opacity: active ? 1 : 0.55,
                fillColor: color,
                fillOpacity: active ? 0.85 : 0.45,
              }}
            >
              <Tooltip>{`${p.name} — ${p.technology ?? '?'} (${p.mw} MW)`}</Tooltip>
            </CircleMarker>
          </span>
        )
      })}
      {grid.nodes.map((n) => (
        <CircleMarker
          key={n.id}
          center={[n.lat, n.lon]}
          radius={selection.has(n.id) ? 9 : 5}
          pathOptions={{
            color: 'black',
            weight: selection.has(n.id) ? 0.8 : 0.6,
            opacity: selection.has(n.id) ? 1 : 0.85,
            fillColor: selection.has(n.id) ? '#2ca02c' : '#7d8d99',
            fillOpacity: selection.has(n.id) ? 0.95 : 0.8,
          }}
        >
          <Tooltip>{n.name}</Tooltip>
        </CircleMarker>
      ))}
    </MapContainer>
  )
}
