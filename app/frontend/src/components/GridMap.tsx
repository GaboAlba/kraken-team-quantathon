import { CircleMarker, MapContainer, Polyline, TileLayer, Tooltip } from 'react-leaflet'
import type { GridEdge, GridPayload } from '../types'

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

export default function GridMap({ grid, selection, subgridEdges }: Props) {
  const byId = new Map(grid.nodes.map((n) => [n.id, n]))
  const subgridPairs = new Set(subgridEdges.map((e) => `${e.u}|${e.v}`))

  return (
    <MapContainer center={[9.9, -84.2]} zoom={8} className="map">
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
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
              weight: inSub ? 5 : 2,
              opacity: inSub ? 0.95 : 0.45,
            }}
          />
        )
      })}
      {grid.plants.map((p) => {
        const sub = p.substation ? byId.get(p.substation) : undefined
        const color = (p.technology && TECH_COLORS[p.technology]) ?? '#7f7f7f'
        return (
          <span key={p.name}>
            {sub && (
              <Polyline
                positions={[[p.lat, p.lon], [sub.lat, sub.lon]]}
                pathOptions={{ color, dashArray: '4 4', weight: 1.5 }}
              />
            )}
            <CircleMarker
              center={[p.lat, p.lon]}
              radius={4 + Math.sqrt(p.mw)}
              pathOptions={{ color: 'black', weight: 0.7, fillColor: color, fillOpacity: 0.85 }}
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
            weight: 0.8,
            fillColor: selection.has(n.id) ? '#2ca02c' : '#555555',
            fillOpacity: 0.95,
          }}
        >
          <Tooltip>{n.name}</Tooltip>
        </CircleMarker>
      ))}
    </MapContainer>
  )
}
