import type { GridPayload, RunRecord, SubgridInfo } from './types'

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`${url}: ${res.status}`)
  return (await res.json()) as T
}

export function fetchGrid(): Promise<GridPayload> {
  return getJson<GridPayload>('/api/grid')
}

export function fetchSubgrid(nodes: string[]): Promise<SubgridInfo> {
  return getJson<SubgridInfo>(`/api/subgrid?nodes=${nodes.join(',')}`)
}

export async function startSimulation(
  nodes: string[],
): Promise<{ run_id: string }> {
  const res = await fetch('/api/simulate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ nodes }),
  })
  if (!res.ok) {
    const body = (await res.json()) as { detail: string }
    throw new Error(body.detail)
  }
  return (await res.json()) as { run_id: string }
}

export function fetchRun(runId: string): Promise<RunRecord> {
  return getJson<RunRecord>(`/api/runs/${runId}`)
}
