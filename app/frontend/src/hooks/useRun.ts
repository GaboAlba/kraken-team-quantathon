import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchRun, startSimulation } from '../api'
import type { RunRecord } from '../types'

export function useRun() {
  const [run, setRun] = useState<RunRecord | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const timer = useRef<ReturnType<typeof setInterval> | null>(null)

  const stop = useCallback(() => {
    if (timer.current) clearInterval(timer.current)
    timer.current = null
  }, [])

  const start = useCallback(async (nodes: string[]) => {
    setError(null)
    setBusy(true)
    try {
      const { run_id } = await startSimulation(nodes)
      timer.current = setInterval(() => {
        fetchRun(run_id)
          .then((r) => {
            setRun(r)
            if (r.status !== 'running') {
              stop()
              setBusy(false)
            }
          })
          .catch((e: Error) => {
            setError(e.message)
            stop()
            setBusy(false)
          })
      }, 2000)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setBusy(false)
    }
  }, [stop])

  useEffect(() => stop, [stop])
  const reset = useCallback(() => { stop(); setRun(null); setBusy(false) }, [stop])
  return { run, start, error, busy, reset }
}
