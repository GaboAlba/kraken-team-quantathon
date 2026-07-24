import { useCallback, useEffect, useRef, useState } from 'react'
import { cancelRun, fetchRun, startSimulation } from '../api'
import type { RunRecord } from '../types'

export function useRun() {
  const [run, setRun] = useState<RunRecord | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const timer = useRef<ReturnType<typeof setInterval> | null>(null)
  const runId = useRef<string | null>(null)
  const gen = useRef(0)

  const stop = useCallback(() => {
    if (timer.current) clearInterval(timer.current)
    timer.current = null
  }, [])

  const start = useCallback(async (nodes: string[]) => {
    gen.current += 1
    const myGen = gen.current
    setError(null)
    setBusy(true)
    try {
      const { run_id } = await startSimulation(nodes)
      runId.current = run_id
      if (gen.current !== myGen) return
      timer.current = setInterval(() => {
        if (gen.current !== myGen) {
          stop()
          return
        }
        fetchRun(run_id)
          .then((r) => {
            if (gen.current !== myGen) return
            setRun(r)
            if (r.status !== 'running') {
              stop()
              setBusy(false)
            }
          })
          .catch((e: Error) => {
            if (gen.current !== myGen) return
            setError(e.message)
            stop()
            setBusy(false)
          })
      }, 2000)
    } catch (e) {
      if (gen.current !== myGen) return
      setError(e instanceof Error ? e.message : String(e))
      setBusy(false)
    }
  }, [stop])

  const cancel = useCallback(async () => {
    if (runId.current === null) return
    try {
      await cancelRun(runId.current)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [])

  useEffect(() => stop, [stop])
  const reset = useCallback(() => {
    gen.current += 1
    stop()
    setRun(null)
    setBusy(false)
  }, [stop])
  return { run, start, cancel, error, busy, reset }
}
