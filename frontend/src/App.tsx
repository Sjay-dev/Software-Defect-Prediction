import { useCallback, useEffect, useState } from 'react'
import { api } from './api'
import { AppShell, type Screen } from './components/AppShell'
import { ErrorPanel, LoadingPanel } from './components/States'
import { ExperimentScreen } from './screens/ExperimentScreen'
import { ModelScreen } from './screens/ModelScreen'
import { PredictionScreen } from './screens/PredictionScreen'
import type { ExperimentResults, Health, Loadable, ModelInfo } from './types'

export default function App() {
  const [screen, setScreen] = useState<Screen>('prediction')
  const [health, setHealth] = useState<Loadable<Health>>({ state: 'loading' })
  const [model, setModel] = useState<Loadable<ModelInfo>>({ state: 'loading' })
  const [experiments, setExperiments] = useState<Loadable<ExperimentResults>>({ state: 'loading' })

  const loadCore = useCallback(() => {
    setHealth({ state: 'loading' })
    setModel({ state: 'loading' })
    void api.health().then((data) => setHealth({ state: 'success', data })).catch((error: Error) => setHealth({ state: 'error', message: error.message }))
    void api.modelInfo().then((data) => setModel({ state: 'success', data })).catch((error: Error) => setModel({ state: 'error', message: error.message }))
  }, [])

  const loadExperiments = useCallback(() => {
    setExperiments({ state: 'loading' })
    void api.experimentResults().then((data) => setExperiments({ state: 'success', data })).catch((error: Error) => setExperiments({ state: 'error', message: error.message }))
  }, [])

  useEffect(() => { loadCore(); loadExperiments() }, [loadCore, loadExperiments])

  return (
    <AppShell active={screen} onNavigate={setScreen} health={health}>
      {screen === 'experiments' ? (
        experiments.state === 'loading' ? <LoadingPanel label="Loading frozen experiment results…" /> : experiments.state === 'error' ? <ErrorPanel message={experiments.message} onRetry={loadExperiments} /> : <ExperimentScreen results={experiments.data} />
      ) : model.state === 'loading' ? (
        <LoadingPanel label="Loading model metadata…" />
      ) : model.state === 'error' ? (
        <ErrorPanel message={model.message} onRetry={loadCore} />
      ) : screen === 'model' ? (
        <ModelScreen model={model.data} />
      ) : (
        <PredictionScreen model={model.data} />
      )}
    </AppShell>
  )
}
