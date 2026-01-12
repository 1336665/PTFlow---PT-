import { useEffect } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { useAuthStore } from './store'
import Layout from './components/Layout'
import LoginPage from './pages/LoginPage'
import DashboardPage from './pages/DashboardPage'
import TorrentsPage from './pages/TorrentsPage'
import InstancesPage from './pages/InstancesPage'
import SitesPage from './pages/SitesPage'
import RSSPage from './pages/RSSPage'
import TasksPage from './pages/TasksPage'
import LimitPage from './pages/LimitPage'
import SettingsPage from './pages/SettingsPage'

function ProtectedRoute({ children }) {
  const isAuthenticated = useAuthStore((state) => state.isAuthenticated)
  
  if (!isAuthenticated) {
    return <Navigate to="/login" replace />
  }
  
  return children
}

function App() {
  const initAuth = useAuthStore((state) => state.initAuth)
  
  useEffect(() => {
    initAuth()
  }, [initAuth])
  
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route
          path="/*"
          element={
            <ProtectedRoute>
              <Layout>
                <Routes>
                  <Route path="/" element={<DashboardPage />} />
                  <Route path="/torrents" element={<TorrentsPage />} />
                  <Route path="/instances" element={<InstancesPage />} />
                  <Route path="/sites" element={<SitesPage />} />
                  <Route path="/rss" element={<RSSPage />} />
                  <Route path="/tasks" element={<TasksPage />} />
                  <Route path="/limit" element={<LimitPage />} />
                  <Route path="/settings" element={<SettingsPage />} />
                </Routes>
              </Layout>
            </ProtectedRoute>
          }
        />
      </Routes>
    </BrowserRouter>
  )
}

export default App
