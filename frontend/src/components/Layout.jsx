import { useState, useEffect } from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import { 
  LayoutDashboard, 
  Download, 
  Server, 
  Globe, 
  Rss, 
  ListChecks,
  Gauge, 
  Settings,
  Menu,
  X,
  LogOut,
  ChevronLeft,
  ChevronRight,
  Zap
} from 'lucide-react'
import { useAuthStore, useUIStore, useStatsStore } from '../store'
import { formatSpeed } from '../utils/helpers'

const navItems = [
  { path: '/', icon: LayoutDashboard, label: '仪表盘' },
  { path: '/torrents', icon: Download, label: '种子管理' },
  { path: '/instances', icon: Server, label: 'qB实例' },
  { path: '/sites', icon: Globe, label: '站点配置' },
  { path: '/rss', icon: Rss, label: 'RSS订阅' },
  { path: '/tasks', icon: ListChecks, label: '任务规则' },
  { path: '/limit', icon: Gauge, label: '动态限速' },
  { path: '/settings', icon: Settings, label: '系统设置' },
]

function Sidebar({ collapsed, onToggle }) {
  const location = useLocation()
  const logout = useAuthStore((state) => state.logout)
  const username = useAuthStore((state) => state.username)
  
  return (
    <aside className={`
      fixed left-0 top-0 h-full bg-dark-900/95 border-r border-dark-800
      flex flex-col z-40 transition-all duration-300
      ${collapsed ? 'w-16' : 'w-64'}
    `}>
      {/* Logo */}
      <div className="h-16 flex items-center justify-between px-4 border-b border-dark-800">
        {!collapsed && (
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-accent-500 to-accent-600 flex items-center justify-center">
              <Zap className="w-5 h-5 text-white" />
            </div>
            <span className="font-display font-bold text-lg gradient-text">PTFlow</span>
          </div>
        )}
        <button
          onClick={onToggle}
          className="p-2 hover:bg-dark-800 rounded-lg transition-colors"
        >
          {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </button>
      </div>
      
      {/* Navigation */}
      <nav className="flex-1 py-4 px-2 space-y-1 overflow-y-auto">
        {navItems.map((item) => {
          const Icon = item.icon
          const isActive = location.pathname === item.path
          
          return (
            <NavLink
              key={item.path}
              to={item.path}
              className={`
                flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-200
                ${isActive 
                  ? 'bg-accent-500/20 text-accent-400 border border-accent-500/30' 
                  : 'text-dark-300 hover:bg-dark-800 hover:text-dark-100 border border-transparent'
                }
              `}
              title={collapsed ? item.label : undefined}
            >
              <Icon className={`w-5 h-5 flex-shrink-0 ${isActive ? 'text-accent-400' : ''}`} />
              {!collapsed && <span className="text-sm font-medium">{item.label}</span>}
            </NavLink>
          )
        })}
      </nav>
      
      {/* User Section */}
      <div className="p-4 border-t border-dark-800">
        {!collapsed && (
          <div className="mb-3 px-2">
            <p className="text-xs text-dark-500">登录账号</p>
            <p className="text-sm font-medium text-dark-200 truncate">{username}</p>
          </div>
        )}
        <button
          onClick={logout}
          className="w-full flex items-center justify-center gap-2 px-3 py-2 
                     text-dark-400 hover:text-red-400 hover:bg-red-500/10 
                     rounded-lg transition-colors"
          title={collapsed ? '退出登录' : undefined}
        >
          <LogOut className="w-4 h-4" />
          {!collapsed && <span className="text-sm">退出登录</span>}
        </button>
      </div>
    </aside>
  )
}

function MobileNav({ isOpen, onClose }) {
  const location = useLocation()
  const logout = useAuthStore((state) => state.logout)
  
  useEffect(() => {
    onClose()
  }, [location.pathname])
  
  if (!isOpen) return null
  
  return (
    <>
      {/* Backdrop */}
      <div 
        className="fixed inset-0 bg-black/60 z-40 lg:hidden"
        onClick={onClose}
      />
      
      {/* Menu */}
      <div className="fixed inset-y-0 left-0 w-64 bg-dark-900 z-50 lg:hidden animate-slide-up">
        <div className="h-16 flex items-center justify-between px-4 border-b border-dark-800">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-accent-500 to-accent-600 flex items-center justify-center">
              <Zap className="w-5 h-5 text-white" />
            </div>
            <span className="font-display font-bold text-lg gradient-text">PTFlow</span>
          </div>
          <button onClick={onClose} className="p-2 hover:bg-dark-800 rounded-lg">
            <X className="w-5 h-5" />
          </button>
        </div>
        
        <nav className="py-4 px-2 space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon
            const isActive = location.pathname === item.path
            
            return (
              <NavLink
                key={item.path}
                to={item.path}
                className={`
                  flex items-center gap-3 px-3 py-3 rounded-lg transition-colors
                  ${isActive 
                    ? 'bg-accent-500/20 text-accent-400' 
                    : 'text-dark-300 hover:bg-dark-800'
                  }
                `}
              >
                <Icon className="w-5 h-5" />
                <span className="font-medium">{item.label}</span>
              </NavLink>
            )
          })}
        </nav>
        
        <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-dark-800">
          <button
            onClick={logout}
            className="w-full flex items-center justify-center gap-2 px-3 py-2.5
                       text-red-400 bg-red-500/10 hover:bg-red-500/20
                       rounded-lg transition-colors"
          >
            <LogOut className="w-4 h-4" />
            <span>退出登录</span>
          </button>
        </div>
      </div>
    </>
  )
}

function TopBar({ onMenuClick }) {
  const stats = useStatsStore((state) => state.stats)
  
  return (
    <header className="h-16 bg-dark-900/80 backdrop-blur-lg border-b border-dark-800 
                       flex items-center justify-between px-4 sticky top-0 z-30">
      <button
        onClick={onMenuClick}
        className="lg:hidden p-2 hover:bg-dark-800 rounded-lg"
      >
        <Menu className="w-5 h-5" />
      </button>
      
      <div className="flex-1" />
      
      {/* Speed Indicators */}
      <div className="flex items-center gap-4 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-green-500 pulse-dot" />
          <span className="text-dark-400">↑</span>
          <span className="text-green-400 font-mono">
            {stats ? formatSpeed(stats.total_up_speed) : '0 B/s'}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-blue-500 pulse-dot" />
          <span className="text-dark-400">↓</span>
          <span className="text-blue-400 font-mono">
            {stats ? formatSpeed(stats.total_dl_speed) : '0 B/s'}
          </span>
        </div>
      </div>
    </header>
  )
}

export default function Layout({ children }) {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const fetchStats = useStatsStore((state) => state.fetchStats)
  
  useEffect(() => {
    fetchStats()
    const interval = setInterval(fetchStats, 5000)
    return () => clearInterval(interval)
  }, [fetchStats])
  
  return (
    <div className="min-h-screen bg-dark-950 bg-grid">
      {/* Desktop Sidebar */}
      <div className="hidden lg:block">
        <Sidebar 
          collapsed={sidebarCollapsed} 
          onToggle={() => setSidebarCollapsed(!sidebarCollapsed)} 
        />
      </div>
      
      {/* Mobile Navigation */}
      <MobileNav isOpen={mobileMenuOpen} onClose={() => setMobileMenuOpen(false)} />
      
      {/* Main Content */}
      <div className={`
        transition-all duration-300
        ${sidebarCollapsed ? 'lg:ml-16' : 'lg:ml-64'}
      `}>
        <TopBar onMenuClick={() => setMobileMenuOpen(true)} />
        <main className="p-4 lg:p-6">
          {children}
        </main>
      </div>
    </div>
  )
}
