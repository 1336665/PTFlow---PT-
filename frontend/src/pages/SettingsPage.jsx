import { useState } from 'react'
import { 
  Settings, 
  Lock, 
  Bell, 
  Database,
  RefreshCw,
  Check,
  Info,
  ExternalLink
} from 'lucide-react'
import { useAuthStore } from '../store'
import api from '../api'

function SettingSection({ icon: Icon, title, description, children }) {
  return (
    <div className="bg-dark-900/50 border border-dark-800 rounded-2xl p-5">
      <div className="flex items-start gap-3 mb-4">
        <div className="p-2.5 rounded-xl bg-dark-800 text-dark-400">
          <Icon className="w-5 h-5" />
        </div>
        <div>
          <h3 className="font-semibold">{title}</h3>
          {description && <p className="text-sm text-dark-500 mt-0.5">{description}</p>}
        </div>
      </div>
      {children}
    </div>
  )
}

export default function SettingsPage() {
  const username = useAuthStore((state) => state.username)
  
  const [passwordForm, setPasswordForm] = useState({
    old_password: '',
    new_password: '',
    confirm_password: ''
  })
  const [passwordLoading, setPasswordLoading] = useState(false)
  const [passwordMessage, setPasswordMessage] = useState({ type: '', text: '' })
  
  const handleChangePassword = async (e) => {
    e.preventDefault()
    
    if (passwordForm.new_password !== passwordForm.confirm_password) {
      setPasswordMessage({ type: 'error', text: '两次输入的密码不一致' })
      return
    }
    
    if (passwordForm.new_password.length < 4) {
      setPasswordMessage({ type: 'error', text: '密码长度至少4位' })
      return
    }
    
    setPasswordLoading(true)
    setPasswordMessage({ type: '', text: '' })
    
    try {
      await api.post('/api/auth/change-password', {
        old_password: passwordForm.old_password,
        new_password: passwordForm.new_password
      })
      setPasswordMessage({ type: 'success', text: '密码修改成功' })
      setPasswordForm({ old_password: '', new_password: '', confirm_password: '' })
    } catch (err) {
      setPasswordMessage({ type: 'error', text: err.response?.data?.detail || '修改失败' })
    }
    
    setPasswordLoading(false)
  }
  
  return (
    <div className="space-y-6 animate-in">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-display font-bold">系统设置</h1>
        <p className="text-dark-400 mt-1">管理账户和系统配置</p>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Account Info */}
        <SettingSection
          icon={Info}
          title="账户信息"
          description="当前登录账户"
        >
          <div className="space-y-3">
            <div className="flex items-center justify-between p-3 bg-dark-800/50 rounded-xl">
              <span className="text-dark-400">用户名</span>
              <span className="font-mono">{username}</span>
            </div>
            <div className="flex items-center justify-between p-3 bg-dark-800/50 rounded-xl">
              <span className="text-dark-400">系统版本</span>
              <span className="font-mono">PTFlow v1.0.0</span>
            </div>
          </div>
        </SettingSection>
        
        {/* Change Password */}
        <SettingSection
          icon={Lock}
          title="修改密码"
          description="更新您的登录密码"
        >
          <form onSubmit={handleChangePassword} className="space-y-4">
            <div>
              <label className="block text-sm text-dark-400 mb-2">当前密码</label>
              <input
                type="password"
                value={passwordForm.old_password}
                onChange={(e) => setPasswordForm({ ...passwordForm, old_password: e.target.value })}
                className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl
                           text-dark-100 placeholder-dark-500
                           focus:border-accent-500 focus:ring-1 focus:ring-accent-500/50"
                placeholder="输入当前密码"
              />
            </div>
            <div>
              <label className="block text-sm text-dark-400 mb-2">新密码</label>
              <input
                type="password"
                value={passwordForm.new_password}
                onChange={(e) => setPasswordForm({ ...passwordForm, new_password: e.target.value })}
                className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl
                           text-dark-100 placeholder-dark-500
                           focus:border-accent-500 focus:ring-1 focus:ring-accent-500/50"
                placeholder="输入新密码"
              />
            </div>
            <div>
              <label className="block text-sm text-dark-400 mb-2">确认新密码</label>
              <input
                type="password"
                value={passwordForm.confirm_password}
                onChange={(e) => setPasswordForm({ ...passwordForm, confirm_password: e.target.value })}
                className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl
                           text-dark-100 placeholder-dark-500
                           focus:border-accent-500 focus:ring-1 focus:ring-accent-500/50"
                placeholder="再次输入新密码"
              />
            </div>
            
            {passwordMessage.text && (
              <div className={`
                px-4 py-3 rounded-xl text-sm
                ${passwordMessage.type === 'error' 
                  ? 'bg-red-500/10 border border-red-500/30 text-red-400' 
                  : 'bg-accent-500/10 border border-accent-500/30 text-accent-400'}
              `}>
                {passwordMessage.text}
              </div>
            )}
            
            <button
              type="submit"
              disabled={passwordLoading}
              className="w-full py-2.5 bg-accent-500 hover:bg-accent-600 text-white rounded-xl 
                         transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
            >
              {passwordLoading ? (
                <RefreshCw className="w-4 h-4 animate-spin" />
              ) : (
                <Check className="w-4 h-4" />
              )}
              修改密码
            </button>
          </form>
        </SettingSection>
        
        {/* About */}
        <SettingSection
          icon={Database}
          title="关于 PTFlow"
          description="PT刷流管理系统"
        >
          <div className="space-y-4">
            <p className="text-sm text-dark-400">
              PTFlow 是一个现代化的PT刷流管理系统，支持多qBittorrent实例管理、
              智能动态限速、RSS自动订阅等功能。
            </p>
            
            <div className="space-y-2">
              <div className="flex items-center justify-between p-3 bg-dark-800/50 rounded-xl">
                <span className="text-dark-400">后端框架</span>
                <span className="text-sm">FastAPI + Python</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-dark-800/50 rounded-xl">
                <span className="text-dark-400">前端框架</span>
                <span className="text-sm">React + TailwindCSS</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-dark-800/50 rounded-xl">
                <span className="text-dark-400">限速算法</span>
                <span className="text-sm">PID + Kalman Filter</span>
              </div>
            </div>
            
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center justify-center gap-2 w-full py-2.5 
                         bg-dark-800 hover:bg-dark-700 text-dark-300 rounded-xl transition-colors"
            >
              <ExternalLink className="w-4 h-4" />
              访问 GitHub
            </a>
          </div>
        </SettingSection>
        
        {/* Features */}
        <SettingSection
          icon={Bell}
          title="功能特性"
          description="系统核心功能"
        >
          <div className="space-y-3">
            <div className="flex items-start gap-3 p-3 bg-dark-800/50 rounded-xl">
              <div className="w-2 h-2 rounded-full bg-accent-400 mt-1.5" />
              <div>
                <p className="text-sm font-medium">多实例管理</p>
                <p className="text-xs text-dark-500">支持同时管理多个qBittorrent客户端</p>
              </div>
            </div>
            <div className="flex items-start gap-3 p-3 bg-dark-800/50 rounded-xl">
              <div className="w-2 h-2 rounded-full bg-blue-400 mt-1.5" />
              <div>
                <p className="text-sm font-medium">智能限速</p>
                <p className="text-xs text-dark-500">基于PID和卡尔曼滤波的动态限速算法</p>
              </div>
            </div>
            <div className="flex items-start gap-3 p-3 bg-dark-800/50 rounded-xl">
              <div className="w-2 h-2 rounded-full bg-orange-400 mt-1.5" />
              <div>
                <p className="text-sm font-medium">RSS订阅</p>
                <p className="text-xs text-dark-500">自动下载新种子，支持过滤规则</p>
              </div>
            </div>
            <div className="flex items-start gap-3 p-3 bg-dark-800/50 rounded-xl">
              <div className="w-2 h-2 rounded-full bg-purple-400 mt-1.5" />
              <div>
                <p className="text-sm font-medium">站点配置</p>
                <p className="text-xs text-dark-500">按站点独立配置限速规则</p>
              </div>
            </div>
          </div>
        </SettingSection>
      </div>
    </div>
  )
}
