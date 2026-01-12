import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Zap, Eye, EyeOff, Loader2 } from 'lucide-react'
import { useAuthStore } from '../store'

export default function LoginPage() {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  
  const login = useAuthStore((state) => state.login)
  const navigate = useNavigate()
  
  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!username || !password) {
      setError('请输入用户名和密码')
      return
    }
    
    setLoading(true)
    setError('')
    
    const result = await login(username, password)
    
    setLoading(false)
    
    if (result.success) {
      navigate('/')
    } else {
      setError(result.error)
    }
  }
  
  return (
    <div className="min-h-screen bg-dark-950 bg-grid flex items-center justify-center p-4">
      {/* Background Effects */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-accent-500/10 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-flame-500/10 rounded-full blur-3xl" />
      </div>
      
      <div className="w-full max-w-md relative">
        {/* Logo */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl 
                          bg-gradient-to-br from-accent-500 to-accent-600 mb-4 
                          shadow-lg shadow-accent-500/30">
            <Zap className="w-8 h-8 text-white" />
          </div>
          <h1 className="font-display text-3xl font-bold gradient-text">PTFlow</h1>
          <p className="text-dark-400 mt-2">PT刷流管理系统</p>
        </div>
        
        {/* Login Form */}
        <div className="bg-dark-900/80 backdrop-blur-xl rounded-2xl border border-dark-800 p-8">
          <h2 className="text-xl font-semibold text-center mb-6">登录账号</h2>
          
          <form onSubmit={handleSubmit} className="space-y-5">
            <div>
              <label className="block text-sm text-dark-400 mb-2">用户名</label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="w-full px-4 py-3 bg-dark-800 border border-dark-700 rounded-xl
                           text-dark-100 placeholder-dark-500
                           focus:border-accent-500 focus:ring-1 focus:ring-accent-500/50
                           transition-all"
                placeholder="请输入用户名"
              />
            </div>
            
            <div>
              <label className="block text-sm text-dark-400 mb-2">密码</label>
              <div className="relative">
                <input
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full px-4 py-3 bg-dark-800 border border-dark-700 rounded-xl
                             text-dark-100 placeholder-dark-500 pr-12
                             focus:border-accent-500 focus:ring-1 focus:ring-accent-500/50
                             transition-all"
                  placeholder="请输入密码"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 p-1 
                             text-dark-400 hover:text-dark-200"
                >
                  {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                </button>
              </div>
            </div>
            
            {error && (
              <div className="px-4 py-3 bg-red-500/10 border border-red-500/30 rounded-xl">
                <p className="text-sm text-red-400">{error}</p>
              </div>
            )}
            
            <button
              type="submit"
              disabled={loading}
              className="w-full py-3 bg-gradient-to-r from-accent-500 to-accent-600 
                         text-white font-medium rounded-xl
                         hover:from-accent-600 hover:to-accent-700
                         focus:ring-2 focus:ring-accent-500/50 focus:ring-offset-2 
                         focus:ring-offset-dark-900
                         disabled:opacity-50 disabled:cursor-not-allowed
                         transition-all duration-200 flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>登录中...</span>
                </>
              ) : (
                <span>登 录</span>
              )}
            </button>
          </form>
          
          <div className="mt-6 pt-6 border-t border-dark-800">
            <p className="text-xs text-center text-dark-500">
              默认账号: admin / admin
            </p>
          </div>
        </div>
        
        {/* Footer */}
        <p className="text-center text-dark-500 text-sm mt-6">
          PTFlow v1.0.0 · 基于 qBittorrent
        </p>
      </div>
    </div>
  )
}
