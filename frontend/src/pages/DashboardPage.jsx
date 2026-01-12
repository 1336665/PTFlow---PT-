import { useEffect, useState } from 'react'
import { 
  Server, 
  Globe, 
  Rss, 
  Download, 
  Upload, 
  Activity,
  TrendingUp,
  Gauge,
  Zap,
  ArrowUp,
  ArrowDown
} from 'lucide-react'
import { AreaChart, Area, XAxis, YAxis, ResponsiveContainer, Tooltip } from 'recharts'
import { useStatsStore, useQBStore, useSiteStore } from '../store'
import { formatSpeed, formatSize, getPhaseInfo } from '../utils/helpers'

function StatCard({ icon: Icon, label, value, subValue, color, trend }) {
  const colorClasses = {
    green: 'from-accent-500/20 to-accent-600/10 border-accent-500/30',
    blue: 'from-blue-500/20 to-blue-600/10 border-blue-500/30',
    orange: 'from-flame-500/20 to-flame-600/10 border-flame-500/30',
    purple: 'from-purple-500/20 to-purple-600/10 border-purple-500/30'
  }
  
  const iconColors = {
    green: 'text-accent-400',
    blue: 'text-blue-400',
    orange: 'text-flame-400',
    purple: 'text-purple-400'
  }
  
  return (
    <div className={`
      bg-gradient-to-br ${colorClasses[color]} 
      border rounded-2xl p-5 card-hover
    `}>
      <div className="flex items-start justify-between">
        <div className={`p-2.5 rounded-xl bg-dark-900/50 ${iconColors[color]}`}>
          <Icon className="w-5 h-5" />
        </div>
        {trend && (
          <div className={`flex items-center gap-1 text-xs ${trend > 0 ? 'text-accent-400' : 'text-red-400'}`}>
            {trend > 0 ? <TrendingUp className="w-3 h-3" /> : <TrendingUp className="w-3 h-3 rotate-180" />}
            <span>{Math.abs(trend)}%</span>
          </div>
        )}
      </div>
      <div className="mt-4">
        <p className="text-2xl font-bold font-mono">{value}</p>
        <p className="text-sm text-dark-400 mt-1">{label}</p>
        {subValue && <p className="text-xs text-dark-500 mt-0.5">{subValue}</p>}
      </div>
    </div>
  )
}

function SpeedChart({ data }) {
  return (
    <div className="bg-dark-900/50 border border-dark-800 rounded-2xl p-5">
      <h3 className="font-semibold mb-4 flex items-center gap-2">
        <Activity className="w-5 h-5 text-accent-400" />
        实时速度
      </h3>
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="uploadGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#22c55e" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#22c55e" stopOpacity={0}/>
              </linearGradient>
              <linearGradient id="downloadGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <XAxis 
              dataKey="time" 
              axisLine={false} 
              tickLine={false}
              tick={{ fill: '#666680', fontSize: 11 }}
            />
            <YAxis 
              axisLine={false} 
              tickLine={false}
              tick={{ fill: '#666680', fontSize: 11 }}
              tickFormatter={(v) => formatSpeed(v)}
              width={60}
            />
            <Tooltip 
              contentStyle={{ 
                background: '#1a1a24', 
                border: '1px solid #434356',
                borderRadius: '8px'
              }}
              formatter={(value) => [formatSpeed(value)]}
            />
            <Area 
              type="monotone" 
              dataKey="upload" 
              stroke="#22c55e" 
              strokeWidth={2}
              fill="url(#uploadGrad)" 
              name="上传"
            />
            <Area 
              type="monotone" 
              dataKey="download" 
              stroke="#3b82f6" 
              strokeWidth={2}
              fill="url(#downloadGrad)" 
              name="下载"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <div className="flex items-center justify-center gap-6 mt-4 text-sm">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-accent-500" />
          <span className="text-dark-400">上传速度</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-blue-500" />
          <span className="text-dark-400">下载速度</span>
        </div>
      </div>
    </div>
  )
}

function LimitStatesList({ states }) {
  if (!states || states.length === 0) {
    return (
      <div className="text-center py-8 text-dark-500">
        暂无正在限速的种子
      </div>
    )
  }
  
  return (
    <div className="space-y-3">
      {states.slice(0, 8).map((state, idx) => {
        const phase = getPhaseInfo(state.phase)
        return (
          <div 
            key={`${state.instance_id}-${state.hash}`}
            className="flex items-center gap-3 p-3 bg-dark-800/50 rounded-xl hover:bg-dark-800 transition-colors"
          >
            <span className="text-lg">{phase.icon}</span>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium truncate">{state.name}</p>
              <div className="flex items-center gap-3 mt-1">
                <span className={`text-xs ${phase.color}`}>{phase.text}</span>
                <span className="text-xs text-dark-500">
                  周期 #{state.cycle_index}
                </span>
                <span className="text-xs text-dark-500">
                  {state.time_left > 0 ? `${Math.floor(state.time_left)}s` : '汇报中'}
                </span>
              </div>
            </div>
            <div className="text-right">
              <p className="text-sm font-mono text-accent-400">
                {state.current_limit > 0 ? formatSpeed(state.current_limit) : 'MAX'}
              </p>
              <p className="text-xs text-dark-500">{state.limit_reason}</p>
            </div>
          </div>
        )
      })}
    </div>
  )
}

export default function DashboardPage() {
  const stats = useStatsStore((state) => state.stats)
  const limitStates = useStatsStore((state) => state.limitStates)
  const fetchStats = useStatsStore((state) => state.fetchStats)
  const fetchLimitStates = useStatsStore((state) => state.fetchLimitStates)
  const fetchInstances = useQBStore((state) => state.fetchInstances)
  const fetchSites = useSiteStore((state) => state.fetchSites)
  
  const [speedHistory, setSpeedHistory] = useState([])
  
  useEffect(() => {
    fetchStats()
    fetchLimitStates()
    fetchInstances()
    fetchSites()
    
    const interval = setInterval(() => {
      fetchStats()
      fetchLimitStates()
    }, 2000)
    
    return () => clearInterval(interval)
  }, [])
  
  useEffect(() => {
    if (stats) {
      setSpeedHistory(prev => {
        const newData = {
          time: new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
          upload: stats.total_up_speed || 0,
          download: stats.total_dl_speed || 0
        }
        const updated = [...prev, newData].slice(-30)
        return updated
      })
    }
  }, [stats])
  
  return (
    <div className="space-y-6 animate-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-display font-bold">仪表盘</h1>
          <p className="text-dark-400 mt-1">系统运行状态概览</p>
        </div>
        <div className="flex items-center gap-2">
          <div className={`
            flex items-center gap-2 px-3 py-1.5 rounded-full text-sm
            ${stats?.limit_engine_running 
              ? 'bg-accent-500/20 text-accent-400 border border-accent-500/30' 
              : 'bg-dark-800 text-dark-400 border border-dark-700'}
          `}>
            <Zap className="w-4 h-4" />
            <span>{stats?.limit_engine_running ? '限速引擎运行中' : '限速引擎已停止'}</span>
          </div>
        </div>
      </div>
      
      {/* Stats Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          icon={Server}
          label="qB实例"
          value={stats?.qb_instances || 0}
          subValue="已连接实例数"
          color="blue"
        />
        <StatCard
          icon={Globe}
          label="站点配置"
          value={stats?.sites || 0}
          subValue="已配置站点数"
          color="purple"
        />
        <StatCard
          icon={Rss}
          label="RSS订阅"
          value={stats?.rss_feeds || 0}
          subValue="活跃订阅数"
          color="orange"
        />
        <StatCard
          icon={Gauge}
          label="限速种子"
          value={stats?.active_limited_torrents || 0}
          subValue="正在限速中"
          color="green"
        />
      </div>
      
      {/* Speed Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-dark-900/50 border border-dark-800 rounded-2xl p-5">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2.5 rounded-xl bg-accent-500/20">
              <ArrowUp className="w-5 h-5 text-accent-400" />
            </div>
            <div>
              <p className="text-sm text-dark-400">总上传速度</p>
              <p className="text-2xl font-bold font-mono text-accent-400">
                {formatSpeed(stats?.total_up_speed || 0)}
              </p>
            </div>
          </div>
          <div className="h-2 bg-dark-800 rounded-full overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-accent-500 to-accent-400 rounded-full transition-all duration-500"
              style={{ width: `${Math.min(100, (stats?.total_up_speed || 0) / 1048576 * 10)}%` }}
            />
          </div>
        </div>
        
        <div className="bg-dark-900/50 border border-dark-800 rounded-2xl p-5">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-2.5 rounded-xl bg-blue-500/20">
              <ArrowDown className="w-5 h-5 text-blue-400" />
            </div>
            <div>
              <p className="text-sm text-dark-400">总下载速度</p>
              <p className="text-2xl font-bold font-mono text-blue-400">
                {formatSpeed(stats?.total_dl_speed || 0)}
              </p>
            </div>
          </div>
          <div className="h-2 bg-dark-800 rounded-full overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-blue-500 to-blue-400 rounded-full transition-all duration-500"
              style={{ width: `${Math.min(100, (stats?.total_dl_speed || 0) / 1048576 * 10)}%` }}
            />
          </div>
        </div>
      </div>
      
      {/* Charts and Lists */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <SpeedChart data={speedHistory} />
        
        <div className="bg-dark-900/50 border border-dark-800 rounded-2xl p-5">
          <h3 className="font-semibold mb-4 flex items-center gap-2">
            <Gauge className="w-5 h-5 text-accent-400" />
            限速状态
          </h3>
          <LimitStatesList states={limitStates} />
        </div>
      </div>
    </div>
  )
}
