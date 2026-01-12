import { useEffect, useState } from 'react'
import { 
  Gauge, 
  Play, 
  Pause, 
  RefreshCw,
  Activity,
  Zap,
  Clock,
  TrendingUp
} from 'lucide-react'
import { useStatsStore, useQBStore } from '../store'
import { formatSpeed, getPhaseInfo } from '../utils/helpers'
import api from '../api'

function LimitStateRow({ state, instances }) {
  const phase = getPhaseInfo(state.phase)
  const instance = instances.find(i => i.id === state.instance_id)
  
  return (
    <tr className="hover:bg-dark-800/50 transition-colors border-b border-dark-800/50">
      <td className="px-4 py-3">
        <div className="max-w-xs">
          <p className="text-sm font-medium truncate" title={state.name}>
            {state.name}
          </p>
          <p className="text-xs text-dark-500">{instance?.name || '-'}</p>
        </div>
      </td>
      <td className="px-4 py-3">
        <div className="flex items-center gap-2">
          <span className="text-lg">{phase.icon}</span>
          <span className={`text-sm font-medium ${phase.color}`}>{phase.text}</span>
        </div>
      </td>
      <td className="px-4 py-3 text-right">
        <span className="text-sm font-mono">
          {state.time_left > 0 ? `${Math.floor(state.time_left)}s` : '汇报中'}
        </span>
      </td>
      <td className="px-4 py-3 text-right">
        <span className="text-sm font-mono text-accent-400">
          {state.current_limit > 0 ? formatSpeed(state.current_limit) : 'MAX'}
        </span>
      </td>
      <td className="px-4 py-3 text-right">
        <span className="text-xs text-dark-400">{state.limit_reason}</span>
      </td>
      <td className="px-4 py-3 text-center">
        <span className={`
          px-2 py-0.5 rounded-full text-xs
          ${state.cycle_synced 
            ? 'bg-accent-500/20 text-accent-400' 
            : 'bg-yellow-500/20 text-yellow-400'}
        `}>
          #{state.cycle_index} {state.cycle_synced ? '✓' : '○'}
        </span>
      </td>
    </tr>
  )
}

function PhaseCard({ phase, count, icon, color }) {
  const colors = {
    orange: 'from-orange-500/20 to-orange-600/10 border-orange-500/30 text-orange-400',
    blue: 'from-blue-500/20 to-blue-600/10 border-blue-500/30 text-blue-400',
    green: 'from-accent-500/20 to-accent-600/10 border-accent-500/30 text-accent-400',
    purple: 'from-purple-500/20 to-purple-600/10 border-purple-500/30 text-purple-400'
  }
  
  return (
    <div className={`bg-gradient-to-br ${colors[color]} border rounded-xl p-4`}>
      <div className="flex items-center justify-between">
        <span className="text-2xl">{icon}</span>
        <span className="text-2xl font-bold font-mono">{count}</span>
      </div>
      <p className="text-sm mt-2 opacity-80">{phase}</p>
    </div>
  )
}

export default function LimitPage() {
  const limitStates = useStatsStore((state) => state.limitStates)
  const fetchLimitStates = useStatsStore((state) => state.fetchLimitStates)
  const instances = useQBStore((state) => state.instances)
  const fetchInstances = useQBStore((state) => state.fetchInstances)
  
  const [engineRunning, setEngineRunning] = useState(true)
  const [loading, setLoading] = useState(false)
  
  useEffect(() => {
    fetchLimitStates()
    fetchInstances()
    checkEngineStatus()
    
    const interval = setInterval(() => {
      fetchLimitStates()
    }, 2000)
    
    return () => clearInterval(interval)
  }, [])
  
  const checkEngineStatus = async () => {
    try {
      const res = await api.get('/api/limit/status')
      setEngineRunning(res.data.running)
    } catch {}
  }
  
  const toggleEngine = async () => {
    setLoading(true)
    try {
      if (engineRunning) {
        await api.post('/api/limit/stop')
      } else {
        await api.post('/api/limit/start')
      }
      await checkEngineStatus()
    } catch (err) {
      alert(err.response?.data?.detail || '操作失败')
    }
    setLoading(false)
  }
  
  // 统计各阶段数量
  const phaseCounts = limitStates.reduce((acc, s) => {
    acc[s.phase] = (acc[s.phase] || 0) + 1
    return acc
  }, {})
  
  return (
    <div className="space-y-6 animate-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-display font-bold">动态限速</h1>
          <p className="text-dark-400 mt-1">实时查看和控制限速状态</p>
        </div>
        
        <div className="flex items-center gap-3">
          <div className={`
            flex items-center gap-2 px-4 py-2 rounded-xl
            ${engineRunning 
              ? 'bg-accent-500/20 border border-accent-500/30' 
              : 'bg-dark-800 border border-dark-700'}
          `}>
            <div className={`w-2 h-2 rounded-full ${engineRunning ? 'bg-accent-400 pulse-dot' : 'bg-dark-600'}`} />
            <span className={engineRunning ? 'text-accent-400' : 'text-dark-400'}>
              {engineRunning ? '引擎运行中' : '引擎已停止'}
            </span>
          </div>
          
          <button
            onClick={toggleEngine}
            disabled={loading}
            className={`
              flex items-center gap-2 px-4 py-2 rounded-xl transition-colors
              ${engineRunning 
                ? 'bg-yellow-500/20 text-yellow-400 hover:bg-yellow-500/30 border border-yellow-500/30' 
                : 'bg-accent-500 text-white hover:bg-accent-600'}
            `}
          >
            {loading ? (
              <RefreshCw className="w-4 h-4 animate-spin" />
            ) : engineRunning ? (
              <Pause className="w-4 h-4" />
            ) : (
              <Play className="w-4 h-4" />
            )}
            {engineRunning ? '暂停引擎' : '启动引擎'}
          </button>
        </div>
      </div>
      
      {/* Phase Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <PhaseCard phase="预热阶段" count={phaseCounts.warmup || 0} icon="🔥" color="orange" />
        <PhaseCard phase="追赶阶段" count={phaseCounts.catch || 0} icon="🏃" color="blue" />
        <PhaseCard phase="稳定阶段" count={phaseCounts.steady || 0} icon="⚖️" color="green" />
        <PhaseCard phase="收尾阶段" count={phaseCounts.finish || 0} icon="🎯" color="purple" />
      </div>
      
      {/* Info Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-dark-900/50 border border-dark-800 rounded-2xl p-5">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-accent-500/20 text-accent-400">
              <Activity className="w-5 h-5" />
            </div>
            <div>
              <p className="text-sm text-dark-400">活跃种子</p>
              <p className="text-xl font-bold font-mono">{limitStates.length}</p>
            </div>
          </div>
        </div>
        
        <div className="bg-dark-900/50 border border-dark-800 rounded-2xl p-5">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-purple-500/20 text-purple-400">
              <Zap className="w-5 h-5" />
            </div>
            <div>
              <p className="text-sm text-dark-400">已同步周期</p>
              <p className="text-xl font-bold font-mono">
                {limitStates.filter(s => s.cycle_synced).length}
              </p>
            </div>
          </div>
        </div>
        
        <div className="bg-dark-900/50 border border-dark-800 rounded-2xl p-5">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-blue-500/20 text-blue-400">
              <Clock className="w-5 h-5" />
            </div>
            <div>
              <p className="text-sm text-dark-400">更新频率</p>
              <p className="text-xl font-bold font-mono">2s</p>
            </div>
          </div>
        </div>
      </div>
      
      {/* Limit States Table */}
      <div className="bg-dark-900/50 border border-dark-800 rounded-2xl overflow-hidden">
        <div className="px-5 py-4 border-b border-dark-800 flex items-center justify-between">
          <h3 className="font-semibold flex items-center gap-2">
            <Gauge className="w-5 h-5 text-accent-400" />
            限速状态详情
          </h3>
          <button
            onClick={fetchLimitStates}
            className="p-2 hover:bg-dark-800 rounded-lg text-dark-400 hover:text-dark-200"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
        
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-dark-800/50 text-left">
              <tr>
                <th className="px-4 py-3 text-sm font-medium text-dark-400">种子</th>
                <th className="px-4 py-3 text-sm font-medium text-dark-400">阶段</th>
                <th className="px-4 py-3 text-sm font-medium text-dark-400 text-right">剩余时间</th>
                <th className="px-4 py-3 text-sm font-medium text-dark-400 text-right">当前限速</th>
                <th className="px-4 py-3 text-sm font-medium text-dark-400 text-right">原因</th>
                <th className="px-4 py-3 text-sm font-medium text-dark-400 text-center">周期</th>
              </tr>
            </thead>
            <tbody>
              {limitStates.length === 0 ? (
                <tr>
                  <td colSpan={6} className="px-4 py-12 text-center text-dark-500">
                    {engineRunning ? '暂无正在限速的种子' : '限速引擎已停止'}
                  </td>
                </tr>
              ) : (
                limitStates.map((state, idx) => (
                  <LimitStateRow 
                    key={`${state.instance_id}-${state.hash}`} 
                    state={state} 
                    instances={instances}
                  />
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
      
      {/* Algorithm Info */}
      <div className="bg-dark-900/50 border border-dark-800 rounded-2xl p-5">
        <h3 className="font-semibold mb-4 flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-accent-400" />
          算法说明
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-dark-400">
          <div>
            <h4 className="text-dark-200 font-medium mb-2">🔥 预热阶段 (Warmup)</h4>
            <p>周期开始，尚未同步汇报时间。使用保守策略，逐步建立速度基准。</p>
          </div>
          <div>
            <h4 className="text-dark-200 font-medium mb-2">🏃 追赶阶段 (Catch)</h4>
            <p>距离汇报 &gt; 120s。如果进度落后，适当放宽限速追赶目标。</p>
          </div>
          <div>
            <h4 className="text-dark-200 font-medium mb-2">⚖️ 稳定阶段 (Steady)</h4>
            <p>距离汇报 30-120s。精确控制速度，确保达标的同时不超速。</p>
          </div>
          <div>
            <h4 className="text-dark-200 font-medium mb-2">🎯 收尾阶段 (Finish)</h4>
            <p>距离汇报 &lt; 30s。最高精度控制，使用卡尔曼滤波预测并微调。</p>
          </div>
        </div>
      </div>
    </div>
  )
}
