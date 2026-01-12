import { useEffect, useState } from 'react'
import { 
  Download, 
  Upload, 
  Pause, 
  Play, 
  Trash2, 
  RefreshCw, 
  Radio,
  Search,
  Filter,
  Plus,
  MoreVertical,
  CheckSquare,
  Square,
  ChevronDown,
  X,
  Link
} from 'lucide-react'
import { useQBStore } from '../store'
import { formatSize, formatSpeed, formatDuration, getTorrentStatus, getPhaseInfo } from '../utils/helpers'

function TorrentRow({ torrent, selected, onSelect, onAction }) {
  const status = getTorrentStatus(torrent.state)
  const limitState = torrent.limit_state
  const phase = limitState ? getPhaseInfo(limitState.phase) : null
  
  return (
    <tr className="group hover:bg-dark-800/50 transition-colors border-b border-dark-800/50">
      <td className="px-4 py-3">
        <button onClick={() => onSelect(torrent.hash)} className="p-1">
          {selected ? (
            <CheckSquare className="w-4 h-4 text-accent-400" />
          ) : (
            <Square className="w-4 h-4 text-dark-500 group-hover:text-dark-300" />
          )}
        </button>
      </td>
      <td className="px-4 py-3">
        <div className="max-w-xs lg:max-w-md">
          <p className="text-sm font-medium truncate" title={torrent.name}>
            {torrent.name}
          </p>
          <div className="flex items-center gap-2 mt-1">
            <span className={`text-xs px-2 py-0.5 rounded-full ${status.bg} ${status.color}`}>
              {status.text}
            </span>
            {phase && (
              <span className={`text-xs ${phase.color}`}>
                {phase.icon} {phase.text}
              </span>
            )}
          </div>
        </div>
      </td>
      <td className="px-4 py-3 text-right">
        <p className="text-sm font-mono">{formatSize(torrent.size)}</p>
      </td>
      <td className="px-4 py-3">
        <div className="w-24">
          <div className="flex items-center justify-between text-xs mb-1">
            <span>{(torrent.progress * 100).toFixed(1)}%</span>
          </div>
          <div className="h-1.5 bg-dark-700 rounded-full overflow-hidden">
            <div 
              className="h-full bg-accent-500 rounded-full transition-all"
              style={{ width: `${torrent.progress * 100}%` }}
            />
          </div>
        </div>
      </td>
      <td className="px-4 py-3 text-right">
        <div className="flex items-center justify-end gap-1 text-sm">
          <Upload className="w-3 h-3 text-accent-400" />
          <span className="font-mono text-accent-400">{formatSpeed(torrent.up_speed)}</span>
        </div>
        <div className="flex items-center justify-end gap-1 text-sm mt-0.5">
          <Download className="w-3 h-3 text-blue-400" />
          <span className="font-mono text-blue-400">{formatSpeed(torrent.dl_speed)}</span>
        </div>
      </td>
      <td className="px-4 py-3 text-right">
        <p className="text-sm font-mono">{torrent.ratio.toFixed(2)}</p>
      </td>
      <td className="px-4 py-3 text-right">
        <p className="text-sm font-mono text-dark-400">
          {torrent.eta > 0 ? formatDuration(torrent.eta) : '∞'}
        </p>
      </td>
      <td className="px-4 py-3">
        <div className="flex items-center justify-end gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            onClick={() => onAction([torrent.hash], torrent.state.includes('paused') ? 'resume' : 'pause')}
            className="p-1.5 hover:bg-dark-700 rounded-lg transition-colors"
            title={torrent.state.includes('paused') ? '继续' : '暂停'}
          >
            {torrent.state.includes('paused') ? (
              <Play className="w-4 h-4 text-accent-400" />
            ) : (
              <Pause className="w-4 h-4 text-yellow-400" />
            )}
          </button>
          <button
            onClick={() => onAction([torrent.hash], 'reannounce')}
            className="p-1.5 hover:bg-dark-700 rounded-lg transition-colors"
            title="强制汇报"
          >
            <Radio className="w-4 h-4 text-blue-400" />
          </button>
          <button
            onClick={() => onAction([torrent.hash], 'delete')}
            className="p-1.5 hover:bg-dark-700 rounded-lg transition-colors"
            title="删除"
          >
            <Trash2 className="w-4 h-4 text-red-400" />
          </button>
        </div>
      </td>
    </tr>
  )
}

function AddTorrentModal({ isOpen, onClose, onSubmit }) {
  const [urls, setUrls] = useState('')
  const [savePath, setSavePath] = useState('')
  const [category, setCategory] = useState('')
  const [paused, setPaused] = useState(false)
  
  if (!isOpen) return null
  
  const handleSubmit = (e) => {
    e.preventDefault()
    onSubmit({ urls, save_path: savePath || undefined, category: category || undefined, paused })
    setUrls('')
    onClose()
  }
  
  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4">
      <div className="bg-dark-900 border border-dark-800 rounded-2xl w-full max-w-lg animate-in">
        <div className="flex items-center justify-between p-4 border-b border-dark-800">
          <h3 className="font-semibold">添加种子</h3>
          <button onClick={onClose} className="p-1 hover:bg-dark-800 rounded-lg">
            <X className="w-5 h-5" />
          </button>
        </div>
        <form onSubmit={handleSubmit} className="p-4 space-y-4">
          <div>
            <label className="block text-sm text-dark-400 mb-2">种子链接 / 磁力链接</label>
            <textarea
              value={urls}
              onChange={(e) => setUrls(e.target.value)}
              className="w-full px-4 py-3 bg-dark-800 border border-dark-700 rounded-xl
                         text-dark-100 placeholder-dark-500 resize-none h-24
                         focus:border-accent-500 focus:ring-1 focus:ring-accent-500/50"
              placeholder="每行一个链接..."
            />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-dark-400 mb-2">保存路径</label>
              <input
                type="text"
                value={savePath}
                onChange={(e) => setSavePath(e.target.value)}
                className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl
                           text-dark-100 placeholder-dark-500
                           focus:border-accent-500 focus:ring-1 focus:ring-accent-500/50"
                placeholder="默认路径"
              />
            </div>
            <div>
              <label className="block text-sm text-dark-400 mb-2">分类</label>
              <input
                type="text"
                value={category}
                onChange={(e) => setCategory(e.target.value)}
                className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl
                           text-dark-100 placeholder-dark-500
                           focus:border-accent-500 focus:ring-1 focus:ring-accent-500/50"
                placeholder="无分类"
              />
            </div>
          </div>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={paused}
              onChange={(e) => setPaused(e.target.checked)}
              className="w-4 h-4 rounded border-dark-600 bg-dark-800 text-accent-500 
                         focus:ring-accent-500/50"
            />
            <span className="text-sm text-dark-300">添加后暂停</span>
          </label>
          <div className="flex justify-end gap-3 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-dark-400 hover:text-dark-200 transition-colors"
            >
              取消
            </button>
            <button
              type="submit"
              className="px-4 py-2 bg-accent-500 hover:bg-accent-600 text-white rounded-lg transition-colors"
            >
              添加
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default function TorrentsPage() {
  const instances = useQBStore((state) => state.instances)
  const selectedInstance = useQBStore((state) => state.selectedInstance)
  const setSelectedInstance = useQBStore((state) => state.setSelectedInstance)
  const torrents = useQBStore((state) => state.torrents)
  const loading = useQBStore((state) => state.loading)
  const fetchInstances = useQBStore((state) => state.fetchInstances)
  const fetchTorrents = useQBStore((state) => state.fetchTorrents)
  const torrentAction = useQBStore((state) => state.torrentAction)
  const addTorrent = useQBStore((state) => state.addTorrent)
  
  const [search, setSearch] = useState('')
  const [statusFilter, setStatusFilter] = useState('all')
  const [selected, setSelected] = useState(new Set())
  const [showAddModal, setShowAddModal] = useState(false)
  
  useEffect(() => {
    fetchInstances()
  }, [])
  
  useEffect(() => {
    if (selectedInstance) {
      fetchTorrents(selectedInstance)
      const interval = setInterval(() => fetchTorrents(selectedInstance), 3000)
      return () => clearInterval(interval)
    }
  }, [selectedInstance])
  
  const filteredTorrents = torrents.filter(t => {
    if (search && !t.name.toLowerCase().includes(search.toLowerCase())) return false
    if (statusFilter !== 'all') {
      if (statusFilter === 'downloading' && !t.state.includes('download')) return false
      if (statusFilter === 'seeding' && !['seeding', 'uploading', 'stalledUP', 'forcedUP'].includes(t.state)) return false
      if (statusFilter === 'paused' && !t.state.includes('paused')) return false
    }
    return true
  })
  
  const handleSelect = (hash) => {
    const newSelected = new Set(selected)
    if (newSelected.has(hash)) {
      newSelected.delete(hash)
    } else {
      newSelected.add(hash)
    }
    setSelected(newSelected)
  }
  
  const handleSelectAll = () => {
    if (selected.size === filteredTorrents.length) {
      setSelected(new Set())
    } else {
      setSelected(new Set(filteredTorrents.map(t => t.hash)))
    }
  }
  
  const handleAction = async (hashes, action, deleteFiles = false) => {
    if (action === 'delete' && !confirm('确定要删除选中的种子吗？')) return
    await torrentAction(selectedInstance, hashes, action, deleteFiles)
    setSelected(new Set())
  }
  
  const handleBatchAction = (action) => {
    if (selected.size === 0) return
    handleAction(Array.from(selected), action)
  }
  
  const handleAddTorrent = async (data) => {
    await addTorrent(selectedInstance, data)
  }
  
  const currentInstance = instances.find(i => i.id === selectedInstance)
  
  return (
    <div className="space-y-6 animate-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-display font-bold">种子管理</h1>
          <p className="text-dark-400 mt-1">管理和监控所有种子任务</p>
        </div>
        
        <div className="flex items-center gap-3">
          {/* Instance Selector */}
          <div className="relative">
            <select
              value={selectedInstance || ''}
              onChange={(e) => setSelectedInstance(Number(e.target.value))}
              className="appearance-none pl-4 pr-10 py-2 bg-dark-800 border border-dark-700 rounded-xl
                         text-dark-100 focus:border-accent-500 focus:ring-1 focus:ring-accent-500/50"
            >
              {instances.map(inst => (
                <option key={inst.id} value={inst.id}>
                  {inst.name} {inst.connected ? '●' : '○'}
                </option>
              ))}
            </select>
            <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-dark-400 pointer-events-none" />
          </div>
          
          <button
            onClick={() => setShowAddModal(true)}
            className="flex items-center gap-2 px-4 py-2 bg-accent-500 hover:bg-accent-600 
                       text-white rounded-xl transition-colors"
          >
            <Plus className="w-4 h-4" />
            <span className="hidden sm:inline">添加种子</span>
          </button>
        </div>
      </div>
      
      {/* Filters & Actions */}
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-dark-500" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="搜索种子..."
            className="w-full pl-10 pr-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl
                       text-dark-100 placeholder-dark-500
                       focus:border-accent-500 focus:ring-1 focus:ring-accent-500/50"
          />
        </div>
        
        <div className="flex items-center gap-2">
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl
                       text-dark-100 focus:border-accent-500"
          >
            <option value="all">全部状态</option>
            <option value="downloading">下载中</option>
            <option value="seeding">做种中</option>
            <option value="paused">已暂停</option>
          </select>
          
          {selected.size > 0 && (
            <div className="flex items-center gap-2 ml-2">
              <span className="text-sm text-dark-400">已选 {selected.size}</span>
              <button
                onClick={() => handleBatchAction('resume')}
                className="p-2 hover:bg-dark-800 rounded-lg text-accent-400"
                title="继续"
              >
                <Play className="w-4 h-4" />
              </button>
              <button
                onClick={() => handleBatchAction('pause')}
                className="p-2 hover:bg-dark-800 rounded-lg text-yellow-400"
                title="暂停"
              >
                <Pause className="w-4 h-4" />
              </button>
              <button
                onClick={() => handleBatchAction('delete')}
                className="p-2 hover:bg-dark-800 rounded-lg text-red-400"
                title="删除"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          )}
        </div>
      </div>
      
      {/* Torrents Table */}
      <div className="bg-dark-900/50 border border-dark-800 rounded-2xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-dark-800/50 text-left">
              <tr>
                <th className="px-4 py-3 w-12">
                  <button onClick={handleSelectAll} className="p-1">
                    {selected.size === filteredTorrents.length && filteredTorrents.length > 0 ? (
                      <CheckSquare className="w-4 h-4 text-accent-400" />
                    ) : (
                      <Square className="w-4 h-4 text-dark-500" />
                    )}
                  </button>
                </th>
                <th className="px-4 py-3 text-sm font-medium text-dark-400">名称</th>
                <th className="px-4 py-3 text-sm font-medium text-dark-400 text-right">大小</th>
                <th className="px-4 py-3 text-sm font-medium text-dark-400">进度</th>
                <th className="px-4 py-3 text-sm font-medium text-dark-400 text-right">速度</th>
                <th className="px-4 py-3 text-sm font-medium text-dark-400 text-right">分享率</th>
                <th className="px-4 py-3 text-sm font-medium text-dark-400 text-right">剩余</th>
                <th className="px-4 py-3 w-24"></th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={8} className="px-4 py-12 text-center text-dark-500">
                    <RefreshCw className="w-6 h-6 animate-spin mx-auto mb-2" />
                    加载中...
                  </td>
                </tr>
              ) : filteredTorrents.length === 0 ? (
                <tr>
                  <td colSpan={8} className="px-4 py-12 text-center text-dark-500">
                    暂无种子
                  </td>
                </tr>
              ) : (
                filteredTorrents.map(torrent => (
                  <TorrentRow
                    key={torrent.hash}
                    torrent={torrent}
                    selected={selected.has(torrent.hash)}
                    onSelect={handleSelect}
                    onAction={handleAction}
                  />
                ))
              )}
            </tbody>
          </table>
        </div>
        
        {/* Footer */}
        <div className="px-4 py-3 border-t border-dark-800 flex items-center justify-between text-sm text-dark-400">
          <span>共 {filteredTorrents.length} 个种子</span>
          <span>
            实例: {currentInstance?.name || '-'} {currentInstance?.connected ? '(已连接)' : '(未连接)'}
          </span>
        </div>
      </div>
      
      {/* Add Torrent Modal */}
      <AddTorrentModal
        isOpen={showAddModal}
        onClose={() => setShowAddModal(false)}
        onSubmit={handleAddTorrent}
      />
    </div>
  )
}
