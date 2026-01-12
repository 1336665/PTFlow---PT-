import { useEffect, useState } from 'react'
import { 
  Globe, 
  Plus, 
  Trash2, 
  Edit2, 
  X, 
  RefreshCw,
  Gauge,
  Shield,
  Zap,
  Check
} from 'lucide-react'
import { useSiteStore } from '../store'
import { formatSpeed } from '../utils/helpers'

function SiteCard({ site, onEdit, onDelete }) {
  return (
    <div className="bg-dark-900/50 border border-dark-800 rounded-2xl p-5 card-hover">
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div className="p-3 rounded-xl bg-purple-500/20 text-purple-400">
            <Globe className="w-6 h-6" />
          </div>
          <div>
            <h3 className="font-semibold text-lg">{site.name}</h3>
            <p className="text-sm text-dark-400 mt-0.5 font-mono">{site.tracker_keyword}</p>
          </div>
        </div>
      </div>
      
      <div className="mt-4 grid grid-cols-2 gap-3">
        <div className="bg-dark-800/50 rounded-xl p-3">
          <div className="flex items-center gap-2 text-accent-400 mb-1">
            <Gauge className="w-4 h-4" />
            <span className="text-xs">目标速度</span>
          </div>
          <p className="font-mono text-sm">{formatSpeed(site.target_speed_kib * 1024)}</p>
        </div>
        <div className="bg-dark-800/50 rounded-xl p-3">
          <div className="flex items-center gap-2 text-blue-400 mb-1">
            <Shield className="w-4 h-4" />
            <span className="text-xs">安全边际</span>
          </div>
          <p className="font-mono text-sm">{(site.safety_margin * 100).toFixed(0)}%</p>
        </div>
      </div>
      
      <div className="mt-3 flex items-center gap-2 flex-wrap">
        {site.enable_limit && (
          <span className="px-2 py-1 bg-accent-500/20 text-accent-400 rounded-lg text-xs">
            限速
          </span>
        )}
        {site.enable_dl_limit && (
          <span className="px-2 py-1 bg-blue-500/20 text-blue-400 rounded-lg text-xs">
            下载限速
          </span>
        )}
        {site.enable_reannounce && (
          <span className="px-2 py-1 bg-orange-500/20 text-orange-400 rounded-lg text-xs">
            汇报优化
          </span>
        )}
      </div>
      
      <div className="mt-4 pt-4 border-t border-dark-800 flex items-center gap-2">
        <button
          onClick={() => onEdit(site)}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-dark-800 text-dark-300 
                     hover:bg-dark-700 rounded-lg text-sm transition-colors"
        >
          <Edit2 className="w-4 h-4" />
          编辑
        </button>
        <button
          onClick={() => onDelete(site.id)}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-red-500/10 text-red-400 
                     hover:bg-red-500/20 rounded-lg text-sm transition-colors"
        >
          <Trash2 className="w-4 h-4" />
          删除
        </button>
      </div>
    </div>
  )
}

function SiteModal({ isOpen, onClose, site, onSubmit }) {
  const [form, setForm] = useState({
    name: '',
    tracker_keyword: '',
    target_speed_kib: 51200,
    safety_margin: 0.98,
    enable_limit: true,
    enable_dl_limit: true,
    enable_reannounce: true,
    cookie: ''
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  
  useEffect(() => {
    if (site) {
      setForm({
        name: site.name,
        tracker_keyword: site.tracker_keyword,
        target_speed_kib: site.target_speed_kib,
        safety_margin: site.safety_margin,
        enable_limit: site.enable_limit,
        enable_dl_limit: site.enable_dl_limit,
        enable_reannounce: site.enable_reannounce,
        cookie: ''
      })
    } else {
      setForm({
        name: '',
        tracker_keyword: '',
        target_speed_kib: 51200,
        safety_margin: 0.98,
        enable_limit: true,
        enable_dl_limit: true,
        enable_reannounce: true,
        cookie: ''
      })
    }
    setError('')
  }, [site, isOpen])
  
  if (!isOpen) return null
  
  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!form.name || !form.tracker_keyword) {
      setError('请填写必填项')
      return
    }
    
    setLoading(true)
    setError('')
    
    try {
      const result = await onSubmit(form, site?.id)
      if (result.success) {
        onClose()
      } else {
        setError(result.error)
      }
    } catch (err) {
      setError(err.message)
    }
    
    setLoading(false)
  }
  
  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4 overflow-y-auto">
      <div className="bg-dark-900 border border-dark-800 rounded-2xl w-full max-w-lg my-8 animate-in">
        <div className="flex items-center justify-between p-4 border-b border-dark-800">
          <h3 className="font-semibold">{site ? '编辑站点' : '添加站点'}</h3>
          <button onClick={onClose} className="p-1 hover:bg-dark-800 rounded-lg">
            <X className="w-5 h-5" />
          </button>
        </div>
        <form onSubmit={handleSubmit} className="p-4 space-y-4">
          <div>
            <label className="block text-sm text-dark-400 mb-2">站点名称 *</label>
            <input
              type="text"
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
              className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl
                         text-dark-100 placeholder-dark-500
                         focus:border-accent-500 focus:ring-1 focus:ring-accent-500/50"
              placeholder="例: U2"
            />
          </div>
          
          <div>
            <label className="block text-sm text-dark-400 mb-2">Tracker关键词 *</label>
            <input
              type="text"
              value={form.tracker_keyword}
              onChange={(e) => setForm({ ...form, tracker_keyword: e.target.value })}
              className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl
                         text-dark-100 placeholder-dark-500
                         focus:border-accent-500 focus:ring-1 focus:ring-accent-500/50"
              placeholder="例: dmhy.org"
            />
            <p className="text-xs text-dark-500 mt-1">用于匹配种子的Tracker地址</p>
          </div>
          
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-dark-400 mb-2">目标速度 (KiB/s)</label>
              <input
                type="number"
                value={form.target_speed_kib}
                onChange={(e) => setForm({ ...form, target_speed_kib: parseInt(e.target.value) || 0 })}
                className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl
                           text-dark-100 placeholder-dark-500
                           focus:border-accent-500 focus:ring-1 focus:ring-accent-500/50"
                placeholder="51200"
              />
              <p className="text-xs text-dark-500 mt-1">{formatSpeed(form.target_speed_kib * 1024)}</p>
            </div>
            <div>
              <label className="block text-sm text-dark-400 mb-2">安全边际</label>
              <input
                type="number"
                step="0.01"
                min="0.8"
                max="1"
                value={form.safety_margin}
                onChange={(e) => setForm({ ...form, safety_margin: parseFloat(e.target.value) || 0.98 })}
                className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl
                           text-dark-100 placeholder-dark-500
                           focus:border-accent-500 focus:ring-1 focus:ring-accent-500/50"
                placeholder="0.98"
              />
              <p className="text-xs text-dark-500 mt-1">推荐 0.95 ~ 0.99</p>
            </div>
          </div>
          
          <div>
            <label className="block text-sm text-dark-400 mb-2">Cookie (可选)</label>
            <input
              type="text"
              value={form.cookie}
              onChange={(e) => setForm({ ...form, cookie: e.target.value })}
              className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl
                         text-dark-100 placeholder-dark-500
                         focus:border-accent-500 focus:ring-1 focus:ring-accent-500/50"
              placeholder="用于获取站点辅助信息"
            />
          </div>
          
          <div className="space-y-3">
            <label className="block text-sm text-dark-400">功能开关</label>
            
            <label className="flex items-center justify-between p-3 bg-dark-800/50 rounded-xl cursor-pointer hover:bg-dark-800 transition-colors">
              <div className="flex items-center gap-3">
                <Gauge className="w-5 h-5 text-accent-400" />
                <div>
                  <p className="text-sm font-medium">动态限速</p>
                  <p className="text-xs text-dark-500">根据汇报周期智能调整上传速度</p>
                </div>
              </div>
              <input
                type="checkbox"
                checked={form.enable_limit}
                onChange={(e) => setForm({ ...form, enable_limit: e.target.checked })}
                className="w-5 h-5 rounded border-dark-600 bg-dark-800 text-accent-500 
                           focus:ring-accent-500/50"
              />
            </label>
            
            <label className="flex items-center justify-between p-3 bg-dark-800/50 rounded-xl cursor-pointer hover:bg-dark-800 transition-colors">
              <div className="flex items-center gap-3">
                <Shield className="w-5 h-5 text-blue-400" />
                <div>
                  <p className="text-sm font-medium">下载限速</p>
                  <p className="text-xs text-dark-500">当上传超速时限制下载</p>
                </div>
              </div>
              <input
                type="checkbox"
                checked={form.enable_dl_limit}
                onChange={(e) => setForm({ ...form, enable_dl_limit: e.target.checked })}
                className="w-5 h-5 rounded border-dark-600 bg-dark-800 text-accent-500 
                           focus:ring-accent-500/50"
              />
            </label>
            
            <label className="flex items-center justify-between p-3 bg-dark-800/50 rounded-xl cursor-pointer hover:bg-dark-800 transition-colors">
              <div className="flex items-center gap-3">
                <Zap className="w-5 h-5 text-orange-400" />
                <div>
                  <p className="text-sm font-medium">汇报优化</p>
                  <p className="text-xs text-dark-500">智能触发强制汇报</p>
                </div>
              </div>
              <input
                type="checkbox"
                checked={form.enable_reannounce}
                onChange={(e) => setForm({ ...form, enable_reannounce: e.target.checked })}
                className="w-5 h-5 rounded border-dark-600 bg-dark-800 text-accent-500 
                           focus:ring-accent-500/50"
              />
            </label>
          </div>
          
          {error && (
            <div className="px-4 py-3 bg-red-500/10 border border-red-500/30 rounded-xl">
              <p className="text-sm text-red-400">{error}</p>
            </div>
          )}
          
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
              disabled={loading}
              className="px-4 py-2 bg-accent-500 hover:bg-accent-600 text-white rounded-lg 
                         transition-colors disabled:opacity-50 flex items-center gap-2"
            >
              {loading && <RefreshCw className="w-4 h-4 animate-spin" />}
              {site ? '保存' : '添加'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default function SitesPage() {
  const sites = useSiteStore((state) => state.sites)
  const fetchSites = useSiteStore((state) => state.fetchSites)
  const addSite = useSiteStore((state) => state.addSite)
  const updateSite = useSiteStore((state) => state.updateSite)
  const deleteSite = useSiteStore((state) => state.deleteSite)
  
  const [modalOpen, setModalOpen] = useState(false)
  const [editSite, setEditSite] = useState(null)
  
  useEffect(() => {
    fetchSites()
  }, [])
  
  const handleOpenModal = (site = null) => {
    setEditSite(site)
    setModalOpen(true)
  }
  
  const handleSubmit = async (data, siteId) => {
    if (siteId) {
      return await updateSite(siteId, data)
    } else {
      return await addSite(data)
    }
  }
  
  const handleDelete = async (id) => {
    if (!confirm('确定要删除此站点配置吗？')) return
    await deleteSite(id)
  }
  
  return (
    <div className="space-y-6 animate-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-display font-bold">站点配置</h1>
          <p className="text-dark-400 mt-1">配置PT站点的限速规则</p>
        </div>
        <button
          onClick={() => handleOpenModal()}
          className="flex items-center gap-2 px-4 py-2 bg-accent-500 hover:bg-accent-600 
                     text-white rounded-xl transition-colors"
        >
          <Plus className="w-4 h-4" />
          添加站点
        </button>
      </div>
      
      {/* Sites Grid */}
      {sites.length === 0 ? (
        <div className="bg-dark-900/50 border border-dark-800 rounded-2xl p-12 text-center">
          <Globe className="w-12 h-12 text-dark-600 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-dark-300 mb-2">暂无站点配置</h3>
          <p className="text-dark-500 mb-4">添加您的第一个PT站点</p>
          <button
            onClick={() => handleOpenModal()}
            className="inline-flex items-center gap-2 px-4 py-2 bg-accent-500 
                       hover:bg-accent-600 text-white rounded-xl transition-colors"
          >
            <Plus className="w-4 h-4" />
            添加站点
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {sites.map(site => (
            <SiteCard
              key={site.id}
              site={site}
              onEdit={handleOpenModal}
              onDelete={handleDelete}
            />
          ))}
        </div>
      )}
      
      {/* Modal */}
      <SiteModal
        isOpen={modalOpen}
        onClose={() => setModalOpen(false)}
        site={editSite}
        onSubmit={handleSubmit}
      />
    </div>
  )
}
