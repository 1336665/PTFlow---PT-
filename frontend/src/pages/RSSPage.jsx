import { useEffect, useState } from 'react'
import { 
  Rss, 
  Plus, 
  Trash2, 
  Edit2, 
  X, 
  RefreshCw,
  Eye,
  ExternalLink,
  Clock,
  Filter
} from 'lucide-react'
import { useRSSStore, useQBStore, useSiteStore } from '../store'
import { formatRelativeTime } from '../utils/helpers'

function FeedCard({ feed, sites, instances, onEdit, onDelete, onPreview }) {
  const site = sites.find(s => s.id === feed.site_id)
  const instance = instances.find(i => i.id === feed.qb_instance_id)
  
  return (
    <div className="bg-dark-900/50 border border-dark-800 rounded-2xl p-5 card-hover">
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div className={`
            p-3 rounded-xl 
            ${feed.auto_download 
              ? 'bg-orange-500/20 text-orange-400' 
              : 'bg-dark-800 text-dark-500'}
          `}>
            <Rss className="w-6 h-6" />
          </div>
          <div>
            <h3 className="font-semibold text-lg">{feed.name}</h3>
            <p className="text-xs text-dark-500 mt-0.5 truncate max-w-[200px]" title={feed.url}>
              {feed.url}
            </p>
          </div>
        </div>
        <div className={`
          px-2.5 py-1 rounded-full text-xs
          ${feed.auto_download 
            ? 'bg-accent-500/20 text-accent-400' 
            : 'bg-dark-800 text-dark-500'}
        `}>
          {feed.auto_download ? '自动下载' : '已暂停'}
        </div>
      </div>
      
      <div className="mt-4 space-y-2 text-sm">
        <div className="flex items-center justify-between">
          <span className="text-dark-500">下载到</span>
          <span className="text-dark-300">{instance?.name || '-'}</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-dark-500">关联站点</span>
          <span className="text-dark-300">{site?.name || '-'}</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-dark-500">保存路径</span>
          <span className="text-dark-300 truncate max-w-[150px]" title={feed.save_path}>
            {feed.save_path || '默认'}
          </span>
        </div>
        {feed.last_check && (
          <div className="flex items-center justify-between">
            <span className="text-dark-500">上次检查</span>
            <span className="text-dark-400 flex items-center gap-1">
              <Clock className="w-3 h-3" />
              {formatRelativeTime(feed.last_check)}
            </span>
          </div>
        )}
      </div>
      
      {(feed.filter_include || feed.filter_exclude) && (
        <div className="mt-3 pt-3 border-t border-dark-800">
          <div className="flex items-center gap-1 text-xs text-dark-500 mb-1">
            <Filter className="w-3 h-3" />
            过滤规则
          </div>
          {feed.filter_include && (
            <p className="text-xs text-accent-400">包含: {feed.filter_include}</p>
          )}
          {feed.filter_exclude && (
            <p className="text-xs text-red-400">排除: {feed.filter_exclude}</p>
          )}
        </div>
      )}
      
      <div className="mt-4 pt-4 border-t border-dark-800 flex items-center gap-2">
        <button
          onClick={() => onPreview(feed.id)}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-500/20 text-blue-400 
                     hover:bg-blue-500/30 rounded-lg text-sm transition-colors"
        >
          <Eye className="w-4 h-4" />
          预览
        </button>
        <button
          onClick={() => onEdit(feed)}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-dark-800 text-dark-300 
                     hover:bg-dark-700 rounded-lg text-sm transition-colors"
        >
          <Edit2 className="w-4 h-4" />
          编辑
        </button>
        <button
          onClick={() => onDelete(feed.id)}
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

function FeedModal({ isOpen, onClose, feed, instances, sites, onSubmit }) {
  const [form, setForm] = useState({
    name: '',
    url: '',
    qb_instance_id: '',
    site_id: '',
    save_path: '',
    category: '',
    auto_download: true,
    filter_include: '',
    filter_exclude: ''
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  
  useEffect(() => {
    if (feed) {
      setForm({
        name: feed.name,
        url: feed.url,
        qb_instance_id: feed.qb_instance_id || '',
        site_id: feed.site_id || '',
        save_path: feed.save_path || '',
        category: feed.category || '',
        auto_download: feed.auto_download,
        filter_include: feed.filter_include || '',
        filter_exclude: feed.filter_exclude || ''
      })
    } else {
      setForm({
        name: '',
        url: '',
        qb_instance_id: instances[0]?.id || '',
        site_id: '',
        save_path: '',
        category: '',
        auto_download: true,
        filter_include: '',
        filter_exclude: ''
      })
    }
    setError('')
  }, [feed, isOpen, instances])
  
  if (!isOpen) return null
  
  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!form.name || !form.url || !form.qb_instance_id) {
      setError('请填写必填项')
      return
    }
    
    setLoading(true)
    setError('')
    
    try {
      const data = {
        ...form,
        qb_instance_id: parseInt(form.qb_instance_id),
        site_id: form.site_id ? parseInt(form.site_id) : null
      }
      const result = await onSubmit(data, feed?.id)
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
          <h3 className="font-semibold">{feed ? '编辑RSS订阅' : '添加RSS订阅'}</h3>
          <button onClick={onClose} className="p-1 hover:bg-dark-800 rounded-lg">
            <X className="w-5 h-5" />
          </button>
        </div>
        <form onSubmit={handleSubmit} className="p-4 space-y-4">
          <div>
            <label className="block text-sm text-dark-400 mb-2">订阅名称 *</label>
            <input
              type="text"
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
              className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl
                         text-dark-100 placeholder-dark-500
                         focus:border-accent-500 focus:ring-1 focus:ring-accent-500/50"
              placeholder="例: U2新种"
            />
          </div>
          
          <div>
            <label className="block text-sm text-dark-400 mb-2">RSS地址 *</label>
            <input
              type="url"
              value={form.url}
              onChange={(e) => setForm({ ...form, url: e.target.value })}
              className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl
                         text-dark-100 placeholder-dark-500
                         focus:border-accent-500 focus:ring-1 focus:ring-accent-500/50"
              placeholder="https://..."
            />
          </div>
          
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-dark-400 mb-2">下载实例 *</label>
              <select
                value={form.qb_instance_id}
                onChange={(e) => setForm({ ...form, qb_instance_id: e.target.value })}
                className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl
                           text-dark-100 focus:border-accent-500"
              >
                <option value="">选择实例</option>
                {instances.map(inst => (
                  <option key={inst.id} value={inst.id}>{inst.name}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm text-dark-400 mb-2">关联站点</label>
              <select
                value={form.site_id}
                onChange={(e) => setForm({ ...form, site_id: e.target.value })}
                className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl
                           text-dark-100 focus:border-accent-500"
              >
                <option value="">无</option>
                {sites.map(site => (
                  <option key={site.id} value={site.id}>{site.name}</option>
                ))}
              </select>
            </div>
          </div>
          
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-dark-400 mb-2">保存路径</label>
              <input
                type="text"
                value={form.save_path}
                onChange={(e) => setForm({ ...form, save_path: e.target.value })}
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
                value={form.category}
                onChange={(e) => setForm({ ...form, category: e.target.value })}
                className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl
                           text-dark-100 placeholder-dark-500
                           focus:border-accent-500 focus:ring-1 focus:ring-accent-500/50"
                placeholder="无分类"
              />
            </div>
          </div>
          
          <div>
            <label className="block text-sm text-dark-400 mb-2">包含关键词 (用逗号分隔)</label>
            <input
              type="text"
              value={form.filter_include}
              onChange={(e) => setForm({ ...form, filter_include: e.target.value })}
              className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl
                         text-dark-100 placeholder-dark-500
                         focus:border-accent-500 focus:ring-1 focus:ring-accent-500/50"
              placeholder="例: 1080p, HEVC"
            />
          </div>
          
          <div>
            <label className="block text-sm text-dark-400 mb-2">排除关键词 (用逗号分隔)</label>
            <input
              type="text"
              value={form.filter_exclude}
              onChange={(e) => setForm({ ...form, filter_exclude: e.target.value })}
              className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl
                         text-dark-100 placeholder-dark-500
                         focus:border-accent-500 focus:ring-1 focus:ring-accent-500/50"
              placeholder="例: CAM, TS"
            />
          </div>
          
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={form.auto_download}
              onChange={(e) => setForm({ ...form, auto_download: e.target.checked })}
              className="w-4 h-4 rounded border-dark-600 bg-dark-800 text-accent-500 
                         focus:ring-accent-500/50"
            />
            <span className="text-sm text-dark-300">自动下载新种子</span>
          </label>
          
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
              {feed ? '保存' : '添加'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

function PreviewModal({ isOpen, onClose, entries, loading }) {
  if (!isOpen) return null
  
  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4">
      <div className="bg-dark-900 border border-dark-800 rounded-2xl w-full max-w-2xl max-h-[80vh] flex flex-col animate-in">
        <div className="flex items-center justify-between p-4 border-b border-dark-800">
          <h3 className="font-semibold">RSS预览</h3>
          <button onClick={onClose} className="p-1 hover:bg-dark-800 rounded-lg">
            <X className="w-5 h-5" />
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-4">
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <RefreshCw className="w-6 h-6 animate-spin text-dark-400" />
            </div>
          ) : entries.length === 0 ? (
            <p className="text-center text-dark-500 py-12">暂无条目</p>
          ) : (
            <div className="space-y-3">
              {entries.map((entry, idx) => (
                <div 
                  key={idx} 
                  className="p-3 bg-dark-800/50 rounded-xl hover:bg-dark-800 transition-colors"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">{entry.title}</p>
                      {entry.published && (
                        <p className="text-xs text-dark-500 mt-1">{entry.published}</p>
                      )}
                    </div>
                    {entry.link && (
                      <a
                        href={entry.link}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="p-1.5 hover:bg-dark-700 rounded-lg text-dark-400 hover:text-dark-200"
                      >
                        <ExternalLink className="w-4 h-4" />
                      </a>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default function RSSPage() {
  const feeds = useRSSStore((state) => state.feeds)
  const fetchFeeds = useRSSStore((state) => state.fetchFeeds)
  const addFeed = useRSSStore((state) => state.addFeed)
  const updateFeed = useRSSStore((state) => state.updateFeed)
  const deleteFeed = useRSSStore((state) => state.deleteFeed)
  const previewFeed = useRSSStore((state) => state.previewFeed)
  
  const instances = useQBStore((state) => state.instances)
  const fetchInstances = useQBStore((state) => state.fetchInstances)
  const sites = useSiteStore((state) => state.sites)
  const fetchSites = useSiteStore((state) => state.fetchSites)
  
  const [modalOpen, setModalOpen] = useState(false)
  const [editFeed, setEditFeed] = useState(null)
  const [previewOpen, setPreviewOpen] = useState(false)
  const [previewEntries, setPreviewEntries] = useState([])
  const [previewLoading, setPreviewLoading] = useState(false)
  
  useEffect(() => {
    fetchFeeds()
    fetchInstances()
    fetchSites()
  }, [])
  
  const handleOpenModal = (feed = null) => {
    setEditFeed(feed)
    setModalOpen(true)
  }
  
  const handleSubmit = async (data, feedId) => {
    if (feedId) {
      return await updateFeed(feedId, data)
    } else {
      return await addFeed(data)
    }
  }
  
  const handleDelete = async (id) => {
    if (!confirm('确定要删除此RSS订阅吗？')) return
    await deleteFeed(id)
  }
  
  const handlePreview = async (id) => {
    setPreviewOpen(true)
    setPreviewLoading(true)
    setPreviewEntries([])
    
    const result = await previewFeed(id)
    setPreviewLoading(false)
    
    if (result.success) {
      setPreviewEntries(result.data)
    }
  }
  
  return (
    <div className="space-y-6 animate-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-display font-bold">RSS订阅</h1>
          <p className="text-dark-400 mt-1">自动下载新种子</p>
        </div>
        <button
          onClick={() => handleOpenModal()}
          className="flex items-center gap-2 px-4 py-2 bg-accent-500 hover:bg-accent-600 
                     text-white rounded-xl transition-colors"
        >
          <Plus className="w-4 h-4" />
          添加订阅
        </button>
      </div>
      
      {/* Feeds Grid */}
      {feeds.length === 0 ? (
        <div className="bg-dark-900/50 border border-dark-800 rounded-2xl p-12 text-center">
          <Rss className="w-12 h-12 text-dark-600 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-dark-300 mb-2">暂无RSS订阅</h3>
          <p className="text-dark-500 mb-4">添加RSS订阅自动下载新种子</p>
          <button
            onClick={() => handleOpenModal()}
            className="inline-flex items-center gap-2 px-4 py-2 bg-accent-500 
                       hover:bg-accent-600 text-white rounded-xl transition-colors"
          >
            <Plus className="w-4 h-4" />
            添加订阅
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {feeds.map(feed => (
            <FeedCard
              key={feed.id}
              feed={feed}
              sites={sites}
              instances={instances}
              onEdit={handleOpenModal}
              onDelete={handleDelete}
              onPreview={handlePreview}
            />
          ))}
        </div>
      )}
      
      {/* Modals */}
      <FeedModal
        isOpen={modalOpen}
        onClose={() => setModalOpen(false)}
        feed={editFeed}
        instances={instances}
        sites={sites}
        onSubmit={handleSubmit}
      />
      
      <PreviewModal
        isOpen={previewOpen}
        onClose={() => setPreviewOpen(false)}
        entries={previewEntries}
        loading={previewLoading}
      />
    </div>
  )
}
