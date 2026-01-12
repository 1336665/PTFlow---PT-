import { useEffect, useState } from 'react'
import { 
  Server, 
  Plus, 
  Trash2, 
  RefreshCw, 
  Edit2, 
  X, 
  Check,
  Plug,
  PlugZap,
  Settings
} from 'lucide-react'
import { useQBStore } from '../store'
import api from '../api'

function InstanceCard({ instance, onEdit, onDelete, onConnect }) {
  return (
    <div className="bg-dark-900/50 border border-dark-800 rounded-2xl p-5 card-hover">
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div className={`
            p-3 rounded-xl 
            ${instance.connected 
              ? 'bg-accent-500/20 text-accent-400' 
              : 'bg-dark-800 text-dark-500'}
          `}>
            <Server className="w-6 h-6" />
          </div>
          <div>
            <h3 className="font-semibold text-lg">{instance.name}</h3>
            <p className="text-sm text-dark-400 mt-0.5">{instance.host}</p>
          </div>
        </div>
        <div className={`
          flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs
          ${instance.connected 
            ? 'bg-accent-500/20 text-accent-400' 
            : 'bg-dark-800 text-dark-500'}
        `}>
          <div className={`w-2 h-2 rounded-full ${instance.connected ? 'bg-accent-400 pulse-dot' : 'bg-dark-600'}`} />
          {instance.connected ? '已连接' : '未连接'}
        </div>
      </div>
      
      <div className="mt-4 pt-4 border-t border-dark-800 flex items-center gap-2">
        {!instance.connected && (
          <button
            onClick={() => onConnect(instance.id)}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-accent-500/20 text-accent-400 
                       hover:bg-accent-500/30 rounded-lg text-sm transition-colors"
          >
            <PlugZap className="w-4 h-4" />
            连接
          </button>
        )}
        <button
          onClick={() => onEdit(instance)}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-dark-800 text-dark-300 
                     hover:bg-dark-700 rounded-lg text-sm transition-colors"
        >
          <Edit2 className="w-4 h-4" />
          编辑
        </button>
        <button
          onClick={() => onDelete(instance.id)}
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

function InstanceModal({ isOpen, onClose, instance, onSubmit }) {
  const [form, setForm] = useState({
    name: '',
    host: '',
    username: '',
    password: ''
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  
  useEffect(() => {
    if (instance) {
      setForm({
        name: instance.name,
        host: instance.host,
        username: instance.username,
        password: ''
      })
    } else {
      setForm({ name: '', host: '', username: '', password: '' })
    }
    setError('')
  }, [instance, isOpen])
  
  if (!isOpen) return null
  
  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!form.name || !form.host || !form.username) {
      setError('请填写必填项')
      return
    }
    
    setLoading(true)
    setError('')
    
    try {
      const result = await onSubmit(form, instance?.id)
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
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4">
      <div className="bg-dark-900 border border-dark-800 rounded-2xl w-full max-w-lg animate-in">
        <div className="flex items-center justify-between p-4 border-b border-dark-800">
          <h3 className="font-semibold">{instance ? '编辑实例' : '添加qB实例'}</h3>
          <button onClick={onClose} className="p-1 hover:bg-dark-800 rounded-lg">
            <X className="w-5 h-5" />
          </button>
        </div>
        <form onSubmit={handleSubmit} className="p-4 space-y-4">
          <div>
            <label className="block text-sm text-dark-400 mb-2">实例名称 *</label>
            <input
              type="text"
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
              className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl
                         text-dark-100 placeholder-dark-500
                         focus:border-accent-500 focus:ring-1 focus:ring-accent-500/50"
              placeholder="例: 主力机"
            />
          </div>
          <div>
            <label className="block text-sm text-dark-400 mb-2">WebUI地址 *</label>
            <input
              type="text"
              value={form.host}
              onChange={(e) => setForm({ ...form, host: e.target.value })}
              className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl
                         text-dark-100 placeholder-dark-500
                         focus:border-accent-500 focus:ring-1 focus:ring-accent-500/50"
              placeholder="http://127.0.0.1:8080"
            />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-dark-400 mb-2">用户名 *</label>
              <input
                type="text"
                value={form.username}
                onChange={(e) => setForm({ ...form, username: e.target.value })}
                className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl
                           text-dark-100 placeholder-dark-500
                           focus:border-accent-500 focus:ring-1 focus:ring-accent-500/50"
                placeholder="admin"
              />
            </div>
            <div>
              <label className="block text-sm text-dark-400 mb-2">密码 {instance ? '' : '*'}</label>
              <input
                type="password"
                value={form.password}
                onChange={(e) => setForm({ ...form, password: e.target.value })}
                className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl
                           text-dark-100 placeholder-dark-500
                           focus:border-accent-500 focus:ring-1 focus:ring-accent-500/50"
                placeholder={instance ? '留空保持不变' : '密码'}
              />
            </div>
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
              {instance ? '保存' : '添加'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default function InstancesPage() {
  const instances = useQBStore((state) => state.instances)
  const fetchInstances = useQBStore((state) => state.fetchInstances)
  const addInstance = useQBStore((state) => state.addInstance)
  const deleteInstance = useQBStore((state) => state.deleteInstance)
  
  const [modalOpen, setModalOpen] = useState(false)
  const [editInstance, setEditInstance] = useState(null)
  
  useEffect(() => {
    fetchInstances()
  }, [])
  
  const handleOpenModal = (instance = null) => {
    setEditInstance(instance)
    setModalOpen(true)
  }
  
  const handleSubmit = async (data, instanceId) => {
    if (instanceId) {
      // 更新
      try {
        const updateData = { ...data }
        if (!updateData.password) delete updateData.password
        await api.put(`/api/qb/instances/${instanceId}`, updateData)
        await fetchInstances()
        return { success: true }
      } catch (err) {
        return { success: false, error: err.response?.data?.detail || '更新失败' }
      }
    } else {
      // 添加
      return await addInstance(data)
    }
  }
  
  const handleDelete = async (id) => {
    if (!confirm('确定要删除此实例吗？')) return
    await deleteInstance(id)
  }
  
  const handleConnect = async (id) => {
    try {
      await api.post(`/api/qb/instances/${id}/connect`)
      await fetchInstances()
    } catch (err) {
      alert(err.response?.data?.detail || '连接失败')
    }
  }
  
  return (
    <div className="space-y-6 animate-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-display font-bold">qBittorrent 实例</h1>
          <p className="text-dark-400 mt-1">管理多个qBittorrent客户端</p>
        </div>
        <button
          onClick={() => handleOpenModal()}
          className="flex items-center gap-2 px-4 py-2 bg-accent-500 hover:bg-accent-600 
                     text-white rounded-xl transition-colors"
        >
          <Plus className="w-4 h-4" />
          添加实例
        </button>
      </div>
      
      {/* Instances Grid */}
      {instances.length === 0 ? (
        <div className="bg-dark-900/50 border border-dark-800 rounded-2xl p-12 text-center">
          <Server className="w-12 h-12 text-dark-600 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-dark-300 mb-2">暂无实例</h3>
          <p className="text-dark-500 mb-4">添加您的第一个 qBittorrent 实例</p>
          <button
            onClick={() => handleOpenModal()}
            className="inline-flex items-center gap-2 px-4 py-2 bg-accent-500 
                       hover:bg-accent-600 text-white rounded-xl transition-colors"
          >
            <Plus className="w-4 h-4" />
            添加实例
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {instances.map(instance => (
            <InstanceCard
              key={instance.id}
              instance={instance}
              onEdit={handleOpenModal}
              onDelete={handleDelete}
              onConnect={handleConnect}
            />
          ))}
        </div>
      )}
      
      {/* Modal */}
      <InstanceModal
        isOpen={modalOpen}
        onClose={() => setModalOpen(false)}
        instance={editInstance}
        onSubmit={handleSubmit}
      />
    </div>
  )
}
