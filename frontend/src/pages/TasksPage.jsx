import { useEffect, useMemo, useState } from 'react'
import {
  Plus,
  Trash2,
  Edit2,
  X,
  RefreshCw,
  Rss,
  Trash,
  Clock,
  Filter
} from 'lucide-react'
import { useAutomationStore, useQBStore } from '../store'
import { formatRelativeTime } from '../utils/helpers'

function SectionHeader({ title, subtitle, onAdd }) {
  return (
    <div className="flex items-center justify-between">
      <div>
        <h2 className="text-xl font-display font-semibold">{title}</h2>
        <p className="text-dark-400 mt-1 text-sm">{subtitle}</p>
      </div>
      <button
        onClick={onAdd}
        className="flex items-center gap-2 px-4 py-2 bg-accent-500 hover:bg-accent-600 
                   text-white rounded-xl transition-colors"
      >
        <Plus className="w-4 h-4" />
        添加任务
      </button>
    </div>
  )
}

function RssTaskCard({ task, rules, instanceName, onEdit, onDelete, onAddRule, onEditRule, onDeleteRule, onPreview }) {
  return (
    <div className="bg-dark-900/50 border border-dark-800 rounded-2xl p-5 space-y-4">
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div className={`p-3 rounded-xl ${task.enabled ? 'bg-orange-500/20 text-orange-400' : 'bg-dark-800 text-dark-500'}`}>
            <Rss className="w-6 h-6" />
          </div>
          <div>
            <h3 className="font-semibold text-lg">{task.name}</h3>
            <p className="text-xs text-dark-500 mt-0.5 truncate max-w-[220px]" title={task.url}>
              {task.url}
            </p>
          </div>
        </div>
        <div className={`px-2.5 py-1 rounded-full text-xs ${task.enabled ? 'bg-accent-500/20 text-accent-400' : 'bg-dark-800 text-dark-500'}`}>
          {task.enabled ? '运行中' : '已暂停'}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2 text-sm">
        <div className="flex items-center justify-between">
          <span className="text-dark-500">下载实例</span>
          <span className="text-dark-300">{instanceName || '-'}</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-dark-500">检查间隔</span>
          <span className="text-dark-300">{task.interval_minutes} 分钟</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-dark-500">保存路径</span>
          <span className="text-dark-300 truncate max-w-[140px]" title={task.save_path}>
            {task.save_path || '默认'}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-dark-500">分类/标签</span>
          <span className="text-dark-300 truncate max-w-[140px]" title={`${task.category || ''} ${task.tags || ''}`}>
            {task.category || '默认'} {task.tags ? `· ${task.tags}` : ''}
          </span>
        </div>
        {task.last_run && (
          <div className="flex items-center gap-1 text-xs text-dark-500">
            <Clock className="w-3 h-3" />
            上次执行: {formatRelativeTime(task.last_run)}
          </div>
        )}
      </div>

      <div className="border-t border-dark-800 pt-3 space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1 text-xs text-dark-500">
            <Filter className="w-3 h-3" />
            规则列表
          </div>
          <button
            onClick={() => onAddRule(task)}
            className="text-xs text-accent-400 hover:text-accent-300"
          >
            + 添加规则
          </button>
        </div>
        {rules.length === 0 ? (
          <p className="text-xs text-dark-500">暂无规则，默认下载所有条目。</p>
        ) : (
          <div className="space-y-2">
            {rules.map(rule => (
              <div key={rule.id} className="flex items-start justify-between bg-dark-800/60 rounded-lg px-3 py-2">
                <div>
                  <p className="text-sm text-dark-200">{rule.name}</p>
                  <p className="text-xs text-dark-500">
                    {rule.include_keywords ? `包含: ${rule.include_keywords}` : '包含: 任意'}
                    {rule.exclude_keywords ? ` | 排除: ${rule.exclude_keywords}` : ''}
                  </p>
                  {(rule.save_path || rule.category || rule.tags) && (
                    <p className="text-xs text-dark-500">
                      覆盖: {rule.save_path || '默认路径'} / {rule.category || '默认分类'}
                      {rule.tags ? ` / ${rule.tags}` : ''}
                    </p>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <button onClick={() => onEditRule(task, rule)} className="text-dark-400 hover:text-dark-200">
                    <Edit2 className="w-4 h-4" />
                  </button>
                  <button onClick={() => onDeleteRule(rule.id)} className="text-red-400 hover:text-red-300">
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="flex items-center gap-2 pt-2">
        <button
          onClick={() => onPreview(task.id)}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-500/20 text-blue-400 
                     hover:bg-blue-500/30 rounded-lg text-sm transition-colors"
        >
          预览
        </button>
        <button
          onClick={() => onEdit(task)}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-dark-800 text-dark-300 
                     hover:bg-dark-700 rounded-lg text-sm transition-colors"
        >
          <Edit2 className="w-4 h-4" />
          编辑
        </button>
        <button
          onClick={() => onDelete(task.id)}
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

function DeleteTaskCard({ task, rules, instanceName, onEdit, onDelete, onAddRule, onEditRule, onDeleteRule }) {
  return (
    <div className="bg-dark-900/50 border border-dark-800 rounded-2xl p-5 space-y-4">
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div className={`p-3 rounded-xl ${task.enabled ? 'bg-red-500/20 text-red-400' : 'bg-dark-800 text-dark-500'}`}>
            <Trash className="w-6 h-6" />
          </div>
          <div>
            <h3 className="font-semibold text-lg">{task.name}</h3>
            <p className="text-xs text-dark-500 mt-0.5">实例: {instanceName || '-'}</p>
          </div>
        </div>
        <div className={`px-2.5 py-1 rounded-full text-xs ${task.enabled ? 'bg-red-500/20 text-red-300' : 'bg-dark-800 text-dark-500'}`}>
          {task.enabled ? '运行中' : '已暂停'}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2 text-sm">
        <div className="flex items-center justify-between">
          <span className="text-dark-500">检查间隔</span>
          <span className="text-dark-300">{task.interval_minutes} 分钟</span>
        </div>
        {task.last_run && (
          <div className="flex items-center gap-1 text-xs text-dark-500">
            <Clock className="w-3 h-3" />
            上次执行: {formatRelativeTime(task.last_run)}
          </div>
        )}
      </div>

      <div className="border-t border-dark-800 pt-3 space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1 text-xs text-dark-500">
            <Filter className="w-3 h-3" />
            规则列表
          </div>
          <button
            onClick={() => onAddRule(task)}
            className="text-xs text-accent-400 hover:text-accent-300"
          >
            + 添加规则
          </button>
        </div>
        {rules.length === 0 ? (
          <p className="text-xs text-dark-500">暂无删种规则。</p>
        ) : (
          <div className="space-y-2">
            {rules.map(rule => (
              <div key={rule.id} className="flex items-start justify-between bg-dark-800/60 rounded-lg px-3 py-2">
                <div>
                  <p className="text-sm text-dark-200">{rule.name}</p>
                  <p className="text-xs text-dark-500">
                    {rule.min_ratio ? `分享率 ≥ ${rule.min_ratio}` : '分享率不限'}
                    {rule.min_seeding_hours ? ` | 做种 ≥ ${rule.min_seeding_hours}h` : ''}
                    {rule.min_uploaded_gb ? ` | 上传 ≥ ${rule.min_uploaded_gb}GB` : ''}
                  </p>
                  {(rule.include_categories || rule.include_tags) && (
                    <p className="text-xs text-dark-500">
                      仅匹配: {rule.include_categories || '全部分类'}
                      {rule.include_tags ? ` / ${rule.include_tags}` : ''}
                    </p>
                  )}
                  {rule.conditions_json && (
                    <p className="text-xs text-blue-300">自定义条件: 已配置</p>
                  )}
                  {rule.delete_files && (
                    <p className="text-xs text-red-400">删除文件</p>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <button onClick={() => onEditRule(task, rule)} className="text-dark-400 hover:text-dark-200">
                    <Edit2 className="w-4 h-4" />
                  </button>
                  <button onClick={() => onDeleteRule(rule.id)} className="text-red-400 hover:text-red-300">
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="flex items-center gap-2 pt-2">
        <button
          onClick={() => onEdit(task)}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-dark-800 text-dark-300 
                     hover:bg-dark-700 rounded-lg text-sm transition-colors"
        >
          <Edit2 className="w-4 h-4" />
          编辑
        </button>
        <button
          onClick={() => onDelete(task.id)}
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

function ModalShell({ title, onClose, children }) {
  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4 overflow-y-auto">
      <div className="bg-dark-900 border border-dark-800 rounded-2xl w-full max-w-lg my-8 animate-in">
        <div className="flex items-center justify-between p-4 border-b border-dark-800">
          <h3 className="font-semibold">{title}</h3>
          <button onClick={onClose} className="p-1 hover:bg-dark-800 rounded-lg">
            <X className="w-5 h-5" />
          </button>
        </div>
        {children}
      </div>
    </div>
  )
}

function RssTaskModal({ isOpen, onClose, task, instances, onSubmit }) {
  const [form, setForm] = useState({
    name: '',
    url: '',
    qb_instance_id: '',
    save_path: '',
    category: '',
    tags: '',
    interval_minutes: 10,
    enabled: true,
    paused: false
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (task) {
      setForm({
        name: task.name,
        url: task.url,
        qb_instance_id: task.qb_instance_id || '',
        save_path: task.save_path || '',
        category: task.category || '',
        tags: task.tags || '',
        interval_minutes: task.interval_minutes || 10,
        enabled: task.enabled,
        paused: task.paused
      })
    } else {
      setForm({
        name: '',
        url: '',
        qb_instance_id: instances[0]?.id || '',
        save_path: '',
        category: '',
        tags: '',
        interval_minutes: 10,
        enabled: true,
        paused: false
      })
    }
    setError('')
  }, [task, isOpen, instances])

  if (!isOpen) return null

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!form.name || !form.url || !form.qb_instance_id) {
      setError('请填写必填项')
      return
    }
    setLoading(true)
    setError('')
    const payload = {
      ...form,
      qb_instance_id: parseInt(form.qb_instance_id, 10),
      interval_minutes: parseInt(form.interval_minutes, 10)
    }
    const result = await onSubmit(payload, task?.id)
    setLoading(false)
    if (result.success) {
      onClose()
    } else {
      setError(result.error)
    }
  }

  return (
    <ModalShell title={task ? '编辑RSS任务' : '添加RSS任务'} onClose={onClose}>
      <form onSubmit={handleSubmit} className="p-4 space-y-4">
        <div>
          <label className="block text-sm text-dark-400 mb-2">任务名称 *</label>
          <input
            type="text"
            value={form.name}
            onChange={(e) => setForm({ ...form, name: e.target.value })}
            className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl
                       text-dark-100 placeholder-dark-500 focus:border-accent-500"
          />
        </div>
        <div>
          <label className="block text-sm text-dark-400 mb-2">RSS地址 *</label>
          <input
            type="url"
            value={form.url}
            onChange={(e) => setForm({ ...form, url: e.target.value })}
            className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl
                       text-dark-100 placeholder-dark-500 focus:border-accent-500"
          />
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm text-dark-400 mb-2">下载实例 *</label>
            <select
              value={form.qb_instance_id}
              onChange={(e) => setForm({ ...form, qb_instance_id: e.target.value })}
              className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl text-dark-100"
            >
              <option value="">选择实例</option>
              {instances.map(inst => (
                <option key={inst.id} value={inst.id}>{inst.name}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm text-dark-400 mb-2">检查间隔(分钟)</label>
            <input
              type="number"
              value={form.interval_minutes}
              onChange={(e) => setForm({ ...form, interval_minutes: e.target.value })}
              className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl text-dark-100"
              min="1"
            />
          </div>
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm text-dark-400 mb-2">保存路径</label>
            <input
              type="text"
              value={form.save_path}
              onChange={(e) => setForm({ ...form, save_path: e.target.value })}
              className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl text-dark-100"
            />
          </div>
          <div>
            <label className="block text-sm text-dark-400 mb-2">分类</label>
            <input
              type="text"
              value={form.category}
              onChange={(e) => setForm({ ...form, category: e.target.value })}
              className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl text-dark-100"
            />
          </div>
        </div>
        <div>
          <label className="block text-sm text-dark-400 mb-2">标签 (逗号分隔)</label>
          <input
            type="text"
            value={form.tags}
            onChange={(e) => setForm({ ...form, tags: e.target.value })}
            className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl text-dark-100"
          />
        </div>
        <div className="flex items-center gap-4">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={form.enabled}
              onChange={(e) => setForm({ ...form, enabled: e.target.checked })}
              className="w-4 h-4 rounded border-dark-600 bg-dark-800 text-accent-500"
            />
            <span className="text-sm text-dark-300">启用任务</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={form.paused}
              onChange={(e) => setForm({ ...form, paused: e.target.checked })}
              className="w-4 h-4 rounded border-dark-600 bg-dark-800 text-accent-500"
            />
            <span className="text-sm text-dark-300">添加时暂停</span>
          </label>
        </div>
        {error && (
          <div className="px-4 py-3 bg-red-500/10 border border-red-500/30 rounded-xl">
            <p className="text-sm text-red-400">{error}</p>
          </div>
        )}
        <div className="flex justify-end gap-3 pt-2">
          <button type="button" onClick={onClose} className="px-4 py-2 text-dark-400 hover:text-dark-200">
            取消
          </button>
          <button
            type="submit"
            disabled={loading}
            className="px-4 py-2 bg-accent-500 hover:bg-accent-600 text-white rounded-lg 
                       transition-colors disabled:opacity-50 flex items-center gap-2"
          >
            {loading && <RefreshCw className="w-4 h-4 animate-spin" />}
            {task ? '保存' : '添加'}
          </button>
        </div>
      </form>
    </ModalShell>
  )
}

function RssRuleModal({ isOpen, onClose, rule, onSubmit }) {
  const [form, setForm] = useState({
    name: '',
    include_keywords: '',
    exclude_keywords: '',
    save_path: '',
    category: '',
    tags: '',
    paused: false
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (rule) {
      setForm({
        name: rule.name,
        include_keywords: rule.include_keywords || '',
        exclude_keywords: rule.exclude_keywords || '',
        save_path: rule.save_path || '',
        category: rule.category || '',
        tags: rule.tags || '',
        paused: rule.paused
      })
    } else {
      setForm({
        name: '',
        include_keywords: '',
        exclude_keywords: '',
        save_path: '',
        category: '',
        tags: '',
        paused: false
      })
    }
    setError('')
  }, [rule, isOpen])

  if (!isOpen) return null

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!form.name) {
      setError('请填写规则名称')
      return
    }
    setLoading(true)
    setError('')
    const result = await onSubmit(form, rule?.id)
    setLoading(false)
    if (result.success) {
      onClose()
    } else {
      setError(result.error)
    }
  }

  return (
    <ModalShell title={rule ? '编辑RSS规则' : '添加RSS规则'} onClose={onClose}>
      <form onSubmit={handleSubmit} className="p-4 space-y-4">
        <div>
          <label className="block text-sm text-dark-400 mb-2">规则名称 *</label>
          <input
            type="text"
            value={form.name}
            onChange={(e) => setForm({ ...form, name: e.target.value })}
            className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl text-dark-100"
          />
        </div>
        <div>
          <label className="block text-sm text-dark-400 mb-2">包含关键词</label>
          <input
            type="text"
            value={form.include_keywords}
            onChange={(e) => setForm({ ...form, include_keywords: e.target.value })}
            className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl text-dark-100"
          />
        </div>
        <div>
          <label className="block text-sm text-dark-400 mb-2">排除关键词</label>
          <input
            type="text"
            value={form.exclude_keywords}
            onChange={(e) => setForm({ ...form, exclude_keywords: e.target.value })}
            className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl text-dark-100"
          />
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm text-dark-400 mb-2">覆盖保存路径</label>
            <input
              type="text"
              value={form.save_path}
              onChange={(e) => setForm({ ...form, save_path: e.target.value })}
              className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl text-dark-100"
            />
          </div>
          <div>
            <label className="block text-sm text-dark-400 mb-2">覆盖分类</label>
            <input
              type="text"
              value={form.category}
              onChange={(e) => setForm({ ...form, category: e.target.value })}
              className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl text-dark-100"
            />
          </div>
        </div>
        <div>
          <label className="block text-sm text-dark-400 mb-2">覆盖标签</label>
          <input
            type="text"
            value={form.tags}
            onChange={(e) => setForm({ ...form, tags: e.target.value })}
            className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl text-dark-100"
          />
        </div>
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={form.paused}
            onChange={(e) => setForm({ ...form, paused: e.target.checked })}
            className="w-4 h-4 rounded border-dark-600 bg-dark-800 text-accent-500"
          />
          <span className="text-sm text-dark-300">添加时暂停</span>
        </label>
        {error && (
          <div className="px-4 py-3 bg-red-500/10 border border-red-500/30 rounded-xl">
            <p className="text-sm text-red-400">{error}</p>
          </div>
        )}
        <div className="flex justify-end gap-3 pt-2">
          <button type="button" onClick={onClose} className="px-4 py-2 text-dark-400 hover:text-dark-200">
            取消
          </button>
          <button
            type="submit"
            disabled={loading}
            className="px-4 py-2 bg-accent-500 hover:bg-accent-600 text-white rounded-lg 
                       transition-colors disabled:opacity-50 flex items-center gap-2"
          >
            {loading && <RefreshCw className="w-4 h-4 animate-spin" />}
            {rule ? '保存' : '添加'}
          </button>
        </div>
      </form>
    </ModalShell>
  )
}

function DeleteTaskModal({ isOpen, onClose, task, instances, onSubmit }) {
  const [form, setForm] = useState({
    name: '',
    qb_instance_id: '',
    interval_minutes: 60,
    enabled: true
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (task) {
      setForm({
        name: task.name,
        qb_instance_id: task.qb_instance_id || '',
        interval_minutes: task.interval_minutes || 60,
        enabled: task.enabled
      })
    } else {
      setForm({
        name: '',
        qb_instance_id: instances[0]?.id || '',
        interval_minutes: 60,
        enabled: true
      })
    }
    setError('')
  }, [task, isOpen, instances])

  if (!isOpen) return null

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!form.name || !form.qb_instance_id) {
      setError('请填写必填项')
      return
    }
    setLoading(true)
    setError('')
    const payload = {
      ...form,
      qb_instance_id: parseInt(form.qb_instance_id, 10),
      interval_minutes: parseInt(form.interval_minutes, 10)
    }
    const result = await onSubmit(payload, task?.id)
    setLoading(false)
    if (result.success) {
      onClose()
    } else {
      setError(result.error)
    }
  }

  return (
    <ModalShell title={task ? '编辑删种任务' : '添加删种任务'} onClose={onClose}>
      <form onSubmit={handleSubmit} className="p-4 space-y-4">
        <div>
          <label className="block text-sm text-dark-400 mb-2">任务名称 *</label>
          <input
            type="text"
            value={form.name}
            onChange={(e) => setForm({ ...form, name: e.target.value })}
            className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl text-dark-100"
          />
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm text-dark-400 mb-2">实例 *</label>
            <select
              value={form.qb_instance_id}
              onChange={(e) => setForm({ ...form, qb_instance_id: e.target.value })}
              className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl text-dark-100"
            >
              <option value="">选择实例</option>
              {instances.map(inst => (
                <option key={inst.id} value={inst.id}>{inst.name}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm text-dark-400 mb-2">检查间隔(分钟)</label>
            <input
              type="number"
              value={form.interval_minutes}
              onChange={(e) => setForm({ ...form, interval_minutes: e.target.value })}
              className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl text-dark-100"
              min="1"
            />
          </div>
        </div>
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={form.enabled}
            onChange={(e) => setForm({ ...form, enabled: e.target.checked })}
            className="w-4 h-4 rounded border-dark-600 bg-dark-800 text-accent-500"
          />
          <span className="text-sm text-dark-300">启用任务</span>
        </label>
        {error && (
          <div className="px-4 py-3 bg-red-500/10 border border-red-500/30 rounded-xl">
            <p className="text-sm text-red-400">{error}</p>
          </div>
        )}
        <div className="flex justify-end gap-3 pt-2">
          <button type="button" onClick={onClose} className="px-4 py-2 text-dark-400 hover:text-dark-200">
            取消
          </button>
          <button
            type="submit"
            disabled={loading}
            className="px-4 py-2 bg-accent-500 hover:bg-accent-600 text-white rounded-lg 
                       transition-colors disabled:opacity-50 flex items-center gap-2"
          >
            {loading && <RefreshCw className="w-4 h-4 animate-spin" />}
            {task ? '保存' : '添加'}
          </button>
        </div>
      </form>
    </ModalShell>
  )
}

function DeleteRuleModal({ isOpen, onClose, rule, onSubmit }) {
  const [form, setForm] = useState({
    name: '',
    min_ratio: '',
    min_seeding_hours: '',
    min_uploaded_gb: '',
    max_size_gb: '',
    include_categories: '',
    exclude_categories: '',
    include_tags: '',
    exclude_tags: '',
    conditions_json: '',
    delete_files: false
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (rule) {
      setForm({
        name: rule.name,
        min_ratio: rule.min_ratio ?? '',
        min_seeding_hours: rule.min_seeding_hours ?? '',
        min_uploaded_gb: rule.min_uploaded_gb ?? '',
        max_size_gb: rule.max_size_gb ?? '',
        include_categories: rule.include_categories || '',
        exclude_categories: rule.exclude_categories || '',
        include_tags: rule.include_tags || '',
        exclude_tags: rule.exclude_tags || '',
        conditions_json: rule.conditions_json || '',
        delete_files: rule.delete_files
      })
    } else {
      setForm({
        name: '',
        min_ratio: '',
        min_seeding_hours: '',
        min_uploaded_gb: '',
        max_size_gb: '',
        include_categories: '',
        exclude_categories: '',
        include_tags: '',
        exclude_tags: '',
        conditions_json: '',
        delete_files: false
      })
    }
    setError('')
  }, [rule, isOpen])

  if (!isOpen) return null

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!form.name) {
      setError('请填写规则名称')
      return
    }
    setLoading(true)
    setError('')
    const payload = {
      ...form,
      min_ratio: form.min_ratio === '' ? null : parseFloat(form.min_ratio),
      min_seeding_hours: form.min_seeding_hours === '' ? null : parseFloat(form.min_seeding_hours),
      min_uploaded_gb: form.min_uploaded_gb === '' ? null : parseFloat(form.min_uploaded_gb),
      max_size_gb: form.max_size_gb === '' ? null : parseFloat(form.max_size_gb),
      conditions_json: form.conditions_json.trim() === '' ? null : form.conditions_json.trim()
    }
    const result = await onSubmit(payload, rule?.id)
    setLoading(false)
    if (result.success) {
      onClose()
    } else {
      setError(result.error)
    }
  }

  return (
    <ModalShell title={rule ? '编辑删种规则' : '添加删种规则'} onClose={onClose}>
      <form onSubmit={handleSubmit} className="p-4 space-y-4">
        <div>
          <label className="block text-sm text-dark-400 mb-2">规则名称 *</label>
          <input
            type="text"
            value={form.name}
            onChange={(e) => setForm({ ...form, name: e.target.value })}
            className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl text-dark-100"
          />
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm text-dark-400 mb-2">最小分享率</label>
            <input
              type="number"
              step="0.1"
              value={form.min_ratio}
              onChange={(e) => setForm({ ...form, min_ratio: e.target.value })}
              className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl text-dark-100"
            />
          </div>
          <div>
            <label className="block text-sm text-dark-400 mb-2">最小做种时长(小时)</label>
            <input
              type="number"
              step="0.1"
              value={form.min_seeding_hours}
              onChange={(e) => setForm({ ...form, min_seeding_hours: e.target.value })}
              className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl text-dark-100"
            />
          </div>
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm text-dark-400 mb-2">最小上传(GB)</label>
            <input
              type="number"
              step="0.1"
              value={form.min_uploaded_gb}
              onChange={(e) => setForm({ ...form, min_uploaded_gb: e.target.value })}
              className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl text-dark-100"
            />
          </div>
          <div>
            <label className="block text-sm text-dark-400 mb-2">最大体积(GB)</label>
            <input
              type="number"
              step="0.1"
              value={form.max_size_gb}
              onChange={(e) => setForm({ ...form, max_size_gb: e.target.value })}
              className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl text-dark-100"
            />
          </div>
        </div>
        <div>
          <label className="block text-sm text-dark-400 mb-2">包含分类</label>
          <input
            type="text"
            value={form.include_categories}
            onChange={(e) => setForm({ ...form, include_categories: e.target.value })}
            className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl text-dark-100"
          />
        </div>
        <div>
          <label className="block text-sm text-dark-400 mb-2">排除分类</label>
          <input
            type="text"
            value={form.exclude_categories}
            onChange={(e) => setForm({ ...form, exclude_categories: e.target.value })}
            className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl text-dark-100"
          />
        </div>
        <div>
          <label className="block text-sm text-dark-400 mb-2">包含标签</label>
          <input
            type="text"
            value={form.include_tags}
            onChange={(e) => setForm({ ...form, include_tags: e.target.value })}
            className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl text-dark-100"
          />
        </div>
        <div>
          <label className="block text-sm text-dark-400 mb-2">排除标签</label>
          <input
            type="text"
            value={form.exclude_tags}
            onChange={(e) => setForm({ ...form, exclude_tags: e.target.value })}
            className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl text-dark-100"
          />
        </div>
        <div>
          <label className="block text-sm text-dark-400 mb-2">自定义条件(JSON)</label>
          <textarea
            rows="4"
            value={form.conditions_json}
            onChange={(e) => setForm({ ...form, conditions_json: e.target.value })}
            className="w-full px-4 py-2.5 bg-dark-800 border border-dark-700 rounded-xl text-dark-100"
            placeholder='[{"field":"ratio","op":">=","value":1},{"field":"seeding_hours","op":">=","value":24}]'
          />
          <p className="text-xs text-dark-500 mt-1">支持字段：ratio、seeding_hours、uploaded_gb、size_gb、category、tags、name、tracker、state、progress。</p>
        </div>
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={form.delete_files}
            onChange={(e) => setForm({ ...form, delete_files: e.target.checked })}
            className="w-4 h-4 rounded border-dark-600 bg-dark-800 text-red-500"
          />
          <span className="text-sm text-dark-300">同时删除文件</span>
        </label>
        {error && (
          <div className="px-4 py-3 bg-red-500/10 border border-red-500/30 rounded-xl">
            <p className="text-sm text-red-400">{error}</p>
          </div>
        )}
        <div className="flex justify-end gap-3 pt-2">
          <button type="button" onClick={onClose} className="px-4 py-2 text-dark-400 hover:text-dark-200">
            取消
          </button>
          <button
            type="submit"
            disabled={loading}
            className="px-4 py-2 bg-accent-500 hover:bg-accent-600 text-white rounded-lg 
                       transition-colors disabled:opacity-50 flex items-center gap-2"
          >
            {loading && <RefreshCw className="w-4 h-4 animate-spin" />}
            {rule ? '保存' : '添加'}
          </button>
        </div>
      </form>
    </ModalShell>
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
                <div key={idx} className="p-3 bg-dark-800/50 rounded-xl hover:bg-dark-800 transition-colors">
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
                        查看
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

export default function TasksPage() {
  const instances = useQBStore((state) => state.instances)
  const fetchInstances = useQBStore((state) => state.fetchInstances)

  const {
    rssTasks,
    rssRules,
    deleteTasks,
    deleteRules,
    fetchRssTasks,
    fetchRssRules,
    addRssTask,
    updateRssTask,
    deleteRssTask,
    addRssRule,
    updateRssRule,
    deleteRssRule,
    previewRssTask,
    fetchDeleteTasks,
    fetchDeleteRules,
    addDeleteTask,
    updateDeleteTask,
    deleteDeleteTask,
    addDeleteRule,
    updateDeleteRule,
    deleteDeleteRule
  } = useAutomationStore()

  const [rssTaskModalOpen, setRssTaskModalOpen] = useState(false)
  const [rssRuleModalOpen, setRssRuleModalOpen] = useState(false)
  const [deleteTaskModalOpen, setDeleteTaskModalOpen] = useState(false)
  const [deleteRuleModalOpen, setDeleteRuleModalOpen] = useState(false)
  const [editingRssTask, setEditingRssTask] = useState(null)
  const [editingDeleteTask, setEditingDeleteTask] = useState(null)
  const [editingRule, setEditingRule] = useState(null)
  const [activeTaskId, setActiveTaskId] = useState(null)
  const [previewOpen, setPreviewOpen] = useState(false)
  const [previewEntries, setPreviewEntries] = useState([])
  const [previewLoading, setPreviewLoading] = useState(false)

  useEffect(() => {
    fetchInstances()
    fetchRssTasks()
    fetchDeleteTasks()
  }, [])

  useEffect(() => {
    rssTasks.forEach(task => {
      fetchRssRules(task.id)
    })
  }, [rssTasks])

  useEffect(() => {
    deleteTasks.forEach(task => {
      fetchDeleteRules(task.id)
    })
  }, [deleteTasks])

  const instancesById = useMemo(() => {
    return instances.reduce((acc, inst) => {
      acc[inst.id] = inst.name
      return acc
    }, {})
  }, [instances])

  const handleRssTaskSubmit = async (data, taskId) => {
    if (taskId) {
      return await updateRssTask(taskId, data)
    }
    return await addRssTask(data)
  }

  const handleRssRuleSubmit = async (data, ruleId) => {
    if (ruleId) {
      return await updateRssRule(ruleId, data, activeTaskId)
    }
    return await addRssRule(activeTaskId, data)
  }

  const handleDeleteTaskSubmit = async (data, taskId) => {
    if (taskId) {
      return await updateDeleteTask(taskId, data)
    }
    return await addDeleteTask(data)
  }

  const handleDeleteRuleSubmit = async (data, ruleId) => {
    if (ruleId) {
      return await updateDeleteRule(ruleId, data, activeTaskId)
    }
    return await addDeleteRule(activeTaskId, data)
  }

  const openRssRuleModal = (task, rule = null) => {
    setActiveTaskId(task.id)
    setEditingRule(rule)
    setRssRuleModalOpen(true)
  }

  const openDeleteRuleModal = (task, rule = null) => {
    setActiveTaskId(task.id)
    setEditingRule(rule)
    setDeleteRuleModalOpen(true)
  }

  const handlePreview = async (taskId) => {
    setPreviewOpen(true)
    setPreviewLoading(true)
    setPreviewEntries([])
    const result = await previewRssTask(taskId)
    setPreviewLoading(false)
    if (result.success) {
      setPreviewEntries(result.data)
    }
  }

  return (
    <div className="space-y-10 animate-in">
      <section className="space-y-6">
        <SectionHeader
          title="RSS任务"
          subtitle="按照规则自动订阅和下载种子"
          onAdd={() => {
            setEditingRssTask(null)
            setRssTaskModalOpen(true)
          }}
        />
        {rssTasks.length === 0 ? (
          <div className="bg-dark-900/50 border border-dark-800 rounded-2xl p-12 text-center">
            <Rss className="w-12 h-12 text-dark-600 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-dark-300 mb-2">暂无RSS任务</h3>
            <p className="text-dark-500 mb-4">添加任务后即可设置规则</p>
            <button
              onClick={() => {
                setEditingRssTask(null)
                setRssTaskModalOpen(true)
              }}
              className="inline-flex items-center gap-2 px-4 py-2 bg-accent-500 
                         hover:bg-accent-600 text-white rounded-xl transition-colors"
            >
              <Plus className="w-4 h-4" />
              添加任务
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {rssTasks.map(task => (
              <RssTaskCard
                key={task.id}
                task={task}
                rules={rssRules[task.id] || []}
                instanceName={instancesById[task.qb_instance_id]}
                onEdit={(data) => {
                  setEditingRssTask(data)
                  setRssTaskModalOpen(true)
                }}
                onDelete={async (id) => {
                  if (!confirm('确定要删除此RSS任务吗？')) return
                  await deleteRssTask(id)
                }}
                onAddRule={openRssRuleModal}
                onEditRule={openRssRuleModal}
                onDeleteRule={async (ruleId) => {
                  if (!confirm('确定要删除此RSS规则吗？')) return
                  await deleteRssRule(ruleId, task.id)
                }}
                onPreview={handlePreview}
              />
            ))}
          </div>
        )}
      </section>

      <section className="space-y-6">
        <SectionHeader
          title="删种任务"
          subtitle="按规则自动清理做种任务"
          onAdd={() => {
            setEditingDeleteTask(null)
            setDeleteTaskModalOpen(true)
          }}
        />
        {deleteTasks.length === 0 ? (
          <div className="bg-dark-900/50 border border-dark-800 rounded-2xl p-12 text-center">
            <Trash className="w-12 h-12 text-dark-600 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-dark-300 mb-2">暂无删种任务</h3>
            <p className="text-dark-500 mb-4">添加任务并设置规则以自动删种</p>
            <button
              onClick={() => {
                setEditingDeleteTask(null)
                setDeleteTaskModalOpen(true)
              }}
              className="inline-flex items-center gap-2 px-4 py-2 bg-accent-500 
                         hover:bg-accent-600 text-white rounded-xl transition-colors"
            >
              <Plus className="w-4 h-4" />
              添加任务
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {deleteTasks.map(task => (
              <DeleteTaskCard
                key={task.id}
                task={task}
                rules={deleteRules[task.id] || []}
                instanceName={instancesById[task.qb_instance_id]}
                onEdit={(data) => {
                  setEditingDeleteTask(data)
                  setDeleteTaskModalOpen(true)
                }}
                onDelete={async (id) => {
                  if (!confirm('确定要删除此删种任务吗？')) return
                  await deleteDeleteTask(id)
                }}
                onAddRule={openDeleteRuleModal}
                onEditRule={openDeleteRuleModal}
                onDeleteRule={async (ruleId) => {
                  if (!confirm('确定要删除此删种规则吗？')) return
                  await deleteDeleteRule(ruleId, task.id)
                }}
              />
            ))}
          </div>
        )}
      </section>

      <RssTaskModal
        isOpen={rssTaskModalOpen}
        onClose={() => setRssTaskModalOpen(false)}
        task={editingRssTask}
        instances={instances}
        onSubmit={handleRssTaskSubmit}
      />
      <RssRuleModal
        isOpen={rssRuleModalOpen}
        onClose={() => setRssRuleModalOpen(false)}
        rule={editingRule}
        onSubmit={handleRssRuleSubmit}
      />
      <DeleteTaskModal
        isOpen={deleteTaskModalOpen}
        onClose={() => setDeleteTaskModalOpen(false)}
        task={editingDeleteTask}
        instances={instances}
        onSubmit={handleDeleteTaskSubmit}
      />
      <DeleteRuleModal
        isOpen={deleteRuleModalOpen}
        onClose={() => setDeleteRuleModalOpen(false)}
        rule={editingRule}
        onSubmit={handleDeleteRuleSubmit}
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
