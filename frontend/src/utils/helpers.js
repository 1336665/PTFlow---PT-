// 格式化文件大小
export function formatSize(bytes, precision = 2) {
  if (bytes === 0) return '0 B'
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
  let value = Math.abs(bytes)
  let unitIndex = 0
  
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024
    unitIndex++
  }
  
  return `${value.toFixed(precision)} ${units[unitIndex]}`
}

// 格式化速度
export function formatSpeed(bytesPerSecond, precision = 1) {
  if (bytesPerSecond === 0) return '0 B/s'
  const units = ['B/s', 'KiB/s', 'MiB/s', 'GiB/s']
  let value = Math.abs(bytesPerSecond)
  let unitIndex = 0
  
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024
    unitIndex++
  }
  
  return `${value.toFixed(precision)} ${units[unitIndex]}`
}

// 格式化时长
export function formatDuration(seconds) {
  if (!seconds || seconds < 0) return '∞'
  seconds = Math.floor(seconds)
  
  if (seconds < 60) return `${seconds}s`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`
  
  const hours = Math.floor(seconds / 3600)
  const mins = Math.floor((seconds % 3600) / 60)
  return `${hours}h ${mins}m`
}

// 格式化日期
export function formatDate(timestamp) {
  if (!timestamp) return '-'
  const date = new Date(timestamp * 1000)
  return date.toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  })
}

// 格式化相对时间
export function formatRelativeTime(timestamp) {
  if (!timestamp) return '-'
  const now = Date.now() / 1000
  const diff = now - timestamp
  
  if (diff < 60) return '刚刚'
  if (diff < 3600) return `${Math.floor(diff / 60)} 分钟前`
  if (diff < 86400) return `${Math.floor(diff / 3600)} 小时前`
  if (diff < 604800) return `${Math.floor(diff / 86400)} 天前`
  
  return formatDate(timestamp)
}

// 获取种子状态文本和颜色
export function getTorrentStatus(state) {
  const statusMap = {
    'downloading': { text: '下载中', color: 'text-blue-400', bg: 'bg-blue-500/20' },
    'uploading': { text: '上传中', color: 'text-green-400', bg: 'bg-green-500/20' },
    'seeding': { text: '做种中', color: 'text-green-400', bg: 'bg-green-500/20' },
    'stalledUP': { text: '等待上传', color: 'text-yellow-400', bg: 'bg-yellow-500/20' },
    'stalledDL': { text: '等待下载', color: 'text-yellow-400', bg: 'bg-yellow-500/20' },
    'pausedUP': { text: '暂停(做种)', color: 'text-gray-400', bg: 'bg-gray-500/20' },
    'pausedDL': { text: '暂停(下载)', color: 'text-gray-400', bg: 'bg-gray-500/20' },
    'queuedUP': { text: '队列(做种)', color: 'text-orange-400', bg: 'bg-orange-500/20' },
    'queuedDL': { text: '队列(下载)', color: 'text-orange-400', bg: 'bg-orange-500/20' },
    'checkingUP': { text: '校验中', color: 'text-purple-400', bg: 'bg-purple-500/20' },
    'checkingDL': { text: '校验中', color: 'text-purple-400', bg: 'bg-purple-500/20' },
    'forcedUP': { text: '强制上传', color: 'text-green-400', bg: 'bg-green-500/20' },
    'forcedDL': { text: '强制下载', color: 'text-blue-400', bg: 'bg-blue-500/20' },
    'metaDL': { text: '获取元数据', color: 'text-cyan-400', bg: 'bg-cyan-500/20' },
    'error': { text: '错误', color: 'text-red-400', bg: 'bg-red-500/20' },
    'missingFiles': { text: '文件丢失', color: 'text-red-400', bg: 'bg-red-500/20' }
  }
  
  return statusMap[state] || { text: state, color: 'text-gray-400', bg: 'bg-gray-500/20' }
}

// 获取限速阶段文本和颜色
export function getPhaseInfo(phase) {
  const phaseMap = {
    'warmup': { text: '预热', color: 'text-orange-400', icon: '🔥' },
    'catch': { text: '追赶', color: 'text-blue-400', icon: '🏃' },
    'steady': { text: '稳定', color: 'text-green-400', icon: '⚖️' },
    'finish': { text: '收尾', color: 'text-purple-400', icon: '🎯' }
  }
  
  return phaseMap[phase] || { text: phase, color: 'text-gray-400', icon: '❓' }
}

// 类名合并
export function cn(...classes) {
  return classes.filter(Boolean).join(' ')
}

// 防抖
export function debounce(func, wait) {
  let timeout
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout)
      func(...args)
    }
    clearTimeout(timeout)
    timeout = setTimeout(later, wait)
  }
}

// 复制到剪贴板
export async function copyToClipboard(text) {
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch {
    return false
  }
}
