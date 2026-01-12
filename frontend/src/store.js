import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import api from './api'

// 认证状态
export const useAuthStore = create(
  persist(
    (set, get) => ({
      token: null,
      username: null,
      isAuthenticated: false,
      
      login: async (username, password) => {
        try {
          const res = await api.post('/api/auth/login', { username, password })
          set({
            token: res.data.token,
            username: res.data.username,
            isAuthenticated: true
          })
          api.defaults.headers.common['Authorization'] = `Bearer ${res.data.token}`
          return { success: true }
        } catch (err) {
          return { success: false, error: err.response?.data?.detail || '登录失败' }
        }
      },
      
      logout: () => {
        set({ token: null, username: null, isAuthenticated: false })
        delete api.defaults.headers.common['Authorization']
      },
      
      initAuth: () => {
        const { token } = get()
        if (token) {
          api.defaults.headers.common['Authorization'] = `Bearer ${token}`
        }
      }
    }),
    {
      name: 'ptflow-auth',
      partialize: (state) => ({ token: state.token, username: state.username, isAuthenticated: state.isAuthenticated })
    }
  )
)

// qB实例状态
export const useQBStore = create((set, get) => ({
  instances: [],
  selectedInstance: null,
  torrents: [],
  loading: false,
  
  fetchInstances: async () => {
    try {
      const res = await api.get('/api/qb/instances')
      set({ instances: res.data })
      if (res.data.length > 0 && !get().selectedInstance) {
        set({ selectedInstance: res.data[0].id })
      }
    } catch (err) {
      console.error('获取实例失败:', err)
    }
  },
  
  setSelectedInstance: (id) => set({ selectedInstance: id }),
  
  fetchTorrents: async (instanceId) => {
    if (!instanceId) return
    set({ loading: true })
    try {
      const res = await api.get(`/api/qb/instances/${instanceId}/torrents`)
      set({ torrents: res.data, loading: false })
    } catch (err) {
      set({ loading: false })
      console.error('获取种子列表失败:', err)
    }
  },
  
  addInstance: async (data) => {
    try {
      const res = await api.post('/api/qb/instances', data)
      await get().fetchInstances()
      return { success: true, data: res.data }
    } catch (err) {
      return { success: false, error: err.response?.data?.detail || '添加失败' }
    }
  },
  
  deleteInstance: async (id) => {
    try {
      await api.delete(`/api/qb/instances/${id}`)
      await get().fetchInstances()
      return { success: true }
    } catch (err) {
      return { success: false, error: err.response?.data?.detail || '删除失败' }
    }
  },
  
  torrentAction: async (instanceId, hashes, action, deleteFiles = false) => {
    try {
      await api.post(`/api/qb/instances/${instanceId}/torrents/action`, {
        hashes, action, delete_files: deleteFiles
      })
      await get().fetchTorrents(instanceId)
      return { success: true }
    } catch (err) {
      return { success: false, error: err.response?.data?.detail || '操作失败' }
    }
  },
  
  addTorrent: async (instanceId, data) => {
    try {
      await api.post(`/api/qb/instances/${instanceId}/torrents/add`, data)
      await get().fetchTorrents(instanceId)
      return { success: true }
    } catch (err) {
      return { success: false, error: err.response?.data?.detail || '添加失败' }
    }
  }
}))

// 站点状态
export const useSiteStore = create((set, get) => ({
  sites: [],
  
  fetchSites: async () => {
    try {
      const res = await api.get('/api/sites')
      set({ sites: res.data })
    } catch (err) {
      console.error('获取站点失败:', err)
    }
  },
  
  addSite: async (data) => {
    try {
      await api.post('/api/sites', data)
      await get().fetchSites()
      return { success: true }
    } catch (err) {
      return { success: false, error: err.response?.data?.detail || '添加失败' }
    }
  },
  
  updateSite: async (id, data) => {
    try {
      await api.put(`/api/sites/${id}`, data)
      await get().fetchSites()
      return { success: true }
    } catch (err) {
      return { success: false, error: err.response?.data?.detail || '更新失败' }
    }
  },
  
  deleteSite: async (id) => {
    try {
      await api.delete(`/api/sites/${id}`)
      await get().fetchSites()
      return { success: true }
    } catch (err) {
      return { success: false, error: err.response?.data?.detail || '删除失败' }
    }
  }
}))

// RSS状态
export const useRSSStore = create((set, get) => ({
  feeds: [],
  
  fetchFeeds: async () => {
    try {
      const res = await api.get('/api/rss')
      set({ feeds: res.data })
    } catch (err) {
      console.error('获取RSS失败:', err)
    }
  },
  
  addFeed: async (data) => {
    try {
      await api.post('/api/rss', data)
      await get().fetchFeeds()
      return { success: true }
    } catch (err) {
      return { success: false, error: err.response?.data?.detail || '添加失败' }
    }
  },
  
  updateFeed: async (id, data) => {
    try {
      await api.put(`/api/rss/${id}`, data)
      await get().fetchFeeds()
      return { success: true }
    } catch (err) {
      return { success: false, error: err.response?.data?.detail || '更新失败' }
    }
  },
  
  deleteFeed: async (id) => {
    try {
      await api.delete(`/api/rss/${id}`)
      await get().fetchFeeds()
      return { success: true }
    } catch (err) {
      return { success: false, error: err.response?.data?.detail || '删除失败' }
    }
  },
  
  previewFeed: async (id) => {
    try {
      const res = await api.get(`/api/rss/${id}/preview`)
      return { success: true, data: res.data }
    } catch (err) {
      return { success: false, error: err.response?.data?.detail || '预览失败' }
    }
  }
}))

// 统计状态
export const useStatsStore = create((set) => ({
  stats: null,
  limitStates: [],
  
  fetchStats: async () => {
    try {
      const res = await api.get('/api/stats')
      set({ stats: res.data })
    } catch (err) {
      console.error('获取统计失败:', err)
    }
  },
  
  fetchLimitStates: async () => {
    try {
      const res = await api.get('/api/limit/states')
      set({ limitStates: res.data })
    } catch (err) {
      console.error('获取限速状态失败:', err)
    }
  },
  
  setLimitStates: (states) => set({ limitStates: states })
}))

// UI状态
export const useUIStore = create((set) => ({
  sidebarOpen: true,
  mobileMenuOpen: false,
  
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
  toggleMobileMenu: () => set((state) => ({ mobileMenuOpen: !state.mobileMenuOpen })),
  closeMobileMenu: () => set({ mobileMenuOpen: false })
}))
