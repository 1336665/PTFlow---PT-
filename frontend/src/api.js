import axios from 'axios'

const api = axios.create({
  baseURL: '',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  }
})

// 响应拦截器
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Token过期，清除登录状态
      localStorage.removeItem('ptflow-auth')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

export default api
