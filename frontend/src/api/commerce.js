import { API_BASE } from './contracts'

function errorMessage(response, body) {
  if (typeof body?.detail === 'string') return body.detail
  if (Array.isArray(body?.detail)) return body.detail.map((item) => item.msg || '请求参数无效').join('；')
  return `请求失败（HTTP ${response.status}）`
}

async function request(path, { token, ...options } = {}) {
  const response = await fetch(`${API_BASE}/commerce${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(options.headers || {}),
    },
  })
  const body = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(errorMessage(response, body))
  return body
}

export function registerAccount(payload) {
  return request('/accounts/register', { method: 'POST', body: JSON.stringify(payload) })
}

export function loginAccount(payload) {
  return request('/accounts/login', { method: 'POST', body: JSON.stringify(payload) })
}

export function listDemoAccounts() {
  return request('/demo/accounts')
}

export function selectDemoAccount(accountId) {
  return request('/demo/select', { method: 'POST', body: JSON.stringify({ account_id: accountId }) })
}

export function getMe(token) {
  return request('/me', { token })
}

export function listProducts(token) {
  return request('/products', { token })
}

export function listPurchases(token) {
  return request('/purchases', { token })
}

export function createProduct(token, payload) {
  return request('/products', { token, method: 'POST', body: JSON.stringify(payload) })
}

export function listConversations(token) {
  return request('/conversations', { token })
}

export function createConversation(token, productId) {
  return request('/conversations', {
    token,
    method: 'POST',
    body: JSON.stringify({ product_id: productId }),
  })
}

export function getConversation(token, conversationId) {
  return request(`/conversations/${encodeURIComponent(conversationId)}`, { token })
}

export function postConversationMessage(token, conversationId, content) {
  const clientMessageId = globalThis.crypto?.randomUUID?.()
    || ('msg-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2))
  return request(`/conversations/${encodeURIComponent(conversationId)}/messages`, {
    token,
    method: 'POST',
    body: JSON.stringify({ content, client_message_id: clientMessageId }),
  })
}

export function updateHandoff(token, conversationId, state) {
  return request(`/conversations/${encodeURIComponent(conversationId)}/handoff`, {
    token, method: 'POST', body: JSON.stringify({ state }),
  })
}

export function sendFeedback(token, conversationId, messageId, rating) {
  return request(`/conversations/${encodeURIComponent(conversationId)}/messages/${encodeURIComponent(messageId)}/feedback`, {
    token, method: 'POST', body: JSON.stringify({ rating }),
  })
}
