const DEFAULT_BACKEND = {
  id: 'echoguide', label: 'EchoGuide Python',
  baseUrl: import.meta.env.VITE_API_URL || '/api'
}

export function createInitialSettings() {
  const saved = readSettings()
  return {
    backend: 'echoguide',
    userId: 'anonymous',
    conversationId: saved.conversationId || '',
    endpoints: { echoguide: saved.endpoints?.echoguide || DEFAULT_BACKEND.baseUrl }
  }
}

export function saveSettings(settings) {
  localStorage.setItem('echoguide.frontend.settings', JSON.stringify(settings))
}

export function backendMeta(settings) {
  const meta = DEFAULT_BACKEND
  return {
    ...meta,
    baseUrl: normalizeBaseUrl(settings.endpoints?.echoguide || meta.baseUrl)
  }
}

export async function requestKnowledgeStats(settings) {
  return requestJson(backendMeta(settings).baseUrl, '/knowledge/stats')
}

export async function requestSearch(settings, query, topK = 5) {
  const params = new URLSearchParams({ query, top_k: String(topK) })
  return requestJson(backendMeta(settings).baseUrl, `/search?${params}`, { method: 'POST' })
}

/**
 * 流式对话（SSE）：POST /chat/stream，逐事件回调。
 *
 * handlers.onEvent(ev) 接收事件对象：
 *   {type:'hello'} {type:'meta',domain,agent} {type:'tool',name,status}
 *   {type:'delta',text} {type:'done',response,...} {type:'error',message}
 */
export async function requestChatStream(settings, message, handlers = {}) {
  const meta = backendMeta(settings)
  const payload = buildChatPayload(settings, message)
  const response = await fetch(`${meta.baseUrl}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify(payload)
  })
  if (!response.ok || !response.body) {
    let detail = `${response.status} ${response.statusText}`
    try {
      const payload = await response.json()
      if (payload?.detail) detail = `${response.status} · ${payload.detail}`
    } catch {
      // 非 JSON 错误响应沿用 HTTP 状态文本。
    }
    throw new Error(detail)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''
  let doneEvent = null

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    // SSE 帧以空行分隔；兼容拆包/粘包
    let idx
    while ((idx = buffer.indexOf('\n\n')) >= 0) {
      const frame = buffer.slice(0, idx)
      buffer = buffer.slice(idx + 2)
      const dataLine = frame.split('\n').find((line) => line.startsWith('data:'))
      if (!dataLine) continue
      const raw = dataLine.slice(5).trim()
      if (!raw) continue
      let ev
      try {
        ev = JSON.parse(raw)
      } catch {
        continue
      }
      if (ev.type === 'done') doneEvent = ev
      // 服务端 error 事件携带真实异常，直接抛出而不是等流结束报笼统文案
      if (ev.type === 'error') {
        if (handlers.onEvent) handlers.onEvent(ev)
        throw new Error(ev.message || '服务端返回错误')
      }
      if (handlers.onEvent) handlers.onEvent(ev)
    }
  }
  if (!doneEvent) throw new Error('连接中断，未收到完成事件')
  return doneEvent
}

export async function addKnowledge(settings, documents) {
  return requestJson(backendMeta(settings).baseUrl, '/knowledge/add', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ documents })
  })
}

export async function uploadKnowledge(settings, file) {
  const form = new FormData()
  form.append('file', file)
  return requestJson(backendMeta(settings).baseUrl, '/knowledge/upload', {
    method: 'POST',
    body: form
  })
}

// ── 个人数据中心（课表 / 待办 / DDL）────────────────────────────────────────

/** 上传 .ics / .json 课表文件导入（按当前 userId）。 */
export async function importScheduleFile(settings, file) {
  const form = new FormData()
  form.append('file', file)
  return requestJson(backendMeta(settings).baseUrl, '/personal/schedule/import/file', {
    method: 'POST',
    body: form
  })
}

/** 当前用户课表（本周周视图）。 */
export async function getSchedule(settings) {
  return requestJson(backendMeta(settings).baseUrl, '/personal/schedule')
}

/** 清空当前用户课表。 */
export async function clearSchedule(settings) {
  return requestJson(backendMeta(settings).baseUrl, '/personal/schedule', {
    method: 'DELETE'
  })
}

/** 待办列表（status: open/done/all）。 */
export async function getTodos(settings, status = 'open') {
  return requestJson(backendMeta(settings).baseUrl, `/personal/todo?status=${status}`)
}

/** 新增待办 / DDL / 考试（kind: todo/ddl/exam，dueAt 可选）。 */
export async function addTodo(settings, content, kind = 'todo', dueAt = '') {
  return requestJson(backendMeta(settings).baseUrl, '/personal/todo', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      content,
      kind,
      due_at: dueAt || null
    })
  })
}

/** 标记完成 / 恢复待办。 */
export async function completeTodo(settings, id, done = true) {
  return requestJson(backendMeta(settings).baseUrl, `/personal/todo/${id}/complete?done=${done}`, {
    method: 'POST'
  })
}

/** 删除待办。 */
export async function deleteTodo(settings, id) {
  return requestJson(backendMeta(settings).baseUrl, `/personal/todo/${id}`, {
    method: 'DELETE'
  })
}

function buildChatPayload(settings, message) {
  return {
    message,
    conv_id: settings.conversationId || undefined
  }
}

export function getCurrentUser(settings) {
  return requestJson(backendMeta(settings).baseUrl, '/auth/me')
}

export function loginUser(settings, username, password) {
  return requestJson(backendMeta(settings).baseUrl, '/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password })
  })
}

export function registerUser(settings, username, password) {
  return requestJson(backendMeta(settings).baseUrl, '/auth/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password })
  })
}

export function logoutUser(settings) {
  return requestJson(backendMeta(settings).baseUrl, '/auth/logout', { method: 'POST' })
}

function normalizeChatResponse(raw) {
  return {
    conversationId: raw.conv_id || raw.conversationId || '',
    response: raw.response || '',
    intent: raw.intent || 'other',
    agentType: raw.agent_type || raw.agentType || '',
    latencyMs: Number(raw.latency_ms ?? raw.latencyMs ?? 0),
    knowledgeUsed: Boolean(raw.knowledge_used ?? raw.knowledgeUsed),
    verified: raw.verified,
    grounded: raw.grounded,
    raw
  }
}

async function requestJson(baseUrl, path, options = {}) {
  const url = `${normalizeBaseUrl(baseUrl)}${path}`
  const response = await fetch(url, { credentials: 'include', ...options })
  const text = await response.text()
  let data = null
  try {
    data = text ? JSON.parse(text) : null
  } catch {
    data = text
  }
  if (!response.ok) {
    const detail = typeof data === 'string' ? data : JSON.stringify(data)
    throw new Error(`${response.status} ${response.statusText}: ${detail}`)
  }
  return data
}

function normalizeBaseUrl(value) {
  return String(value || '').replace(/\/+$/, '')
}

function readSettings() {
  try {
    return JSON.parse(localStorage.getItem('echoguide.frontend.settings') || '{}')
  } catch {
    return {}
  }
}
