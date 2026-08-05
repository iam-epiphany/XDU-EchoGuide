const DEFAULT_BACKENDS = {
  python: {
    id: 'python',
    label: 'Python',
    baseUrl: import.meta.env.VITE_PYTHON_API_URL || '/api/python',
    port: '8100'
  },
  java: {
    id: 'java',
    label: 'Java',
    baseUrl: import.meta.env.VITE_JAVA_API_URL || '/api/java',
    port: '8080'
  }
}

export function createInitialSettings() {
  const saved = readSettings()
  return {
    backend: saved.backend || 'python',
    userId: saved.userId || 'u1001',
    conversationId: saved.conversationId || '',
    endpoints: {
      python: saved.endpoints?.python || DEFAULT_BACKENDS.python.baseUrl,
      java: saved.endpoints?.java || DEFAULT_BACKENDS.java.baseUrl
    }
  }
}

export function saveSettings(settings) {
  localStorage.setItem('echoguide.frontend.settings', JSON.stringify(settings))
}

export function backendMeta(type, settings) {
  const meta = DEFAULT_BACKENDS[type] || DEFAULT_BACKENDS.java
  return {
    ...meta,
    baseUrl: normalizeBaseUrl(settings.endpoints[type] || meta.baseUrl)
  }
}

export async function requestHealth(type, settings) {
  return requestJson(backendMeta(type, settings).baseUrl, '/health')
}

export async function requestMonitor(type, settings) {
  return requestJson(backendMeta(type, settings).baseUrl, '/monitor')
}

export async function requestKnowledgeStats(type, settings) {
  return requestJson(backendMeta(type, settings).baseUrl, '/knowledge/stats')
}

export async function requestSearch(type, settings, query, topK = 5) {
  const params = new URLSearchParams({ query, topK: String(topK) })
  return requestJson(backendMeta(type, settings).baseUrl, `/search?${params}`, { method: 'POST' })
}

export async function requestChat(type, settings, message) {
  const meta = backendMeta(type, settings)
  const payload = buildChatPayload(type, settings, message)
  const raw = await requestJson(meta.baseUrl, '/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
  return normalizeChatResponse(type, raw)
}

/**
 * 流式对话（SSE）：POST /chat/stream，逐事件回调。
 *
 * handlers.onEvent(ev) 接收事件对象：
 *   {type:'hello'} {type:'meta',domain,agent} {type:'tool',name,status}
 *   {type:'delta',text} {type:'done',response,...} {type:'error',message}
 */
export async function requestChatStream(type, settings, message, handlers = {}) {
  const meta = backendMeta(type, settings)
  const payload = buildChatPayload(type, settings, message)
  const response = await fetch(`${meta.baseUrl}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
  if (!response.ok || !response.body) {
    throw new Error(`${response.status} ${response.statusText}`)
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
      if (handlers.onEvent) handlers.onEvent(ev)
    }
  }
  if (!doneEvent) throw new Error('连接中断，未收到完成事件')
  return doneEvent
}

export async function addKnowledge(type, settings, documents) {
  return requestJson(backendMeta(type, settings).baseUrl, '/knowledge/add', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ documents })
  })
}

export async function uploadKnowledge(type, settings, file) {
  const form = new FormData()
  form.append('file', file)
  return requestJson(backendMeta(type, settings).baseUrl, '/knowledge/upload', {
    method: 'POST',
    body: form
  })
}

// ── 个人数据中心（课表 / 待办 / DDL）────────────────────────────────────────

/** 上传 .ics / .json 课表文件导入（按当前 userId）。 */
export async function importScheduleFile(type, settings, file) {
  const form = new FormData()
  form.append('file', file)
  form.append('user_id', settings.userId || 'anonymous')
  return requestJson(backendMeta(type, settings).baseUrl, '/personal/schedule/import/file', {
    method: 'POST',
    body: form
  })
}

/** 当前用户课表（本周周视图）。 */
export async function getSchedule(type, settings) {
  const userId = settings.userId || 'anonymous'
  return requestJson(backendMeta(type, settings).baseUrl, `/personal/schedule?user_id=${encodeURIComponent(userId)}`)
}

/** 清空当前用户课表。 */
export async function clearSchedule(type, settings) {
  const userId = settings.userId || 'anonymous'
  return requestJson(backendMeta(type, settings).baseUrl, `/personal/schedule?user_id=${encodeURIComponent(userId)}`, {
    method: 'DELETE'
  })
}

/** 待办列表（status: open/done/all）。 */
export async function getTodos(type, settings, status = 'open') {
  const userId = settings.userId || 'anonymous'
  return requestJson(backendMeta(type, settings).baseUrl, `/personal/todo?user_id=${encodeURIComponent(userId)}&status=${status}`)
}

/** 新增待办 / DDL / 考试（kind: todo/ddl/exam，dueAt 可选）。 */
export async function addTodo(type, settings, content, kind = 'todo', dueAt = '') {
  return requestJson(backendMeta(type, settings).baseUrl, '/personal/todo', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_id: settings.userId || 'anonymous',
      content,
      kind,
      due_at: dueAt || null
    })
  })
}

/** 标记完成 / 恢复待办。 */
export async function completeTodo(type, settings, id, done = true) {
  const userId = settings.userId || 'anonymous'
  return requestJson(backendMeta(type, settings).baseUrl, `/personal/todo/${id}/complete?user_id=${encodeURIComponent(userId)}&done=${done}`, {
    method: 'POST'
  })
}

/** 删除待办。 */
export async function deleteTodo(type, settings, id) {
  const userId = settings.userId || 'anonymous'
  return requestJson(backendMeta(type, settings).baseUrl, `/personal/todo/${id}?user_id=${encodeURIComponent(userId)}`, {
    method: 'DELETE'
  })
}

function buildChatPayload(type, settings, message) {
  if (type === 'python') {
    return {
      message,
      user_id: settings.userId || 'anonymous',
      conv_id: settings.conversationId || undefined
    }
  }
  return {
    message,
    user_id: settings.userId || 'anonymous',
    conversation_id: settings.conversationId || undefined
  }
}

function normalizeChatResponse(type, raw) {
  return {
    backend: type,
    conversationId: raw.conversation_id || raw.conversationId || raw.conv_id || '',
    response: raw.response || '',
    intent: raw.intent || 'other',
    agentType: raw.agent_type || raw.agentType || '',
    escalated: Boolean(raw.escalated),
    latencyMs: Number(raw.latency_ms ?? raw.latencyMs ?? 0),
    knowledgeUsed: Boolean(raw.knowledge_used ?? raw.knowledgeUsed),
    verified: raw.verified,
    grounded: raw.grounded,
    raw
  }
}

async function requestJson(baseUrl, path, options = {}) {
  const url = `${normalizeBaseUrl(baseUrl)}${path}`
  const response = await fetch(url, options)
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
