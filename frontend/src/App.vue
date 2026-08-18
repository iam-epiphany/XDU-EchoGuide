<template>
  <main class="app-shell">
    <!-- ── 顶部品牌条 ──────────────────────────────────────────────────────── -->
    <header class="topbar">
      <div class="brand">
        <div class="brand-mark">西电</div>
        <div class="brand-text">
          <h1>西电校园助手</h1>
          <p>EchoGuide · 选课 / 校车 / 食堂 / 教务 / 办事一站问</p>
        </div>
      </div>
      <div class="topbar-actions">
        <button class="kb-button" @click="openSchedule">
          <span class="kb-icon">📅</span> 我的课表
        </button>
        <button class="kb-button" @click="openTodos">
          <span class="kb-icon">✅</span> 待办
        </button>
        <button class="kb-button" @click="openObs">
          <span class="kb-icon">📡</span> 监控
        </button>
        <button v-if="authUser?.role === 'admin'" class="kb-button" @click="openKb">
          <span class="kb-icon">📚</span> 知识库
        </button>
        <div v-if="authUser" class="account-chip">
          <span class="account-dot"></span>
          <span>{{ authUser.username }}</span>
          <small v-if="authUser.role === 'admin'">管理员</small>
          <button class="account-action" @click="doLogout">退出</button>
        </div>
        <button v-else class="login-button" @click="openAuth('login')">登录</button>
      </div>
    </header>

    <!-- ── 主对话区 ────────────────────────────────────────────────────────── -->
    <section class="chat-area" ref="chatArea">
      <!-- 欢迎页（无对话时） -->
      <div v-if="messages.length === 0" class="welcome">
        <div class="welcome-logo">西电</div>
        <h2>你好，同学！</h2>
        <p class="welcome-sub">选课、校车、食堂、奖学金、教务系统……校园问题都可以问我</p>
        <div class="topic-grid">
          <button
            v-for="topic in topics"
            :key="topic.title"
            class="topic-card"
            @click="askTopic(topic.question)"
          >
            <span class="topic-icon">{{ topic.icon }}</span>
            <strong>{{ topic.title }}</strong>
            <small>{{ topic.desc }}</small>
          </button>
        </div>
        <p class="welcome-tip">💡 点击上面的问题可以直接提问，也可以自己输入</p>
      </div>

      <!-- 消息流（有对话时） -->
      <div v-else class="messages" ref="messageList">
        <article v-for="item in messages" :key="item.id" :class="['message', item.role]">
          <div class="message-meta">
            <span class="message-role">{{ item.role === 'user' ? '你' : '西电校园助手' }}</span>
            <small v-if="item.meta">{{ item.meta }}</small>
          </div>
          <!-- 工具调用过程徽标（Agentic RAG 可视化） -->
          <div v-if="item.toolStatus" class="tool-badge">{{ item.toolStatus }}</div>
          <span class="message-text" v-html="renderMarkdownMemo(item.content)"></span><span v-if="item.streaming" class="stream-cursor">▍</span>
          <details v-if="debugMode && item.execution" class="execution-details" open>
            <summary>执行详情</summary>
            <div class="execution-grid">
              <span><b>路径</b>{{ item.execution.mode || '-' }}</span>
              <span><b>配置</b>{{ item.execution.profile || '-' }}</span>
              <span><b>分类</b>{{ item.execution.classifier_stage || '-' }}</span>
              <span><b>模型</b>{{ item.execution.model || '-' }}</span>
              <span><b>Agent</b>{{ (item.execution.agents || []).join(' → ') || '-' }}</span>
              <span><b>工具</b>{{ (item.execution.tools || []).join(', ') || '-' }}</span>
              <span><b>Token</b>{{ item.execution.input_tokens || 0 }} in / {{ item.execution.output_tokens || 0 }} out</span>
              <span><b>Trace</b>{{ item.execution.trace_id || '-' }}</span>
            </div>
            <p class="execution-reason">{{ item.execution.complexity_reason }}</p>
            <ol v-if="item.execution.tasks?.length" class="task-list">
              <li v-for="task in item.execution.tasks" :key="task.id">
                <b>{{ task.id }}</b> · {{ task.domain }} · {{ task.action }} ·
                {{ task.profile || '-' }} · {{ task.status }} ·
                {{ task.duration_ms || 0 }} ms · {{ (task.input_tokens || 0) + (task.output_tokens || 0) }} tok
                <small v-if="task.depends_on?.length">依赖 {{ task.depends_on.join(', ') }}</small>
                <small v-if="task.tools?.length">工具 {{ task.tools.join(', ') }}</small>
              </li>
            </ol>
          </details>
        </article>
        <div v-if="busy && !streamingMessage" class="message assistant typing">
          <div class="typing-dots"><i></i><i></i><i></i></div>
        </div>
      </div>
    </section>

    <!-- ── 底部输入区 ──────────────────────────────────────────────────────── -->
    <footer class="composer-wrap">
      <form class="composer" @submit.prevent="sendMessage">
        <textarea
          v-model="draft"
          rows="2"
          placeholder="输入问题，例如：这学期选课什么时候开始？"
          @keydown.enter.exact.prevent="sendMessage"
        ></textarea>
        <button :disabled="busy || !draft.trim()">{{ busy ? '思考中…' : '发送' }}</button>
      </form>
      <p class="composer-hint">内容基于校园公开信息整理，具体事项请以学校官方通知为准</p>
    </footer>

    <!-- ── 知识库弹窗（开发者工具，收在角落） ───────────────────────────────── -->
    <div v-if="showKb" class="modal-mask" @click.self="closeKb">
      <div class="modal">
        <div class="modal-head">
          <h2>知识库</h2>
          <button class="modal-close" @click="closeKb">✕</button>
        </div>

        <section class="kb-section">
          <div class="panel-heading">
            <h3>检索</h3>
            <span class="pill soft">{{ knowledgeCount }} 个片段</span>
          </div>
          <div class="inline-form">
            <input v-model="searchQuery" placeholder="输入关键词，如：校车、选课" @keydown.enter.prevent="searchKnowledge" />
            <button @click="searchKnowledge" :disabled="busyData || !searchQuery.trim()">检索</button>
          </div>
          <div class="result-list">
            <article v-for="item in searchResults" :key="item.id || item.title" class="result-item">
              <strong>{{ item.title || '未命名结果' }}</strong>
              <span>相关度 {{ item.score ?? '-' }}</span>
              <p>{{ item.content }}</p>
            </article>
            <p v-if="searched && searchResults.length === 0" class="no-result">没有检索到相关内容</p>
          </div>
        </section>

        <section class="kb-section">
          <div class="panel-heading">
            <h3>导入知识</h3>
          </div>
          <label>
            <span>标题</span>
            <input v-model="docTitle" placeholder="如：校车时刻说明" />
          </label>
          <label>
            <span>内容</span>
            <textarea v-model="docContent" rows="4" placeholder="输入知识库内容"></textarea>
          </label>
          <div class="actions">
            <button @click="submitKnowledge" :disabled="busyData || !docTitle.trim() || !docContent.trim()">添加文档</button>
            <label class="file-button">
              上传文件
              <input type="file" accept=".txt,.md,.json" @change="handleUpload" />
            </label>
          </div>
        </section>

        <p v-if="statusText" class="kb-status">{{ statusText }}</p>
      </div>
    </div>

    <!-- ── 我的课表弹窗 ─────────────────────────────────────────────────────── -->
    <div v-if="showSchedule" class="modal-mask" @click.self="closeSchedule">
      <div class="modal modal-wide">
        <div class="modal-head">
          <h2>我的课表</h2>
          <button class="modal-close" @click="closeSchedule">✕</button>
        </div>

        <section class="kb-section">
          <div class="panel-heading">
            <h3>导入课表</h3>
            <span class="pill soft">{{ authUser?.username }}</span>
          </div>
          <p class="schedule-tip">
            教务系统导出 <code>.ics</code> 日历文件（选课 → 导出课表）或 <code>.json</code> 课表上传，
            导入后即可问「今天有什么课？」
          </p>
          <div class="actions">
            <label class="file-button">
              上传 .ics / .json
              <input type="file" accept=".ics,.json" @change="handleScheduleUpload" />
            </label>
            <button class="danger-button" v-if="scheduleCourses.length" @click="doClearSchedule" :disabled="busyData">清空课表</button>
          </div>
          <p v-if="scheduleMsg" class="kb-status">{{ scheduleMsg }}</p>
        </section>

        <section class="kb-section" v-if="scheduleCourses.length">
          <div class="panel-heading">
            <h3>本周课程（{{ scheduleInSemester ? '第 ' + scheduleWeekNum + ' 周' : '假期（未开学）' }}）</h3>
            <span class="pill soft">{{ scheduleCourses.length }} 门</span>
          </div>
          <div class="schedule-grid">
            <div v-for="(courses, day) in scheduleByDay" :key="day" class="schedule-day">
              <h4>{{ day }}</h4>
              <template v-if="courses.length">
                <p v-for="c in courses" :key="c.course + c.start_time" class="schedule-item">
                  <strong>{{ c.start_time }}-{{ c.end_time }}</strong>
                  <span>{{ c.course }}</span>
                  <small>{{ c.location || '地点未填' }}</small>
                </p>
              </template>
              <p v-else class="schedule-empty">—</p>
            </div>
          </div>
        </section>

        <p v-if="!scheduleCourses.length && scheduleLoaded" class="kb-status">
          还没有课程。上传教务系统导出的 .ics 文件即可导入课表。
        </p>
      </div>
    </div>

    <!-- ── 待办弹窗 ─────────────────────────────────────────────────────────── -->
    <div v-if="showTodos" class="modal-mask" @click.self="closeTodos">
      <div class="modal">
        <div class="modal-head">
          <h2>待办 / DDL / 考试</h2>
          <button class="modal-close" @click="closeTodos">✕</button>
        </div>

        <section class="kb-section">
          <div class="panel-heading"><h3>新增</h3></div>
          <div class="todo-form">
            <input v-model="newTodoContent" placeholder="事项内容，如：交实验报告" @keydown.enter.prevent="doAddTodo" />
            <select v-model="newTodoKind">
              <option value="todo">待办</option>
              <option value="ddl">截止任务</option>
              <option value="exam">考试</option>
            </select>
            <input v-model="newTodoDue" type="date" title="截止日期（可选）" />
            <button @click="doAddTodo" :disabled="busyData || !newTodoContent.trim()">添加</button>
          </div>
          <p v-if="todoMsg" class="kb-status">{{ todoMsg }}</p>
        </section>

        <section class="kb-section">
          <div class="panel-heading">
            <h3>列表</h3>
            <span class="pill soft">{{ todos.length }} 条</span>
          </div>
          <div class="result-list">
            <article v-for="t in todos" :key="t.id" :class="['todo-item', { done: t.done }]">
              <div class="todo-main">
                <strong>{{ t.content }}</strong>
                <span class="pill">{{ kindLabel(t.kind) }}</span>
                <small v-if="t.due_at">截止 {{ t.due_at }}</small>
              </div>
              <div class="todo-actions">
                <button class="mini" @click="doCompleteTodo(t)">{{ t.done ? '恢复' : '完成' }}</button>
                <button class="mini danger-button" @click="doDeleteTodo(t)">删除</button>
              </div>
            </article>
            <p v-if="todos.length === 0" class="no-result">暂无待办，从聊天里说「帮我记个待办」也可以添加</p>
          </div>
        </section>
      </div>
    </div>

    <!-- ── 可观测性：Agent 统计 + 告警 + Trace 瀑布 ─────────────────────────── -->
    <div v-if="showObs" class="modal-mask" @click.self="closeObs">
      <div class="modal obs-modal">
        <div class="modal-head">
          <h2>可观测性</h2>
          <button class="modal-close" @click="closeObs">✕</button>
        </div>
        <p v-if="obsError" class="obs-error">{{ obsError }}</p>

        <section class="kb-section">
          <div class="panel-heading">
            <h3>Agent 实时统计</h3>
            <button class="kb-button mini" @click="loadObs" :disabled="obsBusy">{{ obsBusy ? '加载中…' : '刷新' }}</button>
          </div>
          <table v-if="obsAgents.length" class="obs-table">
            <thead>
              <tr><th>Agent</th><th>请求</th><th>成功率</th><th>均延迟</th><th>路由分</th><th>在途</th></tr>
            </thead>
            <tbody>
              <tr v-for="a in obsAgents" :key="a.key">
                <td>{{ a.key }}</td>
                <td>{{ a.total }}</td>
                <td :class="a.total && a.success_rate < 0.9 ? 'bad' : ''">{{ (a.success_rate * 100).toFixed(0) }}%</td>
                <td :class="a.avg_ms > 3000 ? 'bad' : ''">{{ Math.round(a.avg_ms) }} ms</td>
                <td>{{ a.routing_score }}</td>
                <td>{{ a.in_flight }}</td>
              </tr>
            </tbody>
          </table>
          <p v-else class="no-result">暂无请求记录 —— 先发几条消息再回来刷新</p>

          <template v-if="obsAlerts.length">
            <h4 class="obs-sub">活跃告警</h4>
            <ul class="obs-list">
              <li v-for="(a, i) in obsAlerts" :key="i" class="obs-alert">
                ⚠ [{{ a.severity }}] {{ a.metric }} = {{ a.value }}（阈值 {{ a.threshold }}）{{ a.message }}
              </li>
            </ul>
          </template>
          <template v-if="obsSuggestions.length">
            <h4 class="obs-sub">优化建议</h4>
            <ul class="obs-list">
              <li v-for="(s, i) in obsSuggestions" :key="i">💡 {{ s.title }}</li>
            </ul>
          </template>
        </section>

        <section class="kb-section">
          <div class="panel-heading">
            <h3>最近 Trace（{{ obsTraces.length }} 条，保留 1 小时）</h3>
          </div>
          <ul class="obs-list traces">
            <li v-for="t in obsTraces" :key="t.trace_id">
              <button class="trace-row" @click="toggleTrace(t.trace_id)">
                <span class="trace-id">{{ t.trace_id }}</span>
                <span>{{ t.name }}</span>
                <span>{{ Math.round(t.duration_ms) }} ms</span>
                <span>{{ (t.spans || []).length }} spans</span>
                <span>{{ fmtTime(t.ts) }}</span>
              </button>
              <div v-if="expandedTrace === t.trace_id" class="trace-detail">
                <p v-if="traceDetail" class="muted">tags: {{ JSON.stringify(traceDetail.tags || {}) }}</p>
                <table class="obs-table">
                  <thead><tr><th>span</th><th>耗时</th><th>meta</th></tr></thead>
                  <tbody>
                    <tr v-for="(s, i) in (traceDetail?.spans || [])" :key="i">
                      <td>{{ s.name }}</td>
                      <td>{{ Math.round(s.duration_ms) }} ms</td>
                      <td class="span-meta">{{ JSON.stringify(s.meta || {}) }}</td>
                    </tr>
                  </tbody>
                </table>
                <p v-if="traceDetail && !traceDetail.spans?.length" class="no-result">该 trace 无 span 记录</p>
              </div>
            </li>
          </ul>
          <p v-if="obsTraces.length === 0" class="no-result">暂无 trace —— 发一条消息后自动产生</p>
        </section>
      </div>
    </div>

    <!-- ── 校园通行证：轻量登录/注册 ──────────────────────────────────────── -->
    <div v-if="showAuth" class="auth-mask" @click.self="closeAuth">
      <section class="auth-card" role="dialog" aria-modal="true" aria-labelledby="auth-title">
        <div class="auth-ticket">
          <span>XD</span>
          <small>ECHO PASS</small>
        </div>
        <div class="auth-content">
          <button class="auth-close" aria-label="关闭" @click="closeAuth">✕</button>
          <p class="auth-eyebrow">西电校园助手 · 个人空间</p>
          <h2 id="auth-title">{{ authMode === 'login' ? '登录校园通行证' : '创建校园通行证' }}</h2>
          <p class="auth-note">登录后，课表、待办和对话记忆只属于你。</p>
          <form class="auth-form" @submit.prevent="submitAuth">
            <label>
              <span>用户名</span>
              <input v-model.trim="authForm.username" autocomplete="username" minlength="3" maxlength="32" required autofocus />
            </label>
            <label>
              <span>密码</span>
              <input v-model="authForm.password" type="password" :autocomplete="authMode === 'login' ? 'current-password' : 'new-password'" minlength="6" maxlength="128" required />
            </label>
            <p v-if="authError" class="auth-error" role="alert">{{ authError }}</p>
            <button class="auth-submit" :disabled="authBusy">
              {{ authBusy ? '处理中…' : authMode === 'login' ? '登录' : '注册并登录' }}
            </button>
          </form>
          <button class="auth-switch" @click="toggleAuthMode">
            {{ authMode === 'login' ? '还没有账号？立即注册' : '已有账号？返回登录' }}
          </button>
        </div>
      </section>
    </div>
  </main>
</template>

<script setup>
import { computed, nextTick, onMounted, reactive, ref, watch } from 'vue'
import { renderMarkdown } from './lib/markdown'
import {
  addKnowledge,
  addTodo,
  backendMeta,
  clearSchedule,
  completeTodo,
  createInitialSettings,
  deleteTodo,
  getSchedule,
  getTodos,
  getCurrentUser,
  importScheduleFile,
  loginUser,
  logoutUser,
  registerUser,
  requestChatStream,
  requestKnowledgeStats,
  requestMonitorSummary,
  requestSearch,
  requestTraceDetail,
  requestTraces,
  saveSettings,
  uploadKnowledge
} from './lib/backends'

const settings = reactive(createInitialSettings())
const messages = ref([])
const draft = ref('')
const busy = ref(false)        // 对话请求进行中（发送按钮/typing 指示器）
const busyData = ref(false)    // 数据操作进行中（知识库/课表/待办），与对话互不阻塞
const statusText = ref('')
const knowledgeCount = ref('-')
const searchQuery = ref('校车')
const searchResults = ref([])
const searched = ref(false)
const docTitle = ref('校车时刻说明')
const docContent = ref('校园穿梭车连接南校区与北校区，工作日班次较多，周末和节假日班次减少，具体时刻以校车管理最新通知为准。')
const showKb = ref(false)
const messageList = ref(null)
const streamingMessage = ref(null)
const debugMode = new URLSearchParams(window.location.search).get('debug') === '1'
const authUser = ref(null)
const showAuth = ref(false)
const authMode = ref('login')
const authBusy = ref(false)
const authError = ref('')
const authForm = reactive({ username: '', password: '' })

// 可观测性：Agent 统计 / 告警 / Trace
const showObs = ref(false)
const obsBusy = ref(false)
const obsError = ref('')
const obsAgents = ref([])
const obsAlerts = ref([])
const obsSuggestions = ref([])
const obsTraces = ref([])
const expandedTrace = ref(null)
const traceDetail = ref(null)

// 个人数据中心：课表
const showSchedule = ref(false)
const scheduleMsg = ref('')
const scheduleCourses = ref([])
const scheduleWeekNum = ref('-')
const scheduleInSemester = ref(true)
const scheduleLoaded = ref(false)

// 个人数据中心：待办 / DDL / 考试
const showTodos = ref(false)
const todos = ref([])
const todoMsg = ref('')
const newTodoContent = ref('')
const newTodoKind = ref('todo')
const newTodoDue = ref('')

// 校园话题推荐（欢迎页卡片）
const topics = [
  { icon: '📅', title: '我的课表', desc: '今天有什么课？', question: '今天有什么课？' },
  { icon: '☀️', title: '天气', desc: '明天要带伞吗？', question: '明天南校区天气怎么样？' },
  { icon: '📖', title: '选课指南', desc: '什么时候选课？怎么选？', question: '这学期选课什么时候开始？' },
  { icon: '🚌', title: '校车时刻', desc: '下一班校车几点？', question: '下一班从南校区到北校区的校车几点？' },
  { icon: '🍜', title: '食堂信息', desc: '几点开门、几点关门？', question: '南校区食堂几点关门？' },
  { icon: '🎓', title: '考试安排', desc: '最近有什么考试？', question: '我最近的考试安排是什么？' },
  { icon: '🏠', title: '宿舍生活', desc: '报修、水电、门禁？', question: '宿舍设施坏了怎么报修？' },
  { icon: '🖥️', title: '教务系统', desc: '登录不上怎么办？', question: '教务系统登录不上怎么办？' },
]

const currentBackend = backendMeta(settings)

// 课表按星期分组（周一到周日），供周视图渲染
const scheduleByDay = computed(() => {
  const days = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
  const groups = Object.fromEntries(days.map((d) => [d, []]))
  for (const c of scheduleCourses.value) {
    const name = days[c.day_of_week] || '周一'
    if (groups[name]) groups[name].push(c)
  }
  return groups
})

watch(
  () => settings.conversationId,
  () => persist()
)

onMounted(async () => {
  await restoreSession()
  loadStats()
})

async function restoreSession() {
  try {
    const data = await getCurrentUser(settings)
    authUser.value = data.authenticated ? data.user : null
    settings.userId = authUser.value?.id || 'anonymous'
  } catch {
    authUser.value = null
    settings.userId = 'anonymous'
  }
}

function openAuth(mode = 'login') {
  authMode.value = mode
  authError.value = ''
  authForm.password = ''
  showAuth.value = true
}

function closeAuth() {
  showAuth.value = false
  authError.value = ''
}

function toggleAuthMode() {
  authMode.value = authMode.value === 'login' ? 'register' : 'login'
  authError.value = ''
}

async function submitAuth() {
  authBusy.value = true
  authError.value = ''
  try {
    const action = authMode.value === 'login' ? loginUser : registerUser
    const data = await action(settings, authForm.username, authForm.password)
    authUser.value = data.user
    settings.userId = data.user.id
    settings.conversationId = ''
    messages.value = []  // 登录后清空匿名会话消息，避免残留挂在新会话下
    authForm.password = ''
    closeAuth()
    persist()
  } catch (error) {
    authError.value = readableError(error)
  } finally {
    authBusy.value = false
  }
}

async function doLogout() {
  try {
    await logoutUser(settings)
  } finally {
    authUser.value = null
    settings.userId = 'anonymous'
    settings.conversationId = ''
    messages.value = []
    persist()
  }
}

function readableError(error) {
  const raw = String(error?.message || error)
  try {
    const jsonStart = raw.indexOf('{')
    if (jsonStart >= 0) {
      const detail = JSON.parse(raw.slice(jsonStart)).detail
      // 只有 detail 是非空字符串时才采用，避免误切正文或返回 undefined
      if (typeof detail === 'string' && detail.trim()) return detail
    }
  } catch {
    // 保留原始错误
  }
  return raw
}

// ── Markdown 渲染缓存（流式 delta 每帧全量重渲，用 memo 避免重复计算）────────
const mdCache = new Map()
const MAX_MD_CACHE = 200
function renderMarkdownMemo(text) {
  const key = text || ''
  if (mdCache.has(key)) return mdCache.get(key)
  const html = renderMarkdown(key)
  if (mdCache.size >= MAX_MD_CACHE) {
    mdCache.delete(mdCache.keys().next().value)
  }
  mdCache.set(key, html)
  return html
}

function persist() {
  saveSettings(settings)
}

function createId() {
  if (globalThis.crypto?.randomUUID) {
    return globalThis.crypto.randomUUID()
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`
}

function askTopic(question) {
  draft.value = question
  sendMessage()
}

async function sendMessage() {
  const content = draft.value.trim()
  if (!content) return
  messages.value.push({ id: createId(), role: 'user', content })
  // 长会话内存保护：只保留最近 100 条
  if (messages.value.length > 100) {
    messages.value = messages.value.slice(-100)
  }
  draft.value = ''
  busy.value = true

  // 流式消息占位（SSE 逐 token 渲染）
  const assistantMsg = {
    id: createId(),
    role: 'assistant',
    content: '',
    meta: '',
    toolStatus: '',
    execution: null,
    streaming: true
  }
  messages.value.push(assistantMsg)
  streamingMessage.value = assistantMsg

  try {
    const done = await requestChatStream(settings, content, {
      onEvent: (ev) => {
        if (ev.type === 'delta') {
          assistantMsg.content += ev.text
        } else if (ev.type === 'tool') {
          if (ev.status === 'start') {
            assistantMsg.toolStatus = `🔍 ${ev.name} · ${ev.input?.query || '检索中…'}`
          } else {
            const titles = (ev.titles || []).slice(0, 3).join(' / ')
            assistantMsg.toolStatus = `✅ 检索完成${titles ? '：' + titles : ''}`
          }
        } else if (ev.type === 'meta') {
          assistantMsg.meta = [ev.domain, ev.action, ev.agent, ev.profile, ev.mode].filter(Boolean).join(' · ')
        }
      }
    })
    if (done.conv_id && !settings.conversationId) {
      settings.conversationId = done.conv_id
      persist()
    }
    assistantMsg.content = done.response
    assistantMsg.execution = done.execution || null
    assistantMsg.streaming = false
    assistantMsg.meta = [
      done.intent,
      done.agent_type,
      done.knowledge_used ? 'RAG' : ''
    ].filter(Boolean).join(' · ')
  } catch (error) {
    assistantMsg.content = error.message
    assistantMsg.meta = '请求失败'
    assistantMsg.streaming = false
  } finally {
    busy.value = false
    streamingMessage.value = null
    await nextTick()
    messageList.value?.scrollTo({ top: messageList.value.scrollHeight, behavior: 'smooth' })
  }
}

// ── 知识库 ──────────────────────────────────────────────────────────────────

function openKb() {
  if (authUser.value?.role !== 'admin') return
  showKb.value = true
  loadStats()
}

function closeKb() {
  showKb.value = false
}

// ── 可观测性：监控 / Trace ────────────────────────────────────────────────────

function openObs() {
  showObs.value = true
  loadObs()
}

function closeObs() {
  showObs.value = false
  expandedTrace.value = null
  traceDetail.value = null
}

async function loadObs() {
  obsBusy.value = true
  obsError.value = ''
  try {
    const [mon, traces] = await Promise.all([
      requestMonitorSummary(settings),
      requestTraces(settings, 20)
    ])
    obsAgents.value = Object.entries(mon.agent_stats || {}).map(([key, v]) => ({ key, ...v }))
    obsAlerts.value = mon.active_alerts || []
    obsSuggestions.value = mon.suggestions || []
    obsTraces.value = traces.traces || []
  } catch (err) {
    obsError.value = String(err?.message || err).slice(0, 300)
    obsAgents.value = []
    obsAlerts.value = []
    obsSuggestions.value = []
    obsTraces.value = []
  } finally {
    obsBusy.value = false
  }
}

async function toggleTrace(traceId) {
  if (expandedTrace.value === traceId) {
    expandedTrace.value = null
    traceDetail.value = null
    return
  }
  expandedTrace.value = traceId
  traceDetail.value = null
  try {
    traceDetail.value = await requestTraceDetail(settings, traceId)
  } catch (err) {
    obsError.value = String(err?.message || err).slice(0, 300)
  }
}

function fmtTime(ts) {
  if (!ts) return '-'
  return new Date(ts * 1000).toLocaleTimeString('zh-CN', { hour12: false })
}

async function loadStats() {
  try {
    const stats = await requestKnowledgeStats(settings)
    knowledgeCount.value = stats.total_chunks ?? stats.totalChunks ?? '-'
  } catch {
    // 后端不可用时保持现状
  }
}

async function searchKnowledge() {
  busyData.value = true
  searched.value = true
  try {
    const data = await requestSearch(settings, searchQuery.value, 5)
    searchResults.value = data.results || []
  } catch (error) {
    statusText.value = error.message
  } finally {
    busyData.value = false
  }
}

async function submitKnowledge() {
  busyData.value = true
  try {
    const data = await addKnowledge(settings, [
      { title: docTitle.value.trim(), content: docContent.value.trim() }
    ])
    statusText.value = data.message || JSON.stringify(data)
    docTitle.value = ''
    docContent.value = ''
    await loadStats()
  } catch (error) {
    statusText.value = error.message
  } finally {
    busyData.value = false
  }
}

async function handleUpload(event) {
  const file = event.target.files?.[0]
  event.target.value = ''
  if (!file) return
  busyData.value = true
  try {
    const data = await uploadKnowledge(settings, file)
    statusText.value = data.message || JSON.stringify(data)
    await loadStats()
  } catch (error) {
    statusText.value = error.message
  } finally {
    busyData.value = false
  }
}

// ── 我的课表 ─────────────────────────────────────────────────────────────────

async function openSchedule() {
  if (!authUser.value) {
    openAuth('login')
    return
  }
  showSchedule.value = true
  scheduleMsg.value = ''
  scheduleLoaded.value = false
  await loadSchedule()
}

function closeSchedule() {
  showSchedule.value = false
}

async function loadSchedule() {
  try {
    const data = await getSchedule(settings)
    scheduleCourses.value = data.courses || []
    scheduleWeekNum.value = data.week_num ?? '-'
    scheduleInSemester.value = data.in_semester !== false
    scheduleLoaded.value = true
  } catch (error) {
    scheduleMsg.value = error.message
    scheduleLoaded.value = true
  }
}

async function handleScheduleUpload(event) {
  const file = event.target.files?.[0]
  event.target.value = ''
  if (!file) return
  busyData.value = true
  scheduleMsg.value = '导入中…'
  try {
    const data = await importScheduleFile(settings, file)
    scheduleMsg.value = data.message || JSON.stringify(data)
    await loadSchedule()
  } catch (error) {
    scheduleMsg.value = error.message
  } finally {
    busyData.value = false
  }
}

async function doClearSchedule() {
  if (!confirm('确定清空课表吗？清空后需要重新导入。')) return
  busyData.value = true
  try {
    const data = await clearSchedule(settings)
    scheduleMsg.value = data.message || '已清空'
    scheduleCourses.value = []
  } catch (error) {
    scheduleMsg.value = error.message
  } finally {
    busyData.value = false
  }
}

// ── 待办 / DDL / 考试 ────────────────────────────────────────────────────────

async function openTodos() {
  if (!authUser.value) {
    openAuth('login')
    return
  }
  showTodos.value = true
  todoMsg.value = ''
  await loadTodos()
}

function closeTodos() {
  showTodos.value = false
}

async function loadTodos() {
  try {
    const data = await getTodos(settings, 'all')
    todos.value = data.todos || []
  } catch (error) {
    todoMsg.value = error.message
  }
}

function kindLabel(kind) {
  return { todo: '待办', ddl: 'DDL', exam: '考试' }[kind] || kind
}

async function doAddTodo() {
  const content = newTodoContent.value.trim()
  if (!content) return
  busyData.value = true
  try {
    await addTodo(settings, content, newTodoKind.value, newTodoDue.value)
    newTodoContent.value = ''
    newTodoDue.value = ''
    todoMsg.value = '已添加'
    await loadTodos()
  } catch (error) {
    todoMsg.value = error.message
  } finally {
    busyData.value = false
  }
}

async function doCompleteTodo(t) {
  try {
    await completeTodo(settings, t.id, !t.done)
    await loadTodos()
  } catch (error) {
    todoMsg.value = error.message
  }
}

async function doDeleteTodo(t) {
  try {
    await deleteTodo(settings, t.id)
    await loadTodos()
  } catch (error) {
    todoMsg.value = error.message
  }
}
</script>
