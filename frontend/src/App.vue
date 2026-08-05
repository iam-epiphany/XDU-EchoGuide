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
        <button class="kb-button" @click="openKb">
          <span class="kb-icon">📚</span> 知识库
        </button>
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
          <span class="message-text" v-html="renderMarkdown(item.content)"></span><span v-if="item.streaming" class="stream-cursor">▍</span>
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
            <button @click="searchKnowledge" :disabled="busy || !searchQuery.trim()">检索</button>
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
            <button @click="submitKnowledge" :disabled="busy || !docTitle.trim() || !docContent.trim()">添加文档</button>
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
            <span class="pill soft">用户 {{ settings.userId }}</span>
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
            <button class="danger-button" v-if="scheduleCourses.length" @click="doClearSchedule" :disabled="busy">清空课表</button>
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
            <button @click="doAddTodo" :disabled="busy || !newTodoContent.trim()">添加</button>
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
  importScheduleFile,
  requestChatStream,
  requestKnowledgeStats,
  requestSearch,
  saveSettings,
  uploadKnowledge
} from './lib/backends'

const settings = reactive(createInitialSettings())
const messages = ref([])
const draft = ref('')
const busy = ref(false)
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

const currentBackend = backendMeta('python', settings)

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

onMounted(() => {
  loadStats()
})

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
  draft.value = ''
  busy.value = true

  // 流式消息占位（SSE 逐 token 渲染）
  const assistantMsg = {
    id: createId(),
    role: 'assistant',
    content: '',
    meta: '',
    toolStatus: '',
    streaming: true
  }
  messages.value.push(assistantMsg)
  streamingMessage.value = assistantMsg

  try {
    const done = await requestChatStream('python', settings, content, {
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
          assistantMsg.meta = [ev.domain, ev.action, ev.agent].filter(Boolean).join(' · ')
        }
      }
    })
    if (done.conv_id && !settings.conversationId) {
      settings.conversationId = done.conv_id
      persist()
    }
    assistantMsg.content = done.response
    assistantMsg.streaming = false
    assistantMsg.meta = [
      done.intent,
      done.agent_type,
      done.knowledge_used ? 'RAG' : '',
      done.escalated ? '转人工' : ''
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
  showKb.value = true
  loadStats()
}

function closeKb() {
  showKb.value = false
}

async function loadStats() {
  try {
    const stats = await requestKnowledgeStats('python', settings)
    knowledgeCount.value = stats.total_chunks ?? stats.totalChunks ?? '-'
  } catch {
    // 后端不可用时保持现状
  }
}

async function searchKnowledge() {
  busy.value = true
  searched.value = true
  try {
    const data = await requestSearch('python', settings, searchQuery.value, 5)
    searchResults.value = data.results || []
  } catch (error) {
    statusText.value = error.message
  } finally {
    busy.value = false
  }
}

async function submitKnowledge() {
  busy.value = true
  try {
    const data = await addKnowledge('python', settings, [
      { title: docTitle.value.trim(), content: docContent.value.trim() }
    ])
    statusText.value = data.message || JSON.stringify(data)
    docTitle.value = ''
    docContent.value = ''
    await loadStats()
  } catch (error) {
    statusText.value = error.message
  } finally {
    busy.value = false
  }
}

async function handleUpload(event) {
  const file = event.target.files?.[0]
  event.target.value = ''
  if (!file) return
  busy.value = true
  try {
    const data = await uploadKnowledge('python', settings, file)
    statusText.value = data.message || JSON.stringify(data)
    await loadStats()
  } catch (error) {
    statusText.value = error.message
  } finally {
    busy.value = false
  }
}

// ── 我的课表 ─────────────────────────────────────────────────────────────────

async function openSchedule() {
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
    const data = await getSchedule('python', settings)
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
  busy.value = true
  scheduleMsg.value = '导入中…'
  try {
    const data = await importScheduleFile('python', settings, file)
    scheduleMsg.value = data.message || JSON.stringify(data)
    await loadSchedule()
  } catch (error) {
    scheduleMsg.value = error.message
  } finally {
    busy.value = false
  }
}

async function doClearSchedule() {
  if (!confirm('确定清空课表吗？清空后需要重新导入。')) return
  busy.value = true
  try {
    const data = await clearSchedule('python', settings)
    scheduleMsg.value = data.message || '已清空'
    scheduleCourses.value = []
  } catch (error) {
    scheduleMsg.value = error.message
  } finally {
    busy.value = false
  }
}

// ── 待办 / DDL / 考试 ────────────────────────────────────────────────────────

async function openTodos() {
  showTodos.value = true
  todoMsg.value = ''
  await loadTodos()
}

function closeTodos() {
  showTodos.value = false
}

async function loadTodos() {
  try {
    const data = await getTodos('python', settings, 'all')
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
  busy.value = true
  try {
    await addTodo('python', settings, content, newTodoKind.value, newTodoDue.value)
    newTodoContent.value = ''
    newTodoDue.value = ''
    todoMsg.value = '已添加'
    await loadTodos()
  } catch (error) {
    todoMsg.value = error.message
  } finally {
    busy.value = false
  }
}

async function doCompleteTodo(t) {
  try {
    await completeTodo('python', settings, t.id, !t.done)
    await loadTodos()
  } catch (error) {
    todoMsg.value = error.message
  }
}

async function doDeleteTodo(t) {
  try {
    await deleteTodo('python', settings, t.id)
    await loadTodos()
  } catch (error) {
    todoMsg.value = error.message
  }
}
</script>
