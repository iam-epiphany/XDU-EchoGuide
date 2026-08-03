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
          <p>{{ item.content }}</p>
        </article>
        <div v-if="busy" class="message assistant typing">
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
  </main>
</template>

<script setup>
import { nextTick, onMounted, reactive, ref, watch } from 'vue'
import {
  addKnowledge,
  backendMeta,
  createInitialSettings,
  requestChat,
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

// 校园话题推荐（欢迎页卡片）
const topics = [
  { icon: '📖', title: '选课指南', desc: '什么时候选课？怎么选？', question: '这学期选课什么时候开始？' },
  { icon: '🚌', title: '校车时刻', desc: '南北校区几点发车？', question: '校车几点发车？' },
  { icon: '🍜', title: '食堂信息', desc: '几点开门、几点关门？', question: '南校区食堂几点关门？' },
  { icon: '🏠', title: '宿舍生活', desc: '报修、水电、门禁？', question: '宿舍设施坏了怎么报修？' },
  { icon: '🎓', title: '奖学金', desc: '什么时候评、怎么申请？', question: '奖学金什么时候评定？' },
  { icon: '🖥️', title: '教务系统', desc: '登录不上怎么办？', question: '教务系统登录不上怎么办？' },
]

const currentBackend = backendMeta('python', settings)

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
  try {
    const response = await requestChat('python', settings, content)
    if (response.conversationId && !settings.conversationId) {
      settings.conversationId = response.conversationId
      persist()
    }
    const meta = [
      response.intent,
      response.agentType,
      response.knowledgeUsed ? 'RAG' : '',
      response.escalated ? '转人工' : ''
    ].filter(Boolean).join(' · ')
    messages.value.push({
      id: createId(),
      role: 'assistant',
      content: response.response,
      meta
    })
  } catch (error) {
    messages.value.push({
      id: createId(),
      role: 'assistant',
      content: error.message,
      meta: '请求失败'
    })
  } finally {
    busy.value = false
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
</script>
