/**
 * 轻量安全 Markdown 渲染 —— 聊天气泡用。
 *
 * 为什么自研而不是引 marked/markdown-it：
 *   - 零第三方依赖（与项目"轻量零依赖"风格一致）
 *   - 输出直接进 v-html，必须安全：块级结构（标题/列表/引用/代码块围栏）
 *     用原始行判断，正文内容一律先 HTML 转义再白名单变换，
 *     用户/LLM 内容中的 <script>、javascript: 链接等永远无法生效
 *   - 覆盖 LLM 回答的常见格式：加粗/斜体/行内代码/链接/标题/列表/引用/代码块
 */
export function escapeHtml(text) {
  return String(text ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

/** 行内格式：`code` / **bold** / *italic* / [text](https://url)。输入必须是已转义文本。 */
function inline(text) {
  let html = text
  // 链接：只允许 http(s)，杜绝 javascript: 等危险协议
  html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
  // 行内代码
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>')
  // 加粗（先粗后斜，避免星号互相吞）
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
  // 斜体
  html = html.replace(/\*([^*\s][^*]*?)\*/g, '<em>$1</em>')
  return html
}

/**
 * 把 Markdown 文本渲染为安全 HTML。
 * 块级：标题 / 列表（ul、ol）/ 引用 / 围栏代码块；普通行合并为段落，<br> 换行。
 */
export function renderMarkdown(text) {
  if (!text) return ''
  const lines = String(text).split('\n')
  const out = []
  let listType = null // 'ul' | 'ol' | null
  let codeLang = null // 围栏代码块模式

  const closeList = () => {
    if (listType) {
      out.push(`</${listType}>`)
      listType = null
    }
  }
  const flushParagraph = (buf) => {
    if (buf.length) {
      out.push(`<p>${buf.map((line) => inline(escapeHtml(line))).join('<br>')}</p>`)
      buf.length = 0
    }
  }

  let para = []
  for (const rawLine of lines) {
    const trimmed = rawLine.trim()

    // 围栏代码块（用原始行判断围栏，正文整体转义）
    if (/^```/.test(trimmed)) {
      flushParagraph(para)
      closeList()
      if (codeLang === null) {
        codeLang = trimmed.replace(/^```/, '').trim() || 'text'
        out.push('<pre><code>')
      } else {
        out.push('</code></pre>')
        codeLang = null
      }
      continue
    }
    if (codeLang !== null) {
      out.push(escapeHtml(rawLine))
      continue
    }

    // 空行 → 段落结束
    if (!trimmed) {
      flushParagraph(para)
      closeList()
      continue
    }

    // 标题：1-4 级统一缩放到气泡里合适的字号
    const heading = /^(#{1,4})\s+(.*)$/.exec(trimmed)
    if (heading) {
      flushParagraph(para)
      closeList()
      out.push(`<h4>${inline(escapeHtml(heading[2]))}</h4>`)
      continue
    }

    // 引用
    if (trimmed.startsWith('>')) {
      flushParagraph(para)
      closeList()
      out.push(`<blockquote>${inline(escapeHtml(trimmed.replace(/^>\s?/, '')))}</blockquote>`)
      continue
    }

    // 无序列表
    const ul = /^[-*•]\s+(.*)$/.exec(trimmed)
    if (ul) {
      flushParagraph(para)
      if (listType !== 'ul') {
        closeList()
        out.push('<ul>')
        listType = 'ul'
      }
      out.push(`<li>${inline(escapeHtml(ul[1]))}</li>`)
      continue
    }

    // 有序列表
    const ol = /^\d+[.)]\s+(.*)$/.exec(trimmed)
    if (ol) {
      flushParagraph(para)
      if (listType !== 'ol') {
        closeList()
        out.push('<ol>')
        listType = 'ol'
      }
      out.push(`<li>${inline(escapeHtml(ol[1]))}</li>`)
      continue
    }

    // 普通行：并入当前段落（多行用 <br> 连接）
    closeList()
    para.push(trimmed)
  }
  flushParagraph(para)
  closeList()
  if (codeLang !== null) {
    out.push('</code></pre>') // 未闭合的代码块兜底
  }
  return out.join('\n')
}
