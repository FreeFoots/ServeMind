<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import {
  createConversation,
  createProduct,
  getConversation,
  listConversations,
  listDemoAccounts,
  listProducts,
  listPurchases,
  postConversationMessage,
  selectDemoAccount,
  updateHandoff,
  sendFeedback,
} from './api/commerce'

const token = ref('')
const account = ref(null)
const booting = ref(true)
const demoAccounts = ref([])
const authBusy = ref(false)
const authError = ref('')

const page = ref('products')
const products = ref([])
const purchases = ref([])
const conversations = ref([])
const selectedProduct = ref(null)
const selectedConversation = ref(null)
const productForm = reactive({ title: '', description: '', sku_id: '', price: '', category: '手机数码' })
const productCategories = ['手机数码', '电脑办公', '家用电器', '家居日用', '母婴童装', '食品酒饮', '美妆护肤', '服饰鞋靴', '汽车用品', '图书文具', '其他商品']
const productBusy = ref(false)
const conversationBusy = ref(false)
const threadLoading = ref(false)
const refreshing = ref(false)
const sending = ref(false)
const draft = ref('')
const dataError = ref('')
const toast = ref('')
const messageScroll = ref(null)
let pollTimer
let toastTimer

const isMerchant = computed(() => account.value?.role === 'merchant')
const merchantProducts = computed(() => products.value.filter((item) => String(item.merchant_id) === String(account.value?.id)))
const visibleProducts = computed(() => isMerchant.value ? merchantProducts.value : products.value)
const orderedConversations = computed(() => [...conversations.value].sort((a, b) => {
  const left = new Date(a.updated_at || a.created_at || 0).getTime()
  const right = new Date(b.updated_at || b.created_at || 0).getTime()
  return right - left
}))
const contextProduct = computed(() => {
  if (page.value === 'conversations' && selectedConversation.value) {
    return selectedConversation.value.product || products.value.find((item) => item.id === selectedConversation.value.product_id) || null
  }
  return selectedProduct.value
})

function accountName(value) {
  return value?.display_name || value?.username || '未命名用户'
}

function merchantName(product) {
  return product?.merchant_display_name || accountName(product?.merchant) || (product?.merchant_id ? '商家 ' + shortId(product.merchant_id) : '商家未提供名称')
}

function priceLabel(product) {
  if (!product?.display_price) return '价格未提供'
  return product.price_basis === 'merchant_declared'
    ? `商家标价 ¥${product.display_price}`
    : `演示标价 ¥${product.display_price}`
}

function shortId(value) {
  const text = String(value || '')
  return text.length > 12 ? text.slice(0, 8) + '…' : text
}

function formatTime(value) {
  if (!value) return '时间未知'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '时间未知'
  return new Intl.DateTimeFormat('zh-CN', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' }).format(date)
}

function statusLabel(value) {
  return value === 'closed' ? '已结束' : value === 'open' ? '进行中' : (value || '进行中')
}

function conversationProductTitle(item) {
  return item.product?.title || products.value.find((product) => product.id === item.product_id)?.title || '商品会话'
}

function senderLabel(message) {
  if (message.sender_type === 'ai') return 'ServeMind AI'
  if (message.sender_type === 'merchant') return accountName(selectedConversation.value?.merchant)
  if (message.sender_type === 'buyer') return accountName(selectedConversation.value?.buyer)
  return '系统'
}

function showToast(value) {
  toast.value = value
  clearTimeout(toastTimer)
  toastTimer = setTimeout(() => { toast.value = '' }, 3000)
}

function signOut() {
  token.value = ''
  account.value = null
  products.value = []
  purchases.value = []
  conversations.value = []
  selectedProduct.value = null
  selectedConversation.value = null
  draft.value = ''
  dataError.value = ''
  authError.value = ''
}

async function refreshData(quiet = false) {
  if (!token.value) return
  const activeToken = token.value
  if (!quiet) refreshing.value = true
  const results = [
    listProducts(token.value),
    listConversations(token.value),
    ...(isMerchant.value ? [] : [listPurchases(token.value)]),
  ]
  const [productResult, conversationResult, purchaseResult] = await Promise.allSettled(results)
  if (token.value !== activeToken) {
    if (!quiet) refreshing.value = false
    return
  }
  const errors = []
  if (productResult.status === 'fulfilled') products.value = productResult.value.items || []
  else errors.push('商品：' + productResult.reason.message)
  if (conversationResult.status === 'fulfilled') conversations.value = conversationResult.value.items || []
  else errors.push('会话：' + conversationResult.reason.message)
  if (!isMerchant.value && purchaseResult?.status === 'fulfilled') purchases.value = purchaseResult.value.items || []
  if (selectedConversation.value?.id) {
    const previousMessageCount = selectedConversation.value.messages?.length || 0
    const scroller = messageScroll.value
    const nearLatest = scroller && scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 80
    try {
      selectedConversation.value = await getConversation(token.value, selectedConversation.value.id)
      if ((selectedConversation.value.messages?.length || 0) > previousMessageCount) {
        if (nearLatest) await scrollToLatest()
        else if (page.value === 'conversations') showToast('这条会话有新消息')
      }
    } catch (error) {
      errors.push('当前会话：' + error.message)
    }
  }
  dataError.value = errors.join('；')
  if (!quiet) refreshing.value = false
}

async function loadDemoAccounts() {
  booting.value = true
  try {
    const response = await listDemoAccounts()
    demoAccounts.value = response.items || []
    authError.value = demoAccounts.value.length ? '' : '尚无可选的演示用户。'
  } catch (error) {
    authError.value = '无法加载演示用户：' + error.message
  } finally {
    booting.value = false
  }
}

async function chooseDemoAccount(item) {
  if (!item?.id || authBusy.value) return
  authError.value = ''
  authBusy.value = true
  try {
    const response = await selectDemoAccount(item.id)
    token.value = response.token
    account.value = response.account
    page.value = response.account?.role === 'merchant' ? 'conversations' : 'products'
    await refreshData()
    showToast('当前使用：' + response.account.display_name)
  } catch (error) {
    authError.value = error.message
  } finally {
    authBusy.value = false
  }
}

function choosePage(nextPage) {
  page.value = nextPage
  dataError.value = ''
  if (nextPage === 'conversations' && selectedConversation.value) scrollToLatest()
}

async function submitProduct() {
  const title = productForm.title.trim()
  const description = productForm.description.trim()
  if (!title || !description || !productForm.price.trim()) {
    dataError.value = '请填写商品名称、描述和价格。'
    return
  }
  productBusy.value = true
  dataError.value = ''
  try {
    const payload = { title, description, category: productForm.category, price: productForm.price.trim() }
    if (productForm.sku_id.trim()) payload.sku_id = productForm.sku_id.trim()
    const product = await createProduct(token.value, payload)
    products.value = [product, ...products.value.filter((item) => item.id !== product.id)]
    selectedProduct.value = product
    productForm.title = ''
    productForm.description = ''
    productForm.sku_id = ''
    productForm.price = ''
    showToast('商品已发布到演示目录')
  } catch (error) {
    dataError.value = '发布失败：' + error.message
  } finally {
    productBusy.value = false
  }
}

async function openConversation(item) {
  if (!item?.id) return
  threadLoading.value = true
  dataError.value = ''
  page.value = 'conversations'
  let loaded = false
  try {
    selectedConversation.value = await getConversation(token.value, item.id)
    loaded = true
  } catch (error) {
    dataError.value = '无法打开会话：' + error.message
  } finally {
    threadLoading.value = false
  }
  if (loaded) await scrollToLatest()
}

async function startConversation(product) {
  if (!product?.id || !account.value || isMerchant.value) return
  conversationBusy.value = true
  dataError.value = ''
  try {
    const conversation = await createConversation(token.value, product.id)
    selectedConversation.value = conversation
    page.value = 'conversations'
    await refreshData(true)
    await scrollToLatest()
    showToast('已进入 AI 支持，会在需要时邀请商家')
  } catch (error) {
    dataError.value = '无法发起会话：' + error.message
  } finally {
    conversationBusy.value = false
  }
}

async function submitMessage() {
  const content = draft.value.trim()
  const conversationId = selectedConversation.value?.id
  if (!content || !conversationId || sending.value) return
  sending.value = true
  dataError.value = ''
  let submitted = false
  try {
    await postConversationMessage(token.value, conversationId, content)
    submitted = true
    draft.value = ''
    selectedConversation.value = await getConversation(token.value, conversationId)
    const response = await listConversations(token.value)
    conversations.value = response.items || []
    await scrollToLatest()
  } catch (error) {
    dataError.value = submitted
      ? '消息已提交，但刷新会话失败。请点右上角“刷新”查看。' 
      : '发送失败：' + error.message
  } finally {
    sending.value = false
  }
}

async function scrollToLatest() {
  await nextTick()
  if (messageScroll.value) messageScroll.value.scrollTop = messageScroll.value.scrollHeight
}

function handoffLabel(state) {
  return { ai_support: 'AI 正在接待', awaiting_merchant: '等待商家回复', merchant_processing: '商家处理中', merchant_replied: '商家已回复', resolved: '已解决' }[state] || 'AI 正在接待'
}

async function changeHandoff(state) {
  try {
    selectedConversation.value = await updateHandoff(token.value, selectedConversation.value.id, state)
    await refreshData(true)
    showToast(state === 'resolved' ? '已标记解决' : '已开始处理')
  } catch (error) { dataError.value = '更新失败：' + error.message }
}

async function rateMessage(message, rating) {
  try {
    await sendFeedback(token.value, selectedConversation.value.id, message.id, rating)
    showToast('反馈已保存，谢谢你帮助改善客服')
  } catch (error) { dataError.value = '反馈保存失败：' + error.message }
}

function handleComposerKeydown(event) {
  if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
    event.preventDefault()
    submitMessage()
  }
}

onMounted(async () => {
  await loadDemoAccounts()
  pollTimer = setInterval(() => {
    if (account.value && !sending.value && !refreshing.value && !threadLoading.value) refreshData(true)
  }, 12000)
})

onBeforeUnmount(() => {
  clearInterval(pollTimer)
  clearTimeout(toastTimer)
})
</script>

<template>
  <div v-if="booting" class="boot-screen">
    <div class="boot-mark">V</div>
    <strong>ServeMind</strong>
    <p>正在加载演示用户…</p>
  </div>

  <main v-else-if="!account" class="auth-screen">
    <section class="auth-story">
      <div class="auth-brand"><span class="brand-icon">S</span><div><strong>ServeMind</strong><small>智能客服协作平台</small></div></div>
      <div class="auth-story-content">
        <span class="eyebrow">买家 × 商品 × 商家</span>
        <h1>围绕一件商品，<br />把问题说清楚。</h1>
        <p>选择买家或商家身份，围绕商品沟通。AI 可以先回答常见问题；需要商家确认时，再把问题交给商家。</p>
        <div class="flow-chips"><span>01 选择用户</span><i>→</i><span>02 选择商品</span><i>→</i><span>03 开始会话</span></div>
      </div>
      <p class="demo-note">仅供本机演示：用户、商家和商品信息均为模拟数据。</p>
    </section>
    <section class="auth-side">
      <div class="auth-card">
        <div class="auth-heading"><span class="eyebrow">本地体验</span><h2>选择一位演示用户</h2><p>选择买家或商家，直接进入对应工作区。无需输入账号和密码。</p></div>
        <div class="demo-account-groups"><section v-for="role in ['buyer', 'merchant']" :key="role"><h3>{{ role === 'buyer' ? '买家' : '商家' }}</h3><div class="demo-account-list"><button v-for="item in demoAccounts.filter((entry) => entry.role === role)" :key="item.id" type="button" :disabled="authBusy" @click="chooseDemoAccount(item)"><span class="avatar">{{ item.display_name.slice(0, 1) }}</span><span>{{ item.display_name }}</span><span aria-hidden="true">↗</span></button></div></section></div>
        <p v-if="authError" class="form-error" role="alert">{{ authError }}</p>
        <p class="auth-footnote">仅在本机演示环境开放快捷选择；正式上线前必须关闭。</p>
      </div>
    </section>
  </main>

  <main v-else class="workspace">
    <aside class="sidebar">
      <div class="sidebar-brand"><span class="brand-icon">S</span><div><strong>ServeMind</strong><small>智能客服协作平台</small></div></div>
      <div class="sidebar-section-label">工作区</div>
      <nav class="sidebar-nav" aria-label="主导航">
        <button type="button" :class="{ active: page === 'products' }" @click="choosePage('products')"><span class="nav-icon">▦</span><span>{{ isMerchant ? '我的商品' : '发现商品' }}</span><span class="nav-count">{{ visibleProducts.length }}</span></button>
        <button type="button" :class="{ active: page === 'conversations' }" @click="choosePage('conversations')"><span class="nav-icon">◌</span><span>{{ isMerchant ? '买家会话' : '我的会话' }}</span><span class="nav-count">{{ conversations.length }}</span></button>
      </nav>
      <div class="sidebar-spacer"></div>
      <div class="side-explainer"><span class="sparkle">✳</span><strong>AI 不代替商家发言</strong><p>它会作为独立参与者提供支持；商家的回复始终由商家账号发出。</p></div>
      <div class="account-box"><div class="avatar">{{ accountName(account).slice(0, 1) }}</div><div class="account-copy"><strong>{{ accountName(account) }}</strong><small>{{ isMerchant ? '商家演示账号' : '买家演示账号' }}</small></div><button type="button" title="切换用户" aria-label="切换用户" @click="signOut">↪</button></div>
    </aside>

    <section class="main-column">
      <header class="toolbar"><div><span class="eyebrow">{{ isMerchant ? 'MERCHANT WORKSPACE' : 'BUYER WORKSPACE' }}</span><h1>{{ page === 'products' ? (isMerchant ? '我的商品' : '发现商品') : (isMerchant ? '买家会话' : '我的会话') }}</h1></div><div class="toolbar-actions"><span class="env-badge"><i></i> 本地演示环境</span><button type="button" class="icon-button" :disabled="refreshing" aria-label="刷新数据" title="刷新数据" @click="refreshData()">{{ refreshing ? '…' : '↻' }}</button></div></header>
      <div v-if="dataError" class="data-error" role="alert"><span>{{ dataError }}</span><button type="button" aria-label="关闭提示" @click="dataError = ''">×</button></div>

      <div v-if="page === 'products'" class="page-scroll">
        <div class="page-intro"><span class="eyebrow">{{ isMerchant ? 'PRODUCT MANAGEMENT' : 'PRODUCT DISCOVERY' }}</span><h2>{{ isMerchant ? '发布商品，等待真实对话' : '从商品开始，联系对应商家' }}</h2><p>{{ isMerchant ? '新商品会出现在买家的商品目录；买家发起的对话会进入你的会话列表。' : '选择一件商品即可创建与该商家的共享会话。AI 会以独立身份加入支持。' }}</p></div>
        <div class="trust-banner"><span>ⓘ</span><p>目录中的商家归属与商品状态是<strong>项目内模拟数据</strong>。AI 会先私下处理买家问题，只有需要人工决策时才向对应商家开放会话。</p></div>
        <section v-if="!isMerchant && purchases.length" class="purchase-strip"><div class="section-head"><div><span class="eyebrow">MY PURCHASES</span><h3>我的已购商品</h3></div><span class="hint">买家专属 · 模拟绑定</span></div><div class="purchase-grid"><article v-for="purchase in purchases" :key="purchase.id" class="purchase-card"><div><span class="product-tag">{{ purchase.product.catalog_label }}</span><h4>{{ purchase.product.title }}</h4><p>{{ purchase.order_alias }} · {{ purchase.fulfillment_status }}</p><p>{{ priceLabel(purchase.product) }}</p></div><button type="button" class="text-button" @click="startConversation(products.find((item) => item.sku_id === purchase.sku_id))">咨询 AI ↗</button></article></div></section>

        <form v-if="isMerchant" class="create-product" @submit.prevent="submitProduct">
          <div class="section-head"><div><span class="eyebrow">NEW PRODUCT</span><h3>发布一件商品</h3></div><span class="hint">商家自行声明 · 未核验</span></div>
          <div class="product-form-grid"><label class="field"><span>商品名称</span><input v-model="productForm.title" placeholder="例如：蓝牙耳机" maxlength="120" required /></label><label class="field"><span>商品类别</span><select v-model="productForm.category"><option v-for="category in productCategories" :key="category" :value="category">{{ category }}</option></select></label><label class="field"><span>SKU 编号（可选）</span><input v-model="productForm.sku_id" placeholder="仅作商家自填标识" maxlength="80" /></label><label class="field"><span>商家标价（元）</span><input v-model="productForm.price" type="number" min="0.01" step="0.01" placeholder="例如：129.90" required /></label></div>
          <label class="field"><span>商品描述</span><textarea v-model="productForm.description" placeholder="请描述商品及买家可能关心的信息" rows="3" maxlength="1000"></textarea></label>
          <div class="form-actions"><small>发布后，买家可围绕该商品发起对话。</small><button class="primary-button" type="submit" :disabled="productBusy">{{ productBusy ? '发布中…' : '发布商品' }} <span aria-hidden="true">↗</span></button></div>
        </form>

        <div class="section-head catalog-head"><div><span class="eyebrow">CATALOG</span><h3>{{ isMerchant ? '我发布的商品' : '可咨询的商品' }}</h3></div><span class="result-count">{{ visibleProducts.length }} 件</span></div>
        <div v-if="!visibleProducts.length" class="empty-state"><div class="empty-icon">▦</div><h3>{{ isMerchant ? '还没有发布商品' : '商品目录暂时为空' }}</h3><p>{{ isMerchant ? '填写上方表单发布第一件商品，买家就能发起会话。' : '需要先由商家演示账号发布商品。随后你可以从商品卡联系商家。' }}</p></div>
        <div v-else class="product-grid"><article v-for="product in visibleProducts" :key="product.id" class="product-card" :class="{ selected: selectedProduct?.id === product.id }"><button type="button" class="product-main" @click="selectedProduct = product"><div class="product-glyph">▧</div><span class="product-tag">{{ product.category }} · {{ product.catalog_status || '在售' }}</span><h4>{{ product.title }}</h4><p>{{ product.description }}</p><div class="product-meta"><span>{{ merchantName(product) }}</span><span>{{ priceLabel(product) }}</span><span v-if="product.sku_id">SKU {{ product.sku_id }}</span></div></button><div class="product-card-foot"><small>{{ formatTime(product.created_at) }} 发布</small><button v-if="!isMerchant" type="button" class="text-button" :disabled="conversationBusy" @click="startConversation(product)">咨询 AI ↗</button><span v-else class="owner-label">我的商品</span></div></article></div>
      </div>

      <div v-else class="conversation-layout">
        <aside class="thread-list"><div class="thread-list-head"><strong>{{ isMerchant ? '待处理与进行中' : '商品会话' }}</strong><span>{{ conversations.length }}</span></div><div v-if="!orderedConversations.length" class="thread-empty"><span>◌</span><strong>还没有会话</strong><p>{{ isMerchant ? '买家从你的商品发起咨询后，会出现在这里。' : '先到“发现商品”中选择商品，再发起会话。' }}</p><button v-if="!isMerchant" type="button" class="text-button" @click="choosePage('products')">去发现商品 ↗</button></div><div v-else class="thread-items"><button v-for="item in orderedConversations" :key="item.id" type="button" class="thread-item" :class="{ active: selectedConversation?.id === item.id }" @click="openConversation(item)"><span class="thread-product-icon">▧</span><span class="thread-item-body"><strong>{{ conversationProductTitle(item) }}</strong><small>{{ isMerchant ? accountName(item.buyer) : accountName(item.merchant) }}</small><em>{{ formatTime(item.updated_at || item.created_at) }}</em></span><span class="thread-status-dot"></span></button></div></aside>

        <section class="thread-body">
          <div v-if="threadLoading" class="thread-placeholder"><span class="loading-ring"></span><p>正在读取会话…</p></div>
          <div v-else-if="!selectedConversation" class="thread-placeholder"><div class="placeholder-icon">◌</div><h2>选择一条会话</h2><p>{{ isMerchant ? '查看买家问题，并以商家身份作出回复。' : '商品、商家与 AI 的讨论会保存在同一条线程。' }}</p></div>
          <template v-else>
            <header class="thread-header"><div class="thread-header-icon">▧</div><div class="thread-header-copy"><span class="eyebrow">{{ selectedConversation.merchant_visible ? 'MERCHANT SHARED' : 'AI PRIVATE SUPPORT' }} · {{ statusLabel(selectedConversation.status) }}</span><h2>{{ contextProduct?.title || '商品会话' }}</h2><p>买家 {{ accountName(selectedConversation.buyer) }} <span>·</span> {{ selectedConversation.merchant_visible ? '商家已介入' : '商家尚未介入' }} <span>·</span> ServeMind AI</p></div><span class="thread-id">#{{ shortId(selectedConversation.id) }}</span></header>
            <div class="handoff-panel">
              <span>{{ handoffLabel(selectedConversation.handoff_state) }}</span>
              <template v-if="isMerchant && selectedConversation.handoff_summary?.question_summary">
                <p>买家需要：{{ selectedConversation.handoff_summary.question_summary }}</p>
                <p v-if="selectedConversation.handoff_summary.order_summary?.fulfillment_status">订单状态：{{ selectedConversation.handoff_summary.order_summary.fulfillment_status }}</p>
                <button v-if="selectedConversation.handoff_state === 'awaiting_merchant'" type="button" class="text-button" @click="changeHandoff('merchant_processing')">开始处理</button>
                <button v-if="selectedConversation.handoff_state !== 'resolved'" type="button" class="text-button" @click="changeHandoff('resolved')">标记已解决</button>
              </template>
            </div>
            <div ref="messageScroll" class="thread-messages"><div class="thread-start"><span>{{ selectedConversation.merchant_visible ? '商家已介入' : 'AI 支持已建立' }}</span><p>{{ selectedConversation.merchant_visible ? '订单摘要与问题摘要已交给对应模拟商家；商家回复才代表商家发言。' : '当前只有你和 AI 可见；需要商家决策时，AI 才会发起升级。' }}</p></div><div v-if="!selectedConversation.messages?.length" class="no-messages">还没有消息。{{ isMerchant ? '等待买家问题升级后再回复。' : '先向 AI 描述你的问题。' }}</div><article v-for="message in selectedConversation.messages || []" :key="message.id" class="message-row" :class="{ mine: message.sender_type === account.role, ai: message.sender_type === 'ai' }"><div class="message-avatar">{{ message.sender_type === 'ai' ? '✳' : senderLabel(message).slice(0, 1) }}</div><div class="message-content"><div class="message-meta"><strong>{{ senderLabel(message) }}</strong><span v-if="message.sender_type === 'ai'" class="ai-label">AI 支持</span><span v-else-if="message.sender_type === 'merchant'" class="merchant-label">商家</span><time>{{ formatTime(message.created_at) }}</time></div><p>{{ message.content }}</p><div v-if="message.sender_type === 'ai'" class="message-feedback"><button type="button" @click="rateMessage(message, 'helpful')">有帮助</button><button type="button" @click="rateMessage(message, 'not_helpful')">没解决</button><button type="button" @click="rateMessage(message, 'incorrect')">信息有误</button></div></div></article></div>
            <form class="composer" @submit.prevent="submitMessage"><label for="message-draft">{{ isMerchant ? '以商家身份回复' : '向 AI 支持提问' }}</label><textarea id="message-draft" v-model="draft" rows="3" :placeholder="isMerchant ? '写下商家的回复，买家会在同一会话看到…' : '先向 AI 描述问题；需要商家处理时会自动升级…'" maxlength="2000" @keydown="handleComposerKeydown"></textarea><div class="composer-foot"><span>{{ isMerchant ? '此消息将标记为商家发言' : 'AI 先私聊处理，升级后商家才可见' }}</span><small>⌘ Enter 发送</small><button type="submit" class="primary-button" :disabled="sending || !draft.trim()">{{ sending ? '发送中…' : '发送消息' }} <span aria-hidden="true">↗</span></button></div></form>
          </template>
        </section>
      </div>
    </section>

    <aside class="context-panel"><template v-if="contextProduct"><div class="context-heading"><span class="eyebrow">CONTEXT</span><h2>这次沟通围绕什么？</h2></div><div class="context-product"><div class="context-glyph">▧</div><span class="product-tag">商家声明 · 未核验</span><h3>{{ contextProduct.title }}</h3><p>{{ contextProduct.description }}</p><div v-if="contextProduct.sku_id" class="context-detail"><span>SKU</span><strong>{{ contextProduct.sku_id }}</strong></div><div class="context-detail"><span>商家</span><strong>{{ merchantName(contextProduct) }}</strong></div></div><div v-if="page === 'conversations' && selectedConversation" class="participant-card"><span class="eyebrow">PARTICIPANTS</span><div><i class="participant-dot buyer-dot"></i><span>买家</span><strong>{{ accountName(selectedConversation.buyer) }}</strong></div><div><i class="participant-dot merchant-dot"></i><span>商家</span><strong>{{ accountName(selectedConversation.merchant) }}</strong></div><div><i class="participant-dot ai-dot"></i><span>AI 支持</span><strong>ServeMind AI</strong></div></div></template><template v-else><div class="context-heading"><span class="eyebrow">HOW IT WORKS</span><h2>一件商品，<br />一条清晰的链路。</h2></div><div class="context-steps"><div><span>1</span><p>商家声明并发布商品</p></div><div><span>2</span><p>买家选择商品发起会话</p></div><div><span>3</span><p>AI 支持，商家亲自回复</p></div></div></template><div class="context-boundary"><strong>沟通提示</strong><p>订单问题请从“我的已购商品”打开对应订单；商品页面没有写明的信息，可以请商家确认。</p></div></aside>
    <div v-if="toast" class="toast" role="status">{{ toast }}</div>
  </main>
</template>
