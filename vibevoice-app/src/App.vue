<template>
  <el-container class="app-container">
    <!-- Sidebar -->
    <el-aside width="240px" class="aside">
      <div class="logo">
        <el-icon :size="28" color="#409eff"><Microphone /></el-icon>
        <span>VibeVoice <small>Desktop</small></span>
      </div>
      
      <el-menu
        :default-active="activeTab"
        class="el-menu-vertical"
        @select="handleSelect"
      >
        <el-menu-item index="asr">
          <el-icon><Document /></el-icon>
          <span>语音转写 (ASR)</span>
        </el-menu-item>
        <el-menu-item index="tts">
          <el-icon><ChatDotRound /></el-icon>
          <span>语音合成 (TTS)</span>
        </el-menu-item>
        <el-divider />
        <el-menu-item index="history">
          <el-icon><Document /></el-icon>
          <span>历史记录</span>
        </el-menu-item>
        <el-divider />
        <el-menu-item index="settings">
          <el-icon><Setting /></el-icon>
          <span>系统设置</span>
        </el-menu-item>
        <el-menu-item index="about">
          <el-icon><InfoFilled /></el-icon>
          <span>关于系统</span>
        </el-menu-item>
      </el-menu>
      
      <div class="status-panel">
        <div class="status-item">
          <span class="dot" :class="{ active: status.asr_loaded }"></span>
          ASR 引擎: {{ status.asr_loaded ? '已就绪' : '待加载' }}
        </div>
        <div class="status-item">
          <span class="dot" :class="{ active: status.tts_loaded }"></span>
          TTS 引擎: {{ status.tts_loaded ? '已就绪' : '待加载' }}
        </div>
        <div class="device-info">
          <el-tag size="small" effect="dark" :type="status.device === 'cuda' ? 'success' : 'warning'">
            设备: {{ status.device.toUpperCase() }}
          </el-tag>
        </div>
      </div>
    </el-aside>

    <!-- Main Content -->
    <el-main class="main-content">
      <!-- ASR Page -->
      <transition name="fade" mode="out-in">
        <div v-if="activeTab === 'asr'" class="page-container" key="asr">
          <div class="page-header">
            <h2>智能语音转写</h2>
            <p>支持单次最长 60 分钟音频，自动识别说话人与时间戳</p>
          </div>

          <el-card class="upload-card" shadow="never">
            <el-upload
              class="asr-uploader"
              drag
              action="#"
              :auto-upload="false"
              :show-file-list="false"
              :on-change="handleFileChange"
              accept=".mp3,.wav,.m4a,.flac"
            >
              <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
              <div class="el-upload__text">
                将音频文件拖到此处，或 <em>点击上传</em>
              </div>
            </el-upload>

            <div v-if="audioFile" class="audio-info">
              <div class="file-meta">
                <el-icon><Headset /></el-icon>
                <span>{{ audioFile.name }}</span>
                <el-button link type="danger" @click="clearAudio">移除</el-button>
              </div>
              <div id="waveform" class="waveform-container"></div>
              <div class="actions">
                <el-button 
                  type="primary" 
                  size="large" 
                  :loading="processing" 
                  @click="startTranscription"
                  icon="VideoPlay"
                >
                  {{ processing ? '正在解析音频并识别...' : '开始智能转写' }}
                </el-button>
                <el-button 
                  v-if="processing" 
                  type="danger" 
                  size="large" 
                  plain 
                  style="margin-left: 12px"
                  @click="cancelTranscription"
                >
                  取消任务
                </el-button>
              </div>
              <div v-if="processing" class="asr-progress">
                <el-progress :percentage="asrProgress" :stroke-width="14" status="success" />
              </div>
            </div>
          </el-card>

          <el-card v-if="transcription || processing" class="result-card" shadow="hover">
            <template #header>
              <div class="card-header">
                <span class="title">转写结果</span>
                <div class="tools">
                  <el-button link icon="CopyDocument" @click="copyTranscription">复制</el-button>
                  <el-button link icon="Download" @click="downloadTranscription">导出</el-button>
                </div>
              </div>
            </template>
            
            <div v-if="processing && !transcription" class="loading-state">
              <el-skeleton :rows="5" animated />
            </div>
            <div v-else class="transcription-content">
              {{ transcription }}
            </div>
          </el-card>
        </div>

        <!-- TTS Page -->
        <div v-else-if="activeTab === 'tts'" class="page-container" key="tts">
          <div class="page-header">
            <h2>长文本语音合成</h2>
            <p>基于 Diffusion 架构，支持多人对话与长达 90 分钟的语音生成</p>
          </div>

          <el-row :gutter="20">
            <el-col :span="16">
              <el-card class="editor-card" shadow="never">
                <el-input
                  v-model="ttsText"
                  type="textarea"
                  :rows="12"
                  placeholder="请输入需要合成的文本内容..."
                  resize="none"
                />
                <div class="text-stats">
                  字数统计: {{ ttsText.length }} 字
                </div>
              </el-card>
            </el-col>
            <el-col :span="8">
              <el-card class="config-card" shadow="never">
                <template #header>合成配置</template>
                <el-form label-position="top">
                  <el-form-item label="角色音色">
                    <el-select v-model="selectedVoice" placeholder="选择音色" style="width: 100%">
                      <el-option
                        v-for="voice in voices"
                        :key="voice"
                        :label="voice"
                        :value="voice"
                      />
                    </el-select>
                  </el-form-item>
                  <el-form-item label="推理步数 (Diffusion Steps)">
                    <el-slider v-model="ttsInferenceSteps" :min="1" :max="50" />
                  </el-form-item>
                  <div class="tts-actions">
                    <el-button 
                      type="primary" 
                      style="width: 100%" 
                      size="large"
                      @click="generateTTS" 
                      :loading="ttsLoading"
                      icon="Microphone"
                    >
                      生成语音
                    </el-button>
                  <div v-if="ttsAudioUrl" style="margin-top:12px">
                    <audio :src="ttsAudioUrl" controls style="width:100%"></audio>
                  </div>
                  </div>
                </el-form>
              </el-card>
            </el-col>
          </el-row>
        </div>

        <!-- History Page -->
        <div v-else-if="activeTab === 'history'" class="page-container" key="history">
          <div class="page-header">
            <h2>历史记录</h2>
            <p>查看最近的转写与合成任务</p>
          </div>

          <el-row :gutter="20">
            <el-col :span="12">
              <el-card shadow="never" class="history-card">
                <template #header>ASR 历史</template>
                <el-table :data="asrHistory" size="small" style="width:100%">
                  <el-table-column prop="id" label="任务ID" width="220" />
                  <el-table-column label="时长" width="100">
                    <template #default="scope">{{ formatDuration(scope.row.total_seconds) }}</template>
                  </el-table-column>
                  <el-table-column label="操作" width="240">
                    <template #default="scope">
                      <el-button size="small" @click="downloadAsr(scope.row)">下载文本</el-button>
                      <el-button size="small" type="danger" @click="deleteAsr(scope.row)">删除</el-button>
                    </template>
                  </el-table-column>
                </el-table>
              </el-card>
            </el-col>
            <el-col :span="12">
              <el-card shadow="never" class="history-card">
                <template #header>TTS 历史</template>
                <el-table :data="ttsHistory" size="small" style="width:100%">
                  <el-table-column prop="id" label="文件名" width="240" />
                  <el-table-column prop="voice" label="音色" width="140" />
                  <el-table-column label="操作" width="220">
                    <template #default="scope">
                      <el-button size="small" @click="playTts(scope.row)">播放</el-button>
                      <el-button size="small" @click="downloadTts(scope.row)">下载</el-button>
                      <el-button size="small" type="danger" @click="deleteTts(scope.row)">删除</el-button>
                    </template>
                  </el-table-column>
                </el-table>
              </el-card>
            </el-col>
          </el-row>
        </div>

        <!-- Settings Page -->
        <div v-else-if="activeTab === 'settings'" class="page-container" key="settings">
          <div class="page-header">
            <h2>系统设置</h2>
            <p>配置模型运行环境与服务端口</p>
          </div>

          <el-card class="settings-card" shadow="never">
            <el-form :model="settings" label-width="140px">
              <el-form-item label="首选推理设备">
                <el-radio-group v-model="settings.device">
                  <el-radio-button label="cuda">GPU (NVIDIA)</el-radio-button>
                  <el-radio-button label="cpu">CPU (兼容模式)</el-radio-button>
                </el-radio-group>
                <div class="form-tip">GPU 模式需要 NVIDIA 显卡及显存支持</div>
              </el-form-item>
              <el-form-item v-if="gpuMismatch">
                <el-alert title="当前系统无可用 CUDA，但选择了 GPU 模式，可能无法正常运行。请切换为 CPU 或安装合适的 CUDA 环境。" type="warning" show-icon />
              </el-form-item>
              <el-form-item label="当前设备信息">
                <div>
                  <el-tag size="small" style="margin-right:8px">设备：{{ status.device.toUpperCase() }}</el-tag>
                  <el-tag v-if="status.cuda_device_name" size="small" type="success">{{ status.cuda_device_name }}</el-tag>
                  <el-tag v-else size="small" type="warning">无 CUDA</el-tag>
                  <div v-if="status.cuda_total_mem_bytes" style="margin-top:6px;color:#64748b;font-size:12px">
                    显存总量：{{ (status.cuda_total_mem_bytes/1024/1024/1024).toFixed(1) }} GB
                  </div>
                </div>
              </el-form-item>
              <el-form-item label="导出目录">
                <div style="display:flex;gap:8px;align-items:center;width:100%">
                  <el-input v-model="settings.exportDir" placeholder="未选择" readonly />
                  <el-button @click="selectExportDir">选择目录</el-button>
                  <el-button type="primary" @click="exportHistory" :disabled="!settings.exportDir">一键导出历史</el-button>
                  <el-button @click="openExportDir" :disabled="!settings.exportDir">打开目录</el-button>
                </div>
                <div class="form-tip">导出 ASR 文本与 TTS 音频到所选目录（自动创建 ASR/TTS 子文件夹）</div>
              </el-form-item>
              <el-form-item label="后端 API 端口">
                <el-input-number v-model="settings.port" :min="1024" :max="65535" />
              </el-form-item>
              <el-divider />
              <el-form-item>
                <el-button type="primary" @click="saveSettings">保存并重启服务</el-button>
                <el-button @click="resetSettings">恢复默认</el-button>
              </el-form-item>
            </el-form>
          </el-card>
        </div>

        <!-- About Page -->
        <div v-else-if="activeTab === 'about'" class="page-container" key="about">
          <div class="page-header">
            <h2>关于 VibeVoice Desktop</h2>
            <p>系统信息与开发者指南</p>
          </div>

          <el-card shadow="never" class="about-card">
            <div class="about-content">
              <h3>VibeVoice Desktop v1.0.0</h3>
              <p>本项目是 VibeVoice 语音系统的官方桌面化封装版本。</p>
              
              <el-divider />
              
              <h4>系统特性</h4>
              <ul>
                <li>基于 Electron + Vue 3 构建的跨平台桌面应用</li>
                <li>集成嵌入式 Python 运行时，无需安装 Python 环境</li>
                <li>支持 GPU (CUDA) 与 CPU 模式一键切换</li>
                <li>全流程 ASR 转写进度监控与任务取消</li>
                <li>历史记录持久化管理与一键批量导出</li>
              </ul>

              <el-divider />

              <h4>运行状态</h4>
              <el-descriptions :column="1" border>
                <el-descriptions-item label="计算设备">
                  <el-tag :type="status.cuda_available ? 'success' : 'info'">
                    {{ status.cuda_available ? 'CUDA 加速已开启' : 'CPU 兼容模式' }}
                  </el-tag>
                </el-descriptions-item>
                <el-descriptions-item label="后端 API 地址">
                  <code>http://127.0.0.1:{{ settings.port }}</code>
                </el-descriptions-item>
                <el-descriptions-item label="ASR 模型状态">
                  {{ status.asr_loaded ? '✅ 已加载' : '⏳ 待加载' }}
                </el-descriptions-item>
                <el-descriptions-item label="TTS 模型状态">
                  {{ status.tts_loaded ? '✅ 已加载' : '⏳ 待加载' }}
                </el-descriptions-item>
              </el-descriptions>
              
              <div style="margin-top: 32px; text-align: center; color: #94a3b8; font-size: 13px;">
                © 2026 VibeVoice Desktop Team. Powered by VibeVoice Core.
              </div>
            </div>
          </el-card>
        </div>
      </transition>
    </el-main>
  </el-container>
</template>

<script lang="ts" setup>
import { ref, onMounted, reactive, watch, nextTick, computed } from 'vue'
import axios from 'axios'
import { ElMessage, ElMessageBox } from 'element-plus'
import WaveSurfer from 'wavesurfer.js'
import { ipcRenderer } from 'electron'

// --- State ---
const activeTab = ref('asr')
const audioFile = ref(null)
const transcription = ref('')
const processing = ref(false)
const asrJobId = ref('')
const asrProgress = ref(0)
let asrTimer: any = null
const ttsText = ref('')
const voices = ref([])
const selectedVoice = ref('')
const ttsInferenceSteps = ref(5)
const ttsLoading = ref(false)
const ttsAudioUrl = ref('')
const status = reactive({ asr_loaded: false, tts_loaded: false, device: 'cpu' })
const settings = reactive({ 
  device: 'cuda', 
  port: 8000,
  exportDir: ''
})
const asrHistory = ref([])
const ttsHistory = ref([])
const SETTINGS_KEY = 'vv-settings'
const gpuMismatch = computed(() => settings.device === 'cuda' && !status.cuda_available)

let wavesurfer = null

// --- Methods ---
const handleSelect = (index) => {
  activeTab.value = index
}

const handleFileChange = (file) => {
  audioFile.value = file.raw
  transcription.value = ''
  initWaveSurfer(file.raw)
}

const clearAudio = () => {
  audioFile.value = null
  transcription.value = ''
  if (wavesurfer) wavesurfer.destroy()
}

const initWaveSurfer = (file) => {
  if (wavesurfer) wavesurfer.destroy()
  
  nextTick(() => {
    wavesurfer = WaveSurfer.create({
      container: '#waveform',
      waveColor: '#dcdfe6',
      progressColor: '#409eff',
      cursorColor: '#409eff',
      barWidth: 2,
      barRadius: 3,
      cursorWidth: 1,
      height: 100,
      plugins: []
    })
    wavesurfer.load(URL.createObjectURL(file))
  })
}

const startTranscription = async () => {
  if (!audioFile.value) return ElMessage.warning('请先上传音频文件')
  
  processing.value = true
  asrProgress.value = 0
  transcription.value = ''
  const formData = new FormData()
  formData.append('file', audioFile.value)
  try {
    const res = await axios.post(`http://127.0.0.1:${settings.port}/asr/start`, formData)
    asrJobId.value = res.data.job_id
    asrTimer = setInterval(async () => {
      try {
        const st = await axios.get(`http://127.0.0.1:${settings.port}/asr/status/${asrJobId.value}`)
        asrProgress.value = st.data.progress
        if (st.data.status === 'done') {
          clearInterval(asrTimer)
          const rr = await axios.get(`http://127.0.0.1:${settings.port}/asr/result/${asrJobId.value}`)
          transcription.value = rr.data.text
          processing.value = false
          ElMessage.success('转写完成')
        } else if (st.data.status === 'error' || st.data.status === 'canceled') {
          clearInterval(asrTimer)
          processing.value = false
          ElMessage.error('任务中断: ' + (st.data.error || st.data.status))
        }
      } catch (e) {
        clearInterval(asrTimer)
        processing.value = false
        ElMessage.error('查询状态失败')
      }
    }, 1000)
  } catch (err) {
    processing.value = false
    ElMessage.error('任务提交失败: ' + (err.response?.data?.detail || err.message))
  }
}

const cancelTranscription = async () => {
  if (!asrJobId.value) return
  try {
    await axios.post(`http://127.0.0.1:${settings.port}/asr/cancel/${asrJobId.value}`)
    if (asrTimer) clearInterval(asrTimer)
    processing.value = false
    ElMessage.success('已请求取消')
  } catch {
    ElMessage.error('取消失败')
  }
}

const copyTranscription = () => {
  navigator.clipboard.writeText(transcription.value)
  ElMessage.success('已复制到剪贴板')
}

const downloadTranscription = () => {
  const blob = new Blob([transcription.value], { type: 'text/plain' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `VibeVoice_ASR_${new Date().getTime()}.txt`
  a.click()
}

const generateTTS = async () => {
  if (!ttsText.value) return ElMessage.warning('请输入需要合成的文本')
  if (!selectedVoice.value) return ElMessage.warning('请选择角色音色')
  
  ttsLoading.value = true
  try {
    const res = await axios.post(`http://127.0.0.1:${settings.port}/tts`, { 
      text: ttsText.value, 
      voice: selectedVoice.value,
      inference_steps: ttsInferenceSteps.value
    })
    ElMessage.success('语音合成成功')
    if (res.data?.url) {
      ttsAudioUrl.value = `http://127.0.0.1:${settings.port}${res.data.url}`
    }
  } catch (err) {
    ElMessage.error('合成失败: ' + (err.response?.data?.detail || err.message))
  } finally {
    ttsLoading.value = false
  }
}

const fetchStatus = async () => {
  try {
    const res = await axios.get(`http://127.0.0.1:${settings.port}/status`)
    status.asr_loaded = res.data.asr_loaded
    status.tts_loaded = res.data.tts_loaded
    status.device = res.data.device
  } catch (e) {
    // Backend might be offline
  }
}

const fetchVoices = async () => {
  try {
    const res = await axios.get(`http://127.0.0.1:${settings.port}/voices`)
    voices.value = res.data.voices
    if (voices.value.length > 0 && !selectedVoice.value) {
      selectedVoice.value = voices.value[0]
    }
  } catch (e) {
    console.error('Failed to fetch voices')
  }
}

const fetchHistory = async () => {
  try {
    const asr = await axios.get(`http://127.0.0.1:${settings.port}/history/asr`)
    asrHistory.value = asr.data.items || []
  } catch {}
  try {
    const tts = await axios.get(`http://127.0.0.1:${settings.port}/history/tts`)
    ttsHistory.value = tts.data.items || []
  } catch {}
}

const downloadAsr = (row) => {
  const a = document.createElement('a')
  a.href = `http://127.0.0.1:${settings.port}/asr/download/${row.id}`
  a.download = `ASR_${row.id}.txt`
  a.click()
}

const playTts = (row) => {
  ttsAudioUrl.value = `http://127.0.0.1:${settings.port}/audio/${row.id}`
}

const downloadTts = (row) => {
  const a = document.createElement('a')
  a.href = `http://127.0.0.1:${settings.port}/audio/${row.id}`
  a.download = row.id
  a.click()
}

const deleteAsr = async (row) => {
  ElMessageBox.confirm('确认删除该 ASR 记录及其文本文件吗？', '确认操作', {
    confirmButtonText: '确定',
    cancelButtonText: '取消',
    type: 'warning'
  }).then(async () => {
    try {
      await axios.delete(`http://127.0.0.1:${settings.port}/history/asr/${row.id}`)
      ElMessage.success('已删除')
      fetchHistory()
    } catch {
      ElMessage.error('删除失败')
    }
  })
}

const deleteTts = async (row) => {
  ElMessageBox.confirm('确认删除该 TTS 音频文件吗？', '确认操作', {
    confirmButtonText: '确定',
    cancelButtonText: '取消',
    type: 'warning'
  }).then(async () => {
    try {
      await axios.delete(`http://127.0.0.1:${settings.port}/history/tts/${row.id}`)
      ElMessage.success('已删除')
      fetchHistory()
    } catch {
      ElMessage.error('删除失败')
    }
  })
}

const saveSettings = () => {
  ElMessageBox.confirm('修改设置需要重启后端服务，确定吗？', '确认操作', {
    confirmButtonText: '确定',
    cancelButtonText: '取消',
    type: 'warning'
  }).then(() => {
    ElMessage.success('设置已保存，正在重启服务...')
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings))
    } catch {}
    ipcRenderer.invoke('vv:restart-backend', { port: settings.port, device: settings.device })
      .then(() => {
        setTimeout(() => {
          fetchStatus()
          fetchVoices()
          fetchHistory()
        }, 1200)
      })
  })
}

const resetSettings = () => {
  settings.device = 'cuda'
  settings.port = 8000
  settings.exportDir = ''
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings))
  } catch {}
}

const loadSettings = () => {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY)
    if (raw) {
      const data = JSON.parse(raw)
      if (typeof data.port === 'number') settings.port = data.port
      if (typeof data.device === 'string') settings.device = data.device
      if (typeof data.exportDir === 'string') settings.exportDir = data.exportDir
    }
  } catch {}
}

const formatDuration = (sec) => {
  if (!sec || isNaN(sec)) return '00:00'
  const s = Math.floor(sec)
  const m = Math.floor(s / 60)
  const r = s % 60
  const mm = String(m).padStart(2, '0')
  const ss = String(r).padStart(2, '0')
  return `${mm}:${ss}`
}
const selectExportDir = async () => {
  const p = await ipcRenderer.invoke('vv:select-export-dir')
  if (p) {
    settings.exportDir = p
    try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings)) } catch {}
    ElMessage.success('已选择导出目录')
  }
}

const exportHistory = async () => {
  if (!settings.exportDir) return ElMessage.warning('请先选择导出目录')
  const res = await ipcRenderer.invoke('vv:export-history', { dir: settings.exportDir })
  if (res?.ok) {
    ElMessage.success(`导出完成：ASR ${res.asr} 条，TTS ${res.tts} 条`)
  } else {
    ElMessage.error('导出失败')
  }
}
const openExportDir = async () => {
  if (!settings.exportDir) return
  await ipcRenderer.invoke('vv:open-path', { dir: settings.exportDir })
}
watch(activeTab, (newTab) => {
  if (newTab === 'tts' && voices.value.length === 0) {
    fetchVoices()
  }
  if (newTab === 'history') {
    fetchHistory()
  }
})

onMounted(() => {
  loadSettings()
  fetchStatus()
  fetchVoices()
  fetchHistory()
  // Retry fetch voices if empty
  const timer = setInterval(() => {
    if (voices.value.length === 0) {
      fetchVoices()
    } else {
      clearInterval(timer)
    }
  }, 5000)
  
  setInterval(fetchStatus, 3000)
})
</script>

<style>
/* Global Resets */
body { margin: 0; font-family: 'Inter', -apple-system, sans-serif; -webkit-font-smoothing: antialiased; }

/* Transitions */
.fade-enter-active, .fade-leave-active { transition: opacity 0.2s ease; }
.fade-enter-from, .fade-leave-to { opacity: 0; }

.app-container { height: 100vh; background-color: #f8fafc; }

.aside {
  background-color: #ffffff;
  border-right: 1px solid #e2e8f0;
  display: flex;
  flex-direction: column;
  padding: 0;
}

.logo {
  height: 70px;
  display: flex;
  align-items: center;
  padding: 0 24px;
  font-size: 18px;
  font-weight: 700;
  color: #1e293b;
  border-bottom: 1px solid #f1f5f9;
}
.logo small { font-weight: 400; color: #64748b; margin-left: 5px; font-size: 12px; }

.el-menu-vertical { border-right: none !important; flex: 1; padding: 12px 0; }
.el-menu-item { height: 50px; line-height: 50px; margin: 4px 12px; border-radius: 8px; }
.el-menu-item.is-active { background-color: #f0f7ff !important; color: #409eff !important; font-weight: 600; }

.status-panel {
  padding: 24px;
  border-top: 1px solid #f1f5f9;
  font-size: 13px;
  color: #64748b;
}
.status-item { display: flex; align-items: center; margin-bottom: 10px; }
.dot { width: 8px; height: 8px; border-radius: 50%; background-color: #cbd5e1; margin-right: 10px; }
.dot.active { background-color: #10b981; box-shadow: 0 0 8px rgba(16, 185, 129, 0.4); }
.device-info { margin-top: 12px; }

.main-content { padding: 40px; max-width: 1000px; margin: 0 auto; }

.page-header { margin-bottom: 32px; }
.page-header h2 { margin: 0 0 8px 0; color: #1e293b; font-size: 24px; }
.page-header p { margin: 0; color: #64748b; font-size: 14px; }

.upload-card, .editor-card, .config-card, .settings-card {
  border-radius: 12px;
  border: 1px solid #e2e8f0;
}

.asr-uploader .el-upload-dragger { border-radius: 12px; padding: 40px; }

.audio-info { margin-top: 24px; }
.file-meta { display: flex; align-items: center; gap: 8px; margin-bottom: 16px; color: #475569; font-size: 14px; }
.waveform-container { background: #f1f5f9; border-radius: 8px; padding: 12px; margin-bottom: 20px; }
.actions { text-align: center; }

.result-card { margin-top: 32px; border-radius: 12px; }
.card-header { display: flex; justify-content: space-between; align-items: center; }
.card-header .title { font-weight: 600; color: #1e293b; }
.transcription-content { white-space: pre-wrap; line-height: 1.8; color: #334155; font-size: 15px; min-height: 100px; }

.text-stats { margin-top: 12px; font-size: 12px; color: #94a3b8; text-align: right; }
.tts-actions { margin-top: 24px; }

.form-tip { font-size: 12px; color: #94a3b8; margin-top: 4px; }
.history-card, .about-card { border-radius: 12px; }
.about-content h3 { color: #1e293b; margin-top: 0; }
.about-content h4 { color: #334155; margin-bottom: 12px; }
.about-content ul { padding-left: 20px; line-height: 2; color: #475569; }
.about-content li { margin-bottom: 8px; }
</style>
