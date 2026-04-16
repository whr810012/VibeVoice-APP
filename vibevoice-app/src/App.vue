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
          <el-icon><Microphone /></el-icon>
          <span>语音转写 (ASR)</span>
        </el-menu-item>
        <el-menu-item index="tts">
          <el-icon><ChatDotRound /></el-icon>
          <span>语音合成 (TTS)</span>
        </el-menu-item>
        <el-divider />
        <el-menu-item index="history">
          <el-icon><Files /></el-icon>
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
      <div class="workspace-status polished-card">
        <div class="workspace-left">
          <el-tag size="small" :type="backendOnline ? 'success' : 'danger'" effect="dark">
            {{ backendOnline ? '后端在线' : '后端离线' }}
          </el-tag>
          <span class="workspace-meta">当前模块：{{ activeTabLabel }}</span>
        </div>
        <div class="workspace-right">
          <span class="workspace-meta">端口 {{ settings.port }}</span>
          <span class="workspace-divider">|</span>
          <span class="workspace-meta">设备 {{ status.device.toUpperCase() }}</span>
          <span class="workspace-divider">|</span>
          <span class="workspace-meta">主题 {{ themeLabel }}</span>
          <span class="workspace-divider">|</span>
          <span class="workspace-meta">更新时间 {{ lastStatusAt || '--:--:--' }}</span>
        </div>
      </div>

      <!-- ASR Page -->
      <transition name="fade" mode="out-in">
        <div v-if="activeTab === 'asr'" class="page-container" key="asr">
          <div class="page-header">
            <h2>智能语音转写</h2>
            <p>面向长音频场景，自动识别说话人与时间戳，提供稳定可追踪的转写流程。</p>
            <div class="header-meta">
              <el-tag size="small" effect="plain" type="info">稳定模式</el-tag>
              <el-tag size="small" effect="plain" :type="status.device === 'cuda' ? 'success' : 'warning'">
                {{ status.device === 'cuda' ? 'GPU 加速' : 'CPU 兼容' }}
              </el-tag>
            </div>
          </div>

          <el-card class="upload-card polished-card" shadow="never">
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
                <el-button link type="danger" icon="Delete" @click="clearAudio">移除</el-button>
              </div>
              <div id="waveform" class="waveform-container"></div>
              <div class="actions">
                <el-button 
                  class="primary-action"
                  type="primary" 
                  size="large" 
                  :loading="processing" 
                  :disabled="!canStartAsr"
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

          <el-card v-if="transcription || processing" class="result-card polished-card" shadow="hover">
            <template #header>
              <div class="card-header">
                <span class="title">转写结果</span>
                <div class="tools">
                  <el-tag size="small" effect="plain" type="info">字数 {{ transcription.length }}</el-tag>
                  <el-button link icon="CopyDocument" :disabled="!hasTranscription" @click="copyTranscription">复制</el-button>
                  <el-button link icon="Download" :disabled="!hasTranscription" @click="downloadTranscription">导出</el-button>
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
            <p>基于 Diffusion 架构，支持长文本语音生成与音色配置。</p>
            <div class="header-meta">
              <el-tag size="small" effect="plain" type="info">高质量模式</el-tag>
              <el-tag size="small" effect="plain">步数 {{ ttsInferenceSteps }}</el-tag>
            </div>
          </div>

          <el-row :gutter="20">
            <el-col :span="16">
              <el-card class="editor-card polished-card" shadow="never">
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
                <div class="text-hint">建议单次控制在 1200 字以内以获得更稳定的生成速度。</div>
              </el-card>
            </el-col>
            <el-col :span="8">
              <el-card class="config-card polished-card" shadow="never">
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
                    <div class="form-tip">当前步数：{{ ttsInferenceSteps }}（更高质量但更慢）</div>
                  </el-form-item>
                  <div class="tts-actions">
                    <el-button 
                      class="primary-action"
                      type="primary" 
                      size="large"
                      @click="generateTTS" 
                      :loading="ttsLoading"
                      :disabled="!canGenerateTts"
                      icon="Microphone"
                    >
                      生成语音
                    </el-button>
                  <div v-if="ttsAudioUrl" class="audio-preview">
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
            <p>统一管理转写与合成任务，支持播放、导出与删除。</p>
            <div class="header-meta">
              <el-tag size="small" effect="plain">ASR {{ asrHistory.length }} 条</el-tag>
              <el-tag size="small" effect="plain">TTS {{ ttsHistory.length }} 条</el-tag>
            </div>
          </div>

          <el-row :gutter="20">
            <el-col :span="12">
              <el-card shadow="never" class="history-card polished-card">
                <template #header>ASR 历史</template>
                <el-table :data="asrHistory" stripe size="small" style="width:100%" empty-text="暂无 ASR 记录">
                  <template #empty>
                    <el-empty description="暂无 ASR 记录" :image-size="84">
                      <el-button text type="primary" @click="goTab('asr')">去创建转写任务</el-button>
                    </el-empty>
                  </template>
                  <el-table-column prop="id" label="任务ID" width="220" />
                  <el-table-column label="时长" width="100">
                    <template #default="scope">{{ formatDuration(scope.row.total_seconds) }}</template>
                  </el-table-column>
                  <el-table-column label="操作" width="240">
                    <template #default="scope">
                      <el-button size="small" icon="Download" @click="downloadAsr(scope.row)">导出文本</el-button>
                      <el-button size="small" icon="Delete" type="danger" @click="deleteAsr(scope.row)">删除</el-button>
                    </template>
                  </el-table-column>
                </el-table>
              </el-card>
            </el-col>
            <el-col :span="12">
              <el-card shadow="never" class="history-card polished-card">
                <template #header>TTS 历史</template>
                <el-table :data="ttsHistory" stripe size="small" style="width:100%" empty-text="暂无 TTS 记录">
                  <template #empty>
                    <el-empty description="暂无 TTS 记录" :image-size="84">
                      <el-button text type="primary" @click="goTab('tts')">去创建合成任务</el-button>
                    </el-empty>
                  </template>
                  <el-table-column prop="id" label="文件名" width="240" />
                  <el-table-column prop="voice" label="音色" width="140" />
                  <el-table-column label="操作" width="220">
                    <template #default="scope">
                      <el-button size="small" icon="VideoPlay" @click="playTts(scope.row)">播放</el-button>
                      <el-button size="small" icon="Download" @click="downloadTts(scope.row)">导出音频</el-button>
                      <el-button size="small" icon="Delete" type="danger" @click="deleteTts(scope.row)">删除</el-button>
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
            <p>配置运行设备、导出目录与服务端口，确保系统稳定运行。</p>
            <div class="header-meta">
              <el-tag size="small" effect="plain" type="warning">高影响操作</el-tag>
              <el-tag size="small" effect="plain">端口 {{ settings.port }}</el-tag>
            </div>
          </div>

          <el-card class="settings-card polished-card" shadow="never">
            <el-form :model="settings" label-width="140px">
              <el-form-item label="界面主题">
                <el-radio-group v-model="settings.theme">
                  <el-radio-button label="system">跟随系统</el-radio-button>
                  <el-radio-button label="light">浅色</el-radio-button>
                  <el-radio-button label="dark">深色</el-radio-button>
                </el-radio-group>
                <div class="form-tip">主题切换即时生效，无需重启服务。</div>
                <div class="theme-actions">
                  <el-button text icon="RefreshLeft" @click="resetAppearance">
                    恢复默认外观（跟随系统）
                  </el-button>
                </div>
              </el-form-item>
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
                <div class="export-row">
                  <el-input v-model="settings.exportDir" placeholder="未选择" readonly />
                  <el-button @click="selectExportDir">选择目录</el-button>
                  <el-button type="primary" :loading="exportingHistory" @click="exportHistory" :disabled="!settings.exportDir">一键导出历史</el-button>
                  <el-button @click="openExportDir" :disabled="!settings.exportDir">打开目录</el-button>
                </div>
                <div class="form-tip">导出 ASR 文本与 TTS 音频到所选目录（自动创建 ASR/TTS 子文件夹）</div>
              </el-form-item>
              <el-form-item label="后端 API 端口">
                <el-input-number v-model="settings.port" :min="1024" :max="65535" />
              </el-form-item>
              <el-divider />
              <el-form-item class="settings-actions">
                <el-button type="primary" icon="RefreshRight" :loading="restartingBackend" @click="saveSettings">保存并重启服务</el-button>
                <el-button icon="Refresh" :disabled="restartingBackend" @click="resetSettings">恢复默认</el-button>
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

          <el-card shadow="never" class="about-card polished-card">
            <div class="about-content">
              <div class="about-hero">
                <h3>VibeVoice Desktop v1.0.0</h3>
                <p>为长音频处理设计的桌面工作台，聚焦“稳定、清晰、可控”的语音生产体验。</p>
                <div class="about-tags">
                  <el-tag effect="plain" size="small">Electron + Vue 3</el-tag>
                  <el-tag effect="plain" size="small">Embedded Python</el-tag>
                  <el-tag effect="plain" size="small">ASR / TTS Workflow</el-tag>
                </div>
              </div>

              <div class="about-grid">
                <div class="about-block">
                  <h4>系统特性</h4>
                  <ul>
                    <li>跨平台桌面应用，界面与任务流程一致</li>
                    <li>集成嵌入式 Python 运行时，降低部署门槛</li>
                    <li>支持 GPU (CUDA) 与 CPU 模式快速切换</li>
                    <li>ASR 转写进度可视化，支持任务取消</li>
                    <li>历史记录可持久化，支持批量导出</li>
                  </ul>
                </div>
                <div class="about-block">
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
                </div>
              </div>

              <el-divider />

              <el-descriptions :column="1" border>
                <el-descriptions-item label="产品定位">桌面端 AI 语音生产力工具</el-descriptions-item>
                <el-descriptions-item label="核心能力">语音转写 / 语音合成 / 历史管理 / 批量导出</el-descriptions-item>
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
const status = reactive({
  asr_loaded: false,
  tts_loaded: false,
  device: 'cpu',
  cuda_available: false,
  cuda_device_name: '',
  cuda_total_mem_bytes: 0
})
const settings = reactive({ 
  theme: 'system',
  device: 'cuda', 
  port: 8000,
  exportDir: ''
})
const asrHistory = ref([])
const ttsHistory = ref([])
const SETTINGS_KEY = 'vv-settings'
const gpuMismatch = computed(() => settings.device === 'cuda' && !status.cuda_available)
const hasTranscription = computed(() => !!transcription.value.trim())
const canStartAsr = computed(() => !!audioFile.value && !processing.value)
const canGenerateTts = computed(() => !!ttsText.value.trim() && !!selectedVoice.value && !ttsLoading.value)
const exportingHistory = ref(false)
const restartingBackend = ref(false)
const backendOnline = ref(false)
const lastStatusAt = ref('')
const themeLabel = computed(() => {
  const map = {
    system: '跟随系统',
    light: '浅色',
    dark: '深色'
  }
  return map[settings.theme] || '跟随系统'
})
const activeTabLabel = computed(() => {
  const map = {
    asr: '语音转写',
    tts: '语音合成',
    history: '历史记录',
    settings: '系统设置',
    about: '关于系统'
  }
  return map[activeTab.value] || '系统'
})

let wavesurfer = null
const notify = (type: 'success' | 'error' | 'warning', message: string) => {
  ElMessage({ type, message, showClose: true, duration: 2200 })
}
const applyTheme = (mode: string) => {
  const root = document.documentElement
  if (mode === 'dark' || mode === 'light') {
    root.setAttribute('data-theme', mode)
    return
  }
  root.removeAttribute('data-theme')
}
const resetAppearance = () => {
  settings.theme = 'system'
  notify('success', '已恢复默认外观')
}
const nowTimeText = () => {
  const d = new Date()
  return d.toTimeString().slice(0, 8)
}

// --- Methods ---
const handleSelect = (index) => {
  activeTab.value = index
}
const goTab = (tab) => {
  activeTab.value = tab
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
  if (!audioFile.value) return notify('warning', '请先上传音频文件')
  
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
          notify('success', '转写完成')
        } else if (st.data.status === 'error' || st.data.status === 'canceled') {
          clearInterval(asrTimer)
          processing.value = false
          notify('error', '任务中断: ' + (st.data.error || st.data.status))
        }
      } catch (e) {
        clearInterval(asrTimer)
        processing.value = false
        notify('error', '查询状态失败')
      }
    }, 1000)
  } catch (err) {
    processing.value = false
    notify('error', '任务提交失败: ' + (err.response?.data?.detail || err.message))
  }
}

const cancelTranscription = async () => {
  if (!asrJobId.value) return
  try {
    await axios.post(`http://127.0.0.1:${settings.port}/asr/cancel/${asrJobId.value}`)
    if (asrTimer) clearInterval(asrTimer)
    processing.value = false
    notify('success', '已请求取消')
  } catch {
    notify('error', '取消失败')
  }
}

const copyTranscription = () => {
  if (!hasTranscription.value) return
  navigator.clipboard.writeText(transcription.value)
  notify('success', '已复制到剪贴板')
}

const downloadTranscription = () => {
  if (!hasTranscription.value) return
  const blob = new Blob([transcription.value], { type: 'text/plain' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `VibeVoice_ASR_${new Date().getTime()}.txt`
  a.click()
}

const generateTTS = async () => {
  if (!ttsText.value) return notify('warning', '请输入需要合成的文本')
  if (!selectedVoice.value) return notify('warning', '请选择角色音色')
  
  ttsLoading.value = true
  try {
    const res = await axios.post(`http://127.0.0.1:${settings.port}/tts`, { 
      text: ttsText.value, 
      voice: selectedVoice.value,
      inference_steps: ttsInferenceSteps.value
    })
    notify('success', '语音合成成功')
    if (res.data?.url) {
      ttsAudioUrl.value = `http://127.0.0.1:${settings.port}${res.data.url}`
    }
  } catch (err) {
    notify('error', '合成失败: ' + (err.response?.data?.detail || err.message))
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
    status.cuda_available = !!res.data.cuda_available
    status.cuda_device_name = res.data.cuda_device_name || ''
    status.cuda_total_mem_bytes = Number(res.data.cuda_total_mem_bytes || 0)
    backendOnline.value = true
    lastStatusAt.value = nowTimeText()
  } catch (e) {
    // Backend might be offline
    backendOnline.value = false
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
  ElMessageBox.confirm('确认删除该 ASR 记录及对应文本文件？此操作不可撤销。', '删除确认', {
    confirmButtonText: '确认删除',
    cancelButtonText: '取消',
    type: 'warning'
  }).then(async () => {
    try {
      await axios.delete(`http://127.0.0.1:${settings.port}/history/asr/${row.id}`)
      notify('success', 'ASR 记录已删除')
      fetchHistory()
    } catch {
      notify('error', '删除 ASR 记录失败')
    }
  })
}

const deleteTts = async (row) => {
  ElMessageBox.confirm('确认删除该 TTS 音频文件？此操作不可撤销。', '删除确认', {
    confirmButtonText: '确认删除',
    cancelButtonText: '取消',
    type: 'warning'
  }).then(async () => {
    try {
      await axios.delete(`http://127.0.0.1:${settings.port}/history/tts/${row.id}`)
      notify('success', 'TTS 记录已删除')
      fetchHistory()
    } catch {
      notify('error', '删除 TTS 记录失败')
    }
  })
}

const saveSettings = () => {
  ElMessageBox.confirm('应用新设置需要重启后端服务，是否继续？', '应用设置', {
    confirmButtonText: '保存并重启',
    cancelButtonText: '取消',
    type: 'warning'
  }).then(() => {
    restartingBackend.value = true
    notify('success', '设置已保存，正在重启服务...')
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings))
    } catch {}
    ipcRenderer.invoke('vv:restart-backend', { port: settings.port, device: settings.device })
      .then(() => {
        setTimeout(() => {
          fetchStatus()
          fetchVoices()
          fetchHistory()
          restartingBackend.value = false
        }, 1200)
      })
      .catch(() => {
        restartingBackend.value = false
        notify('error', '重启服务失败')
      })
  })
}

const resetSettings = () => {
  settings.theme = 'system'
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
      if (typeof data.theme === 'string') settings.theme = data.theme
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
    notify('success', '已选择导出目录')
  }
}

const exportHistory = async () => {
  if (!settings.exportDir) return notify('warning', '请先选择导出目录')
  exportingHistory.value = true
  try {
    const res = await ipcRenderer.invoke('vv:export-history', { dir: settings.exportDir })
    if (res?.ok) {
      notify('success', `导出完成：ASR ${res.asr} 条，TTS ${res.tts} 条`)
    } else {
      notify('error', '导出失败')
    }
  } finally {
    exportingHistory.value = false
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

watch(() => settings.theme, (theme) => {
  applyTheme(theme)
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings))
  } catch {}
})

onMounted(() => {
  loadSettings()
  applyTheme(settings.theme)
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
:root {
  --vv-bg: #f8fafc;
  --vv-bg-soft: #f1f5f9;
  --vv-surface: #ffffff;
  --vv-surface-elevated: #ffffffcc;
  --vv-border: #e2e8f0;
  --vv-text-main: #0f172a;
  --vv-text-secondary: #64748b;
  --vv-text-muted: #94a3b8;
  --vv-brand: #409eff;
  --vv-radius: 12px;
  --vv-shadow-soft: 0 4px 16px rgba(15, 23, 42, 0.04);
  --vv-space-1: 8px;
  --vv-space-2: 12px;
  --vv-space-3: 16px;
  --vv-space-4: 24px;
  --vv-space-5: 32px;

  /* Element Plus theme tokens */
  --el-color-primary: var(--vv-brand);
  --el-color-success: #16a34a;
  --el-color-warning: #d97706;
  --el-color-danger: #dc2626;
  --el-text-color-primary: var(--vv-text-main);
  --el-text-color-regular: var(--vv-text-secondary);
  --el-border-color: var(--vv-border);
  --el-bg-color: var(--vv-surface);
  --el-fill-color-light: var(--vv-bg-soft);
  --el-fill-color-blank: var(--vv-surface);
}

/* Global Resets */
body {
  margin: 0;
  font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
  -webkit-font-smoothing: antialiased;
  color: var(--vv-text-main);
}

/* Transitions */
.fade-enter-active, .fade-leave-active { transition: opacity 0.2s ease; }
.fade-enter-from, .fade-leave-to { opacity: 0; }

.app-container {
  height: 100vh;
  background:
    radial-gradient(circle at top right, rgba(59, 130, 246, 0.08), transparent 42%),
    var(--vv-bg);
}

.aside {
  background-color: var(--vv-surface-elevated);
  border-right: 1px solid var(--vv-border);
  display: flex;
  flex-direction: column;
  padding: 0;
  backdrop-filter: blur(8px);
}

.logo {
  height: 70px;
  display: flex;
  align-items: center;
  padding: 0 24px;
  font-size: 18px;
  font-weight: 700;
  color: #0b1220;
  border-bottom: 1px solid var(--vv-bg-soft);
}
.logo small { font-weight: 500; color: var(--vv-text-secondary); margin-left: 6px; font-size: 12px; }

.el-menu-vertical { border-right: none !important; flex: 1; padding: 12px 0; }
.el-menu-item {
  height: 50px;
  line-height: 50px;
  margin: 4px 12px;
  border-radius: 8px;
  transition: all 0.2s ease;
}
.el-menu-item:hover { background-color: #f8fafc; }
.el-menu-item.is-active { background-color: #f0f7ff !important; color: #409eff !important; font-weight: 600; }

.status-panel {
  padding: 24px;
  border-top: 1px solid var(--vv-bg-soft);
  font-size: 13px;
  color: var(--vv-text-secondary);
}
.status-item { display: flex; align-items: center; margin-bottom: 10px; }
.dot { width: 8px; height: 8px; border-radius: 50%; background-color: #cbd5e1; margin-right: 10px; }
.dot.active { background-color: #10b981; box-shadow: 0 0 8px rgba(16, 185, 129, 0.4); }
.device-info { margin-top: 12px; }

.main-content {
  padding: var(--vv-space-5) 36px;
  width: min(1180px, 100%);
  margin: 0 auto;
}
.workspace-status {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: var(--vv-space-2);
  border: 1px solid var(--vv-border);
  border-radius: var(--vv-radius);
  background: var(--vv-surface-elevated);
  padding: 10px 14px;
  margin-bottom: var(--vv-space-4);
}
.workspace-left, .workspace-right {
  display: flex;
  align-items: center;
  gap: var(--vv-space-1);
  flex-wrap: wrap;
}
.workspace-meta {
  color: var(--vv-text-secondary);
  font-size: 12px;
}
.workspace-divider {
  color: #cbd5e1;
}

.page-header { margin-bottom: var(--vv-space-5); }
.page-header h2 { margin: 0 0 8px 0; color: var(--vv-text-main); font-size: 26px; letter-spacing: 0.2px; font-weight: 700; }
.page-header p { margin: 0; color: var(--vv-text-secondary); font-size: 14px; }
.header-meta {
  margin-top: var(--vv-space-2);
  display: flex;
  gap: var(--vv-space-1);
  flex-wrap: wrap;
}

.upload-card, .editor-card, .config-card, .settings-card {
  border-radius: var(--vv-radius);
  border: 1px solid var(--vv-border);
}
.polished-card {
  box-shadow: var(--vv-shadow-soft);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.polished-card:hover {
  transform: translateY(-1px);
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
}

.asr-uploader .el-upload-dragger {
  border-radius: 12px;
  padding: 40px;
  border: 1px dashed #cbd5e1;
  background: linear-gradient(180deg, var(--vv-surface), #f8fbff);
  transition: all 0.2s ease;
}
.asr-uploader .el-upload-dragger:hover {
  border-color: #60a5fa;
  transform: translateY(-1px);
}

.audio-info { margin-top: var(--vv-space-4); }
.file-meta { display: flex; align-items: center; gap: 8px; margin-bottom: 16px; color: var(--vv-text-secondary); font-size: 14px; }
.waveform-container { background: var(--vv-bg-soft); border-radius: 8px; padding: 12px; margin-bottom: 20px; }
.actions {
  display: flex;
  justify-content: center;
  gap: 12px;
  flex-wrap: wrap;
}
.primary-action { min-width: 200px; }

.result-card { margin-top: var(--vv-space-5); border-radius: 12px; }
.card-header { display: flex; justify-content: space-between; align-items: center; }
.card-header .title { font-weight: 600; color: var(--vv-text-main); }
.card-header .tools {
  display: flex;
  align-items: center;
  gap: 6px;
}
.transcription-content {
  white-space: pre-wrap;
  line-height: 1.85;
  color: var(--vv-text-main);
  font-size: 15px;
  min-height: 100px;
  background: var(--vv-surface);
  border: 1px solid var(--vv-border);
  border-radius: 10px;
  padding: 14px;
}
.loading-state {
  border-radius: 10px;
  border: 1px dashed var(--vv-border);
  background: var(--vv-surface);
  padding: 14px;
}

.text-stats { margin-top: var(--vv-space-2); font-size: 12px; color: var(--vv-text-muted); text-align: right; }
.text-hint { margin-top: 6px; font-size: 12px; color: #64748b; text-align: right; }
.tts-actions { margin-top: var(--vv-space-4); }
.audio-preview { margin-top: 12px; }
.export-row {
  display: flex;
  gap: 8px;
  align-items: center;
  width: 100%;
}
.settings-actions .el-form-item__content { gap: 10px; }

.form-tip {
  font-size: 12px;
  color: var(--vv-text-muted);
  margin-top: 4px;
  line-height: 1.5;
}
.theme-actions {
  margin-top: 6px;
}
.history-card, .about-card { border-radius: 12px; }
.history-card .el-table th.el-table__cell {
  background: var(--vv-bg);
  color: var(--vv-text-main);
}
.history-card .el-table tr:hover > td.el-table__cell {
  background: #f8fbff;
}
.history-card .el-empty {
  padding: 24px 0;
}
.about-hero {
  background: linear-gradient(180deg, #fbfdff, #f8fbff);
  border: 1px solid var(--vv-border);
  border-radius: 12px;
  padding: 16px;
  margin-bottom: 16px;
}
.about-tags {
  margin-top: 10px;
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}
.about-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
}
.about-block {
  border: 1px solid var(--vv-border);
  border-radius: 12px;
  padding: 14px;
  background: var(--vv-surface);
}
.about-content h3 { color: var(--vv-text-main); margin-top: 0; }
.about-content h4 { color: var(--vv-text-main); margin-bottom: 12px; }
.about-content ul { padding-left: 20px; line-height: 2; color: var(--vv-text-secondary); }
.about-content li { margin-bottom: 8px; }

.el-button {
  transition: all 0.18s ease;
}
.el-tag {
  border-radius: 999px;
  font-weight: 600;
}
.el-button:focus-visible,
.el-input__inner:focus-visible,
.el-textarea__inner:focus-visible,
.el-upload-dragger:focus-visible {
  outline: 2px solid color-mix(in srgb, var(--vv-brand) 45%, white);
  outline-offset: 2px;
}
.el-button:not(.is-disabled):hover {
  transform: translateY(-1px);
}
.el-button:not(.is-disabled):active {
  transform: translateY(0);
}
.el-button--primary:not(.is-disabled):hover {
  filter: saturate(1.05) brightness(1.02);
}

@media (prefers-color-scheme: dark) {
  :root:not([data-theme='light']) {
    --vv-bg: #0b1220;
    --vv-bg-soft: #111a2e;
    --vv-surface: #111a2b;
    --vv-surface-elevated: #111a2bd9;
    --vv-border: #22314a;
    --vv-text-main: #e6edf8;
    --vv-text-secondary: #9db0ce;
    --vv-text-muted: #7f93b2;
    --vv-shadow-soft: 0 8px 24px rgba(0, 0, 0, 0.35);

    --el-color-primary: #5ea8ff;
    --el-color-success: #34d399;
    --el-color-warning: #fbbf24;
    --el-color-danger: #f87171;
    --el-text-color-primary: var(--vv-text-main);
    --el-text-color-regular: var(--vv-text-secondary);
    --el-border-color: var(--vv-border);
    --el-bg-color: var(--vv-surface);
    --el-fill-color-light: #16233a;
    --el-fill-color-blank: var(--vv-surface);
  }

  :root:not([data-theme='light']) .app-container {
    background:
      radial-gradient(circle at top right, rgba(64, 158, 255, 0.18), transparent 42%),
      var(--vv-bg);
  }

  :root:not([data-theme='light']) .el-menu-item:hover {
    background-color: #17243a;
  }

  :root:not([data-theme='light']) .el-menu-item.is-active {
    background-color: #173050 !important;
  }

  :root:not([data-theme='light']) .asr-uploader .el-upload-dragger,
  :root:not([data-theme='light']) .about-hero {
    background: #121f36;
  }

  :root:not([data-theme='light']) .history-card .el-table tr:hover > td.el-table__cell {
    background: #162742;
  }

  :root:not([data-theme='light']) .transcription-content,
  :root:not([data-theme='light']) .loading-state,
  :root:not([data-theme='light']) .about-block {
    background: #111a2b;
  }

  :root:not([data-theme='light']) .workspace-divider {
    color: #3a4f72;
  }
}

:root[data-theme='dark'] {
  --vv-bg: #0b1220;
  --vv-bg-soft: #111a2e;
  --vv-surface: #111a2b;
  --vv-surface-elevated: #111a2bd9;
  --vv-border: #22314a;
  --vv-text-main: #e6edf8;
  --vv-text-secondary: #9db0ce;
  --vv-text-muted: #7f93b2;
  --vv-shadow-soft: 0 8px 24px rgba(0, 0, 0, 0.35);

  --el-color-primary: #5ea8ff;
  --el-color-success: #34d399;
  --el-color-warning: #fbbf24;
  --el-color-danger: #f87171;
  --el-text-color-primary: var(--vv-text-main);
  --el-text-color-regular: var(--vv-text-secondary);
  --el-border-color: var(--vv-border);
  --el-bg-color: var(--vv-surface);
  --el-fill-color-light: #16233a;
  --el-fill-color-blank: var(--vv-surface);
}

:root[data-theme='dark'] .app-container {
  background:
    radial-gradient(circle at top right, rgba(64, 158, 255, 0.18), transparent 42%),
    var(--vv-bg);
}

:root[data-theme='dark'] .el-menu-item:hover {
  background-color: #17243a;
}

:root[data-theme='dark'] .el-menu-item.is-active {
  background-color: #173050 !important;
}

:root[data-theme='dark'] .asr-uploader .el-upload-dragger,
:root[data-theme='dark'] .about-hero {
  background: #121f36;
}

:root[data-theme='dark'] .history-card .el-table tr:hover > td.el-table__cell {
  background: #162742;
}

:root[data-theme='dark'] .transcription-content,
:root[data-theme='dark'] .loading-state,
:root[data-theme='dark'] .about-block {
  background: #111a2b;
}

:root[data-theme='dark'] .workspace-divider {
  color: #3a4f72;
}

@media (prefers-contrast: more) {
  :root {
    --vv-border: #7f8ea8;
    --vv-text-secondary: #344256;
    --vv-text-muted: #4b5d78;
    --vv-shadow-soft: 0 0 0 2px rgba(15, 23, 42, 0.22);
  }

  .polished-card,
  .workspace-status,
  .transcription-content,
  .loading-state,
  .about-block {
    box-shadow: none;
    border-width: 2px;
  }

  .el-button--primary {
    border: 2px solid color-mix(in srgb, var(--el-color-primary) 70%, black);
  }
}

@media (forced-colors: active) {
  .polished-card,
  .workspace-status,
  .transcription-content,
  .loading-state,
  .about-block {
    forced-color-adjust: auto;
    border: 1px solid CanvasText;
    box-shadow: none;
  }

  .workspace-divider {
    color: CanvasText;
  }
}

@media (max-width: 1024px) {
  .main-content {
    padding: 20px;
    width: 100%;
  }
}

@media (max-width: 768px) {
  .app-container {
    display: block;
    height: auto;
    min-height: 100vh;
  }
  .aside {
    width: 100% !important;
    border-right: none;
    border-bottom: 1px solid var(--vv-border);
  }
  .status-panel {
    display: none;
  }
  .page-header h2 {
    font-size: 22px;
  }
  .workspace-status {
    align-items: flex-start;
    flex-direction: column;
  }
  .header-meta {
    margin-top: 10px;
  }
  .card-header {
    align-items: flex-start;
    gap: 8px;
    flex-direction: column;
  }
  .about-grid {
    grid-template-columns: 1fr;
  }
  .history-card .el-table .cell {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .export-row {
    flex-direction: column;
    align-items: stretch;
  }
  .primary-action {
    width: 100%;
  }
}
</style>
