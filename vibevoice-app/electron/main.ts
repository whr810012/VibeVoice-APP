import { app, BrowserWindow, ipcMain, dialog, shell } from 'electron'
import { spawn, ChildProcess } from 'child_process'
import path from 'path'
import { platform } from 'os'
import fs from 'fs'
import fsp from 'fs/promises'

let mainWindow: BrowserWindow | null = null
let pyProc: ChildProcess | null = null

function startBackend(port?: number, device?: 'cpu' | 'cuda') {
  const isDev = !app.isPackaged
  let pyPath = 'python'
  let scriptPath = path.join(app.getAppPath(), 'backend', 'server.py')
  
  if (!isDev) {
    // In production, use embedded Python runtime + server.py from resources
    pyPath = path.join(process.resourcesPath, 'python-win', 'python.exe')
    scriptPath = path.join(process.resourcesPath, 'backend', 'server.py')
  }
  
  console.log('Starting backend process...')
  const args = scriptPath ? [scriptPath] : []
  
  const env = {
    ...process.env,
    VV_PORT: port ? String(port) : (process.env.VV_PORT || '8000'),
    VV_DEVICE: device ? device : (process.env.VV_DEVICE || '')
  }
  // In production, assist embedded Python by setting PYTHONHOME/PYTHONPATH
  if (!isDev) {
    const pyHome = path.join(process.resourcesPath, 'python-win')
    const vibevoiceLib = path.join(process.resourcesPath, 'vibevoice')
    env.PYTHONHOME = pyHome
    env.PYTHONPATH = [path.join(pyHome, 'Lib'), path.join(pyHome, 'Lib', 'site-packages'), vibevoiceLib, process.resourcesPath].join(path.delimiter)
    env.PATH = `${pyHome};${path.join(pyHome, 'Scripts')};${env.PATH || ''}`
  }

  pyProc = spawn(pyPath, args, {
    stdio: 'pipe',
    shell: true,
    env
  })
  
  pyProc.stdout?.on('data', (data) => {
    console.log(`Backend: ${data}`)
  })
  
  pyProc.stderr?.on('data', (data) => {
    console.error(`Backend Error: ${data}`)
  })
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
    title: "VibeVoice Desktop"
  })

  if (app.isPackaged) {
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'))
  } else {
    mainWindow.loadURL(process.env.VITE_DEV_SERVER_URL as string)
  }
}

app.whenReady().then(() => {
  startBackend()
  createWindow()
})

ipcMain.handle('vv:restart-backend', async (_e, payload: { port?: number, device?: 'cpu'|'cuda' }) => {
  if (pyProc) {
    try { pyProc.kill() } catch {}
    pyProc = null
  }
  startBackend(payload?.port, payload?.device)
  return 'ok'
})

ipcMain.handle('vv:select-export-dir', async () => {
  const res = await dialog.showOpenDialog({ properties: ['openDirectory', 'createDirectory'] })
  if (res.canceled || res.filePaths.length === 0) return null
  return res.filePaths[0]
})

function resolveTmpDir() {
  const cands = [
    path.join(app.getAppPath(), 'backend', 'tmp'),
    path.join(process.resourcesPath || '', 'backend', 'tmp')
  ]
  for (const p of cands) {
    try {
      if (fs.existsSync(p)) return p
    } catch {}
  }
  return path.join(app.getAppPath(), 'backend', 'tmp')
}

ipcMain.handle('vv:export-history', async (_e, payload: { dir: string }) => {
  const targetRoot = payload?.dir
  if (!targetRoot) return { ok: false, error: 'no_dir' }
  const tmpDir = resolveTmpDir()
  const asrSrc = await fsp.readdir(tmpDir).catch(() => [])
  let asrCount = 0
  let ttsCount = 0
  const asrOut = path.join(targetRoot, 'ASR')
  const ttsOut = path.join(targetRoot, 'TTS')
  await fsp.mkdir(asrOut, { recursive: true }).catch(() => {})
  await fsp.mkdir(ttsOut, { recursive: true }).catch(() => {})
  for (const name of asrSrc) {
    try {
      if (name.startsWith('asr_') && name.endsWith('.txt')) {
        await fsp.copyFile(path.join(tmpDir, name), path.join(asrOut, name))
        asrCount++
      } else if (name.startsWith('tts_') && name.endsWith('.wav')) {
        await fsp.copyFile(path.join(tmpDir, name), path.join(ttsOut, name))
        ttsCount++
      }
    } catch {}
  }
  return { ok: true, asr: asrCount, tts: ttsCount }
})

ipcMain.handle('vv:open-path', async (_e, payload: { dir: string }) => {
  if (!payload?.dir) return { ok: false }
  try {
    const res = await shell.openPath(payload.dir)
    return { ok: res === '' }
  } catch {
    return { ok: false }
  }
})

app.on('window-all-closed', () => {
  if (pyProc) {
    pyProc.kill()
  }
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

app.on('will-quit', () => {
  if (pyProc) {
    pyProc.kill()
  }
})
