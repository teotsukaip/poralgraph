import * as THREE from "three";
import { FBXLoader } from "three/addons/loaders/FBXLoader.js";
import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.17";

const IPD = 0.063;
const MODEL_URL = "./copzombie_l_actisdato.fbx";
const MODEL_YAW = 0;
const HEAD_AT = 0.93;
const MODEL_HEIGHT = 1.75;
const NEAR = 0.05;
const FAR = 40;
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.17/wasm";
const TASK_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const $ = (id) => document.getElementById(id);

const ui = {
  status: $("status-text"),
  dot: $("dot"),
  depth: $("depth"),
  inch: $("inch"),
  inchLabel: $("inch-label"),
  gain: $("gain"),
  gainLabel: $("gain-label"),
  calibrate: $("calibrate"),
  anaglyph: $("anaglyph"),
  mouseMode: $("mouse-mode"),
  fullscreen: $("fullscreen"),
  start: $("start"),
  startMouse: $("start-mouse"),
  intro: $("intro"),
  hint: $("hint"),
  panel: $("panel"),
  preview: $("preview"),
  previewCanvas: $("preview-canvas"),
  calOverlay: $("cal-overlay"),
  calCount: $("cal-count"),
  loading: $("loading"),
  bar: $("bar"),
  loadLabel: $("load-label"),
};

const renderer = new THREE.WebGLRenderer({
  canvas: $("view"),
  antialias: true,
  alpha: false,
  powerPreference: "high-performance",
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x07080b, 1);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x07080b);
scene.fog = new THREE.Fog(0x07080b, 6.5, 14);

const camera = new THREE.PerspectiveCamera(35, 1, NEAR, FAR);
camera.matrixAutoUpdate = true;

const world = new THREE.Group();
scene.add(world);

const eye = new THREE.Vector3(0, 0, 0.55);
const eyeTarget = new THREE.Vector3(0, 0, 0.55);
const headUp = new THREE.Vector2(0, 1);
let screenW = 0.3;
let screenH = 0.19;
let gain = 1;
let anaglyph = false;
let mouseMode = false;
let walking = true;
let started = false;
let focalPx = 480;
let focalLocked = false;
let xBias = 0;
let yMountBias = 0;
let overrideEye = null;
let lastSample = null;

const clock = new THREE.Clock();
const filterX = new OneEuro(1.2, 0.9);
const filterY = new OneEuro(1.2, 0.9);
const filterZ = new OneEuro(0.7, 0.45);
const filterRoll = new OneEuro(1.0, 0.3);

const actor = {
  holder: new THREE.Group(),
  mixer: null,
  height: MODEL_HEIGHT,
  floorY: 0,
  waypoint: 0,
  meeting: false,
  meetLeft: 0,
  paused: false,
};
world.add(actor.holder);

const route = [
  { x: -0.58, z: -2.45 },
  { x: 0.58, z: -2.45 },
  { x: 0.4, z: -1.45 },
  { x: 0.08, z: -0.92, meet: true },
  { x: -0.18, z: -0.92, meet: true },
  { x: -0.5, z: -1.55 },
];

buildRoom();
loadModel();
window.addEventListener("resize", resize);

const keys = new Set();
window.addEventListener("keydown", (event) => {
  keys.add(event.key);
  if (event.key === "f" || event.key === "F") toggleFullscreen();
  if (event.key === "a" || event.key === "A") setAnaglyph(!anaglyph);
  if (event.key === "m" || event.key === "M") setMouseMode(!mouseMode);
  if (event.key === "c" || event.key === "C") beginCalibration();
  if (event.key === " ") {
    event.preventDefault();
    walking = !walking;
  }
  if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(event.key)) {
    event.preventDefault();
    setMouseMode(true);
  }
});
window.addEventListener("keyup", (event) => keys.delete(event.key));

window.addEventListener("pointermove", (event) => {
  if (!mouseMode) return;
  const nx = (event.clientX / window.innerWidth) * 2 - 1;
  const ny = -((event.clientY / window.innerHeight) * 2 - 1);
  eyeTarget.set(nx * 0.26 * gain, ny * 0.18 * gain, eyeTarget.z);
});
window.addEventListener("wheel", (event) => {
  if (!mouseMode) return;
  event.preventDefault();
  eyeTarget.z = THREE.MathUtils.clamp(eyeTarget.z + event.deltaY * 0.0008, 0.28, 1.05);
}, { passive: false });

ui.inch.addEventListener("input", () => {
  ui.inchLabel.textContent = Number(ui.inch.value).toFixed(1);
  resize();
});
ui.gain.addEventListener("input", () => {
  gain = Number(ui.gain.value);
  ui.gainLabel.textContent = gain.toFixed(2);
});
ui.anaglyph.addEventListener("click", () => setAnaglyph(!anaglyph));
ui.mouseMode.addEventListener("click", () => setMouseMode(!mouseMode));
ui.fullscreen.addEventListener("click", toggleFullscreen);
ui.calibrate.addEventListener("click", beginCalibration);
ui.start.addEventListener("click", () => begin(false));
ui.startMouse.addEventListener("click", () => begin(true));

let video = null;
let landmarker = null;
let previewCtx = ui.previewCanvas.getContext("2d");
let detectReady = false;
let trackingLost = 0;
let calibrating = false;

const rtLeft = new THREE.WebGLRenderTarget(8, 8);
const rtRight = new THREE.WebGLRenderTarget(8, 8);
rtLeft.texture.colorSpace = THREE.SRGBColorSpace;
rtRight.texture.colorSpace = THREE.SRGBColorSpace;
const quadScene = new THREE.Scene();
const quadCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
const quadMaterial = new THREE.ShaderMaterial({
  uniforms: {
    tLeft: { value: rtLeft.texture },
    tRight: { value: rtRight.texture },
  },
  vertexShader: `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = vec4(position.xy, 0.0, 1.0);
    }
  `,
  fragmentShader: `
    varying vec2 vUv;
    uniform sampler2D tLeft;
    uniform sampler2D tRight;
    void main() {
      vec4 leftEye = texture2D(tLeft, vUv);
      vec4 rightEye = texture2D(tRight, vUv);
      gl_FragColor = vec4(leftEye.r, rightEye.g, rightEye.b, 1.0);
    }
  `,
  depthTest: false,
  depthWrite: false,
});
quadScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), quadMaterial));

const siren = new THREE.PointLight(0x3366ff, 0, 6, 2);
siren.position.set(1.1, 1.5, -2.6);
world.add(siren);
const sirenWarm = new THREE.PointLight(0xff5530, 0, 6, 2);
sirenWarm.position.set(-1.1, 1.5, -2.6);
world.add(sirenWarm);

window.__portal = {
  setEye(x, y, z) {
    overrideEye = new THREE.Vector3(x, y, z);
  },
  clearEye() {
    overrideEye = null;
  },
  eye: () => eye.clone(),
  screen: () => ({ w: screenW, h: screenH }),
  actor,
};

resize();
requestAnimationFrame(frame);

function buildRoom() {
  const grid = makeGridTexture("#12141a", "#2a3344");
  const floorGrid = makeGridTexture("#0c0d11", "#242017");
  floorGrid.repeat.set(6, 8);

  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(8, 10),
    new THREE.MeshStandardMaterial({ map: floorGrid, roughness: 0.92, metalness: 0.04 })
  );
  floor.rotation.x = -Math.PI / 2;
  floor.position.set(0, 0, -3.2);
  floor.receiveShadow = true;
  world.add(floor);

  const wallMap = grid.clone();
  wallMap.repeat.set(3, 2);
  wallMap.needsUpdate = true;
  const wallMat = new THREE.MeshStandardMaterial({ map: wallMap, roughness: 0.96, metalness: 0.02 });

  const back = new THREE.Mesh(new THREE.PlaneGeometry(6.4, 3.2), wallMat);
  back.position.set(0, 1.55, -4.15);
  back.receiveShadow = true;
  world.add(back);

  const sideGeo = new THREE.PlaneGeometry(6.2, 3.2);
  const left = new THREE.Mesh(sideGeo, wallMat);
  left.position.set(-2.4, 1.55, -1.4);
  left.rotation.y = Math.PI / 2;
  world.add(left);
  const right = new THREE.Mesh(sideGeo, wallMat);
  right.position.set(2.4, 1.55, -1.4);
  right.rotation.y = -Math.PI / 2;
  world.add(right);

  const ceiling = new THREE.Mesh(
    new THREE.PlaneGeometry(6.4, 6.2),
    new THREE.MeshStandardMaterial({ color: 0x101218, roughness: 1 })
  );
  ceiling.rotation.x = Math.PI / 2;
  ceiling.position.set(0, 3.15, -1.4);
  world.add(ceiling);

  const door = new THREE.Mesh(
    new THREE.PlaneGeometry(1.05, 2.15),
    new THREE.MeshStandardMaterial({ color: 0x1a140f, roughness: 0.8, emissive: 0x2a130c, emissiveIntensity: 0.35 })
  );
  door.position.set(0.15, 1.08, -4.12);
  world.add(door);

  const rail = new THREE.Mesh(
    new THREE.BoxGeometry(4.6, 0.045, 0.045),
    new THREE.MeshStandardMaterial({ color: 0x8d6a45, roughness: 0.45, metalness: 0.35 })
  );
  rail.position.set(0, 1.62, -4.05);
  world.add(rail);

  const bulb = new THREE.Mesh(
    new THREE.SphereGeometry(0.045, 20, 12),
    new THREE.MeshBasicMaterial({ color: 0xffd2a8 })
  );
  bulb.position.set(-0.35, 1.72, -2.15);
  world.add(bulb);
  const bulbLight = new THREE.PointLight(0xffc48a, 8, 5.5, 2);
  bulbLight.position.copy(bulb.position);
  world.add(bulbLight);

  for (const pole of [
    [-0.85, -1.7],
    [0.9, -1.7],
    [-0.45, -3.15],
    [0.55, -3.15],
  ]) {
    const mesh = new THREE.Mesh(
      new THREE.CylinderGeometry(0.025, 0.025, 3.05, 8),
      new THREE.MeshStandardMaterial({ color: 0x23262e, roughness: 0.6, metalness: 0.4 })
    );
    mesh.position.set(pole[0], 1.52, pole[1]);
    mesh.castShadow = true;
    world.add(mesh);
  }

  scene.add(new THREE.AmbientLight(0x8d93a3, 0.35));
  scene.add(new THREE.HemisphereLight(0x9eb4c8, 0x1a120d, 0.55));

  const key = new THREE.DirectionalLight(0xfff1dd, 2.4);
  key.position.set(0.7, 3.4, 1.6);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  key.shadow.camera.near = 0.4;
  key.shadow.camera.far = 12;
  key.shadow.camera.left = -4;
  key.shadow.camera.right = 4;
  key.shadow.camera.top = 4;
  key.shadow.camera.bottom = -4;
  key.shadow.bias = -0.0004;
  scene.add(key);
  key.target.position.set(0, 1.2, -1.4);
  scene.add(key.target);

  const rim = new THREE.DirectionalLight(0x7aa2ff, 1.15);
  rim.position.set(-1.6, 2.2, -3.4);
  scene.add(rim);

  actor.shadow = new THREE.Mesh(
    new THREE.CircleGeometry(0.34, 28),
    new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.38, depthWrite: false })
  );
  actor.shadow.rotation.x = -Math.PI / 2;
  actor.shadow.position.y = 0.015;
  world.add(actor.shadow);
}

function makeGridTexture(fill, line) {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = fill;
  ctx.fillRect(0, 0, 512, 512);
  ctx.strokeStyle = line;
  ctx.lineWidth = 2;
  for (let i = 0; i <= 512; i += 64) {
    ctx.beginPath();
    ctx.moveTo(i, 0);
    ctx.lineTo(i, 512);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, i);
    ctx.lineTo(512, i);
    ctx.stroke();
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 8;
  return texture;
}

function loadModel() {
  const loader = new FBXLoader();
  loader.load(MODEL_URL, (model) => {
    prepareModel(model);
    actor.holder.add(model);
    if (model.animations?.length) {
      const clip = model.animations.reduce((best, item) => (item.duration > best.duration ? item : best));
      neutralizeRootMotion(clip);
      actor.mixer = new THREE.AnimationMixer(model);
      const action = actor.mixer.clipAction(clip);
      action.setLoop(THREE.LoopRepeat, Infinity);
      action.play();
      window.__portal.clip = { name: clip.name, duration: clip.duration, tracks: clip.tracks.length };
    }
    actor.holder.position.set(route[0].x, 0, route[0].z);
    frameCharacter(model);
    ui.loading.hidden = true;
    setStatus("顔を画面に向けてください", "warn");
  }, (event) => {
    if (!event.lengthComputable) return;
    const ratio = event.loaded / event.total;
    ui.bar.style.width = `${Math.round(ratio * 100)}%`;
    ui.loadLabel.textContent = `${Math.round(ratio * 100)}%`;
  }, (error) => {
    console.error(error);
    const message = error?.message || String(error);
    ui.loading.hidden = true;
    ui.loadLabel.textContent = message;
    setStatus("モデルの読み込みに失敗しました", "warn");
  });
}

function prepareModel(model) {
  model.traverse((child) => {
    if (!child.isMesh) return;
    child.castShadow = true;
    child.receiveShadow = true;
    child.frustumCulled = false;
    const materials = Array.isArray(child.material) ? child.material : [child.material];
    for (const material of materials) {
      if (!material) continue;
      if (material.opacity !== undefined && material.opacity < 0.05) {
        material.opacity = 1;
        material.transparent = false;
      }
      material.side = THREE.FrontSide;
      if (material.map) material.map.colorSpace = THREE.SRGBColorSpace;
      material.needsUpdate = true;
    }
  });

  const bounds = new THREE.Box3().setFromObject(model);
  const size = bounds.getSize(new THREE.Vector3());
  const scale = MODEL_HEIGHT / Math.max(size.y, 0.0001);
  model.scale.multiplyScalar(scale);
  model.updateMatrixWorld(true);

  const fitted = new THREE.Box3().setFromObject(model);
  model.position.y -= fitted.min.y;
  actor.footOffset = model.position.y;
  actor.height = fitted.max.y - fitted.min.y;
  window.__portal.model = { rawHeight: size.y, height: actor.height, scale };
}

function frameCharacter() {
  const headLocal = actor.height * HEAD_AT;
  world.position.y = -headLocal;
}

function neutralizeRootMotion(clip) {
  for (const track of clip.tracks) {
    if (!track.name.endsWith(".position")) continue;
    if (!/hips|root|armature/i.test(track.name)) continue;
    const values = track.values;
    const x0 = values[0];
    const z0 = values[2];
    for (let i = 0; i < values.length; i += 3) {
      values[i] = x0;
      values[i + 2] = z0;
    }
  }
}

function updateActor(dt) {
  const goal = route[actor.waypoint];
  const pos = actor.holder.position;
  actor.shadow.position.x = pos.x;
  actor.shadow.position.z = pos.z;
  actor.stepping = false;

  if (!walking || actor.paused) {
    faceUser(dt, 4);
    return;
  }

  if (actor.meeting) {
    faceUser(dt, 5);
    actor.meetLeft -= dt;
    if (actor.meetLeft <= 0) {
      actor.meeting = false;
      actor.waypoint = (actor.waypoint + 1) % route.length;
    }
    return;
  }

  const dx = goal.x - pos.x;
  const dz = goal.z - pos.z;
  const dist = Math.hypot(dx, dz);
  if (dist < 0.05) {
    if (goal.meet) {
      actor.meeting = true;
      actor.meetLeft = 3.2;
    } else {
      actor.waypoint = (actor.waypoint + 1) % route.length;
    }
    return;
  }

  const step = Math.min(dist, 0.52 * dt);
  pos.x += (dx / dist) * step;
  pos.z += (dz / dist) * step;
  actor.stepping = step > 0;
  const yaw = Math.atan2(dx, dz) + MODEL_YAW;
  actor.holder.rotation.y = dampAngle(actor.holder.rotation.y, yaw, dt, 6);
}

const gait = {
  phase: 0,
  weight: 0,
  bones: null,
  swing: new THREE.Quaternion(),
  axis: new THREE.Vector3(1, 0, 0),
};

function applyGait(dt) {
  const model = actor.holder.children[0];
  if (!model) return;
  if (!gait.bones) {
    const get = (name) => actor.holder.getObjectByName(name);
    gait.bones = [
      [get("mixamorigLeftArm"), 0.32],
      [get("mixamorigRightArm"), -0.32],
      [get("mixamorigLeftUpLeg"), -0.36],
      [get("mixamorigRightUpLeg"), 0.36],
    ];
  }
  const stepping = Boolean(actor.stepping);
  gait.weight += ((stepping ? 1 : 0) - gait.weight) * (1 - Math.exp(-5 * dt));
  gait.phase += dt * (stepping ? 7.4 : 1.2);
  const swing = Math.sin(gait.phase) * gait.weight;
  model.position.y = (actor.footOffset || 0) + Math.abs(Math.sin(gait.phase)) * 0.028 * gait.weight;
  for (const [bone, amount] of gait.bones) {
    if (!bone) continue;
    gait.swing.setFromAxisAngle(gait.axis, swing * amount);
    bone.quaternion.multiply(gait.swing);
  }
}

function faceUser(dt, sharpness) {
  const pos = actor.holder.position;
  const yaw = Math.atan2(eye.x - pos.x, eye.z - pos.z) + MODEL_YAW;
  actor.holder.rotation.y = dampAngle(actor.holder.rotation.y, yaw, dt, sharpness);
}

async function begin(useMouse) {
  started = true;
  ui.intro.hidden = true;
  ui.hint.hidden = false;
  ui.hint.textContent = useMouse
    ? "マウスを動かすと視点が動きます。ホイールで前後、矢印キーでも動けます"
    : "顔を動かして、窓の向こうを覗いてください";
  ui.panel.hidden = false;
  toggleFullscreen();
  if (useMouse) {
    setMouseMode(true);
    setStatus("マウスで視点を動かしています", "");
    return;
  }
  setStatus("カメラを起動しています", "");
  try {
    await startCamera();
    ui.calibrate.disabled = false;
    ui.preview.hidden = false;
    setStatus("両目を探しています", "warn");
  } catch (error) {
    console.error(error);
    setMouseMode(true);
    setStatus("カメラを使えないため、マウス視点です", "warn");
  }
}

async function startCamera() {
  video = document.createElement("video");
  video.playsInline = true;
  video.muted = true;
  video.autoplay = true;
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: "user",
      width: { ideal: 640 },
      height: { ideal: 480 },
      frameRate: { ideal: 30, max: 30 },
    },
  });
  video.srcObject = stream;
  await video.play();
  if (!focalLocked && video.videoWidth) {
    focalPx = focalFromFov(video.videoWidth, 62);
  }

  const files = await FilesetResolver.forVisionTasks(WASM_URL);
  try {
    landmarker = await FaceLandmarker.createFromOptions(files, landmarkerOptions("GPU"));
  } catch (error) {
    console.warn("GPU delegate failed, using CPU", error);
    landmarker = await FaceLandmarker.createFromOptions(files, landmarkerOptions("CPU"));
  }
  detectReady = true;
  if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(onVideoFrame);
}

function landmarkerOptions(delegate) {
  return {
    baseOptions: { modelAssetPath: TASK_URL, delegate },
    runningMode: "VIDEO",
    numFaces: 1,
    minFaceDetectionConfidence: 0.5,
    minFacePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  };
}

let lastDetectTs = -1;
function onVideoFrame() {
  detectFace();
  if (video?.requestVideoFrameCallback) video.requestVideoFrameCallback(onVideoFrame);
}

function detectFace() {
  if (!detectReady || !video || video.readyState < 2) return;
  let timestamp = performance.now();
  if (timestamp <= lastDetectTs) timestamp = lastDetectTs + 1;
  lastDetectTs = timestamp;
  let result;
  try {
    result = landmarker.detectForVideo(video, timestamp);
  } catch (error) {
    console.error(error);
    return;
  }
  const landmarks = result.faceLandmarks?.[0];
  drawPreview(landmarks);
  if (!landmarks) return;

  const sample = eyesFromLandmarks(landmarks, video.videoWidth, video.videoHeight);
  if (!sample) return;
  if (calibrating) lastSample = sample;
  lastSample = sample;
  trackingLost = 0;

  if (mouseMode || overrideEye) {
    setStatus("両目を追跡中", "live");
    return;
  }
  const mountY = screenH / 2 + 0.012;
  const sx = sample.x * gain + xBias;
  const sy = mountY + yMountBias + sample.y * gain;
  const sz = THREE.MathUtils.clamp(sample.z, 0.24, 1.15);
  eyeTarget.set(sx, sy, sz);
  headUp.set(sample.upX, sample.upY);
  setStatus("両目を追跡中", "live");
}

function eyesFromLandmarks(landmarks, width, height) {
  const left = irisCenter(landmarks, 468, 33, 133);
  const right = irisCenter(landmarks, 473, 263, 362);
  if (!left || !right) return null;

  const leftPx = { x: left.x * width, y: left.y * height };
  const rightPx = { x: right.x * width, y: right.y * height };
  const ipdPx = Math.hypot(leftPx.x - rightPx.x, leftPx.y - rightPx.y);
  if (ipdPx < 10 || ipdPx > 420) return null;

  const z = (focalPx * IPD) / ipdPx;
  if (!Number.isFinite(z) || z < 0.15 || z > 2.2) return null;

  const midX = (leftPx.x + rightPx.x) / 2;
  const midY = (leftPx.y + rightPx.y) / 2;
  const x = -((midX - width / 2) * IPD) / ipdPx;
  const y = -((midY - height / 2) * IPD) / ipdPx;

  const dx = leftPx.x - rightPx.x;
  const dy = leftPx.y - rightPx.y;
  const len = Math.hypot(dy, dx) || 1;

  return { x, y, z, ipdPx, upX: -dy / len, upY: dx / len };
}

function irisCenter(landmarks, irisIndex, outerIndex, innerIndex) {
  if (landmarks[irisIndex]) return landmarks[irisIndex];
  const outer = landmarks[outerIndex];
  const inner = landmarks[innerIndex];
  if (!outer || !inner) return null;
  return { x: (outer.x + inner.x) / 2, y: (outer.y + inner.y) / 2 };
}

function drawPreview(landmarks) {
  const canvas = ui.previewCanvas;
  const ctx = previewCtx;
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.save();
  ctx.translate(w, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, w, h);
  ctx.restore();
  if (!landmarks) return;
  const dots = [468, 473, 33, 133, 263, 362];
  ctx.fillStyle = "#e7a35a";
  for (const index of dots) {
    const landmark = landmarks[index];
    if (!landmark) continue;
    const x = (1 - landmark.x) * w;
    const y = landmark.y * h;
    ctx.beginPath();
    ctx.arc(x, y, index >= 468 ? 3 : 1.6, 0, Math.PI * 2);
    ctx.fill();
  }
}

function beginCalibration() {
  if (!detectReady || calibrating) return;
  calibrating = true;
  ui.calOverlay.hidden = false;
  let count = 3;
  ui.calCount.textContent = String(count);
  const timer = setInterval(() => {
    count -= 1;
    if (count <= 0) {
      clearInterval(timer);
      applyCalibration();
      calibrating = false;
      ui.calOverlay.hidden = true;
      return;
    }
    ui.calCount.textContent = String(count);
  }, 1000);
}

function applyCalibration() {
  if (!lastSample) {
    setStatus("顔を検出してから較正してください", "warn");
    return;
  }
  focalPx = (lastSample.ipdPx * 0.5) / IPD;
  focalLocked = true;
  xBias = -lastSample.x * gain;
  yMountBias = -(screenH / 2 + 0.012) - lastSample.y * gain;
  setStatus("較正しました", "live");
}

function updateEye(dt) {
  if (overrideEye) {
    eye.copy(overrideEye);
    ui.depth.textContent = `${Math.round(eye.z * 100)} cm`;
    return;
  }

  if (mouseMode) {
    const speed = 0.55;
    if (keys.has("ArrowLeft")) eyeTarget.x -= speed * dt;
    if (keys.has("ArrowRight")) eyeTarget.x += speed * dt;
    if (keys.has("ArrowUp")) eyeTarget.y += speed * dt;
    if (keys.has("ArrowDown")) eyeTarget.y -= speed * dt;
    eyeTarget.x = THREE.MathUtils.clamp(eyeTarget.x, -0.45, 0.45);
    eyeTarget.y = THREE.MathUtils.clamp(eyeTarget.y, -0.35, 0.35);
  } else if (started && detectReady) {
    trackingLost += dt;
    if (trackingLost > 0.45) {
      const time = performance.now() / 1000;
      eyeTarget.x = Math.sin(time * 0.45) * 0.02;
      eyeTarget.y = Math.sin(time * 0.3) * 0.015;
      if (trackingLost > 0.8) setStatus("顔が見つかりません", "warn");
    }
  } else {
    const time = performance.now() / 1000;
    eyeTarget.set(Math.sin(time * 0.35) * 0.14, Math.sin(time * 0.23) * 0.07, 0.52 + Math.sin(time * 0.18) * 0.04);
    headUp.set(Math.sin(time * 0.2) * 0.04, 1);
  }

  eye.x = filterX.filter(eyeTarget.x, dt);
  eye.y = filterY.filter(eyeTarget.y, dt);
  eye.z = filterZ.filter(eyeTarget.z, dt);
  const smoothUpX = filterRoll.filter(headUp.x, dt);
  headUp.x = smoothUpX;
  ui.depth.textContent = `${Math.round(eye.z * 100)} cm`;
}

function applyProjection(position, upX, upY) {
  const depth = Math.max(position.z, 0.2);
  const scale = NEAR / depth;
  const left = (-screenW / 2 - position.x) * scale;
  const right = (screenW / 2 - position.x) * scale;
  const bottom = (-screenH / 2 - position.y) * scale;
  const top = (screenH / 2 - position.y) * scale;
  camera.position.copy(position);
  camera.position.z = depth;
  const up = new THREE.Vector3(upX, upY, 0);
  if (up.lengthSq() < 1e-4 || up.y < 0.25) up.set(0, 1, 0);
  camera.up.copy(up.normalize());
  camera.lookAt(position.x, position.y, 0);
  camera.near = NEAR;
  camera.far = FAR;
  camera.projectionMatrix.makePerspective(left, right, top, bottom, NEAR, FAR, renderer.coordinateSystem);
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
  camera.updateMatrixWorld(true);
}

function renderView() {
  const upX = headUp.x;
  const upY = headUp.y;
  if (!anaglyph) {
    applyProjection(eye, upX, upY);
    renderer.setRenderTarget(null);
    renderer.render(scene, camera);
    return;
  }

  const right = new THREE.Vector3(upY, -upX, 0);
  if (right.lengthSq() < 1e-4) right.set(1, 0, 0);
  right.normalize();
  const leftEye = eye.clone().addScaledVector(right, -IPD / 2);
  const rightEye = eye.clone().addScaledVector(right, IPD / 2);
  leftEye.z = eye.z;
  rightEye.z = eye.z;

  applyProjection(leftEye, upX, upY);
  renderer.setRenderTarget(rtLeft);
  renderer.render(scene, camera);
  applyProjection(rightEye, upX, upY);
  renderer.setRenderTarget(rtRight);
  renderer.render(scene, camera);
  renderer.setRenderTarget(null);
  quadMaterial.uniforms.tLeft.value = rtLeft.texture;
  quadMaterial.uniforms.tRight.value = rtRight.texture;
  renderer.render(quadScene, quadCamera);
}

function frame() {
  requestAnimationFrame(frame);
  const dt = Math.min(clock.getDelta(), 0.05);
  if (!video?.requestVideoFrameCallback && detectReady) detectFace();
  updateEye(dt);
  updateActor(dt);
  if (actor.mixer) actor.mixer.update(dt);
  applyGait(dt);

  const time = performance.now() / 1000;
  siren.intensity = 1.4 + Math.sin(time * 3.2) * 1.4;
  sirenWarm.intensity = 1.4 + Math.sin(time * 3.2 + Math.PI) * 1.4;

  renderView();
}

function resize() {
  const size = viewportMeters(Number(ui.inch.value));
  screenW = size.w;
  screenH = size.h;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  const width = renderer.domElement.width;
  const height = renderer.domElement.height;
  rtLeft.setSize(width, height);
  rtRight.setSize(width, height);
  if (!focalLocked) focalPx = focalFromFov(video?.videoWidth || 640, 62);
}

function viewportMeters(inches) {
  const diagonal = inches * 0.0254;
  const displayW = Math.max(screen.width, 1);
  const displayH = Math.max(screen.height, 1);
  const aspect = displayW / displayH;
  const displayHeight = diagonal / Math.sqrt(aspect * aspect + 1);
  const displayWidth = displayHeight * aspect;
  return {
    w: displayWidth * (window.innerWidth / displayW),
    h: displayHeight * (window.innerHeight / displayH),
  };
}

function focalFromFov(widthPx, fovDegrees) {
  const fov = (fovDegrees * Math.PI) / 180;
  return (widthPx / 2) / Math.tan(fov / 2);
}

function setAnaglyph(next) {
  anaglyph = next;
  ui.anaglyph.setAttribute("aria-pressed", String(next));
}

function setMouseMode(next) {
  mouseMode = next;
  ui.mouseMode.setAttribute("aria-pressed", String(next));
  if (next) {
    setStatus("マウスで視点を動かしています", "");
    if (started) ui.hint.textContent = "マウスを動かすと視点が動きます。ホイールで前後、矢印キーでも動けます";
  } else if (started) {
    ui.hint.textContent = "顔を動かして、窓の向こうを覗いてください";
  }
}

function setStatus(text, mode) {
  ui.status.textContent = text;
  ui.dot.classList.toggle("live", mode === "live");
  ui.dot.classList.toggle("warn", mode === "warn");
}

function toggleFullscreen() {
  if (document.fullscreenElement) {
    document.exitFullscreen?.();
    return;
  }
  document.documentElement.requestFullscreen?.().catch(() => {});
}

function dampAngle(current, target, dt, sharpness) {
  let delta = target - current;
  while (delta > Math.PI) delta -= Math.PI * 2;
  while (delta < -Math.PI) delta += Math.PI * 2;
  const t = 1 - Math.exp(-sharpness * dt);
  let next = current + delta * t;
  while (next > Math.PI) next -= Math.PI * 2;
  while (next < -Math.PI) next += Math.PI * 2;
  return next;
}

function OneEuro(minCutoff, beta) {
  const x = new LowPass();
  const dx = new LowPass();
  let last = null;
  return {
    filter(value, dt) {
      const freq = dt > 1e-4 ? 1 / dt : 60;
      const derivative = last == null ? 0 : (value - last) * freq;
      last = value;
      const filteredDerivative = dx.filter(derivative, alpha(freq, 1));
      const cutoff = minCutoff + beta * Math.abs(filteredDerivative);
      return x.filter(value, alpha(freq, cutoff));
    },
  };
}

function LowPass() {
  let y = null;
  return {
    filter(value, a) {
      y = y == null ? value : a * value + (1 - a) * y;
      return y;
    },
  };
}

function alpha(freq, cutoff) {
  const tau = 1 / (2 * Math.PI * cutoff);
  return 1 / (1 + tau / freq);
}
