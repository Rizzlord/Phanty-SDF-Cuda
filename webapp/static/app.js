const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const fileInfo = document.getElementById('file-info');
const fileName = document.getElementById('file-name');
const fileSize = document.getElementById('file-size');
const loadDefaultBtn = document.getElementById('load-default-btn');
const resSlider = document.getElementById('res-slider');
const resVal = document.getElementById('res-val');
const scaleSlider = document.getElementById('scale-slider');
const scaleVal = document.getElementById('scale-val');
const voxelizeFirstCb = document.getElementById('voxelize-first-cb');
const voxelResSlider = document.getElementById('voxel-res-slider');
const voxelResVal = document.getElementById('voxel-res-val');
const startBtn = document.getElementById('start-btn');
const downloadBtn = document.getElementById('download-btn');
const spinner = document.getElementById('spinner');
const tabInput = document.getElementById('tab-input');
const tabOutput = document.getElementById('tab-output');
const resetCamBtn = document.getElementById('reset-cam-btn');
const wireframeBtn = document.getElementById('wireframe-btn');
const loadingOverlay = document.getElementById('loading-overlay');
const statsDashboard = document.getElementById('stats-dashboard');
const canvasContainer = document.getElementById('canvas-3d');

let selectedFile = null;
let outputBlob = null;
let scene, camera, renderer, controls;
let inputModel = null;
let outputModel = null;
let activeTab = 'input';
let isWireframe = false;

function init3D() {
    scene = new THREE.Scene();
    
    camera = new THREE.PerspectiveCamera(45, canvasContainer.clientWidth / canvasContainer.clientHeight, 0.01, 1000);
    camera.position.set(2.5, 2.0, 3.0);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(canvasContainer.clientWidth, canvasContainer.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.outputEncoding = THREE.sRGBEncoding;
    canvasContainer.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.target.set(0, 0, 0);

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight1.position.set(5, 10, 7.5);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x6366f1, 0.4);
    dirLight2.position.set(-5, -5, -5);
    scene.add(dirLight2);

    const gridHelper = new THREE.GridHelper(4, 20, 0x6366f1, 0x334155);
    gridHelper.position.y = -1.0;
    scene.add(gridHelper);

    window.addEventListener('resize', onWindowResize);
    animate();
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

function onWindowResize() {
    if (!canvasContainer) return;
    camera.aspect = canvasContainer.clientWidth / canvasContainer.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(canvasContainer.clientWidth, canvasContainer.clientHeight);
}

function centerAndScaleModel(group) {
    const box = new THREE.Box3().setFromObject(group);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    const scale = maxDim > 0 ? (1.6 / maxDim) : 1.0;

    group.position.sub(center.multiplyScalar(scale));
    group.scale.setScalar(scale);
}

function applyWireframe(object, wire) {
    object.traverse((child) => {
        if (child.isMesh && child.material) {
            child.material.wireframe = wire;
        }
    });
}

function loadGltfToScene(arrayBuffer, isInput) {
    const loader = new THREE.GLTFLoader();
    loader.parse(arrayBuffer, '', (gltf) => {
        const model = gltf.scene;
        centerAndScaleModel(model);
        applyWireframe(model, isWireframe);

        if (isInput) {
            if (inputModel) scene.remove(inputModel);
            inputModel = model;
            if (activeTab === 'input') scene.add(inputModel);
        } else {
            if (outputModel) scene.remove(outputModel);
            outputModel = model;
            if (activeTab === 'output') scene.add(outputModel);
        }
    }, (error) => {
        console.error(error);
    });
}

function updateFileUI(file) {
    selectedFile = file;
    fileName.textContent = file.name;
    const mb = (file.size / (1024 * 1024)).toFixed(2);
    fileSize.textContent = `${mb} MB`;
    fileInfo.classList.remove('hidden');
    startBtn.disabled = false;

    const reader = new FileReader();
    reader.onload = (e) => {
        loadGltfToScene(e.target.result, true);
        switchTab('input');
    };
    reader.readAsArrayBuffer(file);
}

function switchTab(tab) {
    activeTab = tab;
    if (tab === 'input') {
        tabInput.classList.add('active');
        tabOutput.classList.remove('active');
        if (outputModel) scene.remove(outputModel);
        if (inputModel) scene.add(inputModel);
    } else {
        tabOutput.classList.add('active');
        tabInput.classList.remove('active');
        if (inputModel) scene.remove(inputModel);
        if (outputModel) scene.add(outputModel);
    }
}

dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        const file = e.dataTransfer.files[0];
        if (file.name.toLowerCase().endsWith('.glb')) {
            updateFileUI(file);
        } else {
            alert('Please drop a valid .glb 3D model file.');
        }
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
        updateFileUI(e.target.files[0]);
    }
});

loadDefaultBtn.addEventListener('click', async () => {
    try {
        loadDefaultBtn.disabled = true;
        loadDefaultBtn.textContent = "Loading test_input.glb...";
        const res = await fetch('/api/default_glb');
        const data = await res.json();
        if (data.success) {
            const byteCharacters = atob(data.glb_base64);
            const byteNumbers = new Array(byteCharacters.length);
            for (let i = 0; i < byteCharacters.length; i++) {
                byteNumbers[i] = byteCharacters.charCodeAt(i);
            }
            const byteArray = new Uint8Array(byteNumbers);
            const blob = new Blob([byteArray], { type: 'model/gltf-binary' });
            const file = new File([blob], data.filename, { type: 'model/gltf-binary' });
            updateFileUI(file);
        } else {
            alert("Failed to load sample model.");
        }
    } catch (err) {
        alert("Error loading sample model: " + err.message);
    } finally {
        loadDefaultBtn.disabled = false;
        loadDefaultBtn.textContent = "Try Sample (test_input.glb ~60MB)";
    }
});

resSlider.addEventListener('input', (e) => {
    resVal.textContent = `${e.target.value}³ Grid`;
});

scaleSlider.addEventListener('input', (e) => {
    scaleVal.textContent = `${parseFloat(e.target.value).toFixed(2)}x`;
});

if (voxelResSlider && voxelResVal) {
    voxelResSlider.addEventListener('input', (e) => {
        voxelResVal.textContent = `${e.target.value}³`;
    });
}

tabInput.addEventListener('click', () => switchTab('input'));
tabOutput.addEventListener('click', () => {
    if (outputModel) switchTab('output');
});

resetCamBtn.addEventListener('click', () => {
    camera.position.set(2.5, 2.0, 3.0);
    controls.target.set(0, 0, 0);
    controls.update();
});

wireframeBtn.addEventListener('click', () => {
    isWireframe = !isWireframe;
    if (inputModel) applyWireframe(inputModel, isWireframe);
    if (outputModel) applyWireframe(outputModel, isWireframe);
});

startBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    try {
        startBtn.disabled = true;
        spinner.classList.remove('hidden');
        loadingOverlay.classList.remove('hidden');

        const backendVal = document.querySelector('input[name="backend"]:checked').value;
        const brickSizeVal = document.querySelector('input[name="brick_size"]:checked').value;
        const removeFloatersVal = document.getElementById('remove-floaters-cb').checked;
        const closeHolesVal = document.getElementById('close-holes-cb').checked;
        const voxelizeFirstVal = voxelizeFirstCb ? voxelizeFirstCb.checked : false;
        const voxelResValInt = voxelResSlider ? voxelResSlider.value : 256;

        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('res', resSlider.value);
        formData.append('backend', backendVal);
        formData.append('brick_size', brickSizeVal);
        formData.append('scale_factor', scaleSlider.value);
        formData.append('remove_floaters', removeFloatersVal);
        formData.append('close_holes', closeHolesVal);
        formData.append('voxelize_first', voxelizeFirstVal);
        formData.append('voxel_res', voxelResValInt);


        const res = await fetch('/api/extract', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();

        if (data.success) {
            const byteCharacters = atob(data.glb_base64);
            const byteNumbers = new Array(byteCharacters.length);
            for (let i = 0; i < byteCharacters.length; i++) {
                byteNumbers[i] = byteCharacters.charCodeAt(i);
            }
            const byteArray = new Uint8Array(byteNumbers);
            outputBlob = new Blob([byteArray], { type: 'model/gltf-binary' });

            loadGltfToScene(byteArray.buffer, false);
            tabOutput.disabled = false;
            switchTab('output');

            downloadBtn.classList.remove('hidden');
            downloadBtn.disabled = false;

            statsDashboard.classList.remove('hidden');
            document.getElementById('stat-total-time').textContent = `${data.stats.total_ms} ms`;
            document.getElementById('stat-vertices').textContent = data.stats.vertex_count.toLocaleString();
            document.getElementById('stat-faces').textContent = data.stats.face_count.toLocaleString();
            document.getElementById('stat-sdf-time').textContent = `${data.stats.sdf_ms} ms`;
            if (data.stats.total_bricks && data.stats.total_bricks > 0) {
                const pct = (100.0 * data.stats.active_bricks / data.stats.total_bricks).toFixed(2);
                document.getElementById('stat-bricks').textContent = `${data.stats.active_bricks.toLocaleString()} / ${data.stats.total_bricks.toLocaleString()} (${pct}%)`;
            } else {
                document.getElementById('stat-bricks').textContent = 'N/A (Dense Backend)';
            }
            if (data.stats.ambiguous_cells && data.stats.ambiguous_cells > 0) {
                document.getElementById('stat-mvdc').textContent = `${data.stats.ambiguous_cells.toLocaleString()} split (${data.stats.split_rejection_count || 0} rejected)`;
            } else {
                document.getElementById('stat-mvdc').textContent = '0 (Single-Vertex or Standard)';
            }
            document.getElementById('stat-upload').textContent = `${data.stats.upload_ms} ms`;
            document.getElementById('stat-marking').textContent = `${data.stats.marking_ms} ms`;
            document.getElementById('stat-compaction').textContent = `${data.stats.compaction_ms} ms`;
            document.getElementById('stat-qef').textContent = `${data.stats.qef_ms} ms`;
            document.getElementById('stat-emit').textContent = `${data.stats.face_emission_ms} ms`;
            document.getElementById('stat-download').textContent = `${data.stats.download_ms} ms`;
        } else {
            alert('Extraction Error: ' + (data.error || 'Unknown error occurred.'));
        }
    } catch (err) {
        alert('Request failed: ' + err.message);
    } finally {
        startBtn.disabled = false;
        spinner.classList.add('hidden');
        loadingOverlay.classList.add('hidden');
    }
});

downloadBtn.addEventListener('click', () => {
    if (!outputBlob) return;
    const url = URL.createObjectURL(outputBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `dual_contouring_${resSlider.value}cube_${selectedFile ? selectedFile.name : 'model.glb'}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
});

init3D();
