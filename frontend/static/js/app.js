document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadSection = document.getElementById('upload-section');
    const resultsSection = document.getElementById('results-section');
    const loader = document.getElementById('loader');
    const btnReset = document.getElementById('btn-reset');
    const tabs = document.querySelectorAll('.tab-btn');
    const editorToolbar = document.getElementById('editor-toolbar');
    
    const canvas = document.getElementById('drawing-canvas');
    const ctx = canvas.getContext('2d');
    const baseImg = document.getElementById('img-orig');
    let currentImageId = "";

    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.style.borderColor = '#2563eb'; });
    dropZone.addEventListener('dragleave', () => dropZone.style.borderColor = '');
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) handleFile(e.target.files[0]);
    });

    async function handleFile(file) {
        if (!file.type.startsWith('image/')) return showToast("Допускаются только изображения (JPG, PNG)", "error");
        currentImageId = file.name.split('.')[0];
        
        dropZone.style.display = 'none';
        loader.style.display = 'block';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/analyze', { method: 'POST', body: formData });
            const data = await response.json();
            
            if (!response.ok) throw new Error(data.error || 'Ошибка при обработке запроса сервером');
            if (data.error) throw new Error(data.error);
            
            renderResults(data);
        } catch (error) {
            showToast(error.message, "error");
            dropZone.style.display = 'block';
        } finally {
            loader.style.display = 'none';
        }
    }

    function renderResults(data) {
        const stats = data.stats;
        document.getElementById('res-final-class').innerText = stats.final_class;
        document.getElementById('res-talc-ratio').innerText = stats.talc_ratio + '%';
        document.getElementById('res-texture').innerText = stats.texture_class;
        document.getElementById('res-conf').innerText = stats.confidence;
        document.getElementById('res-time').innerText = stats.processing_time;
        document.getElementById('res-rec').innerText = stats.recommendation;
        document.getElementById('res-grind').innerText = stats.grinding;

        const card = document.getElementById('final-class-card');
        card.className = 'stat-card'; // Сброс классов
        if(stats.talc_ratio >= 10) card.classList.add('red');
        else if(stats.texture_class.includes('Труднообогатимая')) card.classList.add('orange');
        else card.classList.add('green');

        const ts = new Date().getTime();
        document.getElementById('img-orig').src = data.images.original + '?t=' + ts;
        document.getElementById('img-ore').src = data.images.ore_mask + '?t=' + ts;
        document.getElementById('img-talc').src = data.images.talc_mask + '?t=' + ts;
        document.getElementById('img-heatmap').src = data.images.heatmap + '?t=' + ts;
        document.getElementById('img-overlay').src = data.images.overlay + '?t=' + ts;

        uploadSection.style.display = 'none';
        resultsSection.style.display = 'flex';
        document.querySelector('.tab-btn[data-target="img-orig"]').click();
    }

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.viewer-img').forEach(img => img.classList.remove('active'));
            
            tab.classList.add('active');
            const targetId = tab.getAttribute('data-target');
            
            if (targetId === 'editor-mode') {
                document.getElementById('img-orig').classList.add('active');
                canvas.classList.add('active');
                editorToolbar.style.display = 'flex';
                initCanvas();
            } else {
                document.getElementById(targetId).classList.add('active');
                editorToolbar.style.display = 'none';
            }
        });
    });

    let isDrawing = false;
    let brushMode = 'brush';
    const btnBrush = document.getElementById('btn-brush');
    const btnEraser = document.getElementById('btn-eraser');
    const brushSizeInput = document.getElementById('brush-size');

    btnBrush.onclick = () => { brushMode = 'brush'; btnBrush.classList.add('active'); btnEraser.classList.remove('active'); };
    btnEraser.onclick = () => { brushMode = 'eraser'; btnEraser.classList.add('active'); btnBrush.classList.remove('active'); };

    function initCanvas() {
        canvas.width = baseImg.clientWidth;
        canvas.height = baseImg.clientHeight;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
    }

    window.addEventListener('resize', () => {
        if (canvas.classList.contains('active')) initCanvas();
    });

    canvas.addEventListener('mousedown', (e) => {
        isDrawing = true;
        draw(e);
    });

    canvas.addEventListener('mousemove', draw);
    window.addEventListener('mouseup', () => { isDrawing = false; ctx.beginPath(); });

    function draw(e) {
        if (!isDrawing) return;
        
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        ctx.lineWidth = brushSizeInput.value;
        
        if (brushMode === 'brush') {
            ctx.globalCompositeOperation = 'source-over';
            ctx.strokeStyle = 'rgba(220, 38, 38, 0.7)';
        } else {
            ctx.globalCompositeOperation = 'destination-out';
            ctx.strokeStyle = 'rgba(0,0,0,1)';
        }

        ctx.lineTo(x, y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x, y);
    }

    document.getElementById('btn-save-mask').onclick = async () => {
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = canvas.width; tempCanvas.height = canvas.height;
        const tempCtx = tempCanvas.getContext('2d');
        
        tempCtx.fillStyle = 'black';
        tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
        tempCtx.drawImage(canvas, 0, 0);

        const maskBase64 = tempCanvas.toDataURL('image/png');
        
        try {
            const res = await fetch('/api/save_mask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ imageId: currentImageId, maskBase64: maskBase64 })
            });
            const data = await res.json();
            if(data.status === 'success') showToast(data.message, "success");
            else showToast(data.error, "error");
        } catch (e) {
            showToast("Ошибка при передаче маски на сервер", "error");
        }
    };

    document.getElementById('btn-retrain').onclick = async () => {
        const btn = document.getElementById('btn-retrain');
        btn.innerText = "Идет обучение...";
        btn.disabled = true;
        showToast("Запущен фоновый процесс переобучения сверточной сети...", "info");

        try {
            const res = await fetch('/api/retrain', { method: 'POST' });
            const data = await res.json();
            
            if(data.status === 'success') {
                showToast(data.message, "success");
                document.getElementById('model-status').innerText = "Модель v1.1 (Веса обновлены)";
            } else {
                showToast("Сбой дообучения: " + data.error, "error");
            }
        } catch (e) {
            showToast("Сбой связи с сервером обучения", "error");
        } finally {
            btn.innerText = "Запустить дообучение";
            btn.disabled = false;
        }
    };

    btnReset.addEventListener('click', () => {
        resultsSection.style.display = 'none';
        uploadSection.style.display = 'block';
        dropZone.style.display = 'block';
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        fileInput.value = '';
    });

    function showToast(message, type="success") {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = 'toast';
        if (type === 'error') toast.style.borderLeftColor = 'var(--accent-red)';
        if (type === 'info') toast.style.borderLeftColor = 'var(--accent-indigo)';
        
        toast.innerText = message;
        container.appendChild(toast);
        
        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }
});