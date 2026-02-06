/**
 * Apple Health Import Functionality
 */

let selectedFile = null;

document.addEventListener('DOMContentLoaded', () => {
    initializeAppleHealthImport();
});

function initializeAppleHealthImport() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const fileSelectBtn = document.getElementById('file-select-btn');
    const importBtn = document.getElementById('import-btn');
    
    if (!dropZone || !fileInput || !fileSelectBtn || !importBtn) {
        return; // Elements not on current page
    }
    
    // File select button
    fileSelectBtn.addEventListener('click', (e) => {
        e.preventDefault();
        fileInput.click();
    });
    
    // File input change
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFileSelect(file);
        }
    });
    
    // Drag and drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });
    
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        
        const file = e.dataTransfer.files[0];
        if (file) {
            handleFileSelect(file);
        }
    });
    
    // Import button
    importBtn.addEventListener('click', handleImport);
}

function handleFileSelect(file) {
    // Validate file type
    if (!file.name.endsWith('.xml')) {
        app.showToast('Please select a valid XML file', 'error');
        return;
    }
    
    // Validate MIME type if available
    if (file.type && file.type !== 'text/xml' && file.type !== 'application/xml' && file.type !== '') {
        app.showToast('Invalid file type. Please select an XML file', 'error');
        return;
    }
    
    selectedFile = file;
    
    // Show file info
    const fileInfo = document.getElementById('file-info');
    const fileName = document.getElementById('file-name');
    const fileSize = document.getElementById('file-size');
    const importBtn = document.getElementById('import-btn');
    
    if (fileInfo && fileName && fileSize) {
        fileInfo.style.display = 'flex';
        fileName.textContent = file.name;
        fileSize.textContent = app.formatBytes(file.size);
    }
    
    if (importBtn) {
        importBtn.disabled = false;
    }
    
    app.showToast('File selected successfully', 'success');
}

async function handleImport() {
    if (!selectedFile) {
        app.showToast('Please select a file first', 'error');
        return;
    }
    
    const personSelect = document.getElementById('import-person');
    const personName = personSelect.value;
    
    const progressContainer = document.getElementById('progress-container');
    const progressFill = document.getElementById('progress-fill');
    const progressText = document.getElementById('progress-text');
    const importBtn = document.getElementById('import-btn');
    const resultsContainer = document.getElementById('import-results');
    
    try {
        // Show progress
        if (progressContainer) {
            progressContainer.style.display = 'block';
            progressFill.style.width = '10%';
            progressText.textContent = 'Uploading file...';
        }
        
        if (importBtn) {
            importBtn.disabled = true;
        }
        
        // Create form data
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('person_name', personName);
        
        // Update progress
        if (progressFill) {
            progressFill.style.width = '30%';
            progressText.textContent = 'Processing XML...';
        }
        
        // Send request
        const response = await fetch(`${app.state.apiUrl}/api/health/import-xml`, {
            method: 'POST',
            body: formData,
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        // Update progress
        if (progressFill) {
            progressFill.style.width = '70%';
            progressText.textContent = 'Importing data...';
        }
        
        const result = await response.json();
        
        // Complete progress
        if (progressFill) {
            progressFill.style.width = '100%';
            progressText.textContent = 'Import complete!';
        }
        
        // Show results
        setTimeout(() => {
            if (progressContainer) {
                progressContainer.style.display = 'none';
            }
            displayImportResults(result);
            app.showToast('Data imported successfully!', 'success');
        }, 1000);
        
    } catch (error) {
        console.error('Import failed:', error);
        app.showToast(`Import failed: ${error.message}`, 'error');
        
        if (progressContainer) {
            progressContainer.style.display = 'none';
        }
        
        if (importBtn) {
            importBtn.disabled = false;
        }
    }
}

function displayImportResults(result) {
    const resultsContainer = document.getElementById('import-results');
    const resultsGrid = document.getElementById('results-grid');
    
    if (!resultsContainer || !resultsGrid) return;
    
    resultsContainer.style.display = 'block';
    
    const { records_imported, statistics } = result;
    
    resultsGrid.innerHTML = `
        <div class="card">
            <h4>Steps</h4>
            <div class="big-stat">${records_imported.steps_days}</div>
            <p class="stat-label">days</p>
        </div>
        <div class="card">
            <h4>Calories</h4>
            <div class="big-stat">${records_imported.calories_days}</div>
            <p class="stat-label">days</p>
        </div>
        <div class="card">
            <h4>Heart Rate</h4>
            <div class="big-stat">${records_imported.heart_rate_days}</div>
            <p class="stat-label">days</p>
        </div>
        <div class="card">
            <h4>Sleep Sessions</h4>
            <div class="big-stat">${records_imported.sleep_sessions}</div>
            <p class="stat-label">sessions</p>
        </div>
        <div class="card">
            <h4>Total Steps</h4>
            <div class="big-stat">${statistics.total_steps.toLocaleString()}</div>
            <p class="stat-label">steps</p>
        </div>
        <div class="card">
            <h4>Avg Steps/Day</h4>
            <div class="big-stat">${statistics.avg_steps_per_day.toLocaleString()}</div>
            <p class="stat-label">steps</p>
        </div>
        <div class="card">
            <h4>Avg Heart Rate</h4>
            <div class="big-stat">${statistics.avg_heart_rate}</div>
            <p class="stat-label">bpm</p>
        </div>
    `;
    
    // Reset for next import
    selectedFile = null;
    const fileInfo = document.getElementById('file-info');
    const importBtn = document.getElementById('import-btn');
    const fileInput = document.getElementById('file-input');
    
    if (fileInfo) {
        fileInfo.style.display = 'none';
    }
    
    if (importBtn) {
        importBtn.disabled = true;
    }
    
    if (fileInput) {
        fileInput.value = '';
    }
}
