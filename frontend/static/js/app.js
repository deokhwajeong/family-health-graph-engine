/**
 * Family Health Dashboard - Main Application
 */

// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// State Management
const state = {
    currentTab: 'dashboard',
    darkMode: localStorage.getItem('darkMode') === 'true',
    apiUrl: localStorage.getItem('apiUrl') || API_BASE_URL,
};

// Initialize Application
document.addEventListener('DOMContentLoaded', () => {
    initializeDarkMode();
    initializeNavigation();
    initializeMobileMenu();
    initializeSettings();
});

// ==========================================
// Dark Mode
// ==========================================
function initializeDarkMode() {
    const darkModeToggle = document.getElementById('darkModeToggle');
    const darkModeSetting = document.getElementById('dark-mode-setting');
    
    // Apply saved dark mode preference
    if (state.darkMode) {
        document.body.classList.add('dark-mode');
        if (darkModeSetting) darkModeSetting.checked = true;
    }
    
    // Toggle button click
    if (darkModeToggle) {
        darkModeToggle.addEventListener('click', toggleDarkMode);
    }
    
    // Settings checkbox
    if (darkModeSetting) {
        darkModeSetting.addEventListener('change', (e) => {
            if (e.target.checked !== state.darkMode) {
                toggleDarkMode();
            }
        });
    }
}

function toggleDarkMode() {
    state.darkMode = !state.darkMode;
    document.body.classList.toggle('dark-mode');
    localStorage.setItem('darkMode', state.darkMode);
    
    const darkModeSetting = document.getElementById('dark-mode-setting');
    if (darkModeSetting) {
        darkModeSetting.checked = state.darkMode;
    }
    
    const icon = document.querySelector('#darkModeToggle i');
    if (icon) {
        icon.className = state.darkMode ? 'fas fa-sun' : 'fas fa-moon';
    }
}

// ==========================================
// Navigation
// ==========================================
function initializeNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const tabName = item.getAttribute('data-tab');
            switchTab(tabName);
            
            // Close mobile menu after selection
            const sidebar = document.getElementById('sidebar');
            if (window.innerWidth <= 768) {
                sidebar.classList.remove('active');
            }
        });
    });
}

function switchTab(tabName) {
    // Update state
    state.currentTab = tabName;
    
    // Update nav items
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('active');
        if (item.getAttribute('data-tab') === tabName) {
            item.classList.add('active');
        }
    });
    
    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    
    const activeTab = document.getElementById(`${tabName}-tab`);
    if (activeTab) {
        activeTab.classList.add('active');
    }
}

// ==========================================
// Mobile Menu
// ==========================================
function initializeMobileMenu() {
    const menuToggle = document.getElementById('mobileMenuToggle');
    const sidebar = document.getElementById('sidebar');
    
    if (menuToggle && sidebar) {
        menuToggle.addEventListener('click', () => {
            sidebar.classList.toggle('active');
        });
        
        // Close menu when clicking outside
        document.addEventListener('click', (e) => {
            if (window.innerWidth <= 768) {
                if (!sidebar.contains(e.target) && !menuToggle.contains(e.target)) {
                    sidebar.classList.remove('active');
                }
            }
        });
    }
}

// ==========================================
// Settings
// ==========================================
function initializeSettings() {
    const apiUrlSetting = document.getElementById('api-url-setting');
    
    if (apiUrlSetting) {
        apiUrlSetting.value = state.apiUrl;
        apiUrlSetting.addEventListener('change', (e) => {
            state.apiUrl = e.target.value;
            localStorage.setItem('apiUrl', state.apiUrl);
            showToast('API URL updated', 'success');
        });
    }
}

// ==========================================
// Toast Notifications
// ==========================================
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icon = type === 'success' ? 'fa-check-circle' : 
                 type === 'error' ? 'fa-exclamation-circle' : 
                 'fa-info-circle';
    
    toast.innerHTML = `
        <i class="fas ${icon}"></i>
        <span>${message}</span>
    `;
    
    container.appendChild(toast);
    
    // Auto remove after 3 seconds
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => {
            toast.remove();
        }, 300);
    }, 3000);
}

// Add slideOut animation
const style = document.createElement('style');
style.textContent = `
    @keyframes slideOut {
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// ==========================================
// API Helper Functions
// ==========================================
async function apiRequest(endpoint, options = {}) {
    const url = `${state.apiUrl}${endpoint}`;
    
    try {
        const response = await fetch(url, {
            ...options,
            headers: {
                ...options.headers,
            },
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        // Check if response is JSON
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
            return await response.json();
        }
        
        return await response.text();
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

// ==========================================
// Utility Functions
// ==========================================
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
}

function formatTime(timeString) {
    if (!timeString) return '';
    const [hours, minutes] = timeString.split(':');
    const hour = parseInt(hours);
    const ampm = hour >= 12 ? 'PM' : 'AM';
    const displayHour = hour % 12 || 12;
    return `${displayHour}:${minutes} ${ampm}`;
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// Export for use in other modules
window.app = {
    state,
    switchTab,
    showToast,
    apiRequest,
    formatDate,
    formatTime,
    formatBytes,
};
