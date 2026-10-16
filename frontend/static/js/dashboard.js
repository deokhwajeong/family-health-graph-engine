/**
 * Dashboard - Charts and Statistics
 */

let weeklyTrendsChart = null;
let sleepChart = null;

// Initialize dashboard when tab is loaded
document.addEventListener('DOMContentLoaded', () => {
    initializeDashboard();
});

function initializeDashboard() {
    createWeeklyTrendsChart();
    createSleepChart();
    loadDashboardData();
}

// ==========================================
// Weekly Trends Chart
// ==========================================
function createWeeklyTrendsChart() {
    const ctx = document.getElementById('weeklyTrendsChart');
    if (!ctx) return;
    
    // Sample data for demonstration
    const labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
    
    weeklyTrendsChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Meals',
                    data: [3, 3, 2, 3, 3, 4, 3],
                    borderColor: '#7ED321',
                    backgroundColor: 'rgba(126, 211, 33, 0.1)',
                    tension: 0.4,
                },
                {
                    label: 'Sleep (hours)',
                    data: [7.5, 8, 7, 7.5, 6.5, 8.5, 9],
                    borderColor: '#4A90E2',
                    backgroundColor: 'rgba(74, 144, 226, 0.1)',
                    tension: 0.4,
                },
                {
                    label: 'Activities',
                    data: [2, 1, 2, 3, 1, 3, 2],
                    borderColor: '#F5A623',
                    backgroundColor: 'rgba(245, 166, 35, 0.1)',
                    tension: 0.4,
                },
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        precision: 0
                    }
                }
            }
        }
    });
}

// ==========================================
// Sleep Chart
// ==========================================
function createSleepChart() {
    const ctx = document.getElementById('sleepChart');
    if (!ctx) return;
    
    const labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
    
    sleepChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Leo',
                    data: [8, 8.5, 7.5, 8, 7, 9, 9.5],
                    backgroundColor: 'rgba(74, 144, 226, 0.7)',
                },
                {
                    label: 'Jordan',
                    data: [7, 7.5, 6.5, 7, 6, 8, 8.5],
                    backgroundColor: 'rgba(80, 227, 194, 0.7)',
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Hours'
                    }
                }
            }
        }
    });
}

// ==========================================
// Load Dashboard Data
// ==========================================
async function loadDashboardData() {
    try {
        // In a real implementation, fetch data from API
        // For now, use mock data
        updateTodayStats();
        updateFamilyActivity();
    } catch (error) {
        console.error('Failed to load dashboard data:', error);
    }
}

function updateTodayStats() {
    // Mock data - in production, fetch from API
    const stats = {
        meals: 3,
        mealsChange: '+2 vs avg',
        sleep: '7.5h',
        sleepChange: '+0.5h',
        activities: 2,
        activitiesChange: 'Same as avg',
        calories: 1800,
        caloriesChange: '+200 kcal'
    };
    
    document.getElementById('today-meals').textContent = stats.meals;
    document.getElementById('meals-change').textContent = stats.mealsChange;
    document.getElementById('today-sleep').textContent = stats.sleep;
    document.getElementById('sleep-change').textContent = stats.sleepChange;
    document.getElementById('today-activities').textContent = stats.activities;
    document.getElementById('activities-change').textContent = stats.activitiesChange;
    document.getElementById('today-calories').textContent = stats.calories;
    document.getElementById('calories-change').textContent = stats.caloriesChange;
}

function updateFamilyActivity() {
    // Mock data - in production, fetch from API
    const leoActivity = document.getElementById('leo-last-activity');
    const jordanActivity = document.getElementById('jordan-last-activity');
    
    if (leoActivity) {
        leoActivity.textContent = 'Last meal: Lunch at 12:30 PM';
    }
    
    if (jordanActivity) {
        jordanActivity.textContent = 'Last activity: Morning walk at 7:00 AM';
    }
}

// ==========================================
// Copy Endpoint Function (for Shortcuts tab)
// ==========================================
window.copyEndpoint = function(endpoint) {
    const baseUrl = document.getElementById('base-url').value;
    const fullUrl = `${baseUrl}${endpoint}`;
    
    navigator.clipboard.writeText(fullUrl).then(() => {
        app.showToast('Endpoint copied to clipboard!', 'success');
    }).catch(err => {
        console.error('Failed to copy:', err);
        app.showToast('Failed to copy endpoint', 'error');
    });
};

// Copy base URL
const copyUrlBtn = document.getElementById('copy-url-btn');
if (copyUrlBtn) {
    copyUrlBtn.addEventListener('click', () => {
        const baseUrl = document.getElementById('base-url').value;
        navigator.clipboard.writeText(baseUrl).then(() => {
            app.showToast('URL copied to clipboard!', 'success');
        }).catch(err => {
            app.showToast('Failed to copy URL', 'error');
        });
    });
}
