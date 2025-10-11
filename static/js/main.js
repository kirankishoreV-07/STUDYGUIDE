// StudyHub Main JavaScript

// Global variables
let apiStatus = {};
let currentUser = null;

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    checkApiStatus();
    setupEventListeners();
    
    // Check API status every 30 seconds
    setInterval(checkApiStatus, 30000);
    
    console.log('StudyHub initialized successfully');
});

/**
 * Initialize the application
 */
function initializeApp() {
    // Set up CSRF token for AJAX requests if needed
    const csrfToken = document.querySelector('meta[name="csrf-token"]');
    if (csrfToken) {
        window.csrfToken = csrfToken.getAttribute('content');
    }
    
    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize Bootstrap popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Add fade-in animation to main content
    const mainContent = document.querySelector('.main-content');
    if (mainContent) {
        mainContent.classList.add('fade-in-up');
    }
}

/**
 * Set up global event listeners
 */
function setupEventListeners() {
    // Handle form submissions
    document.addEventListener('submit', function(e) {
        const form = e.target;
        if (form.classList.contains('prevent-double-submit')) {
            const submitButton = form.querySelector('button[type="submit"]');
            if (submitButton) {
                setTimeout(() => {
                    submitButton.disabled = true;
                    submitButton.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
                }, 100);
            }
        }
    });
    
    // Handle navigation clicks
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('nav-link') || e.target.closest('.nav-link')) {
            // Add loading state to navigation
            const navLink = e.target.classList.contains('nav-link') ? e.target : e.target.closest('.nav-link');
            if (!navLink.classList.contains('dropdown-toggle')) {
                addLoadingState(navLink);
            }
        }
    });
    
    // Handle keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Escape key to close modals
        if (e.key === 'Escape') {
            const modals = document.querySelectorAll('.modal.show');
            modals.forEach(modal => {
                const modalInstance = bootstrap.Modal.getInstance(modal);
                if (modalInstance) {
                    modalInstance.hide();
                }
            });
        }
        
        // Ctrl/Cmd + K for search (future feature)
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            // Future: Open search modal
        }
    });
}

/**
 * Check API status
 */
async function checkApiStatus() {
    try {
        const response = await fetch('/api/status');
        apiStatus = await response.json();
        updateApiStatusIndicators();
    } catch (error) {
        console.error('Failed to check API status:', error);
        apiStatus = { error: true };
        updateApiStatusIndicators();
    }
}

/**
 * Update API status indicators
 */
function updateApiStatusIndicators() {
    const statusElements = document.querySelectorAll('.api-status-indicator');
    statusElements.forEach(element => {
        if (apiStatus.error) {
            element.className = 'api-status-indicator status-offline';
            element.title = 'API services unavailable';
        } else if (apiStatus.gemini_api && apiStatus.ollama_available) {
            element.className = 'api-status-indicator status-online';
            element.title = 'All services online';
        } else {
            element.className = 'api-status-indicator status-checking';
            element.title = 'Some services unavailable';
        }
    });
}

/**
 * Show API status modal
 */
function showApiStatus() {
    const modal = document.getElementById('apiStatusModal');
    const content = document.getElementById('apiStatusContent');
    
    if (apiStatus.error) {
        content.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-circle me-2"></i>
                <strong>Unable to check API status</strong>
                <p class="mb-0">Please check your connection and try again.</p>
            </div>
        `;
    } else {
        content.innerHTML = `
            <div class="row">
                <div class="col-md-6">
                    <div class="service-status">
                        <div class="d-flex align-items-center mb-2">
                            <div class="status-indicator ${apiStatus.gemini_api ? 'status-online' : 'status-offline'} me-2"></div>
                            <strong>Gemini AI</strong>
                        </div>
                        <p class="small text-muted mb-3">
                            ${apiStatus.gemini_api ? 'Connected and ready for summarization and quiz generation' : 'Not available - check API key configuration'}
                        </p>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="service-status">
                        <div class="d-flex align-items-center mb-2">
                            <div class="status-indicator ${apiStatus.ollama_available ? 'status-online' : 'status-offline'} me-2"></div>
                            <strong>Ollama LLM</strong>
                        </div>
                        <p class="small text-muted mb-3">
                            ${apiStatus.ollama_available ? 'Connected and ready for PDF chat' : 'Not available - check Ollama installation'}
                        </p>
                    </div>
                </div>
            </div>
            <div class="row">
                <div class="col-md-6">
                    <div class="service-status">
                        <div class="d-flex align-items-center mb-2">
                            <div class="status-indicator ${apiStatus.embeddings_loaded ? 'status-online' : 'status-checking'} me-2"></div>
                            <strong>Embeddings Model</strong>
                        </div>
                        <p class="small text-muted mb-3">
                            ${apiStatus.embeddings_loaded ? 'Loaded and ready' : 'Loading on demand'}
                        </p>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="service-status">
                        <div class="d-flex align-items-center mb-2">
                            <div class="status-indicator status-online me-2"></div>
                            <strong>Web Server</strong>
                        </div>
                        <p class="small text-muted mb-3">Flask server running on localhost:5000</p>
                    </div>
                </div>
            </div>
            <hr>
            <div class="system-info">
                <h6>System Information</h6>
                <ul class="small mb-0">
                    <li><strong>Last Updated:</strong> ${apiStatus.timestamp ? new Date(apiStatus.timestamp).toLocaleString() : 'Unknown'}</li>
                    <li><strong>Session Active:</strong> Yes</li>
                    <li><strong>Storage:</strong> Local filesystem</li>
                </ul>
            </div>
        `;
    }
    
    const modalInstance = new bootstrap.Modal(modal);
    modalInstance.show();
}

/**
 * Show help modal
 */
function showHelp() {
    const modal = document.getElementById('helpModal');
    const modalInstance = new bootstrap.Modal(modal);
    modalInstance.show();
}

/**
 * Show about modal
 */
function showAbout() {
    const modal = document.getElementById('aboutModal');
    const modalInstance = new bootstrap.Modal(modal);
    modalInstance.show();
}

/**
 * Add loading state to element
 */
function addLoadingState(element) {
    element.classList.add('loading');
    const originalText = element.innerHTML;
    element.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Loading...';
    
    setTimeout(() => {
        element.classList.remove('loading');
        element.innerHTML = originalText;
    }, 1000);
}

/**
 * Show notification toast
 */
function showToast(message, type = 'info', duration = 5000) {
    const toastContainer = getOrCreateToastContainer();
    
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white bg-${type} border-0`;
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'assertive');
    toast.setAttribute('aria-atomic', 'true');
    
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                <i class="fas fa-${getIconForType(type)} me-2"></i>
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;
    
    toastContainer.appendChild(toast);
    
    const bsToast = new bootstrap.Toast(toast, { delay: duration });
    bsToast.show();
    
    // Remove from DOM after hiding
    toast.addEventListener('hidden.bs.toast', () => {
        toast.remove();
    });
}

/**
 * Get or create toast container
 */
function getOrCreateToastContainer() {
    let container = document.querySelector('.toast-container');
    if (!container) {
        container = document.createElement('div');
        container.className = 'toast-container position-fixed top-0 end-0 p-3';
        container.style.zIndex = '9999';
        document.body.appendChild(container);
    }
    return container;
}

/**
 * Get icon for toast type
 */
function getIconForType(type) {
    switch (type) {
        case 'success': return 'check-circle';
        case 'danger': return 'exclamation-circle';
        case 'warning': return 'exclamation-triangle';
        case 'info': return 'info-circle';
        default: return 'info-circle';
    }
}

/**
 * Format file size
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Validate file types
 */
function validateFileType(file, allowedTypes) {
    const fileExtension = file.name.split('.').pop().toLowerCase();
    return allowedTypes.includes(fileExtension);
}

/**
 * Debounce function
 */
function debounce(func, wait, immediate) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            timeout = null;
            if (!immediate) func(...args);
        };
        const callNow = immediate && !timeout;
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
        if (callNow) func(...args);
    };
}

/**
 * Throttle function
 */
function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

/**
 * Copy text to clipboard
 */
async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        showToast('Copied to clipboard!', 'success', 2000);
        return true;
    } catch (err) {
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        textArea.style.top = '-999999px';
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        
        try {
            document.execCommand('copy');
            showToast('Copied to clipboard!', 'success', 2000);
            return true;
        } catch (err) {
            showToast('Failed to copy to clipboard', 'danger');
            return false;
        } finally {
            document.body.removeChild(textArea);
        }
    }
}

/**
 * Download text as file
 */
function downloadTextFile(text, filename) {
    const blob = new Blob([text], { type: 'text/plain' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
    showToast(`Downloaded ${filename}`, 'success');
}

/**
 * Handle AJAX errors
 */
function handleAjaxError(error, context = '') {
    console.error(`AJAX Error ${context}:`, error);
    
    let message = 'An error occurred. Please try again.';
    
    if (error.response) {
        switch (error.response.status) {
            case 400:
                message = 'Invalid request. Please check your input.';
                break;
            case 401:
                message = 'Authentication required. Please refresh the page.';
                break;
            case 403:
                message = 'Access denied. You do not have permission.';
                break;
            case 404:
                message = 'The requested resource was not found.';
                break;
            case 429:
                message = 'Too many requests. Please wait and try again.';
                break;
            case 500:
                message = 'Server error. Please try again later.';
                break;
            case 503:
                message = 'Service temporarily unavailable.';
                break;
        }
    } else if (!navigator.onLine) {
        message = 'No internet connection. Please check your network.';
    }
    
    showToast(message, 'danger');
}

/**
 * Make API request with error handling
 */
async function apiRequest(url, options = {}) {
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
        },
    };
    
    // Add CSRF token if available
    if (window.csrfToken) {
        defaultOptions.headers['X-CSRFToken'] = window.csrfToken;
    }
    
    const finalOptions = { ...defaultOptions, ...options };
    
    try {
        const response = await fetch(url, finalOptions);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
            return await response.json();
        } else {
            return await response.text();
        }
    } catch (error) {
        handleAjaxError(error, `requesting ${url}`);
        throw error;
    }
}

/**
 * Session management
 */
const Session = {
    get: (key) => {
        try {
            return JSON.parse(sessionStorage.getItem(key));
        } catch {
            return sessionStorage.getItem(key);
        }
    },
    
    set: (key, value) => {
        const stringValue = typeof value === 'string' ? value : JSON.stringify(value);
        sessionStorage.setItem(key, stringValue);
    },
    
    remove: (key) => {
        sessionStorage.removeItem(key);
    },
    
    clear: () => {
        sessionStorage.clear();
    }
};

/**
 * Local storage management
 */
const Storage = {
    get: (key) => {
        try {
            return JSON.parse(localStorage.getItem(key));
        } catch {
            return localStorage.getItem(key);
        }
    },
    
    set: (key, value) => {
        const stringValue = typeof value === 'string' ? value : JSON.stringify(value);
        localStorage.setItem(key, stringValue);
    },
    
    remove: (key) => {
        localStorage.removeItem(key);
    },
    
    clear: () => {
        localStorage.clear();
    }
};

// Export functions for use in other scripts
window.StudyHub = {
    showToast,
    copyToClipboard,
    downloadTextFile,
    apiRequest,
    Session,
    Storage,
    formatFileSize,
    validateFileType,
    debounce,
    throttle
};

// Console welcome message
console.log(`
    ╔══════════════════════════════════════╗
    ║            StudyHub v1.0             ║
    ║        AI-Powered Learning           ║
    ║                                      ║
    ║  🚀 Successfully Initialized         ║
    ║  📚 PDF Chat Ready                   ║
    ║  🎬 Summarizer Ready                 ║
    ║  ❓ Quiz Generator Ready             ║
    ╚══════════════════════════════════════╝
    
    Available features:
    • StudyHub.showToast(message, type)
    • StudyHub.copyToClipboard(text)
    • StudyHub.downloadTextFile(text, filename)
    • StudyHub.apiRequest(url, options)
    • StudyHub.Session / StudyHub.Storage
`);