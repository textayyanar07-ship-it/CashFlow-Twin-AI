// Placeholder URL - Replace with the actual deployed URL of your Streamlit app
const STREAMLIT_URL = "http://localhost:8501"; 

const loadingScreen = document.getElementById('loading-screen');
const errorScreen = document.getElementById('error-screen');
const appFrame = document.getElementById('app-frame');
const retryBtn = document.getElementById('retry-btn');
const statusText = document.querySelector('.status-text');

let isAppLoaded = false;

// Initialize Capacitor Plugins (Using window.Capacitor)
const initApp = async () => {
    // Hide Capacitor native splash screen once our web wrapper is ready
    if (window.Capacitor && window.Capacitor.Plugins.SplashScreen) {
        try {
            await window.Capacitor.Plugins.SplashScreen.hide();
        } catch (e) {
            console.warn("Splash screen plugin not available");
        }
    }
    
    // Register network status listener
    if (window.Capacitor && window.Capacitor.Plugins.Network) {
        window.Capacitor.Plugins.Network.addListener('networkStatusChange', status => {
            console.log('Network status changed', status);
            if (!status.connected && !isAppLoaded) {
                showErrorScreen();
            } else if (status.connected && !isAppLoaded) {
                loadApp();
            }
        });
    }

    loadApp();
};

const checkConnection = async () => {
    if (window.Capacitor && window.Capacitor.Plugins.Network) {
        const status = await window.Capacitor.Plugins.Network.getStatus();
        return status.connected;
    }
    // Fallback for browser testing
    return navigator.onLine;
};

const loadApp = async () => {
    showLoadingScreen("Checking connection...");
    
    const isConnected = await checkConnection();
    
    if (!isConnected) {
        showErrorScreen();
        return;
    }

    showLoadingScreen("Loading CashFlow Twin AI...");
    
    // Attempt to load the iframe
    appFrame.src = STREAMLIT_URL;
    
    appFrame.onload = () => {
        isAppLoaded = true;
        hideScreens();
    };

    appFrame.onerror = () => {
        showErrorScreen();
    };
    
    // Fallback timeout in case onload doesn't fire correctly for some cross-origin issues
    setTimeout(() => {
        if (!isAppLoaded) {
            // Assume it loaded or give control to user
            isAppLoaded = true;
            hideScreens();
        }
    }, 8000); 
};

const showLoadingScreen = (text) => {
    if(statusText) statusText.innerText = text;
    errorScreen.classList.remove('active');
    loadingScreen.classList.add('active');
};

const showErrorScreen = () => {
    loadingScreen.classList.remove('active');
    errorScreen.classList.add('active');
};

const hideScreens = () => {
    loadingScreen.classList.remove('active');
    errorScreen.classList.remove('active');
};

// Retry button event listener
retryBtn.addEventListener('click', () => {
    loadApp();
});

// App Plugin - Handle Android Back Button
if (window.Capacitor && window.Capacitor.Plugins.App) {
    window.Capacitor.Plugins.App.addListener('backButton', () => {
        // Simple implementation: try to go back in iframe history, or exit app
        if(isAppLoaded) {
           // We can't easily read iframe history due to CORS, so we either do nothing
           // or minimize the app. Let's minimize the app.
           window.Capacitor.Plugins.App.minimizeApp();
        } else {
           window.Capacitor.Plugins.App.exitApp();
        }
    });
}

// Start app
document.addEventListener("DOMContentLoaded", () => {
    initApp();
});
