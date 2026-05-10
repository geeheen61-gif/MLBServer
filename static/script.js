document.addEventListener('DOMContentLoaded', () => {
    // Tab Switching Logic
    const navItems = document.querySelectorAll('.nav-item');
    const panels = document.querySelectorAll('.panel');

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const tabId = item.getAttribute('data-tab');
            
            // Update Navigation State
            navItems.forEach(i => i.classList.remove('active'));
            item.classList.add('active');
            
            // Update Panel State
            panels.forEach(p => p.classList.remove('active'));
            const targetPanel = document.getElementById(tabId);
            if (targetPanel) {
                targetPanel.classList.add('active');
            }
            
            // Sync Color Theme
            const themeColor = item.style.getPropertyValue('--tab-color');
            document.documentElement.style.setProperty('--accent', themeColor);
        });
    });

    // API Interaction Handler
    const loader = document.getElementById('loader');

    async function callEngine(endpoint, payload) {
        loader.classList.remove('hidden');
        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            if (!response.ok) throw new Error(`HTTP Error: ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error('Engine Connection Error:', error);
            showNotification('CRITICAL: FAILED TO CONNECT TO AI CORE', 'error');
        } finally {
            loader.classList.add('hidden');
        }
    }

    function showNotification(msg, type) {
        // Simple alert for now, could be a toast in the future
        alert(msg);
    }

    // Dynamic Result Rendering
    function injectAnalytics(containerId, data, themeColor) {
        const container = document.getElementById(containerId);
        container.classList.remove('hidden');
        
        // 1. Build Metric Grid
        let metricsMarkup = '';
        if (data.metrics) {
            metricsMarkup = `<div class="metric-grid">`;
            for (const [key, val] of Object.entries(data.metrics)) {
                metricsMarkup += `
                    <div class="metric-card" style="border-left: 3px solid ${themeColor}">
                        <span class="metric-label">${key.replace(/_/g, ' ')}</span>
                        <span class="metric-value" style="color: ${themeColor}">${val}</span>
                    </div>
                `;
            }
            metricsMarkup += `</div>`;
        }

        // 2. Parse Intelligence Report
        const rawSummary = data.summary || "Inconclusive data. Please verify inputs.";
        let analysisBody = rawSummary;
        let predictionMarkup = "";

        // Standardized Delimiter Support
        const separator = "### OVERALL MATCH PREDICTION ###";
        if (rawSummary.includes(separator)) {
            const fragments = rawSummary.split(separator);
            analysisBody = fragments[0].trim();
            predictionMarkup = `
                <div class="prediction-footer">
                    <div class="prediction-header">
                        <i class="fas fa-diamond-turn-right"></i> FINAL PROBABILITY VERDICT
                    </div>
                    <div class="prediction-content">${fragments[1].trim()}</div>
                </div>
            `;
        }

        // 3. Inject into DOM
        container.innerHTML = `
            ${metricsMarkup}
            <div class="report-card">
                <div class="report-title">INTELLIGENCE REPORT</div>
                <div class="report-text">${analysisBody.replace(/\n/g, '<br>')}</div>
            </div>
            ${predictionMarkup}
        `;
        
        // Smooth entry
        container.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // Action Triggers
    document.getElementById('analyze-batsman').addEventListener('click', async () => {
        const payload = {
            player_name: document.getElementById('batter-name').value,
            sportsbook_odds: parseFloat(document.getElementById('odds').value),
            pitcher_hr9: parseFloat(document.getElementById('hr9').value),
            park_factor: parseFloat(document.getElementById('park-factor').value),
            is_home: document.getElementById('is-home').checked
        };
        const results = await callEngine('/predict_batter', payload);
        if (results && results.player_info) {
             document.getElementById('stadium-name').value = results.player_info.current_stadium;
             document.getElementById('park-factor').value = results.player_info.park_factor;
        }
        if (results) injectAnalytics('batsman-results', results, '#6366F1');
    });

    document.getElementById('analyze-pitcher').addEventListener('click', async () => {
        const payload = {
            pitcher_name: document.getElementById('pitcher-name').value
        };
        const results = await callEngine('/predict_pitcher', payload);
        if (results) injectAnalytics('pitcher-results', results, '#F43F5E');
    });

    document.getElementById('analyze-ballpark').addEventListener('click', async () => {
        const payload = {
            stadium_name: document.getElementById('stadium-name').value
        };
        const results = await callEngine('/park_history', payload);
        if (results && results.metrics && results.metrics.hr_factor) {
            document.getElementById('park-factor').value = results.metrics.hr_factor;
        }
        if (results) injectAnalytics('ballpark-results', results, '#F59E0B');
    });
});
