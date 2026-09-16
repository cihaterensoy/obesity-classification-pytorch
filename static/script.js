document.addEventListener('DOMContentLoaded', () => {
    // --- Slider value displays ---
    const sliders = [
        { id: 'age', valId: 'age-val', suffix: '' },
        { id: 'height', valId: 'height-val', suffix: ' cm' },
        { id: 'weight', valId: 'weight-val', suffix: ' kg' },
        { id: 'fcvc', valId: 'fcvc-val', map: (v) => v <= 1.4 ? 'Az (1)' : v <= 2.4 ? 'Orta (2)' : 'Çok (3)' },
        { id: 'ncp', valId: 'ncp-val', suffix: ' Öğün' },
        { id: 'ch2o', valId: 'ch2o-val', suffix: ' L' },
        { id: 'faf', valId: 'faf-val', map: (v) => v == 0 ? 'Hiç (0 Gün)' : v <= 1.5 ? 'Haftada 1-2 Gün' : 'Haftada 3+ Gün' },
        { id: 'tue', valId: 'tue-val', suffix: ' Saat' }
    ];

    sliders.forEach(item => {
        const input = document.getElementById(item.id);
        const display = document.getElementById(item.valId);
        
        if (input && display) {
            const updateVal = () => {
                if (item.map) {
                    display.textContent = item.map(parseFloat(input.value));
                } else {
                    display.textContent = input.value + (item.suffix || '');
                }
            };
            input.addEventListener('input', updateVal);
            updateVal();
        }
    });

    // --- Form Submission ---
    const form = document.getElementById('prediction-form');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = submitBtn.querySelector('.btn-text');
    const btnLoader = submitBtn.querySelector('.btn-loader');
    const resultsCard = document.getElementById('results-card');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Show loading state
        submitBtn.disabled = true;
        btnText.style.display = 'none';
        btnLoader.style.display = 'inline-block';

        const formData = new FormData(form);
        const payload = {};

        formData.forEach((value, key) => {
            // Convert numeric values
            if (['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE'].includes(key)) {
                payload[key] = parseFloat(value);
            } else {
                payload[key] = value;
            }
        });

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (data.success) {
                renderResults(data);
            } else {
                alert('Tahmin hesaplanırken bir hata oluştu: ' + (data.error || 'Bilinmeyen hata'));
            }
        } catch (error) {
            console.error('Fetch error:', error);
            alert('Sunucu ile iletişim kurulamadı. Lütfen sunucunun çalıştığından emin olun.');
        } finally {
            submitBtn.disabled = false;
            btnText.style.display = 'inline-block';
            btnLoader.style.display = 'none';
        }
    });

    function renderResults(data) {
        // Show results card
        resultsCard.style.display = 'block';
        resultsCard.scrollIntoView({ behavior: 'smooth' });

        // Hero info
        document.getElementById('predicted-badge').textContent = '✅ Tahmin Başarılı';
        document.getElementById('predicted-class-tr').textContent = data.predicted_class_tr;
        document.getElementById('predicted-description').textContent = data.description;

        // BMI
        const bmiValEl = document.getElementById('bmi-value');
        const bmiStatusEl = document.getElementById('bmi-status');
        
        bmiValEl.textContent = data.bmi;
        if (data.bmi < 18.5) {
            bmiStatusEl.textContent = 'Zayıf (< 18.5)';
            bmiValEl.style.color = '#38bdf8';
        } else if (data.bmi < 25) {
            bmiStatusEl.textContent = 'Normal (18.5 - 24.9)';
            bmiValEl.style.color = '#34d399';
        } else if (data.bmi < 30) {
            bmiStatusEl.textContent = 'Fazla Kilolu (25 - 29.9)';
            bmiValEl.style.color = '#fbbf24';
        } else {
            bmiStatusEl.textContent = 'Obezite (≥ 30)';
            bmiValEl.style.color = '#f43f5e';
        }

        // Probabilities
        const container = document.getElementById('probabilities-container');
        container.innerHTML = '';

        data.probabilities.forEach((item, index) => {
            const probItem = document.createElement('div');
            probItem.className = 'prob-item';

            const isTop = index === 0;
            const pctColor = isTop ? '#818cf8' : '#9ca3af';

            probItem.innerHTML = `
                <div class="prob-info">
                    <span class="prob-name">${isTop ? '🏆 ' : ''}${item.class_tr}</span>
                    <span class="prob-pct" style="color: ${pctColor}">${item.probability}%</span>
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width: 0%; ${isTop ? 'background: linear-gradient(135deg, #6366f1, #ec4899);' : 'background: rgba(255,255,255,0.2);'}"></div>
                </div>
            `;

            container.appendChild(probItem);

            // Animate bar width
            setTimeout(() => {
                const fill = probItem.querySelector('.prob-bar-fill');
                fill.style.width = item.probability + '%';
            }, 50 * index + 100);
        });

        // Insights
        const insightsContainer = document.getElementById('insights-container');
        insightsContainer.innerHTML = '';

        if (data.insights && data.insights.length > 0) {
            data.insights.forEach(insightText => {
                const insightCard = document.createElement('div');
                insightCard.className = 'insight-card';
                insightCard.textContent = insightText;
                insightsContainer.appendChild(insightCard);
            });
        }
    }
});
