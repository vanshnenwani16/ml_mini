document.addEventListener('DOMContentLoaded', function() {
    const validateForm = document.getElementById('validateForm');
    const trainForm = document.getElementById('trainForm');
    const predictForm = document.getElementById('predictForm');
    const targetColumnSelect = document.getElementById('targetColumn');
    let currentFile = null;

    // Validate Dataset
    validateForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        const fileInput = document.getElementById('datasetFile');
        const file = fileInput.files[0];
        if (!file) return;

        currentFile = file;
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/validate', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            
            const resultsDiv = document.getElementById('validationResults');
            if (data.status === 'success') {
                resultsDiv.className = 'alert alert-success';
                resultsDiv.innerHTML = `
                    <h6>Dataset Validation Successful!</h6>
                    <p>Rows: ${data.info.rows}</p>
                    <p>Columns: ${data.info.columns}</p>
                `;
                
                // Populate target column dropdown
                targetColumnSelect.innerHTML = '<option value="">Select target column</option>';
                data.info.features.forEach(feature => {
                    const option = document.createElement('option');
                    option.value = feature;
                    option.textContent = feature;
                    targetColumnSelect.appendChild(option);
                });
                
                targetColumnSelect.disabled = false;
                trainForm.querySelector('button').disabled = false;
            } else {
                resultsDiv.className = 'alert alert-danger';
                resultsDiv.textContent = data.message;
            }
            resultsDiv.classList.remove('d-none');
        } catch (error) {
            console.error('Error:', error);
        }
    });

    // Train Model
    trainForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        if (!currentFile) return;

        const formData = new FormData();
        formData.append('file', currentFile);
        formData.append('target_column', targetColumnSelect.value);

        try {
            const response = await fetch('/api/train', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            
            const resultsDiv = document.getElementById('trainingResults');
            if (data.status === 'success') {
                resultsDiv.className = 'alert alert-success';
                resultsDiv.textContent = data.message;
                
                // Enable prediction section
                document.getElementById('predictionFile').disabled = false;
                predictForm.querySelector('button').disabled = false;
                
                // Store the features for validation
                window.trainedFeatures = data.features;
            } else {
                resultsDiv.className = 'alert alert-danger';
                resultsDiv.textContent = data.message;
            }
            resultsDiv.classList.remove('d-none');
        } catch (error) {
            console.error('Error:', error);
        }
    });

    // Validate prediction file when selected
    document.getElementById('predictionFile').addEventListener('change', async function(e) {
        const file = e.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/validate', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            
            if (data.status === 'success' && window.trainedFeatures) {
                // Check if all required features are present
                const missingFeatures = window.trainedFeatures.filter(
                    feature => !data.info.features.includes(feature)
                );
                
                if (missingFeatures.length > 0) {
                    const resultsDiv = document.getElementById('predictionResults');
                    resultsDiv.className = 'alert alert-danger';
                    resultsDiv.textContent = `Missing required features: ${missingFeatures.join(', ')}`;
                    resultsDiv.classList.remove('d-none');
                    predictForm.querySelector('button').disabled = true;
                } else {
                    predictForm.querySelector('button').disabled = false;
                    const resultsDiv = document.getElementById('predictionResults');
                    resultsDiv.className = 'alert alert-success';
                    resultsDiv.textContent = 'Test file validated successfully. Ready for predictions.';
                    resultsDiv.classList.remove('d-none');
                }
            }
        } catch (error) {
            console.error('Error:', error);
        }
    });

    // Make Prediction
    predictForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const fileInput = document.getElementById('predictionFile');
        const file = fileInput.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            
            const resultsDiv = document.getElementById('predictionResults');
            const tableDiv = document.getElementById('predictionsTable');
            
            if (data.status === 'success') {
                resultsDiv.className = 'alert alert-success';
                resultsDiv.innerHTML = `
                    <h6>Prediction Summary:</h6>
                    <p>Total Predictions: ${data.summary.count}</p>
                    <p>Mean: ${data.summary.mean.toFixed(4)}</p>
                    <p>Range: ${data.summary.min.toFixed(4)} to ${data.summary.max.toFixed(4)}</p>
                `;

                // Create table with predictions
                let tableHTML = `
                    <div class="table-responsive">
                        <table class="table table-striped table-hover">
                            <thead>
                                <tr>
                                    ${Object.keys(data.predictions[0]).map(key => 
                                        `<th>${key}</th>`).join('')}
                                </tr>
                            </thead>
                            <tbody>
                                ${data.predictions.map(row => `
                                    <tr>
                                        ${Object.values(row).map(value => 
                                            `<td>${typeof value === 'number' ? value.toFixed(4) : value}</td>`
                                        ).join('')}
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                `;
                tableDiv.innerHTML = tableHTML;
            } else {
                resultsDiv.className = 'alert alert-danger';
                resultsDiv.textContent = data.message;
                tableDiv.innerHTML = '';
            }
            resultsDiv.classList.remove('d-none');
        } catch (error) {
            console.error('Error:', error);
        }
    });

    // Analyze Dataset -> Render multiple plots using Plotly
    const analyzeForm = document.getElementById('analyzeForm');
    const analysisResults = document.getElementById('analysisResults');
    analyzeForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        const fileInput = document.getElementById('analyzeFile');
        const file = fileInput.files[0] || currentFile;
        if (!file) {
            alert('Please upload or validate a dataset first to analyze.');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/analyze', { method: 'POST', body: formData });
            const data = await response.json();
            if (data.status !== 'success') {
                analysisResults.innerHTML = `<div class="alert alert-danger">${data.message}</div>`;
                return;
            }

            // Clear previous results
            analysisResults.innerHTML = '';

            // Helper to create plot container
            function makePlotContainer(title) {
                const wrapper = document.createElement('div');
                wrapper.className = 'mb-4';
                const h = document.createElement('h6');
                h.textContent = title;
                wrapper.appendChild(h);
                const div = document.createElement('div');
                div.style.width = '100%';
                div.style.height = '400px';
                wrapper.appendChild(div);
                analysisResults.appendChild(wrapper);
                return div;
            }

            // Histograms
            if (data.histograms && Object.keys(data.histograms).length > 0) {
                for (const col of Object.keys(data.histograms)) {
                    const hist = data.histograms[col];
                    const bins = hist.bins;
                    const counts = hist.counts;
                    // compute bin centers
                    const centers = [];
                    for (let i = 0; i < bins.length - 1; i++) centers.push((bins[i] + bins[i+1]) / 2);
                    const container = makePlotContainer(`Histogram: ${col}`);
                    Plotly.newPlot(container, [{ x: centers, y: counts, type: 'bar', marker: { color: '#4e79a7' } }], {margin: {t:30}});
                }
            }

            // Boxplot stats (display as small stat cards)
            if (data.boxplots && Object.keys(data.boxplots).length > 0) {
                const statsWrapper = document.createElement('div');
                statsWrapper.className = 'row mb-4';
                for (const col of Object.keys(data.boxplots)) {
                    const b = data.boxplots[col];
                    const card = document.createElement('div');
                    card.className = 'col-md-4 mb-3';
                    card.innerHTML = `
                        <div class="card p-2">
                            <div class="card-body">
                                <h6 class="card-title">Box: ${col}</h6>
                                <p class="card-text mb-1">min: ${b.min.toFixed(4)}</p>
                                <p class="card-text mb-1">Q1: ${b.q1.toFixed(4)}</p>
                                <p class="card-text mb-1">median: ${b.median.toFixed(4)}</p>
                                <p class="card-text mb-1">Q3: ${b.q3.toFixed(4)}</p>
                                <p class="card-text mb-0">max: ${b.max.toFixed(4)}</p>
                            </div>
                        </div>
                    `;
                    statsWrapper.appendChild(card);
                }
                analysisResults.appendChild(statsWrapper);
            }

            // Correlation heatmap
            if (data.correlation && data.correlation.matrix) {
                const cols = data.correlation.columns;
                const z = data.correlation.matrix;
                const container = makePlotContainer('Correlation Heatmap');
                Plotly.newPlot(container, [{ z: z, x: cols, y: cols, type: 'heatmap', colorscale: 'RdBu', reversescale: true }], {margin:{t:30}});
            }

            // Top correlated pairs -> scatter + regression line
            if (data.pair_samples && data.pair_samples.length > 0) {
                for (const ps of data.pair_samples) {
                    const x = ps.x;
                    const y = ps.y;
                    const container = makePlotContainer(`Scatter: ${ps.x_name} vs ${ps.y_name}`);
                    // compute linear regression (least squares)
                    let slope = 0, intercept = 0;
                    if (x.length > 1) {
                        const n = x.length;
                        const mx = x.reduce((a,b)=>a+b,0)/n;
                        const my = y.reduce((a,b)=>a+b,0)/n;
                        let num = 0, den = 0;
                        for (let i=0;i<n;i++){ num += (x[i]-mx)*(y[i]-my); den += (x[i]-mx)*(x[i]-mx); }
                        slope = den === 0 ? 0 : num/den;
                        intercept = my - slope*mx;
                    }
                    const tracePoints = { x: x, y: y, mode: 'markers', type: 'scatter', marker: { size: 6 }, name: 'data' };
                    // regression line
                    const lineX = [Math.min(...x), Math.max(...x)];
                    const lineY = lineX.map(v => slope * v + intercept);
                    const traceLine = { x: lineX, y: lineY, mode: 'lines', line: { color: '#ff7f0e' }, name: `fit (slope=${slope.toFixed(3)})` };
                    Plotly.newPlot(container, [tracePoints, traceLine], {margin:{t:30}});
                }
            }

            // Categorical count plots
            if (data.categorical_counts && Object.keys(data.categorical_counts).length > 0) {
                for (const col of Object.keys(data.categorical_counts)) {
                    const cat = data.categorical_counts[col];
                    const container = makePlotContainer(`Categorical Counts: ${col}`);
                    Plotly.newPlot(container, [{ x: cat.labels, y: cat.counts, type: 'bar', marker: { color: '#59a14f' } }], {margin:{t:30}});
                }
            }

            // Time series plot (first numeric series)
            if (data.time_series) {
                const ts = data.time_series;
                const index = ts.index;
                const values = ts.values;
                // choose first numeric column to plot
                const keys = Object.keys(values);
                if (keys.length > 0) {
                    const first = keys[0];
                    const container = makePlotContainer(`Time Series (${ts.datetime_col}): ${first}`);
                    Plotly.newPlot(container, [{ x: index, y: values[first], mode: 'lines+markers', line: { color: '#4e79a7' } }], {margin:{t:30}});
                }
            }

        } catch (error) {
            console.error('Analyze error:', error);
            analysisResults.innerHTML = `<div class="alert alert-danger">Analysis failed: ${error}</div>`;
        }
    });
});