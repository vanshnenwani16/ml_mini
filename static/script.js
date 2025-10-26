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
});