document.addEventListener('DOMContentLoaded', async () => {
    const predictionForm = document.getElementById('predictionForm');
    const districtSelect = document.getElementById('district');
    const areaSelect = document.getElementById('area');
    const waterSelect = document.getElementById('water_supply');
    const electricitySelect = document.getElementById('electricity');
    const predictBtn = document.getElementById('predictBtn');
    const btnText = predictBtn.querySelector('.btn-text');
    const loader = predictBtn.querySelector('.loader-dots');
    const resultCard = document.getElementById('resultCard');
    const predictedPriceElem = document.getElementById('predictedPrice');
    const resetBtn = document.getElementById('resetBtn');

    let areasByDistrict = {};

    try {
        const response = await fetch('/options');
        const options = await response.json();

        options.districts.forEach(d => {
            const opt = document.createElement('option');
            opt.value = d;
            opt.textContent = d;
            districtSelect.appendChild(opt);
        });

        areasByDistrict = options.areas_by_district;
        options.water_supply.forEach(w => {
            const opt = document.createElement('option');
            opt.value = w;
            opt.textContent = w;
            waterSelect.appendChild(opt);
        });

        options.electricity.forEach(e => {
            const opt = document.createElement('option');
            opt.value = e;
            opt.textContent = e;
            electricitySelect.appendChild(opt);
        });

    } catch (err) {
        console.error('Error fetching options:', err);
    }

    districtSelect.addEventListener('change', () => {
        const selectedDistrict = districtSelect.value;
        areaSelect.innerHTML = '<option value="" disabled selected>Select Area</option>';

        if (selectedDistrict && areasByDistrict[selectedDistrict]) {
            areasByDistrict[selectedDistrict].forEach(a => {
                const opt = document.createElement('option');
                opt.value = a;
                opt.textContent = a;
                areaSelect.appendChild(opt);
            });
            areaSelect.disabled = false;
        } else {
            areaSelect.disabled = true;
        }
    });

    predictionForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        btnText.style.display = 'none';
        loader.style.display = 'flex';
        predictBtn.disabled = true;

        const formData = new FormData(predictionForm);
        const data = {
            district: formData.get('district'),
            area: formData.get('area'),
            perch: parseFloat(formData.get('perch')),
            bedrooms: parseInt(formData.get('bedrooms')),
            bathrooms: parseInt(formData.get('bathrooms')),
            kitchen_area_sqft: parseInt(formData.get('kitchen_area_sqft')),
            parking_spots: parseInt(formData.get('parking_spots')),
            has_garden: formData.get('has_garden') === 'on',
            has_ac: formData.get('has_ac') === 'on',
            water_supply: formData.get('water_supply'),
            electricity: formData.get('electricity'),
            floors: parseInt(formData.get('floors')),
            year_built: parseInt(formData.get('year_built'))
        };

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (result.status === 'success') {
                predictionForm.classList.add('hidden');
                resultCard.classList.remove('hidden');
                predictedPriceElem.textContent = result.formatted_prediction;

                window.scrollTo({ top: 0, behavior: 'smooth' });
            } else {
                alert('Error: ' + result.message);
            }
        } catch (err) {
            console.error('Prediction error:', err);
            alert('An error occurred during prediction.');
        } finally {
            btnText.style.display = 'block';
            loader.style.display = 'none';
            predictBtn.disabled = false;
        }
    });

    resetBtn.addEventListener('click', () => {
        predictionForm.reset();
        areaSelect.disabled = true;
        areaSelect.innerHTML = '<option value="" disabled selected>Select Area</option>';
        resultCard.classList.add('hidden');
        predictionForm.classList.remove('hidden');
    });
});
