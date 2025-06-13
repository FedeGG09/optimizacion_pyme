ChatGPT Plus
js
Copiar
Editar
// app.js
// Archivo principal que orquesta la carga de los módulos frontend.
// Carga upload.js, metrics.js y dashboard.js, y expone initPredictionByFields()
// para poblar y manejar el formulario de predicción.

console.log('Demo Sales Forecasting cargado');

document.addEventListener('DOMContentLoaded', () => {
  const scripts = [
    '/static/js/upload.js',
    '/static/js/metrics.js',
    '/static/js/dashboard.js'
  ];

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src   = src;
      s.async = false; // respetar orden
      s.onload  = () => {
        console.log(`Módulo cargado: ${src}`);
        resolve();
      };
      s.onerror = () => reject(new Error(`Error cargando ${src}`));
      document.body.appendChild(s);
    });
  }

  (async () => {
    try {
      for (const src of scripts) {
        await loadScript(src);
      }
      console.log('Todos los módulos frontend han sido cargados.');
      // No llamamos initPredictionByFields aquí — lo haremos en upload.js tras subir el CSV.
    } catch (err) {
      console.error('Error cargando módulos frontend:', err);
    }
  })();
});


// Esta función pobla los selects y maneja el submit del formulario de predicción.
// Debe llamarse tras un upload exitoso: window.initPredictionByFields()
async function initPredictionByFields() {
  const regionSel      = document.getElementById("region");
  const productNameSel = document.getElementById("product-name");
  const subcatSel      = document.getElementById("subcat");
  const dateInput      = document.getElementById("date");
  const modelSel       = document.getElementById("model-select");
  const resultDiv      = document.getElementById("prediction-result");
  const errorDiv       = document.getElementById("prediction-error");
  const form           = document.getElementById("prediction-form");

  // Limpiar mensajes previos
  if (errorDiv)  errorDiv.textContent  = "";
  if (resultDiv) resultDiv.textContent = "";

  if (!form || !regionSel || !productNameSel || !subcatSel) {
    console.warn("Elementos de predicción no encontrados.");
    return;
  }

  // 1) Cargar listas para dropdowns
  try {
    const [regions, products, subcats] = await Promise.all([
      fetch('/metadata/regions').then(r => r.ok ? r.json() : Promise.reject(r.status)),
      fetch('/metadata/products').then(r => r.ok ? r.json() : Promise.reject(r.status)),
      fetch('/metadata/subcategories').then(r => r.ok ? r.json() : Promise.reject(r.status))
    ]);

    regionSel.innerHTML      = regions.map(r => `<option value="${r}">${r}</option>`).join('');
    productNameSel.innerHTML = products.map(p => `<option value="${p}">${p}</option>`).join('');
    subcatSel.innerHTML      = subcats.map(s => `<option value="${s}">${s}</option>`).join('');
  } catch (err) {
    console.error('Error cargando metadatos:', err);
    errorDiv.textContent = 'No se pudo cargar las listas de selección.';
    return;
  }

  // 2) Manejar envío del formulario
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    resultDiv.textContent = "Calculando…";
    errorDiv.textContent  = "";

    const payload = {
      region:        regionSel.value,
      product_name:  productNameSel.value,
      sub_category:  subcatSel.value,
      order_date:    dateInput.value,
      model:         modelSel.value
    };

    try {
      const resp = await fetch("/predict/by_fields", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(payload)
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `Error ${resp.status}`);
      }
      const { prediction } = await resp.json();
      resultDiv.textContent = `Predicción ${modelSel.value.toUpperCase()}: ${prediction.toFixed(2)}`;
    } catch (err) {
      errorDiv.textContent  = err.message;
      resultDiv.textContent = "";
      console.error('Error en predict/by_fields:', err);
    }
  }, { once: true });
}

// Hacemos accesible la función para upload.js
window.initPredictionByFields = initPredictionByFields;
