console.log('Demo Sales Forecasting cargado');

document.addEventListener('DOMContentLoaded', () => {
  // 1) Carga dinámica de módulos: upload, metrics y dashboard
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
      // 2) Una vez cargados, inicializamos la lógica de predicción simplificada
      await initPredictionByFields();
    } catch (err) {
      console.error(err);
    }
  })();
});

async function initPredictionByFields() {
  const form       = document.getElementById("prediction-form");
  const regionSel  = document.getElementById("region");
  const productSel = document.getElementById("product");
  const subcatSel  = document.getElementById("subcat");
  const dateInput  = document.getElementById("date");
  const modelSel   = document.getElementById("model-select");
  const resultDiv  = document.getElementById("prediction-result");
  const errorDiv   = document.getElementById("prediction-error");

  if (!form) {
    console.warn("No encontré #prediction-form en el DOM");
    return;
  }

  try {
    // 1) Carga listas para los dropdowns
    const [regions, products, subcats] = await Promise.all([
      fetch('/metadata/regions').then(r => r.json()),
      fetch('/metadata/products').then(r => r.json()),
      fetch('/metadata/subcategories').then(r => r.json())
    ]);

    // 2) Rellenar selects
    regions.forEach(r => regionSel.add(new Option(r, r)));
    products.forEach(p => productSel.add(new Option(p, p)));
    subcats.forEach(s => subcatSel.add(new Option(s, s)));
  } catch (err) {
    console.error("Error cargando metadata:", err);
    errorDiv.textContent = "No se pudo cargar las listas de selección.";
    return;
  }

  // 3) Manejar submit del formulario con POST JSON
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    resultDiv.textContent = "Calculando…";
    errorDiv.textContent  = "";

    // Prepara el payload
    const payload = {
      region:       regionSel.value,
      product_id:   productSel.value,
      sub_category: subcatSel.value,
      order_date:   dateInput.value,
      model:        modelSel.value
    };

    try {
      const resp = await fetch("/predict/by_fields", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
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
      console.error(err);
    }
  });
}

