from flask import Flask, render_template

app = Flask(__name__)

# Datos de ejemplo para los casos de uso
casos_uso = [
    {
        'id': 'desarrollo',
        'titulo': 'Detección de fraudes en transacciones financieras',
        'industria': 'Tecnología',
        'problema': 'Identificar transacciones fraudulentas en tiempo real',
        'algoritmo': 'Random Forest',
        'beneficios': 'Reducción del 85% en fraudes no detectados',
        'empresa': 'Visa Inc.',
        'referencias': 'Smith, J. (2022). Machine Learning in Finance. Journal of Financial Technology, 15(2), 112-125.',
    },
    {
     'id': 'salud',
        'titulo': 'Google/Verily – Detección de Retinopatía Diabética',
        'industria': 'Salud',
        'problema': 'La retinopatia diabetica es una de las principales causas de cegera evitable en el mundo. millones de pacientescon diabetes requieren exámenes oftalmologico regulares, pero existe una escasez global de especialistas que dificulta la deteccion temprana, sobre todo en zonas rurales o con pocos recursos medicos.',
        'algoritmo': 'Gloogle Research y Verily desarrollaron un sistema de redes neuronales convolucionales profundas (CNN) entrenado con mas de 120,000 imagenes de fondo de ojo, cuidadosamente etiquetadas por oftamologos certificados.El modelo clasifica las severidad de la retinopatia diabetica y el edema macular diabetico de forma automatica.',
        'beneficios': 'El algoritmo logró una sensibilidad de hasta 97% y una especificidad del 93%, comparable al desempeño de oftalmólogos expertos. Gracias a esto, se ha implementado en hospitales de India y Tailandia, logrando tamizajes masivos y diagnósticos tempranos, reduciendo el riesgo de ceguera y permitiendo que los especialistas se concentren en los casos más graves.',
        'empresa': 'Google Research y Verily Life Sciences, en colaboración con el Aravind Eye Hospital (India) y el Rajavithi Hospital (Tailandia).',
        'referencias': 'Google Research Asia-Pacific. (2024, octubre 17). Cómo la IA está haciendo que la atención para salvar la vista sea más accesible en entornos con recursos limitados. *Google Blog*. https://blog.google/around-the-globe/google-asia/arda-diabetic-retinopathy-india-thailand/'+'Aravind Eye Hospitals, Google LLC, & Verily Life Sciences LLC. (2023). Rendimiento de un algoritmo de aprendizaje profundo para la retinopatía diabética en India. *PMC*. https://pmc.ncbi.nlm.nih.gov/articles/PMC11923701/'+'Andi, H. (2019, febrero 27). Google, Verily utilizar el aprendizaje automático para detectar la enfermedad ocular diabética. *Fierce Healthcare*. https://www.fiercehealthcare.com/tech/google-verily-develop-ai-tool-to-screen-for-diabetic-eye-disease'
     },
    {
        'id': 'entretenimiento',
        'titulo': 'Sistemas de recomendación de contenido',
        'industria': 'Entretenimiento',
        'problema': 'Personalización de recomendaciones para usuarios',
        'algoritmo': 'Filtrado colaborativo',
        'beneficios': 'Incremento del 30% en tiempo de visualización',
        'empresa': 'Netflix',
        'referencias': 'Robinson, M. (2020). Algorithmic Recommendations. Digital Media Journal, 12(4), 78-92.',
    },
    {
        'id': 'retail',
        'titulo': 'Predictión de demanda en retail',
        'industria': 'Comercio minorista',
        'problema': 'Optimización de inventario y almacenamiento',
        'algoritmo': 'Regresión lineal múltiple',
        'beneficios': 'Reducción del 20% en costos de inventario',
        'empresa': 'Amazon',
        'referencias': 'Wilson, A. (2023). Predictive Analytics in Retail. Supply Chain Management, 19(1), 33-47.',
    }
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/caso1')
def caso1():
    caso = casos_uso[0]  # Primer caso
    return render_template('caso1.html', caso=caso)

@app.route('/caso2')
def caso2():
    caso = casos_uso[1]  # Segundo caso
    return render_template('caso2.html', caso=caso)

@app.route('/caso3')
def caso3():
    caso = casos_uso[2]  # Tercer caso
    return render_template('caso3.html', caso=caso)

@app.route('/caso4')
def caso4():
    caso = casos_uso[3]  # Cuarto caso
    return render_template('caso4.html', caso=caso)

if __name__ == '__main__':
    app.run(debug=True)
