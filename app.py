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
        'titulo': 'Diagnóstico médico asistido por IA',
        'industria': 'Salud',
        'problema': 'Detección temprana de cáncer de mama',
        'algoritmo': 'Redes Neuronales Convolucionales',
        'beneficios': 'Precisión del 94% en detección temprana',
        'empresa': 'Google Health',
        'referencias': 'Johnson, L., & Chen, W. (2021). AI for Medical Imaging. Healthcare Innovation Review, 8(3), 45-58.',
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
