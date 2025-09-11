from flask import Flask, render_template, request
import RegresionLineal


app = Flask(__name__)

# Datos de ejemplo para los casos de uso
casos_uso = [
    {
        'id': 'entretenimiento',
        'titulo': 'Predicción de Abandono de Clientes (Churn) en Netflix',
        'industria': 'Streaming y Entretenimiento',
        'problema': 'El principal problema que se resuelve es la pérdida de ingresos debido a la cancelación de suscripciones. Adquirir un nuevo cliente es mucho más costoso que retener a uno existente. Por lo tanto, las empresas como Netflix necesitan identificar de manera proactiva a los usuarios que probablemente cancelarán su servicio (esto se conoce como "churn"). Al predecir quiénes están en riesgo, la empresa puede implementar estrategias de retención dirigidas para evitar que el cliente se vaya.',
        'algoritmo': 'Este es un problema de clasificación binaria (el cliente cancelará o no). Los algoritmos más comunes incluyen:\n- Regresión Logística\n- Máquinas de Soporte Vectorial (SVM)\n- Bosques Aleatorios (Random Forest), uno de los más efectivos.\nLos modelos se entrenan con datos históricos de los usuarios, como la frecuencia de inicio de sesión, el tiempo de visualización, los géneros preferidos y el historial de pagos.',
        'beneficios': '- Reducción de la Tasa de Abandono: Permite actuar antes de que el cliente se vaya, mejorando la retención.\n- Aumento de Ingresos: Retener más clientes asegura un flujo de ingresos más estable.\n- Optimización del Marketing: Las campañas de retención se dirigen solo a los clientes en riesgo, ahorrando costos.\n- Mejora de la Experiencia del Cliente: Las acciones de retención, como recomendaciones personalizadas, mejoran la satisfacción.',
        'empresa': 'Netflix es un ejemplo paradigmático. La compañía utiliza masivamente los datos de sus usuarios para alimentar sus algoritmos de recomendación y, de forma crucial, sus modelos de predicción de abandono para sostener su modelo de negocio.',
        'referencias': 'Ahmed, A. (2022). *Predicting Customer Churn in Python*. DataCamp. Recuperado de https://www.datacamp.com/tutorial/predicting-customer-churn-in-python\nJain, A. (2023, 10 de octubre). *How Netflix Uses Data Science to Enhance User Experience and Drive Business Growth*. Medium. Recuperado de https://medium.com/@anushka.jain992/how-netflix-uses-data-science-to-enhance-user-experience-and-drive-business-growth-4852c486411a'
   },
    {
        'id': 'salud',
        'titulo': 'Google/Verily – Detección de Retinopatía Diabética',
        'industria': 'Salud',
        'problema': 'La retinopatia diabetica es una de las principales causas de cegera evitable en el mundo. millones de pacientescon diabetes requieren exámenes oftalmologico regulares, pero existe una escasez global de especialistas que dificulta la deteccion temprana, sobre todo en zonas rurales o con pocos recursos medicos.',
        'algoritmo': 'Gloogle Research y Verily desarrollaron un sistema de redes neuronales convolucionales profundas (CNN) entrenado con mas de 120,000 imagenes de fondo de ojo, cuidadosamente etiquetadas por oftamologos certificados.El modelo clasifica las severidad de la retinopatia diabetica y el edema macular diabetico de forma automatica.',
        'beneficios': 'El algoritmo logró una sensibilidad de hasta 97% y una especificidad del 93%, comparable al desempeño de oftalmólogos expertos. Gracias a esto, se ha implementado en hospitales de India y Tailandia, logrando tamizajes masivos y diagnósticos tempranos, reduciendo el riesgo de ceguera y permitiendo que los especialistas se concentren en los casos más graves.',
        'empresa': 'Google Research y Verily Life Sciences, en colaboración con el Aravind Eye Hospital (India) y el Rajavithi Hospital (Tailandia).',
        'referencias': 'Google Research Asia-Pacific. (2024, octubre 17). Cómo la IA está haciendo que la atención para salvar la vista sea más accesible en entornos con recursos limitados. *Google Blog*. https://blog.google/around-the-globe/google-asia/arda-diabetic-retinopathy-india-thailand/'+'Aravind Eye Hospitals, Google LLC, & Verily Life Sciences LLC. (2023). Rendimiento de un algoritmo de aprendizaje profundo para la retinopatía diabética en India. *PMC*. https://pmc.ncbi.nlm.nih.gov/articles/PMC11923701/'+'Andi, H. (2019, febrero 27). Google, Verily utilizar el aprendizaje automático para detectar la enfermedad ocular diabética. *Fierce Healthcare*. https://www.fiercehealthcare.com/tech/google-verily-develop-ai-tool-to-screen-for-diabetic-eye-disease'    },
    {
        'id': 'educación',
        'titulo': 'Modelo de predicción del éxito de Saber Pro (y trabajos relacionados).',
        'industria': 'Educación superior',
        'problema': 'Identificar qué factores (académicos y socioeconómicos) influyen en que un estudiante obtenga resultado por encima del promedio en la prueba Saber Pro — utilidad para políticas educativas y detección temprana de riesgo académico.',
        'algoritmo': 'Árboles de decisión (CART) como modelo supervisado de clasificación; estudios relacionados en Colombia también usan redes neuronales, regresión logística y bosques aleatorios según el alcance.',
        'beneficios': 'Identificación de variables influyentes, ej: puntaje en Saber 11 — especialmente la sección de Estudios Sociales —, características socioeconómicas y rasgos institucionales. '
        'Permite priorizar intervenciones educativas y diseñar estrategias de apoyo a estudiantes en riesgo.'
        'En estudios con grandes conjuntos de datos, ej: alrededor de 246.436 registros en análisis nacionales se obtuvieron modelos con rendimiento útil para análisis agregados y para decisiones institucionales.',
        'empresa': 'Trabajo académico de autores vinculados a Universidad EAFIT',
        'referencias': 'Pérez Bernal, G., Toro Villegas, L., & Toro, M. (2020). Saber Pro success prediction model using decision tree based learning. arXiv. https://arxiv.org/abs/2006.01322. '
        'Bernal, G. P., Villegas, L. T., & Toro, M. (2020). Saber Pro success prediction model using decision tree based learning. ResearchGate. https://www.researchgate.net/publication/341851879_Saber_Pro_success_prediction_model_using_decision_tree_based_learning. '
        'Vanegas-Ayala, S. C., Leal-Lara, D. D., & Barón-Velandia, J. (2022). Predicción rendimiento estudiantes pruebas Saber Pro en pandemia junto con las características socioeconómicas. Revista TIA — Tecnología, Investigación y Academia, 9(2), 5-16. Recuperado de https://revistas.udistrital.edu.co/index.php/tia/article/download/19446/18260/116618.',
    },
    {
        'id': 'servicios',
        'titulo': 'Detección de fraude en pagos (PayPal)',
        'industria': 'Servicios financieros',
        'problema': 'El fraude en transacciones digitales representa pérdidas millonaria para plataforma de pago y usuarios. Los sistemas tradicionales, basados en reglas fijas, no lograban adaptarse a la rapidez con la que evolucionan los patrones de fraude. Estos ocasionaba un gran número de falsos positivos, afectando la confianza de los clientes y aumentando los costos operativos.',
        'algoritmo': 'PayPal utiliza modelos de clasificación supervisada entrenados con millones de transacciones etiquetadas como "fraudulentas" o "legítimas". Entre los algoritmos aplicados destacan:\n'
                    '- Gradient Boosting (para mejorar precisión en datos desbalanceados).\n'
                    '- Redes neuronales profundas (Deep Learning) (para detectar patrones complejos y no lineales).\n'
                    '- Ensembles de modelos (combinación de varios clasificadores para aumentar la robustez).\n'
                    'Estos modelos permiten la detección en tiempo real de operaciones sospechosas, adaptándose dinámicamente a las nuevas estrategias de fraude.',
        'beneficios': '- Mayor precisión en la detección de fraude, reduciendo falsos positivos.\n'
                    '- Prevención en tiempo real, protegiendo tanto a la empresa como a sus clientes.\n'
                    '- Optimización de recursos, al reducir revisiones manuales.\n'
                    '- Mejor experiencia de usuario, ya que menos transacciones legítimas son bloqueadas.',
        'empresa': 'PayPal Holdings, Inc. es una de las plataformas de pagos digitales más grandes del mundo, fundada en 1998 y con más de 430 millones de usuarios activos en 200 países. La compañía ha sido pionera en la integración de Machine Learning en sus sistemas de seguridad, conslidándose como un referente mundial en prevención de fraude financiero.',
        'referencias': 'Algotive. 5 ejemplos de Machine Learning que usas en tu día a día y no lo sabías:https://www.algotive.ai/es-mx/blog/5-ejemplos-de-machine-learning-que-usas-en-tu-dia-a-dia-y-no-lo-sabias#PayPal\n'
                        'Stripe. Cómo funciona el machine learning para prevenir el fraude en pagos: https://stripe.com/es/resources/more/how-machine-learning-works-for-payment-fraud-detection-and-prevention?utm_source=chatgpt.com\n'
                        'SEON. Machine Learning para detectar fraude: https://seon.io/es/recursos/machine-learning-para-detectar-fraude/?utm_source=chatgpt.com\n'
                        'InteractiveChaos.Gradient Boosting: https://interactivechaos.com/es/manual/tutorial-de-machine-learning/gradient-boosting\n'
                        'IBM. ¿Qué es el deep learning?: https://www.ibm.com/es-es/think/topics/deep-learning\n'
                        'BuiltIn. Modelos de conjunto ¿Qué son y cuándo utilizarlos?: https://builtin-com.translate.goog/machine-learning/ensemble-model?_x_tr_sl=en&_x_tr_tl=es&_x_tr_hl=es&_x_tr_pto=wa\n'
                        'Wikipedia. PayPal: https://es.wikipedia.org/wiki/PayPal\n'
                        'PayPal. History & facts: https://about.pypl.com/who-we-are/history-and-facts/default.aspx\n',
    },
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/casos')
def casos():
    return render_template('casos.html')

@app.route('/caso1')
def caso1():
    caso = casos_uso[0]
    return render_template('caso1.html', caso=caso)

@app.route('/caso2')
def caso2():
    caso = casos_uso[1]
    return render_template('caso2.html', caso=caso)

@app.route('/caso3')
def caso3():
    caso = casos_uso[2]
    return render_template('caso3.html', caso=caso)

@app.route('/caso4')
def caso4():
    caso = casos_uso[3]
    return render_template('caso4.html', caso=caso)

@app.route('/rl')
def rl():
    return render_template('rl_index.html')

@app.route('/rl_conceptos')
def rl_conceptos():
    return render_template('rl_conceptos.html')

@app.route('/rl_practico', methods=['GET', 'POST'])
def rl_practico():
    # 1. Generar el gráfico (siempre)
    plot_url = RegresionLineal.PlotGraph()
    
    # 2. Inicializar variables para el resultado de la predicción
    result = None
    autos = None
    transporte = None

    # 3. Lógica para manejar el formulario (POST)
    if request.method == 'POST':
        autos = float(request.form['autos'])
        transporte = float(request.form['transporte'])
        result = RegresionLineal.PredecirContaminacion(autos, transporte)
        
    # 4. Renderizar plantilla con variables
    return render_template(
        'rl_practico.html',
        plot_url=plot_url,
        result=result,
        autos=autos,
        transporte=transporte
    )

if __name__ == '__main__':
    app.run(debug=True)
