import pandas as pd
import matplotlib.pyplot as mp
from sklearn.linear_model import LinearRegression
import io
import base64

data = {
    "Inasistencias": [2, 0, 1, 3, 0, 4, 0, 5, 1, 0, 2, 6, 0, 7, 0, 4, 1, 8, 0, 9],
    "Horas de Estudio": [10, 15, 12, 8, 14, 5, 16, 7, 11, 13, 9, 4, 18, 3, 17, 6, 14, 2, 20, 1],
    "Nota Final": [3.8, 4.2, 3.6, 3, 4.5, 2.5, 4.8, 2.8, 3.7, 4, 3.2, 2.2, 5, 1.8, 4.9, 2.7, 4.4, 1.5, 5, 1]
}

df = pd.DataFrame(data)


x = df [["Horas de Estudio", "Inasistencias"]]
y = df [["Nota Final"]]

model = LinearRegression()
model.fit(x,y)

def CalculateGrade(hours):
    result = model.predict([[hours]])[0]
    return result

def PlotGraph():
    # 1. Lógica para generar el gráfico (se ejecuta siempre)
    img = io.BytesIO()
    
    mp.figure(figsize=(10, 6))
    mp.scatter(x, y, color='blue', label='Datos reales')
    mp.plot(x, model.predict(x), color='red', label='Línea de regresión')
    mp.title('Regresión Lineal - Horas de estudio vs Calificación')
    mp.xlabel('Horas de estudio')
    mp.ylabel('Calificación')
    mp.legend()
    mp.grid(True)
    
    mp.savefig(img, format='png')
    img.seek(0)
    
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return plot_url