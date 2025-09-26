import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Usar backend sin GUI para evitar problemas de threading
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')
# Configurar matplotlib para mejor rendimiento en servidor web
plt.ioff()  # Desactivar modo interactivo

# Configurar semilla para reproducibilidad
np.random.seed(42)

# Dataset: Predicción de abandono escolar universitario
# Variables: edad, promedio_secundaria, ingresos_familiares, horas_estudio, actividades_extra, soporte_familiar
# Clase objetivo: abandona (0=No abandona, 1=Abandona)

# Generar dataset sintético basado en factores reales de abandono escolar
def generate_dataset():
    np.random.seed(42)
    n_samples = 1000
    
    # Variables independientes
    edad = np.random.normal(19, 1.5, n_samples).clip(17, 25)  # Edad entre 17-25
    promedio_secundaria = np.random.normal(3.5, 0.8, n_samples).clip(2.0, 5.0)  # Promedio 2.0-5.0
    ingresos_familiares = np.random.lognormal(8, 0.5, n_samples).clip(500000, 5000000)  # Ingresos en pesos
    horas_estudio = np.random.exponential(15, n_samples).clip(0, 40)  # Horas semanales de estudio
    actividades_extra = np.random.binomial(1, 0.4, n_samples)  # 1=Participa en actividades, 0=No participa
    soporte_familiar = np.random.binomial(1, 0.7, n_samples)  # 1=Buen soporte, 0=Soporte limitado
    
    # Variable objetivo con lógica realista
    # Mayor probabilidad de abandono si:
    # - Promedio bajo, pocos ingresos, pocas horas de estudio, sin actividades extra, sin soporte familiar
    abandona_prob = (
        0.1 +  # Probabilidad base
        0.3 * (promedio_secundaria < 3.0) +  # Promedio bajo aumenta riesgo
        0.2 * (ingresos_familiares < 1000000) +  # Bajos ingresos aumentan riesgo
        0.25 * (horas_estudio < 10) +  # Pocas horas de estudio aumentan riesgo
        0.15 * (1 - actividades_extra) +  # Sin actividades extra aumenta riesgo
        0.4 * (1 - soporte_familiar)  # Sin soporte familiar aumenta mucho el riesgo
    )
    
    # Agregar algo de ruido aleatorio
    abandona_prob += np.random.normal(0, 0.1, n_samples)
    abandona_prob = np.clip(abandona_prob, 0, 1)
    
    abandona = np.random.binomial(1, abandona_prob, n_samples)
    
    # Crear DataFrame
    df = pd.DataFrame({
        'edad': edad.round(1),
        'promedio_secundaria': promedio_secundaria.round(2),
        'ingresos_familiares': ingresos_familiares.round(0).astype(int),
        'horas_estudio': horas_estudio.round(1),
        'actividades_extra': actividades_extra,
        'soporte_familiar': soporte_familiar,
        'abandona': abandona
    })
    
    return df

# Cargar y preparar los datos
data = generate_dataset()

# Documentación de las clases
print("=== DOCUMENTACIÓN DEL DATASET ===")
print("Variable objetivo: 'abandona'")
print("  0 = No abandona (clase negativa)")
print("  1 = Abandona (clase positiva)")
print("\nVariables independientes:")
print("- edad: Edad del estudiante (17-25 años)")
print("- promedio_secundaria: Promedio de calificaciones en secundaria (2.0-5.0)")
print("- ingresos_familiares: Ingresos familiares mensuales en pesos")
print("- horas_estudio: Horas de estudio semanales")
print("- actividades_extra: Participación en actividades extracurriculares (0=No, 1=Sí)")
print("- soporte_familiar: Nivel de soporte familiar (0=Limitado, 1=Bueno)")
print("\nDistribución de clases:")
print(data['abandona'].value_counts())
print(f"Proporción de abandono: {data['abandona'].mean():.3f}")

# Separar características y variable objetivo
X = data.drop('abandona', axis=1)
y = data['abandona']

# División entrenamiento/prueba (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Crear pipeline con preprocesamiento
# Para Decision Trees no se requiere escalado, pero se incluye por completitud
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Escalado estándar
    ('classifier', DecisionTreeClassifier(
        random_state=42,
        max_depth=10,  # Limitar profundidad para evitar overfitting
        min_samples_split=20,  # Mínimo de muestras para dividir
        min_samples_leaf=10,   # Mínimo de muestras en hoja
        class_weight='balanced'  # Balancear clases
    ))
])

# Entrenar el modelo
pipeline.fit(X_train, y_train)

# Realizar predicciones en conjunto de prueba
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

def evaluate():
    """
    Evalúa el modelo y retorna métricas junto con visualizaciones
    
    Returns:
        tuple: (accuracy, report_dict, confusion_matrix_url, roc_curve_url)
    """
    # Calcular exactitud
    accuracy = accuracy_score(y_test, y_pred)
    
    # Generar reporte de clasificación
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Crear figura con subplots para matriz de confusión y ROC
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Matriz de confusión
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                ax=ax1, cbar=True,
                xticklabels=['No Abandona (0)', 'Abandona (1)'],
                yticklabels=['No Abandona (0)', 'Abandona (1)'])
    ax1.set_xlabel('Predicción')
    ax1.set_ylabel('Real')
    ax1.set_title('Matriz de Confusión\nÁrbol de Decisión')
    
    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    ax2.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Tasa de Falsos Positivos')
    ax2.set_ylabel('Tasa de Verdaderos Positivos')
    ax2.set_title('Curva ROC\nÁrbol de Decisión')
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Convertir a imagen base64
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    combined_url = base64.b64encode(img_buffer.getvalue()).decode('utf8')
    plt.close()
    
    # Crear visualización del árbol (solo parte del árbol para legibilidad)
    plt.figure(figsize=(20, 12))
    plot_tree(pipeline.named_steps['classifier'], 
              feature_names=X.columns,
              class_names=['No Abandona', 'Abandona'],
              filled=True, 
              rounded=True,
              max_depth=3,  # Solo mostrar los primeros 3 niveles
              fontsize=10)
    plt.title('Estructura del Árbol de Decisión (Primeros 3 niveles)', fontsize=16)
    
    # Convertir árbol a imagen base64
    tree_buffer = BytesIO()
    plt.savefig(tree_buffer, format='png', dpi=150, bbox_inches='tight')
    tree_buffer.seek(0)
    tree_url = base64.b64encode(tree_buffer.getvalue()).decode('utf8')
    plt.close()
    
    return accuracy, report, combined_url, tree_url

def predict_label(features, threshold=0.5):
    """
    Predice la etiqueta para nuevas características
    
    Args:
        features (list): Lista con [edad, promedio_secundaria, ingresos_familiares, 
                        horas_estudio, actividades_extra, soporte_familiar]
        threshold (float): Umbral de decisión (default: 0.5)
    
    Returns:
        tuple: (prediccion_texto, probabilidad)
    """
    # Convertir a formato adecuado
    features_df = pd.DataFrame([features], columns=X.columns)
    
    # Obtener probabilidades
    probabilities = pipeline.predict_proba(features_df)[0]
    prob_abandona = probabilities[1]  # Probabilidad de abandono
    
    # Determinar predicción basada en el threshold
    prediction = "Sí" if prob_abandona >= threshold else "No"
    
    return prediction, prob_abandona

def get_feature_importance():
    """
    Retorna la importancia de las características
    
    Returns:
        dict: Diccionario con la importancia de cada característica
    """
    importances = pipeline.named_steps['classifier'].feature_importances_
    feature_importance = dict(zip(X.columns, importances))
    
    # Ordenar por importancia
    feature_importance = dict(sorted(feature_importance.items(), 
                                   key=lambda x: x[1], reverse=True))
    
    return feature_importance

def get_threshold_interpretation(threshold):
    """
    Genera interpretación del umbral seleccionado
    
    Args:
        threshold (float): Umbral de decisión
    
    Returns:
        str: Interpretación del umbral
    """
    if threshold < 0.3:
        return f"Con threshold={threshold}, el modelo es muy sensible y identificará más casos de riesgo de abandono, pero también más falsos positivos."
    elif threshold < 0.5:
        return f"Con threshold={threshold}, aumenta la sensibilidad del modelo para detectar estudiantes en riesgo de abandono."
    elif threshold == 0.5:
        return f"Con threshold={threshold} (valor estándar), el modelo balancea entre sensibilidad y especificidad."
    elif threshold < 0.7:
        return f"Con threshold={threshold}, el modelo es más conservador y solo predice abandono con mayor certeza."
    else:
        return f"Con threshold={threshold}, el modelo es muy conservador y solo identifica casos de muy alto riesgo de abandono."

# Información adicional del dataset
dataset_info = {
    'name': 'Predicción de Abandono Escolar Universitario',
    'target_variable': 'abandona',
    'classes': {
        0: 'No Abandona',
        1: 'Abandona (Clase Positiva)'
    },
    'features': {
        'edad': {
            'type': 'numérica',
            'description': 'Edad del estudiante (17-25 años)',
            'scaling_required': True
        },
        'promedio_secundaria': {
            'type': 'numérica', 
            'description': 'Promedio de calificaciones en secundaria (2.0-5.0)',
            'scaling_required': True
        },
        'ingresos_familiares': {
            'type': 'numérica',
            'description': 'Ingresos familiares mensuales en pesos',
            'scaling_required': True
        },
        'horas_estudio': {
            'type': 'numérica',
            'description': 'Horas de estudio semanales',
            'scaling_required': True
        },
        'actividades_extra': {
            'type': 'categórica binaria',
            'description': 'Participación en actividades extracurriculares (0=No, 1=Sí)',
            'scaling_required': False
        },
        'soporte_familiar': {
            'type': 'categórica binaria',
            'description': 'Nivel de soporte familiar (0=Limitado, 1=Bueno)',
            'scaling_required': False
        }
    },
    'n_samples': len(data),
    'n_features': len(X.columns),
    'class_distribution': data['abandona'].value_counts().to_dict()
}

if __name__ == "__main__":
    # Ejemplo de uso
    print("\n=== EVALUACIÓN DEL MODELO ===")
    accuracy, report, matriz_url, tree_url = evaluate()
    print(f"Exactitud: {accuracy:.4f}")
    print(f"Precisión promedio: {np.mean([report['0']['precision'], report['1']['precision']]):.4f}")
    print(f"Recall promedio: {np.mean([report['0']['recall'], report['1']['recall']]):.4f}")
    
    print("\n=== IMPORTANCIA DE CARACTERÍSTICAS ===")
    importance = get_feature_importance()
    for feature, imp in importance.items():
        print(f"{feature}: {imp:.4f}")
    
    print("\n=== EJEMPLO DE PREDICCIÓN ===")
    # Ejemplo: estudiante de 20 años, promedio 3.0, ingresos bajos, pocas horas de estudio, sin actividades, sin soporte
    ejemplo_features = [20, 3.0, 800000, 8, 0, 0]  # Alto riesgo de abandono
    pred, prob = predict_label(ejemplo_features, threshold=0.5)
    print(f"Estudiante ejemplo - Predicción: {pred}, Probabilidad de abandono: {prob:.4f}")
    print(get_threshold_interpretation(0.5))