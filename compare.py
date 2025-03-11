import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

def extract_results_from_data(previous_data, current_data):
    """
    Extrae los resultados de los datos proporcionados
    
    Args:
        previous_data: String con los resultados anteriores
        current_data: String con los resultados actuales
        
    Returns:
        tuple: (resultados_anteriores, resultados_actuales)
    """
    # Procesar datos anteriores
    previous_results = []
    lines = previous_data.strip().split('\n')
    for i in range(0, len(lines), 2):
        if i+1 < len(lines) and lines[i].startswith('Esperado:') and lines[i+1].startswith('Resultado:'):
            expected = lines[i].replace('Esperado:', '').strip()
            predicted = lines[i+1].replace('Resultado:', '').strip()
            
            if expected and predicted:
                previous_results.append({
                    'expected': expected,
                    'predicted': predicted,
                    'match': expected == predicted,
                })
            elif expected:  # Si no hay predicción (está vacío)
                previous_results.append({
                    'expected': expected,
                    'predicted': '',
                    'match': False,
                })
    
    # Procesar datos actuales
    current_results = []
    lines = current_data.strip().split('\n')
    for i in range(0, len(lines)):
        if lines[i].startswith('Esperado:') and i+1 < len(lines) and lines[i+1].startswith('Predicho:'):
            expected = lines[i].replace('Esperado:', '').strip()
            predicted = lines[i+1].replace('Predicho:', '').strip()
            match = True if i+2 < len(lines) and 'COINCIDE' in lines[i+2] else False
            
            current_results.append({
                'expected': expected,
                'predicted': predicted,
                'match': match,
            })
    
    return previous_results, current_results

def create_confusion_matrices(previous_results, current_results):
    """
    Crea las matrices de confusión para ambos conjuntos de resultados
    
    Args:
        previous_results: Lista de diccionarios con resultados anteriores
        current_results: Lista de diccionarios con resultados actuales
        
    Returns:
        dict: Matrices de confusión y métricas
    """
    # Preparar datos para matriz anterior
    prev_y_true = []
    prev_y_pred = []
    
    for result in previous_results:
        expected_id = result['expected']
        predicted_id = result['predicted']
        
        # Detectabilidad (1 si detectó algún ID, 0 si no)
        prev_y_true.append(1)  # Asumimos que todos deberían tener ID
        prev_y_pred.append(1 if predicted_id else 0)
    
    # Preparar datos para matriz actual
    curr_y_true = []
    curr_y_pred = []
    
    for result in current_results:
        expected_id = result['expected']
        predicted_id = result['predicted']
        
        # Detectabilidad (1 si detectó algún ID, 0 si no)
        curr_y_true.append(1)  # Asumimos que todos deberían tener ID
        curr_y_pred.append(1 if predicted_id else 0)
    
    # Calcular matrices
    prev_cm = confusion_matrix(prev_y_true, prev_y_pred, labels=[0, 1])
    curr_cm = confusion_matrix(curr_y_true, curr_y_pred, labels=[0, 1])
    
    # Calcular exactitud de IDs
    prev_exact_matches = sum(1 for r in previous_results if r['match'])
    prev_total = len(previous_results)
    prev_id_accuracy = prev_exact_matches / prev_total if prev_total > 0 else 0
    
    curr_exact_matches = sum(1 for r in current_results if r['match'])
    curr_total = len(current_results)
    curr_id_accuracy = curr_exact_matches / curr_total if curr_total > 0 else 0
    
    # Calcular métricas para conjunto anterior
    if prev_cm.shape == (2, 2):
        prev_tn, prev_fp, prev_fn, prev_tp = prev_cm.ravel()
    else:
        # Manejar caso donde la matriz no es 2x2
        prev_tp = sum(1 for y_t, y_p in zip(prev_y_true, prev_y_pred) if y_t == 1 and y_p == 1)
        prev_tn = sum(1 for y_t, y_p in zip(prev_y_true, prev_y_pred) if y_t == 0 and y_p == 0)
        prev_fp = sum(1 for y_t, y_p in zip(prev_y_true, prev_y_pred) if y_t == 0 and y_p == 1)
        prev_fn = sum(1 for y_t, y_p in zip(prev_y_true, prev_y_pred) if y_t == 1 and y_p == 0)
    
    # Calcular métricas para conjunto actual
    if curr_cm.shape == (2, 2):
        curr_tn, curr_fp, curr_fn, curr_tp = curr_cm.ravel()
    else:
        # Manejar caso donde la matriz no es 2x2
        curr_tp = sum(1 for y_t, y_p in zip(curr_y_true, curr_y_pred) if y_t == 1 and y_p == 1)
        curr_tn = sum(1 for y_t, y_p in zip(curr_y_true, curr_y_pred) if y_t == 0 and y_p == 0)
        curr_fp = sum(1 for y_t, y_p in zip(curr_y_true, curr_y_pred) if y_t == 0 and y_p == 1)
        curr_fn = sum(1 for y_t, y_p in zip(curr_y_true, curr_y_pred) if y_t == 1 and y_p == 0)
    
    # Calcular métricas
    prev_metrics = {
        'accuracy': (prev_tp + prev_tn) / (prev_tp + prev_tn + prev_fp + prev_fn) if (prev_tp + prev_tn + prev_fp + prev_fn) > 0 else 0,
        'precision': prev_tp / (prev_tp + prev_fp) if (prev_tp + prev_fp) > 0 else 0,
        'recall': prev_tp / (prev_tp + prev_fn) if (prev_tp + prev_fn) > 0 else 0,
        'id_accuracy': prev_id_accuracy,
        'tp': prev_tp,
        'tn': prev_tn,
        'fp': prev_fp,
        'fn': prev_fn
    }
    
    curr_metrics = {
        'accuracy': (curr_tp + curr_tn) / (curr_tp + curr_tn + curr_fp + curr_fn) if (curr_tp + curr_tn + curr_fp + curr_fn) > 0 else 0,
        'precision': curr_tp / (curr_tp + curr_fp) if (curr_tp + curr_fp) > 0 else 0,
        'recall': curr_tp / (curr_tp + curr_fn) if (curr_tp + curr_fn) > 0 else 0,
        'id_accuracy': curr_id_accuracy,
        'tp': curr_tp,
        'tn': curr_tn,
        'fp': curr_fp,
        'fn': curr_fn
    }
    
    prev_metrics['f1_score'] = 2 * (prev_metrics['precision'] * prev_metrics['recall']) / (prev_metrics['precision'] + prev_metrics['recall']) if (prev_metrics['precision'] + prev_metrics['recall']) > 0 else 0
    curr_metrics['f1_score'] = 2 * (curr_metrics['precision'] * curr_metrics['recall']) / (curr_metrics['precision'] + curr_metrics['recall']) if (curr_metrics['precision'] + curr_metrics['recall']) > 0 else 0
    
    return {
        'previous': {
            'confusion_matrix': prev_cm,
            'metrics': prev_metrics
        },
        'current': {
            'confusion_matrix': curr_cm,
            'metrics': curr_metrics
        }
    }

def analyze_errors(previous_results, current_results):
    """
    Analiza los errores y patrones en ambos conjuntos de resultados
    
    Args:
        previous_results: Lista de diccionarios con resultados anteriores
        current_results: Lista de diccionarios con resultados actuales
        
    Returns:
        dict: Análisis de errores
    """
    # Extraer errores
    prev_errors = [r for r in previous_results if not r['match']]
    curr_errors = [r for r in current_results if not r['match']]
    
    # Categorizar tipos de errores anteriores
    prev_error_types = {}
    for error in prev_errors:
        expected = error['expected']
        predicted = error['predicted']
        
        if not predicted:
            error_type = 'No detección'
        elif len(expected) != len(predicted):
            error_type = 'Longitud diferente'
        elif sum(1 for i in range(min(len(expected), len(predicted))) if expected[i] != predicted[i]) == 1:
            error_type = '1 dígito incorrecto'
        else:
            error_type = 'Múltiples dígitos incorrectos'
        
        if error_type in prev_error_types:
            prev_error_types[error_type].append(error)
        else:
            prev_error_types[error_type] = [error]
    
    # Categorizar tipos de errores actuales
    curr_error_types = {}
    for error in curr_errors:
        expected = error['expected']
        predicted = error['predicted']
        
        if not predicted:
            error_type = 'No detección'
        elif len(expected) != len(predicted):
            error_type = 'Longitud diferente'
        elif sum(1 for i in range(min(len(expected), len(predicted))) if expected[i] != predicted[i]) == 1:
            error_type = '1 dígito incorrecto'
        else:
            error_type = 'Múltiples dígitos incorrectos'
        
        if error_type in curr_error_types:
            curr_error_types[error_type].append(error)
        else:
            curr_error_types[error_type] = [error]
    
    # Analizar mejoras
    improved = []
    for prev_err in prev_errors:
        expected = prev_err['expected']
        # Buscar si el mismo ID esperado ahora es correcto
        for curr_res in current_results:
            if curr_res['expected'] == expected and curr_res['match']:
                improved.append({
                    'expected': expected,
                    'previous_prediction': prev_err['predicted'],
                    'current_prediction': curr_res['predicted']
                })
                break
    
    return {
        'previous': {
            'total': len(prev_errors),
            'by_type': prev_error_types
        },
        'current': {
            'total': len(curr_errors),
            'by_type': curr_error_types
        },
        'improvements': {
            'total': len(improved),
            'details': improved
        }
    }

def visualize_confusion_matrices(matrix_data):
    """
    Visualiza las matrices de confusión
    
    Args:
        matrix_data: Diccionario con datos de matrices de confusión
    
    Returns:
        matplotlib.figure.Figure: Figura con visualizaciones
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Matriz anterior
    sns.heatmap(
        matrix_data['previous']['confusion_matrix'],
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['No ID', 'ID'],
        yticklabels=['No ID', 'ID'],
        ax=ax1
    )
    ax1.set_xlabel('Predicción')
    ax1.set_ylabel('Valor Real')
    ax1.set_title('Matriz de Confusión - Resultados Anteriores')
    
    # Matriz actual
    sns.heatmap(
        matrix_data['current']['confusion_matrix'],
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['No ID', 'ID'],
        yticklabels=['No ID', 'ID'],
        ax=ax2
    )
    ax2.set_xlabel('Predicción')
    ax2.set_ylabel('Valor Real')
    ax2.set_title('Matriz de Confusión - Resultados Actuales')
    
    plt.tight_layout()
    return fig

def visualize_metrics_comparison(matrix_data):
    """
    Visualiza la comparación de métricas entre los dos conjuntos
    
    Args:
        matrix_data: Diccionario con datos de matrices y métricas
    
    Returns:
        matplotlib.figure.Figure: Figura con visualizaciones
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'id_accuracy']
    metric_labels = ['Exactitud', 'Precisión', 'Sensibilidad', 'F1-Score', 'Exactitud ID']
    
    prev_values = [matrix_data['previous']['metrics'][m] * 100 for m in metrics]
    curr_values = [matrix_data['current']['metrics'][m] * 100 for m in metrics]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, prev_values, width, label='Anterior', color='#ff9999')
    bars2 = ax.bar(x + width/2, curr_values, width, label='Actual', color='#66b3ff')
    
    ax.set_ylim(0, 105)
    ax.set_ylabel('Porcentaje (%)')
    ax.set_title('Comparación de Métricas')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    
    # Añadir etiquetas de valor
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    add_labels(bars1)
    add_labels(bars2)
    
    plt.tight_layout()
    return fig

def generate_comprehensive_report(previous_results, current_results, matrix_data, error_analysis):
    """
    Genera un informe completo con todos los análisis
    
    Args:
        previous_results: Lista de resultados anteriores
        current_results: Lista de resultados actuales
        matrix_data: Datos de matrices de confusión
        error_analysis: Análisis de errores
        
    Returns:
        str: Informe completo
    """
    # Métricas resumidas
    prev_metrics = matrix_data['previous']['metrics']
    curr_metrics = matrix_data['current']['metrics']
    
    prev_accuracy = sum(1 for r in previous_results if r['match']) / len(previous_results) * 100
    curr_accuracy = sum(1 for r in current_results if r['match']) / len(current_results) * 100
    
    report = []
    report.append("ANÁLISIS COMPARATIVO DEL SISTEMA TRUCKID")
    report.append("=" * 50)
    report.append("")
    
    # Resumen general
    report.append("RESUMEN GENERAL")
    report.append("-" * 30)
    report.append(f"Total de imágenes analizadas: {len(previous_results)}")
    report.append("")
    report.append(f"Resultados anteriores:")
    report.append(f"  • Predicciones correctas: {sum(1 for r in previous_results if r['match'])}")
    report.append(f"  • Precisión global: {prev_accuracy:.2f}%")
    report.append(f"  • Total de errores: {error_analysis['previous']['total']}")
    report.append("")
    report.append(f"Resultados actuales:")
    report.append(f"  • Predicciones correctas: {sum(1 for r in current_results if r['match'])}")
    report.append(f"  • Precisión global: {curr_accuracy:.2f}%")
    report.append(f"  • Total de errores: {error_analysis['current']['total']}")
    report.append("")
    report.append(f"Mejora absoluta: {curr_accuracy - prev_accuracy:+.2f}%")
    report.append("")
    
    # Análisis de matrices de confusión
    report.append("ANÁLISIS DE MATRICES DE CONFUSIÓN")
    report.append("-" * 30)
    report.append("Matriz de confusión anterior:")
    report.append("                Predicción")
    report.append("                No ID    ID")
    report.append(f"Real No ID     {prev_metrics['tn']:5d}   {prev_metrics['fp']:5d}")
    report.append(f"     ID        {prev_metrics['fn']:5d}   {prev_metrics['tp']:5d}")
    report.append("")
    report.append("Matriz de confusión actual:")
    report.append("                Predicción")
    report.append("                No ID    ID")
    report.append(f"Real No ID     {curr_metrics['tn']:5d}   {curr_metrics['fp']:5d}")
    report.append(f"     ID        {curr_metrics['fn']:5d}   {curr_metrics['tp']:5d}")
    report.append("")
    
    # Comparación de métricas
    report.append("COMPARACIÓN DE MÉTRICAS")
    report.append("-" * 30)
    report.append(f"{'Métrica':<15} {'Anterior':>10} {'Actual':>10} {'Diferencia':>12}")
    report.append("-" * 50)
    report.append(f"{'Exactitud':<15} {prev_metrics['accuracy']*100:>9.2f}% {curr_metrics['accuracy']*100:>9.2f}% {(curr_metrics['accuracy']-prev_metrics['accuracy'])*100:>+11.2f}%")
    report.append(f"{'Precisión':<15} {prev_metrics['precision']*100:>9.2f}% {curr_metrics['precision']*100:>9.2f}% {(curr_metrics['precision']-prev_metrics['precision'])*100:>+11.2f}%")
    report.append(f"{'Sensibilidad':<15} {prev_metrics['recall']*100:>9.2f}% {curr_metrics['recall']*100:>9.2f}% {(curr_metrics['recall']-prev_metrics['recall'])*100:>+11.2f}%")
    report.append(f"{'F1-Score':<15} {prev_metrics['f1_score']*100:>9.2f}% {curr_metrics['f1_score']*100:>9.2f}% {(curr_metrics['f1_score']-prev_metrics['f1_score'])*100:>+11.2f}%")
    report.append(f"{'Exactitud ID':<15} {prev_metrics['id_accuracy']*100:>9.2f}% {curr_metrics['id_accuracy']*100:>9.2f}% {(curr_metrics['id_accuracy']-prev_metrics['id_accuracy'])*100:>+11.2f}%")
    report.append("")
    
    # Análisis de errores
    report.append("ANÁLISIS DE ERRORES")
    report.append("-" * 30)
    
    # Errores anteriores por tipo
    report.append("Errores anteriores por tipo:")
    for error_type, errors in error_analysis['previous']['by_type'].items():
        report.append(f"  • {error_type}: {len(errors)} casos ({len(errors)/error_analysis['previous']['total']*100:.1f}%)")
        for i, err in enumerate(errors[:3]):  # Mostrar hasta 3 ejemplos
            report.append(f"    - Esperado: {err['expected']}, Predicho: {err['predicted'] or 'No detectado'}")
        if len(errors) > 3:
            report.append(f"    - ... y {len(errors) - 3} más")
    report.append("")
    
    # Errores actuales por tipo
    if error_analysis['current']['total'] > 0:
        report.append("Errores actuales por tipo:")
        for error_type, errors in error_analysis['current']['by_type'].items():
            report.append(f"  • {error_type}: {len(errors)} casos ({len(errors)/error_analysis['current']['total']*100:.1f}%)")
            for i, err in enumerate(errors[:3]):  # Mostrar hasta 3 ejemplos
                report.append(f"    - Esperado: {err['expected']}, Predicho: {err['predicted'] or 'No detectado'}")
            if len(errors) > 3:
                report.append(f"    - ... y {len(errors) - 3} más")
    else:
        report.append("No se encontraron errores en los resultados actuales.")
    report.append("")
    
    # Errores corregidos
    report.append("ANÁLISIS DE MEJORAS")
    report.append("-" * 30)
    total_prev_errors = error_analysis['previous']['total']
    improvement_rate = error_analysis['improvements']['total'] / total_prev_errors * 100 if total_prev_errors > 0 else 0
    
    report.append(f"Se corrigieron {error_analysis['improvements']['total']} de {total_prev_errors} errores previos ({improvement_rate:.1f}%)")
    report.append("")
    report.append("Detalle de errores corregidos:")
    for i, improvement in enumerate(error_analysis['improvements']['details']):
        report.append(f"  {i+1}. Esperado: {improvement['expected']}")
        report.append(f"     Anterior: {improvement['previous_prediction'] or 'No detectado'}")
        report.append(f"     Actual: {improvement['current_prediction']}")
    report.append("")
    
    # Conclusión
    report.append("CONCLUSIÓN")
    report.append("-" * 30)
    if curr_accuracy == 100:
        report.append("¡EXCELENTE! Se ha alcanzado una precisión perfecta del 100%.")
        report.append(f"Se han corregido todos los {total_prev_errors} errores que existían anteriormente.")
        report.append("El sistema TruckID 100 ha demostrado una capacidad perfecta para identificar números de camiones mineros.")
    elif curr_accuracy > prev_accuracy:
        report.append(f"RESULTADO POSITIVO: Se ha logrado una mejora de {curr_accuracy - prev_accuracy:+.2f}% en la precisión.")
        report.append(f"Se corrigieron {error_analysis['improvements']['total']} de {total_prev_errors} errores anteriores.")
        if error_analysis['current']['total'] > 0:
            report.append(f"Aún quedan {error_analysis['current']['total']} errores por resolver.")
    else:
        report.append(f"RESULTADO SIN CAMBIOS: La precisión se mantiene en {curr_accuracy:.2f}%.")
    
    return "\n".join(report)

def main():
    # Datos de los resultados anteriores (primera página del primer PDF)
    previous_data = """Esperado: 144
Resultado: 143
Esperado: 74
Resultado: 74
Esperado: 460
Resultado: 00
Esperado: 105
Resultado: 262
Esperado: 160
Resultado: 160
Esperado: 302
Resultado:
Esperado: 314
Resultado: 12345
Esperado: 336
Resultado: 336
Esperado: 351
Resultado: 351
Esperado: 56
Resultado: 56
Esperado: 204
Resultado: 204
Esperado: 115
Resultado: 115
Esperado: 144
Resultado: 144
Esperado: 109
Resultado: 109
Esperado: 45
Resultado: 45
Esperado: 691
Resultado: 691
Esperado: 717
Resultado: 717
Esperado: 81
Resultado: 91"""

    # Datos de los resultados actuales (primer PDF)
    current_data = """Esperado: 144
Predicho: 144
 COINCIDE
Esperado: 74
Predicho: 74
 COINCIDE
Esperado: 460
Predicho: 460
 COINCIDE
Esperado: 105
Predicho: 105
 COINCIDE
Esperado: 160
Predicho: 160
 COINCIDE
Esperado: 302
Predicho: 302
 COINCIDE
Esperado: 314
Predicho: 314
 COINCIDE
Esperado: 336
Predicho: 336
 COINCIDE
Esperado: 351
Predicho: 351
 COINCIDE
Esperado: 56
Predicho: 56
 COINCIDE
Esperado: 204
Predicho: 204
 COINCIDE
Esperado: 115
Predicho: 115
 COINCIDE
Esperado: 144
Predicho: 144
 COINCIDE
Esperado: 109
Predicho: 109
 COINCIDE
Esperado: 45
Predicho: 45
 COINCIDE
Esperado: 691
Predicho: 691
 COINCIDE
Esperado: 717
Predicho: 717
 COINCIDE
Esperado: 81
Predicho: 81
 COINCIDE"""

    # Extraer resultados
    previous_results, current_results = extract_results_from_data(previous_data, current_data)
    
    # Crear matrices de confusión
    matrix_data = create_confusion_matrices(previous_results, current_results)
    
    # Analizar errores
    error_analysis = analyze_errors(previous_results, current_results)
    
    # Generar informe completo
    report = generate_comprehensive_report(previous_results, current_results, matrix_data, error_analysis)
    
    # Visualizar matrices de confusión
    confusion_matrices_fig = visualize_confusion_matrices(matrix_data)
    
    # Visualizar comparación de métricas
    metrics_comparison_fig = visualize_metrics_comparison(matrix_data)
    
    # Imprimir informe
    print(report)
    
    # Guardar visualizaciones
    confusion_matrices_fig.savefig("matrices_confusion.png", dpi=150, bbox_inches='tight')
    metrics_comparison_fig.savefig("comparacion_metricas.png", dpi=150, bbox_inches='tight')
    
    # Guardar informe en archivo
    with open("informe_analisis_truckid.txt", "w") as f:
        f.write(report)
    
    return previous_results, current_results, matrix_data, error_analysis

if __name__ == "__main__":
    main()