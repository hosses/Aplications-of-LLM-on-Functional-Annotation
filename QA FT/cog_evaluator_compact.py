#!/usr/bin/env python3
"""
COG Evaluator Compacto - Compatible con procesador y fine-tuner
Evalúa modelos fine-tuneados en datos de prueba COG
"""

import pandas as pd
import numpy as np
import subprocess
import json
import argparse
import logging
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import Counter
from tqdm import tqdm

# Configuración
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class COGEvaluator:
    def __init__(self):
        """Inicializar evaluador COG compacto"""
        
        # Categorías COG
        self.cog_categories = {
            'J': 'Translation, ribosomal structure and biogenesis',
            'A': 'RNA processing and modification', 'K': 'Transcription',
            'L': 'Replication, recombination and repair', 'B': 'Chromatin structure and dynamics',
            'D': 'Cell cycle control, cell division', 'O': 'Molecular chaperones and related functions',
            'M': 'Cell wall/membrane/envelope biogenesis', 'N': 'Cell motility',
            'P': 'Inorganic ion transport and metabolism', 'T': 'Signal transduction mechanisms',
            'U': 'Intracellular trafficking, secretion', 'V': 'Defense mechanisms',
            'W': 'Extracellular structures', 'C': 'Energy production and conversion',
            'G': 'Carbohydrate transport and metabolism', 'E': 'Amino acid transport and metabolism',
            'F': 'Nucleotide transport and metabolism', 'H': 'Coenzyme transport and metabolism',
            'I': 'Lipid transport and metabolism', 'Q': 'Secondary metabolites biosynthesis',
            'R': 'General function prediction only', 'S': 'Function unknown'
        }
        
        self.results = []
        self.evaluation_metrics = {}
    
    def check_model_availability(self, model_name: str) -> bool:
        """Verificar modelo disponible en Ollama"""
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
            available_models = [line.split()[0] for line in result.stdout.split('\n')[1:] if line.strip()]
            
            # Verificar coincidencias
            model_found = (model_name in available_models or 
                          f"{model_name}:latest" in available_models or
                          model_name.replace(':latest', '') in available_models)
            
            logger.info(f"Modelo '{model_name}': {'✅ Disponible' if model_found else '❌ No encontrado'}")
            return model_found
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("❌ Error accediendo a Ollama")
            return False
    
    def load_test_data(self, test_file: str) -> pd.DataFrame:
        """Cargar datos de prueba - Compatible con procesador COG"""
        logger.info(f"📊 Cargando datos de prueba: {test_file}")
        
        try:
            # Detectar separador
            with open(test_file, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                separator = '\t' if '\t' in first_line else ','
            
            df = pd.read_csv(test_file, sep=separator)
            logger.info(f"✅ Datos cargados: {len(df)} ejemplos")
            logger.info(f"📋 Columnas: {list(df.columns)}")
            
            # Mapear columnas según formato del procesador COG
            column_mapping = self._detect_column_mapping(df)
            
            # Crear DataFrame estandarizado
            df_standard = pd.DataFrame()
            df_standard['protein_id'] = df[column_mapping['protein_id']]
            df_standard['sequence'] = df[column_mapping['sequence']]
            df_standard['true_category'] = df[column_mapping['category']]
            
            # Generar prompts si no existen
            if 'prompt' in df.columns:
                df_standard['prompt'] = df['prompt']
            else:
                logger.info("🧬 Generando prompts automáticamente...")
                df_standard['prompt'] = df_standard.apply(self._generate_prompt, axis=1)
            
            # Agregar información adicional si existe
            for col in ['organism', 'gene_name', 'footprint']:
                if col in df.columns:
                    df_standard[col] = df[col]
            
            # Validar categorías
            valid_categories = set(self.cog_categories.keys())
            df_final = df_standard[df_standard['true_category'].isin(valid_categories)].reset_index(drop=True)
            
            if len(df_final) < len(df_standard):
                logger.warning(f"⚠️ Filtradas {len(df_standard) - len(df_final)} muestras con categorías inválidas")
            
            # Mostrar estadísticas
            category_counts = df_final['true_category'].value_counts().sort_index()
            logger.info(f"📈 Distribución: {len(category_counts)} categorías, {category_counts.min()}-{category_counts.max()} muestras/cat")
            
            return df_final
            
        except Exception as e:
            logger.error(f"❌ Error cargando datos: {e}")
            raise
    
    def _detect_column_mapping(self, df: pd.DataFrame) -> Dict[str, str]:
        """Detectar mapeo de columnas automáticamente"""
        mapping = {}
        
        # Detectar columna de protein_id
        for col in ['protein_id', 'id']:
            if col in df.columns:
                mapping['protein_id'] = col
                break
        else:
            # Crear IDs si no existen
            df['protein_id'] = [f"protein_{i}" for i in range(len(df))]
            mapping['protein_id'] = 'protein_id'
        
        # Detectar columna de secuencia
        for col in ['sequence', 'full_sequence']:
            if col in df.columns:
                mapping['sequence'] = col
                break
        else:
            raise ValueError("❌ No se encontró columna de secuencia")
        
        # Detectar columna de categoría
        for col in ['category', 'true_category']:
            if col in df.columns:
                mapping['category'] = col
                break
        else:
            raise ValueError("❌ No se encontró columna de categoría")
        
        return mapping
    
    def _generate_prompt(self, row) -> str:
        """Generar prompt automático - Compatible con fine-tuner"""
        sequence = row['sequence']
        protein_id = row.get('protein_id', 'unknown')
        organism = row.get('organism', 'Unknown')
        gene_name = row.get('gene_name', '')
        
        categories_text = "\n".join([f"{k}: {v}" for k, v in self.cog_categories.items()])
        
        prompt = f"""You are an expert in protein functional classification using COG categories.

Analyze this protein sequence and predict its COG functional category based on sequence motifs and domains.

COG Categories:
{categories_text}

Protein sequence: {sequence}

Provide a brief analysis and predict the single letter COG category.

Analysis:"""
        
        return prompt
    
    def predict_single(self, model_name: str, prompt: str, timeout: int = 60) -> Tuple[str, str, float]:
        """Hacer predicción individual"""
        start_time = time.time()
        
        try:
            result = subprocess.run(
                ['ollama', 'run', model_name],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            response_time = time.time() - start_time
            
            if result.returncode == 0:
                response = result.stdout.strip()
                predicted_category = self._extract_category(response)
                return response, predicted_category, response_time
            else:
                return "", "S", response_time  # Default a "Function unknown"
                
        except subprocess.TimeoutExpired:
            return "", "S", timeout
        except Exception as e:
            return "", "S", time.time() - start_time
    
    def _extract_category(self, response: str) -> str:
        """Extraer categoría COG de respuesta"""
        
        # Patrones de búsqueda
        patterns = [
            r'Prediction:\s*([A-Z])',
            r'Category:\s*([A-Z])',
            r'^([A-Z])$',  # Línea con solo una letra
            r'\b([A-Z])\b'  # Letra como palabra completa
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                category = match.upper()
                if category in self.cog_categories:
                    return category
        
        # Fallback: buscar cualquier categoría válida
        for category in self.cog_categories.keys():
            if category in response.upper():
                return category
        
        return "S"  # Function unknown por defecto
    
    def evaluate_model(self, model_name: str, test_df: pd.DataFrame,
                      max_samples: Optional[int] = None, timeout: int = 60) -> List[Dict]:
        """Evaluar modelo con barra de progreso"""
        
        logger.info(f"🤖 Evaluando modelo: {model_name}")
        
        if not self.check_model_availability(model_name):
            raise ValueError(f"Modelo {model_name} no disponible")
        
        # Limitar muestras si es necesario
        if max_samples and len(test_df) > max_samples:
            test_df = test_df.sample(n=max_samples, random_state=42)
            logger.info(f"📊 Limitando a {max_samples} muestras")
        
        model_results = []
        
        # Evaluar con barra de progreso
        with tqdm(test_df.iterrows(), total=len(test_df), desc=f"🧪 Evaluando {model_name}") as pbar:
            for idx, row in pbar:
                # Hacer predicción
                response, predicted_category, response_time = self.predict_single(
                    model_name, row['prompt'], timeout
                )
                
                # Guardar resultado
                result = {
                    'model': model_name,
                    'protein_id': row['protein_id'],
                    'predicted_category': predicted_category,
                    'true_category': row['true_category'],
                    'correct': predicted_category == row['true_category'],
                    'response_time': response_time,
                    'sequence_length': len(row['sequence']),
                    'response': response,
                    'prompt': row['prompt']
                }
                
                model_results.append(result)
                
                # Actualizar progreso
                correct_so_far = sum(1 for r in model_results if r['correct'])
                accuracy_so_far = correct_so_far / len(model_results)
                avg_time = np.mean([r['response_time'] for r in model_results])
                
                pbar.set_postfix({
                    'Acc': f"{accuracy_so_far:.3f}",
                    'Time': f"{avg_time:.1f}s"
                })
        
        logger.info(f"✅ Evaluación completada: {model_name}")
        return model_results
    
    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calcular métricas de evaluación"""
        if not results:
            return {}
        
        # Métricas básicas
        total = len(results)
        correct = sum(1 for r in results if r['correct'])
        accuracy = correct / total
        avg_time = np.mean([r['response_time'] for r in results])
        
        # Métricas por categoría
        true_cats = [r['true_category'] for r in results]
        pred_cats = [r['predicted_category'] for r in results]
        
        category_metrics = {}
        for category in self.cog_categories.keys():
            tp = sum(1 for t, p in zip(true_cats, pred_cats) if t == category and p == category)
            fp = sum(1 for t, p in zip(true_cats, pred_cats) if t != category and p == category)
            fn = sum(1 for t, p in zip(true_cats, pred_cats) if t == category and p != category)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            category_metrics[category] = {
                'precision': precision, 'recall': recall, 'f1': f1,
                'support': true_cats.count(category)
            }
        
        # Métricas macro
        valid_metrics = [m for m in category_metrics.values() if m['support'] > 0]
        macro_precision = np.mean([m['precision'] for m in valid_metrics])
        macro_recall = np.mean([m['recall'] for m in valid_metrics])
        macro_f1 = np.mean([m['f1'] for m in valid_metrics])
        
        # Errores más comunes
        errors = [f"{t}→{p}" for t, p in zip(true_cats, pred_cats) if t != p]
        error_distribution = dict(Counter(errors).most_common(10))
        
        return {
            'accuracy': accuracy,
            'total_predictions': total,
            'correct_predictions': correct,
            'avg_response_time': avg_time,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'category_metrics': category_metrics,
            'error_distribution': error_distribution
        }
    
    def save_results(self, all_results: List[Dict], all_metrics: Dict, output_prefix: str = "cog_evaluation") -> Tuple[str, str]:
        """Guardar resultados en archivos"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Guardar CSV con resultados
        csv_file = f"{output_prefix}_{timestamp}.csv"
        df_results = pd.DataFrame(all_results)
        
        # Reordenar columnas
        column_order = ['model', 'protein_id', 'predicted_category', 'true_category', 'correct', 'response_time']
        other_columns = [col for col in df_results.columns if col not in column_order]
        df_results = df_results[column_order + other_columns]
        
        df_results.to_csv(csv_file, index=False)
        logger.info(f"📊 Resultados CSV: {csv_file}")
        
        # Guardar métricas JSON
        json_file = f"{output_prefix}_metrics_{timestamp}.json"
        complete_data = {
            'metrics': all_metrics,
            'evaluation_timestamp': timestamp,
            'summary': self._create_summary(all_metrics)
        }
        
        with open(json_file, 'w') as f:
            json.dump(complete_data, f, indent=2)
        logger.info(f"📈 Métricas JSON: {json_file}")
        
        return csv_file, json_file
    
    def _create_summary(self, all_metrics: Dict) -> Dict:
        """Crear resumen de evaluación"""
        if 'individual_models' not in all_metrics:
            return {}
        
        models = list(all_metrics['individual_models'].keys())
        
        # Encontrar mejor modelo por métrica
        best_accuracy = max(models, key=lambda m: all_metrics['individual_models'][m]['accuracy'])
        best_f1 = max(models, key=lambda m: all_metrics['individual_models'][m]['macro_f1'])
        fastest = min(models, key=lambda m: all_metrics['individual_models'][m]['avg_response_time'])
        
        return {
            'models_evaluated': models,
            'best_accuracy': {
                'model': best_accuracy,
                'value': all_metrics['individual_models'][best_accuracy]['accuracy']
            },
            'best_f1': {
                'model': best_f1,
                'value': all_metrics['individual_models'][best_f1]['macro_f1']
            },
            'fastest': {
                'model': fastest,
                'value': all_metrics['individual_models'][fastest]['avg_response_time']
            }
        }
    
    def print_summary(self, metrics: Dict, model_name: str):
        """Imprimir resumen compacto"""
        print(f"\n🎯 {model_name.upper()}")
        print("-" * 40)
        print(f"📊 Accuracy: {metrics['accuracy']:.3f}")
        print(f"🏆 F1-Score: {metrics['macro_f1']:.3f}")
        print(f"⏱️ Tiempo: {metrics['avg_response_time']:.2f}s")
        print(f"✅ Correctas: {metrics['correct_predictions']}/{metrics['total_predictions']}")
        
        # Top errores
        if metrics['error_distribution']:
            print(f"❌ Errores comunes:")
            for error, count in list(metrics['error_distribution'].items())[:3]:
                print(f"   {error}: {count}x")
    
    def run_evaluation(self, models: List[str], test_file: str, max_samples: Optional[int] = None,
                      output_prefix: str = "cog_evaluation", timeout: int = 60) -> Tuple[str, str]:
        """Ejecutar evaluación completa"""
        
        logger.info("🚀 INICIANDO EVALUACIÓN COG")
        logger.info(f"🤖 Modelos: {models}")
        logger.info(f"📚 Datos: {test_file}")
        
        # Cargar datos
        test_df = self.load_test_data(test_file)
        
        # Verificar modelos
        available_models = [m for m in models if self.check_model_availability(m)]
        
        if not available_models:
            raise ValueError("❌ No hay modelos disponibles")
        
        # Evaluar modelos
        all_results = []
        all_metrics = {'individual_models': {}}
        
        for model_name in available_models:
            model_results = self.evaluate_model(model_name, test_df, max_samples, timeout)
            metrics = self.calculate_metrics(model_results)
            
            all_results.extend(model_results)
            all_metrics['individual_models'][model_name] = metrics
            
            self.print_summary(metrics, model_name)
        
        # Comparación si hay múltiples modelos
        if len(available_models) > 1:
            print(f"\n🏁 COMPARACIÓN")
            print("=" * 50)
            
            summary = self._create_summary(all_metrics)
            print(f"🎯 Mejor Accuracy: {summary['best_accuracy']['model']} ({summary['best_accuracy']['value']:.3f})")
            print(f"🏆 Mejor F1: {summary['best_f1']['model']} ({summary['best_f1']['value']:.3f})")
            print(f"⚡ Más rápido: {summary['fastest']['model']} ({summary['fastest']['value']:.2f}s)")
            
            # Tabla comparativa
            print(f"\n📋 TABLA:")
            print(f"{'Modelo':<20} {'Accuracy':<10} {'F1':<10} {'Tiempo':<10}")
            print("-" * 50)
            for model in available_models:
                m = all_metrics['individual_models'][model]
                print(f"{model:<20} {m['accuracy']:<10.3f} {m['macro_f1']:<10.3f} {m['avg_response_time']:<10.2f}")
        
        # Guardar resultados
        csv_file, json_file = self.save_results(all_results, all_metrics, output_prefix)
        
        print(f"\n💾 ARCHIVOS GENERADOS:")
        print(f"  📊 {csv_file}")
        print(f"  📈 {json_file}")
        
        return csv_file, json_file

def main():
    """Función principal compacta"""
    parser = argparse.ArgumentParser(description='COG Evaluator Compacto - Compatible con procesador y fine-tuner')
    
    parser.add_argument('--models', nargs='+', required=True, help='Modelos a evaluar')
    parser.add_argument('--test-file', required=True, help='Archivo CSV con datos de prueba')
    parser.add_argument('--max-samples', type=int, help='Máximo muestras por modelo')
    parser.add_argument('--output-prefix', default='cog_evaluation', help='Prefijo archivos salida')
    parser.add_argument('--timeout', type=int, default=60, help='Timeout por predicción (s)')
    parser.add_argument('--list-models', action='store_true', help='Listar modelos Ollama')
    parser.add_argument('--preview-data', action='store_true', help='Preview de datos')
    
    args = parser.parse_args()
    
    evaluator = COGEvaluator()
    
    # Listar modelos
    if args.list_models:
        print("🤖 MODELOS DISPONIBLES:")
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
            for line in result.stdout.split('\n')[1:]:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 1:
                        print(f"  ✅ {parts[0]}")
        except:
            print("❌ Error accediendo a Ollama")
        return
    
    # Preview de datos
    if args.preview_data:
        print("📊 PREVIEW DATOS:")
        try:
            test_df = evaluator.load_test_data(args.test_file)
            
            print(f"\n📈 ESTADÍSTICAS:")
            print(f"  Total: {len(test_df)} muestras")
            print(f"  Categorías: {len(test_df['true_category'].unique())}")
            
            seq_lengths = test_df['sequence'].str.len()
            print(f"  Longitud secuencias: {seq_lengths.min()}-{seq_lengths.max()} aa (avg: {seq_lengths.mean():.1f})")
            
            print(f"\n📋 DISTRIBUCIÓN:")
            category_counts = test_df['true_category'].value_counts().head(5)
            for cat, count in category_counts.items():
                pct = (count / len(test_df)) * 100
                print(f"  {cat}: {count} ({pct:.1f}%)")
            
            print(f"\n🔍 EJEMPLO PROMPT:")
            print(test_df['prompt'].iloc[0][:300] + "...")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        return
    
    # Ejecutar evaluación
    try:
        print("🧬 COG EVALUATOR COMPACTO")
        print("=" * 50)
        print(f"🤖 Modelos: {', '.join(args.models)}")
        print(f"📚 Datos: {args.test_file}")
        if args.max_samples:
            print(f"📊 Muestras: {args.max_samples}")
        
        start_time = time.time()
        csv_file, json_file = evaluator.run_evaluation(
            models=args.models,
            test_file=args.test_file,
            max_samples=args.max_samples,
            output_prefix=args.output_prefix,
            timeout=args.timeout
        )
        total_time = time.time() - start_time
        
        print(f"\n🎉 EVALUACIÓN COMPLETADA!")
        print(f"⏰ Tiempo total: {total_time:.1f}s")
        
        print(f"\n🚀 ANÁLISIS RÁPIDO:")
        print(f"  import pandas as pd")
        print(f"  import json")
        print(f"  ")
        print(f"  # Cargar resultados")
        print(f"  df = pd.read_csv('{csv_file}')")
        print(f"  with open('{json_file}', 'r') as f:")
        print(f"      metrics = json.load(f)")
        print(f"  ")
        print(f"  # Accuracy por modelo")
        print(f"  accuracy_by_model = df.groupby('model')['correct'].mean()")
        print(f"  print(accuracy_by_model)")
        
        print(f"\n💡 COMANDOS ÚTILES:")
        print(f"  # Preview rápido:")
        print(f"  python {__file__} --test-file {args.test_file} --preview-data")
        print(f"  ")
        print(f"  # Evaluación rápida (50 muestras):")
        print(f"  python {__file__} --models {' '.join(args.models)} --test-file {args.test_file} --max-samples 50")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Evaluación interrumpida")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        print(f"\n🔧 POSIBLES SOLUCIONES:")
        print(f"  1. Verificar Ollama: ollama list")
        print(f"  2. Verificar datos: --preview-data")
        print(f"  3. Verificar modelos: --list-models")
        print(f"  4. Probar con menos muestras: --max-samples 10")

if __name__ == "__main__":
    main()