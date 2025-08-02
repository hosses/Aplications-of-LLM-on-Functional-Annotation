import os
import pandas as pd
import numpy as np
import requests
import json
import time
import re
import argparse
import sys
import sqlite3
import faiss
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
import logging
from dataclasses import dataclass

# Para embeddings de proteínas
import torch
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class COGPrediction:
    """Resultado de predicción COG"""
    sequence: str
    rag_prediction: Dict
    ollama_prediction: Dict
    ensemble_prediction: Dict
    confidence_score: float
    processing_time: float

class ProteinEmbedder:
    """Clase para generar embeddings de proteínas usando modelos pre-entrenados"""
    
    def __init__(self, model_name="facebook/esm2_t6_8M_UR50D", device="gpu"):
        self.device = device
        self.model_name = model_name
        
        logger.info(f"🧬 Cargando modelo de embeddings: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(device)
            self.model.eval()
            
            # Obtener dimensión de embeddings
            with torch.no_grad():
                test_input = self.tokenizer("M", return_tensors="pt").to(device)
                test_output = self.model(**test_input)
                self.embedding_dim = test_output.last_hidden_state.shape[-1]
            
            logger.info(f"✅ Modelo cargado. Dimensión: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelo: {e}")
            raise e
    
    def encode(self, sequences: List[str]) -> np.ndarray:
        """Genera embeddings para una lista de secuencias"""
        embeddings = []
        
        with torch.no_grad():
            for sequence in sequences:
                # Truncar secuencias muy largas
                seq = sequence[:1000] if len(sequence) > 1000 else sequence
                
                try:
                    inputs = self.tokenizer(
                        seq, 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True,
                        max_length=1022
                    ).to(self.device)
                    
                    outputs = self.model(**inputs)
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    embeddings.extend(embedding)
                    
                except Exception as e:
                    logger.warning(f"Error procesando secuencia: {e}")
                    embeddings.append(np.zeros(self.embedding_dim))
        
        return np.array(embeddings)

class COGRAGRetriever:
    """Sistema de recuperación para el RAG COG"""
    
    def __init__(self, rag_path: str = "RAG_COG"):
        self.rag_path = Path(rag_path)
        
        if not self.rag_path.exists():
            raise FileNotFoundError(f"No se encontró el directorio RAG: {rag_path}")
        
        logger.info(f"🔍 Cargando RAG COG desde: {self.rag_path}")
        
        # Cargar configuración
        config_path = self.rag_path / "config.json"
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Cargar índice FAISS
        faiss_path = str(self.rag_path / "cog_rag.index")
        self.index = faiss.read_index(faiss_path)
        
        # Conectar a base de datos
        db_path = self.rag_path / "cog_rag.db"
        self.conn = sqlite3.connect(db_path)
        
        # Inicializar embedder
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = ProteinEmbedder(
            model_name=self.config['model_name'], 
            device=device
        )
        
        logger.info(f"✅ RAG COG cargado: {self.config['total_entries']:,} entradas")
    
    def predict_category(self, query_sequence: str, k: int = 20) -> Dict:
        """Predice la categoría COG usando RAG"""
        # Generar embedding de la consulta
        query_embedding = self.embedder.encode([query_sequence])
        
        # Buscar en el índice
        distances, indices = self.index.search(
            query_embedding.astype(np.float32), k * 2
        )
        
        # Convertir distancias a similitudes
        similarities = 1 / (1 + distances[0])
        
        # Recuperar información de la base de datos
        similar_sequences = []
        for idx, similarity in zip(indices[0], similarities):
            if similarity >= 0.3:  # Umbral mínimo
                cursor = self.conn.execute(
                    "SELECT * FROM cog_entries WHERE id = ?", (int(idx),)
                )
                row = cursor.fetchone()
                
                if row:
                    similar_sequences.append({
                        'similarity': float(similarity),
                        'protein_id': row[1],
                        'cog_id': row[4],
                        'category_id': row[5],
                        'category_description': row[6],
                        'functional_group': row[7],
                        'organism': row[13]
                    })
                    
                    if len(similar_sequences) >= k:
                        break
        
        if not similar_sequences:
            return {
                'predicted_category': 'S',
                'category_description': 'Function unknown',
                'functional_group': 'POORLY CHARACTERIZED',
                'confidence': 0.0,
                'evidence': [],
                'method': 'RAG'
            }
        
        # Votar por categoría ponderado por similitud
        from collections import defaultdict
        category_votes = defaultdict(float)
        total_weight = 0
        
        for seq_info in similar_sequences:
            weight = seq_info['similarity']
            category = seq_info['category_id']
            category_votes[category] += weight
            total_weight += weight
        
        # Normalizar scores
        category_scores = {
            cat: score / total_weight 
            for cat, score in category_votes.items()
        }
        
        # Predicción final
        predicted_category = max(category_scores.keys(), key=category_scores.get)
        confidence = category_scores[predicted_category]
        
        return {
            'predicted_category': predicted_category,
            'category_description': self.config['categories'].get(predicted_category, 'Unknown'),
            'functional_group': self.config['functional_groups'].get(predicted_category, 'Unknown'),
            'confidence': float(confidence),
            'evidence': similar_sequences[:5],
            'category_scores': {k: float(v) for k, v in category_scores.items()},
            'method': 'RAG'
        }

class OllamaClient:
    """Cliente para interactuar con Ollama API"""
    
    def __init__(self, host="localhost", port=11434):
        self.base_url = f"http://{host}:{port}"
        self.api_url = f"{self.base_url}/api"
        self.available_models = self._get_available_models()
        
        logger.info(f"🦙 Cliente Ollama inicializado: {self.base_url}")
    
    def _get_available_models(self) -> List[str]:
        """Obtiene lista de modelos disponibles en Ollama"""
        try:
            response = requests.get(f"{self.api_url}/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            return []
        except Exception as e:
            logger.warning(f"⚠️ Error conectando con Ollama: {e}")
            return []
    
    def is_model_available(self, model_name: str) -> bool:
        """Verifica si un modelo específico está disponible"""
        return model_name in self.available_models
    
    def generate_text(self, model: str, prompt: str, **kwargs) -> Optional[Dict]:
        """Genera texto usando Ollama"""
        if not self.is_model_available(model):
            logger.error(f"❌ Modelo {model} no disponible")
            return None
        
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            **kwargs
        }
        
        try:
            response = requests.post(f"{self.api_url}/generate", json=data, timeout=60)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"❌ Error Ollama: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error en generación: {e}")
            return None

class EnhancedCOGPromptGenerator:
    """Generador de prompts mejorado que incluye evidencia del RAG"""
    
    def __init__(self):
        self.cog_categories = {
            'J': 'Translation, ribosomal structure and biogenesis',
            'A': 'RNA processing and modification',
            'K': 'Transcription', 
            'L': 'Replication, recombination and repair',
            'B': 'Chromatin structure and dynamics',
            'D': 'Cell cycle control, cell division',
            'O': 'Molecular chaperones and related functions',
            'M': 'Cell wall/membrane/envelope biogenesis',
            'N': 'Cell motility',
            'P': 'Inorganic ion transport and metabolism',
            'T': 'Signal transduction mechanisms',
            'U': 'Intracellular trafficking, secretion',
            'V': 'Defense mechanisms',
            'W': 'Extracellular structures',
            'C': 'Energy production and conversion',
            'G': 'Carbohydrate transport and metabolism',
            'E': 'Amino acid transport and metabolism',
            'F': 'Nucleotide transport and metabolism',
            'H': 'Coenzyme transport and metabolism',
            'I': 'Lipid transport and metabolism',
            'Q': 'Secondary metabolites biosynthesis',
            'R': 'General function prediction only',
            'S': 'Function unknown'
        }
        
        self.functional_groups = {
            'J': 'INFORMATION STORAGE AND PROCESSING',
            'A': 'INFORMATION STORAGE AND PROCESSING',
            'K': 'INFORMATION STORAGE AND PROCESSING',
            'L': 'INFORMATION STORAGE AND PROCESSING',
            'B': 'INFORMATION STORAGE AND PROCESSING',
            'D': 'CELLULAR PROCESSES AND SIGNALING',
            'O': 'CELLULAR PROCESSES AND SIGNALING',
            'M': 'CELLULAR PROCESSES AND SIGNALING',
            'N': 'CELLULAR PROCESSES AND SIGNALING',
            'P': 'CELLULAR PROCESSES AND SIGNALING',
            'T': 'CELLULAR PROCESSES AND SIGNALING',
            'U': 'CELLULAR PROCESSES AND SIGNALING',
            'V': 'CELLULAR PROCESSES AND SIGNALING',
            'W': 'CELLULAR PROCESSES AND SIGNALING',
            'C': 'METABOLISM',
            'G': 'METABOLISM',
            'E': 'METABOLISM',
            'F': 'METABOLISM',
            'H': 'METABOLISM',
            'I': 'METABOLISM',
            'Q': 'METABOLISM',
            'R': 'POORLY CHARACTERIZED',
            'S': 'POORLY CHARACTERIZED'
        }
    
    def generate_enhanced_prompt(self, sequence: str, rag_evidence: List[Dict] = None) -> str:
        """Genera prompt mejorado con evidencia del RAG"""
        
        base_prompt = f"""You are an expert protein function prediction system. Analyze the given protein sequence and predict its COG functional category and group.

COG CATEGORIES AND GROUPS:

INFORMATION STORAGE AND PROCESSING:
- J: Translation, ribosomal structure and biogenesis
- A: RNA processing and modification
- K: Transcription
- L: Replication, recombination and repair
- B: Chromatin structure and dynamics

CELLULAR PROCESSES AND SIGNALING:
- D: Cell cycle control, cell division
- O: Molecular chaperones and related functions
- M: Cell wall/membrane/envelope biogenesis
- N: Cell motility
- P: Inorganic ion transport and metabolism
- T: Signal transduction mechanisms
- U: Intracellular trafficking, secretion
- V: Defense mechanisms
- W: Extracellular structures

METABOLISM:
- C: Energy production and conversion
- G: Carbohydrate transport and metabolism
- E: Amino acid transport and metabolism
- F: Nucleotide transport and metabolism
- H: Coenzyme transport and metabolism
- I: Lipid transport and metabolism
- Q: Secondary metabolites biosynthesis

POORLY CHARACTERIZED:
- R: General function prediction only
- S: Function unknown

PROTEIN TO ANALYZE:
Sequence: {sequence}
Length: {len(sequence)} amino acids"""

        # Agregar evidencia del RAG si está disponible
        if rag_evidence and len(rag_evidence) > 0:
            base_prompt += f"\n\nSIMILAR PROTEINS FOUND (for reference):"
            for i, evidence in enumerate(rag_evidence[:3], 1):
                base_prompt += f"""
Evidence {i}:
- Similarity: {evidence['similarity']:.3f}
- Category: {evidence['category_id']} ({evidence['category_description']})
- Group: {evidence['functional_group']}
- Organism: {evidence['organism']}
- COG ID: {evidence['cog_id']}"""

        base_prompt += f"""

ANALYSIS INSTRUCTIONS:
1. Examine the amino acid sequence for characteristic patterns and motifs
2. Consider the sequence length and composition
3. If similar proteins are provided, use them as supporting evidence
4. Make your best prediction based on sequence analysis

REQUIRED OUTPUT FORMAT:
Analysis: [Brief explanation of your reasoning, including any relevant sequence features or similarity to known proteins]
Category: [Single letter: J,A,K,L,B,D,O,M,N,P,T,U,V,W,C,G,E,F,H,I,Q,R,S]
Group: [Full group name: INFORMATION STORAGE AND PROCESSING, CELLULAR PROCESSES AND SIGNALING, METABOLISM, or POORLY CHARACTERIZED]
Confidence: [High/Medium/Low]"""

        return base_prompt
    
    def extract_prediction_from_response(self, response: str) -> Dict:
        """Extrae la predicción estructurada de la respuesta del LLM"""
        response = response.strip()
        
        # Buscar categoría
        category_match = re.search(r'Category:\s*([A-S])\b', response, re.IGNORECASE)
        category = category_match.group(1).upper() if category_match else 'S'
        
        # Buscar grupo
        group_patterns = [
            r'Group:\s*(INFORMATION STORAGE AND PROCESSING|CELLULAR PROCESSES AND SIGNALING|METABOLISM|POORLY CHARACTERIZED)',
            r'Group:\s*(Information Storage and Processing|Cellular Processes and Signaling|Metabolism|Poorly Characterized)'
        ]
        
        group = 'POORLY CHARACTERIZED'  # Default
        for pattern in group_patterns:
            group_match = re.search(pattern, response, re.IGNORECASE)
            if group_match:
                group = group_match.group(1).upper()
                break
        
        # Buscar confianza
        confidence_match = re.search(r'Confidence:\s*(High|Medium|Low)', response, re.IGNORECASE)
        confidence = confidence_match.group(1).title() if confidence_match else 'Low'
        
        # Extraer análisis
        analysis_match = re.search(r'Analysis:\s*([^\n]+(?:\n[^Category]+)*)', response, re.IGNORECASE)
        analysis = analysis_match.group(1).strip() if analysis_match else "No analysis provided"
        
        return {
            'predicted_category': category,
            'category_description': self.cog_categories.get(category, 'Unknown'),
            'functional_group': group,
            'confidence': confidence,
            'analysis': analysis,
            'raw_response': response,
            'method': 'Ollama'
        }

class RAGOllamaCOGPredictor:
    """Predictor híbrido que combina RAG y Ollama"""
    
    def __init__(self, rag_path: str = "RAG_COG", ollama_host: str = "localhost", 
                 ollama_port: int = 11434, preferred_model: str = None):
        
        # Inicializar componentes
        self.rag_retriever = COGRAGRetriever(rag_path)
        self.ollama_client = OllamaClient(ollama_host, ollama_port)
        self.prompt_generator = EnhancedCOGPromptGenerator()
        
        # Seleccionar modelo preferido o el mejor disponible
        self.model = self._select_best_model(preferred_model)
        
        logger.info(f"🤖 Predictor híbrido inicializado")
        logger.info(f"🔍 RAG: {self.rag_retriever.config['total_entries']:,} entradas")
        logger.info(f"🦙 Ollama: {self.model}")
    
    def _select_best_model(self, preferred_model: str = None) -> str:
        """Selecciona el mejor modelo disponible"""
        if preferred_model and self.ollama_client.is_model_available(preferred_model):
            return preferred_model
        
        # Orden de preferencia de modelos
        preferred_models = [
            "llama3.2:3b", "llama3.1:8b", "mistral:7b", "qwen2.5:7b",
            "gemma2:9b", "deepseek-coder:6.7b", "phi3:medium"
        ]
        
        for model in preferred_models:
            if self.ollama_client.is_model_available(model):
                return model
        
        # Si no hay modelos preferidos, usar el primero disponible
        if self.ollama_client.available_models:
            return self.ollama_client.available_models[0]
        
        raise RuntimeError("No hay modelos de Ollama disponibles")
    
    def predict(self, sequence: str, use_rag: bool = True, use_ollama: bool = True) -> COGPrediction:
        """Realiza predicción completa usando RAG y/o Ollama"""
        start_time = time.time()
        
        logger.info(f"🧬 Analizando secuencia de {len(sequence)} aminoácidos...")
        
        rag_prediction = None
        ollama_prediction = None
        
        # Predicción con RAG
        if use_rag:
            logger.info("🔍 Ejecutando predicción RAG...")
            rag_prediction = self.rag_retriever.predict_category(sequence)
        
        # Predicción con Ollama
        if use_ollama:
            logger.info(f"🦙 Ejecutando predicción Ollama ({self.model})...")
            
            # Generar prompt con evidencia del RAG si está disponible
            evidence = rag_prediction.get('evidence', []) if rag_prediction else []
            prompt = self.prompt_generator.generate_enhanced_prompt(sequence, evidence)
            
            # Parámetros optimizados
            params = {
                'temperature': 0.1,
                'top_p': 0.9,
                'top_k': 40
            }
            
            # Generar respuesta
            result = self.ollama_client.generate_text(self.model, prompt, **params)
            
            if result and 'response' in result:
                ollama_prediction = self.prompt_generator.extract_prediction_from_response(result['response'])
            else:
                ollama_prediction = {
                    'predicted_category': 'S',
                    'category_description': 'Function unknown',
                    'functional_group': 'POORLY CHARACTERIZED',
                    'confidence': 'Low',
                    'analysis': 'Error in model response',
                    'method': 'Ollama'
                }
        
        # Crear predicción ensemble
        ensemble_prediction = self._create_ensemble_prediction(rag_prediction, ollama_prediction)
        
        # Calcular confianza general
        confidence_score = self._calculate_overall_confidence(rag_prediction, ollama_prediction, ensemble_prediction)
        
        processing_time = time.time() - start_time
        
        return COGPrediction(
            sequence=sequence,
            rag_prediction=rag_prediction,
            ollama_prediction=ollama_prediction,
            ensemble_prediction=ensemble_prediction,
            confidence_score=confidence_score,
            processing_time=processing_time
        )
    
    def _create_ensemble_prediction(self, rag_pred: Dict, ollama_pred: Dict) -> Dict:
        """Crea predicción ensemble combinando RAG y Ollama"""
        if not rag_pred and not ollama_pred:
            return {
                'predicted_category': 'S',
                'category_description': 'Function unknown',
                'functional_group': 'POORLY CHARACTERIZED',
                'confidence': 'Low',
                'method': 'Ensemble'
            }
        
        if not rag_pred:
            return {**ollama_pred, 'method': 'Ensemble (Ollama only)'}
        
        if not ollama_pred:
            return {**rag_pred, 'method': 'Ensemble (RAG only)'}
        
        # Ambos métodos disponibles - crear ensemble
        rag_category = rag_pred['predicted_category']
        ollama_category = ollama_pred['predicted_category']
        
        # Si coinciden, usar esa categoría con alta confianza
        if rag_category == ollama_category:
            return {
                'predicted_category': rag_category,
                'category_description': rag_pred['category_description'],
                'functional_group': rag_pred['functional_group'],
                'confidence': 'High',
                'method': 'Ensemble (Agreement)',
                'agreement': True
            }
        
        # Si no coinciden, priorizar RAG si tiene alta confianza
        if rag_pred.get('confidence', 0) > 0.7:
            return {
                'predicted_category': rag_category,
                'category_description': rag_pred['category_description'],
                'functional_group': rag_pred['functional_group'],
                'confidence': 'Medium',
                'method': 'Ensemble (RAG priority)',
                'agreement': False,
                'alternative': ollama_category
            }
        
        # Si RAG tiene baja confianza, usar Ollama
        return {
            'predicted_category': ollama_category,
            'category_description': ollama_pred['category_description'],
            'functional_group': ollama_pred['functional_group'],
            'confidence': 'Medium',
            'method': 'Ensemble (Ollama priority)',
            'agreement': False,
            'alternative': rag_category
        }
    
    def _calculate_overall_confidence(self, rag_pred: Dict, ollama_pred: Dict, ensemble_pred: Dict) -> float:
        """Calcula confianza general de la predicción"""
        confidence_score = 0.5  # Base
        
        if rag_pred and ollama_pred:
            # Ambos métodos disponibles
            if rag_pred['predicted_category'] == ollama_pred['predicted_category']:
                confidence_score = 0.9  # Alta confianza por acuerdo
            else:
                confidence_score = 0.6  # Confianza media por desacuerdo
            
            # Ajustar por confianza del RAG
            if isinstance(rag_pred.get('confidence'), float):
                confidence_score *= (0.5 + 0.5 * rag_pred['confidence'])
        
        elif rag_pred:
            # Solo RAG
            if isinstance(rag_pred.get('confidence'), float):
                confidence_score = 0.4 + 0.5 * rag_pred['confidence']
            else:
                confidence_score = 0.6
        
        elif ollama_pred:
            # Solo Ollama
            ollama_conf = ollama_pred.get('confidence', 'Low')
            if ollama_conf == 'High':
                confidence_score = 0.8
            elif ollama_conf == 'Medium':
                confidence_score = 0.6
            else:
                confidence_score = 0.4
        
        return min(confidence_score, 1.0)

def print_prediction_results(prediction: COGPrediction):
    """Imprime los resultados de la predicción de manera formateda"""
    print(f"\n{'='*80}")
    print(f"🧬 ANÁLISIS DE SECUENCIA PROTEICA")
    print(f"{'='*80}")
    
    print(f"📊 Información básica:")
    print(f"   Longitud: {len(prediction.sequence)} aminoácidos")
    print(f"   Secuencia: {prediction.sequence[:50]}{'...' if len(prediction.sequence) > 50 else ''}")
    print(f"   Tiempo de procesamiento: {prediction.processing_time:.2f} segundos")
    
    print(f"\n🎯 PREDICCIÓN FINAL (ENSEMBLE):")
    ensemble = prediction.ensemble_prediction
    print(f"   🏷️ Categoría: {ensemble['predicted_category']}")
    print(f"   📝 Descripción: {ensemble['category_description']}")
    print(f"   🔬 Grupo funcional: {ensemble['functional_group']}")
    print(f"   📈 Confianza: {ensemble['confidence']}")
    print(f"   🤖 Método: {ensemble['method']}")
    
    if 'agreement' in ensemble:
        if ensemble['agreement']:
            print(f"   ✅ Concordancia: RAG y Ollama coinciden")
        else:
            print(f"   ⚠️ Discordancia: Alternativa es {ensemble.get('alternative', 'N/A')}")
    
    print(f"   🎯 Confianza general: {prediction.confidence_score:.3f}")
    
    # Detalles del RAG
    if prediction.rag_prediction:
        print(f"\n🔍 DETALLES RAG:")
        rag = prediction.rag_prediction
        print(f"   Categoría predicha: {rag['predicted_category']} ({rag['category_description']})")
        print(f"   Confianza RAG: {rag['confidence']:.3f}")
        
        if rag.get('evidence'):
            print(f"   📋 Evidencia (Top 3 proteínas similares):")
            for i, evidence in enumerate(rag['evidence'][:3], 1):
                print(f"      {i}. Similitud: {evidence['similarity']:.3f} | "
                      f"Cat: {evidence['category_id']} | "
                      f"COG: {evidence['cog_id']} | "
                      f"Organismo: {evidence['organism'][:30]}...")
        
        if rag.get('category_scores'):
            print(f"   📊 Scores por categoría (Top 5):")
            sorted_scores = sorted(rag['category_scores'].items(), key=lambda x: x[1], reverse=True)
            for cat, score in sorted_scores[:5]:
                print(f"      {cat}: {score:.3f}")
    
    # Detalles de Ollama
    if prediction.ollama_prediction:
        print(f"\n🦙 DETALLES OLLAMA:")
        ollama = prediction.ollama_prediction
        print(f"   Categoría predicha: {ollama['predicted_category']} ({ollama['category_description']})")
        print(f"   Confianza Ollama: {ollama['confidence']}")
        
        if ollama.get('analysis'):
            print(f"   🧠 Análisis del modelo:")
            # Formatear el análisis para que no sea muy largo
            analysis = ollama['analysis'][:300] + "..." if len(ollama['analysis']) > 300 else ollama['analysis']
            print(f"      {analysis}")
    
    print(f"\n{'='*80}")

def parse_arguments():
    """Parsea argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description='Predictor híbrido COG usando RAG + Ollama para secuencias individuales',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

1. Predicción interactiva (recomendado):
   python rag_ollama_predictor.py

2. Predicción de secuencia específica:
   python rag_ollama_predictor.py --sequence "MKKLFAGASTYIVADLAGGVQAEVFVEMGDAHTLLGLVAARTGISLDAFASQLPDVLA"

3. Solo usar RAG (sin Ollama):
   python rag_ollama_predictor.py --sequence "MKKL..." --rag-only

4. Solo usar Ollama (sin RAG):
   python rag_ollama_predictor.py --sequence "MKKL..." --ollama-only

5. Especificar modelo de Ollama:
   python rag_ollama_predictor.py --sequence "MKKL..." --model llama3.2:3b

6. Especificar ruta del RAG:
   python rag_ollama_predictor.py --sequence "MKKL..." --rag-path /path/to/RAG_COG

FUNCIONALIDADES:
✅ Predicción híbrida RAG + Ollama con ensemble
✅ Análisis detallado de evidencia y confianza
✅ Modo interactivo para múltiples consultas
✅ Selección automática del mejor modelo disponible
✅ Prompt mejorado con contexto de proteínas similares
        """
    )
    
    parser.add_argument(
        '--sequence', '-s',
        type=str,
        help='Secuencia de proteína a analizar'
    )
    
    parser.add_argument(
        '--rag-path',
        type=str,
        default='RAG_COG',
        help='Ruta al directorio RAG COG (default: RAG_COG)'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        help='Modelo específico de Ollama a usar'
    )
    
    parser.add_argument(
        '--ollama-host',
        type=str,
        default='localhost',
        help='Host de Ollama (default: localhost)'
    )
    
    parser.add_argument(
        '--ollama-port',
        type=int,
        default=11434,
        help='Puerto de Ollama (default: 11434)'
    )
    
    parser.add_argument(
        '--rag-only',
        action='store_true',
        help='Solo usar RAG para predicción'
    )
    
    parser.add_argument(
        '--ollama-only',
        action='store_true',
        help='Solo usar Ollama para predicción'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Modo interactivo para múltiples consultas'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Mostrar información detallada adicional'
    )
    
    return parser.parse_args()

def interactive_mode(predictor: RAGOllamaCOGPredictor):
    """Modo interactivo para consultas múltiples"""
    print(f"\n🤖 MODO INTERACTIVO - Predictor COG Híbrido")
    print(f"{'='*60}")
    print(f"📝 Ingresa secuencias de proteínas para análisis")
    print(f"💡 Comandos especiales:")
    print(f"   'quit' o 'exit' - Salir")
    print(f"   'help' - Mostrar ayuda")
    print(f"   'stats' - Estadísticas del RAG")
    print(f"   'models' - Modelos disponibles")
    print(f"{'='*60}\n")
    
    session_count = 0
    
    while True:
        try:
            # Solicitar input
            sequence = input("🧬 Ingresa secuencia (o comando): ").strip()
            
            if not sequence:
                continue
            
            # Procesar comandos especiales
            if sequence.lower() in ['quit', 'exit', 'q']:
                print(f"\n👋 Cerrando predictor. Consultas realizadas: {session_count}")
                break
            
            elif sequence.lower() in ['help', 'h']:
                print(f"\n📖 AYUDA:")
                print(f"   - Pega una secuencia de aminoácidos (ej: MKKLFAGA...)")
                print(f"   - El sistema usará RAG + Ollama para predicción")
                print(f"   - Comandos: quit, help, stats, models")
                print(f"   - Formatos aceptados: secuencia continua o con espacios")
                continue
            
            elif sequence.lower() in ['stats', 'statistics']:
                stats = predictor.rag_retriever.get_statistics()
                print(f"\n📊 ESTADÍSTICAS DEL RAG:")
                print(f"   📈 Total entradas: {stats['total_entries']:,}")
                print(f"   🧮 Dimensión embeddings: {stats['embedding_dimension']}")
                print(f"   🤖 Modelo embeddings: {stats['model_name']}")
                print(f"   🏷️ Categorías: {len(stats['categories'])}")
                continue
            
            elif sequence.lower() in ['models', 'model']:
                print(f"\n🦙 MODELOS OLLAMA DISPONIBLES:")
                for model in predictor.ollama_client.available_models:
                    mark = "✅" if model == predictor.model else "  "
                    print(f"   {mark} {model}")
                print(f"\n🎯 Modelo actual: {predictor.model}")
                continue
            
            # Limpiar secuencia
            clean_sequence = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', sequence.upper())
            
            if len(clean_sequence) < 10:
                print(f"⚠️ Secuencia muy corta ({len(clean_sequence)} aa). Mínimo: 10 aminoácidos")
                continue
            
            if len(clean_sequence) > 2000:
                print(f"⚠️ Secuencia muy larga ({len(clean_sequence)} aa). Máximo: 2000 aminoácidos")
                continue
            
            # Realizar predicción
            session_count += 1
            print(f"\n🔬 Consulta #{session_count} - Analizando {len(clean_sequence)} aminoácidos...")
            
            prediction = predictor.predict(clean_sequence)
            print_prediction_results(prediction)
            
            # Preguntar si quiere continuar
            print(f"\n💡 ¿Otra consulta? (Enter para continuar, 'quit' para salir)")
            
        except KeyboardInterrupt:
            print(f"\n\n👋 Predictor interrumpido. Consultas realizadas: {session_count}")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print(f"💡 Intenta de nuevo con una secuencia válida")
            continue

def validate_sequence(sequence: str) -> Tuple[bool, str, str]:
    """Valida una secuencia de proteína"""
    if not sequence:
        return False, "Secuencia vacía", ""
    
    # Limpiar secuencia
    clean_sequence = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', sequence.upper())
    
    if not clean_sequence:
        return False, "No se encontraron aminoácidos válidos", ""
    
    if len(clean_sequence) < 10:
        return False, f"Secuencia muy corta ({len(clean_sequence)} aa). Mínimo: 10", clean_sequence
    
    if len(clean_sequence) > 5000:
        return False, f"Secuencia muy larga ({len(clean_sequence)} aa). Máximo: 5000", clean_sequence
    
    # Verificar porcentaje de aminoácidos válidos
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
    valid_count = sum(1 for aa in clean_sequence if aa in valid_aa)
    valid_percentage = (valid_count / len(clean_sequence)) * 100
    
    if valid_percentage < 90:
        return False, f"Muchos caracteres inválidos ({valid_percentage:.1f}% válidos)", clean_sequence
    
    return True, "Secuencia válida", clean_sequence

def main():
    """Función principal"""
    args = parse_arguments()
    
    print(f"🧬 Predictor COG Híbrido (RAG + Ollama)")
    print(f"{'='*50}")
    
    try:
        # Inicializar predictor
        logger.info(f"🚀 Inicializando predictor híbrido...")
        predictor = RAGOllamaCOGPredictor(
            rag_path=args.rag_path,
            ollama_host=args.ollama_host,
            ollama_port=args.ollama_port,
            preferred_model=args.model
        )
        
        # Configurar métodos a usar
        use_rag = not args.ollama_only
        use_ollama = not args.rag_only
        
        print(f"\n⚙️ CONFIGURACIÓN:")
        print(f"   🔍 RAG: {'✅ Activado' if use_rag else '❌ Desactivado'}")
        print(f"   🦙 Ollama: {'✅ Activado' if use_ollama else '❌ Desactivado'}")
        if use_ollama:
            print(f"   🤖 Modelo: {predictor.model}")
        print(f"   📁 RAG Path: {args.rag_path}")
        
        # Modo interactivo
        if args.interactive or not args.sequence:
            interactive_mode(predictor)
            return
        
        # Modo de secuencia única
        logger.info(f"🧬 Modo de predicción única")
        
        # Validar secuencia
        is_valid, message, clean_sequence = validate_sequence(args.sequence)
        
        if not is_valid:
            print(f"\n❌ Error en secuencia: {message}")
            if clean_sequence:
                print(f"🧹 Secuencia limpia: {clean_sequence[:50]}...")
                print(f"💡 ¿Quieres usar la secuencia limpia? (y/n)")
                response = input().strip().lower()
                if response not in ['y', 'yes', 'sí', 's']:
                    return
                args.sequence = clean_sequence
            else:
                return
        else:
            args.sequence = clean_sequence
        
        print(f"\n✅ {message}")
        print(f"📏 Longitud: {len(args.sequence)} aminoácidos")
        
        # Realizar predicción
        prediction = predictor.predict(args.sequence, use_rag=use_rag, use_ollama=use_ollama)
        
        # Mostrar resultados
        print_prediction_results(prediction)
        
        # Información adicional en modo verbose
        if args.verbose:
            print(f"\n🔍 INFORMACIÓN ADICIONAL:")
            
            if prediction.rag_prediction and 'category_scores' in prediction.rag_prediction:
                print(f"\n📊 Scores completos RAG:")
                for cat, score in sorted(prediction.rag_prediction['category_scores'].items(), 
                                       key=lambda x: x[1], reverse=True):
                    desc = predictor.rag_retriever.config['categories'].get(cat, 'Unknown')
                    print(f"   {cat}: {score:.3f} ({desc})")
            
            if prediction.ollama_prediction and 'raw_response' in prediction.ollama_prediction:
                print(f"\n🦙 Respuesta completa de Ollama:")
                print(f"   {prediction.ollama_prediction['raw_response'][:500]}...")
        
        print(f"\n💡 Para más consultas, usa: python {__file__} --interactive")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"💡 Asegúrate de que el RAG esté construido en: {args.rag_path}")
        print(f"💡 Ejecuta primero: python rag_cog_builder.py --build")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"❌ Error inesperado: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()