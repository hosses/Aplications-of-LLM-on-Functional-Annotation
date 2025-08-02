import sqlite3
import numpy as np
import pandas as pd
import json
import gzip
from pathlib import Path
from Bio import SeqIO
import csv
from typing import Dict, List, Tuple
import argparse

class EmbeddingDataTracer:
    """Rastreador de datos usados para crear embeddings en el RAG COG"""
    
    def __init__(self, rag_path: str = "RAG_COG", cog_base_path: str = None):
        self.rag_path = Path(rag_path)
        self.cog_base_path = Path(cog_base_path) if cog_base_path else None
        
        # Archivos del RAG
        self.db_path = self.rag_path / "cog_rag.db"
        self.config_path = self.rag_path / "config.json"
        self.embeddings_path = self.rag_path / "embeddings.npy"
        
        print(f"🔍 Analizando RAG: {self.rag_path}")
        
        # Verificar archivos
        if not self.db_path.exists():
            raise FileNotFoundError(f"Base de datos no encontrada: {self.db_path}")
        
        # Conectar a la base de datos
        self.conn = sqlite3.connect(self.db_path)
        
        # Cargar configuración
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            print(f"✅ Configuración cargada")
        else:
            self.config = {}
            print("⚠️ No se encontró archivo de configuración")
    
    def analyze_rag_creation_process(self):
        """Analiza el proceso de creación del RAG"""
        print(f"\n📊 ANÁLISIS DEL PROCESO DE CREACIÓN DEL RAG")
        print("=" * 60)
        
        # Información de la configuración
        if self.config:
            print(f"🔧 Configuración del RAG:")
            print(f"   📈 Total de entradas: {self.config.get('total_entries', 'N/A'):,}")
            print(f"   🧮 Dimensión embeddings: {self.config.get('embedding_dim', 'N/A')}")
            print(f"   🤖 Modelo usado: {self.config.get('model_name', 'N/A')}")
            print(f"   📅 Fecha de creación: {self.config.get('build_date', 'N/A')}")
        
        # Analizar contenido de la base de datos
        print(f"\n📊 Contenido de la base de datos:")
        
        # Contar entradas por categoría
        cursor = self.conn.execute("""
            SELECT category_id, COUNT(*) as count,
                   AVG(sequence_length) as avg_seq_len,
                   AVG(functional_length) as avg_func_len,
                   MIN(membership_class) as min_quality,
                   MAX(membership_class) as max_quality
            FROM cog_entries 
            GROUP BY category_id
            ORDER BY count DESC
        """)
        
        category_data = cursor.fetchall()
        total_entries = sum(row[1] for row in category_data)
        
        print(f"   📈 Total de secuencias en RAG: {total_entries:,}")
        print(f"   🏷️ Categorías COG: {len(category_data)}")
        
        print(f"\n📋 Distribución por categoría (Top 10):")
        for i, (cat, count, avg_seq, avg_func, min_qual, max_qual) in enumerate(category_data[:10], 1):
            percentage = (count / total_entries) * 100
            print(f"   {i:2d}. {cat}: {count:,} secuencias ({percentage:.1f}%) "
                  f"| Longitud: {avg_seq:.0f} → {avg_func:.0f} (footprint)")
    
    def trace_data_sources(self):
        """Rastrea las fuentes de datos originales"""
        print(f"\n📁 RASTREANDO FUENTES DE DATOS ORIGINALES")
        print("=" * 60)
        
        # Archivos COG esperados
        expected_files = {
            'cog-24.cog.csv': 'Mapeo proteína → COG con footprints',
            'cog-24.def.tab': 'Definiciones de COGs y categorías',
            'COGorg24.faa.gz': 'Secuencias proteicas completas',
            'cog-24.org.csv': 'Información de organismos',
            'cog-24.fun.tab': 'Categorías funcionales COG'
        }
        
        if self.cog_base_path and self.cog_base_path.exists():
            print(f"🔍 Verificando archivos fuente en: {self.cog_base_path}")
            
            for filename, description in expected_files.items():
                file_path = self.cog_base_path / filename
                if file_path.exists():
                    if filename.endswith('.gz'):
                        try:
                            with gzip.open(file_path, 'rt') as f:
                                # Contar líneas aproximadamente
                                line_count = sum(1 for _ in f)
                            size_mb = file_path.stat().st_size / (1024 * 1024)
                            print(f"   ✅ {filename}: {line_count:,} secuencias, {size_mb:.1f} MB")
                        except:
                            size_mb = file_path.stat().st_size / (1024 * 1024)
                            print(f"   ✅ {filename}: {size_mb:.1f} MB")
                    else:
                        try:
                            with open(file_path, 'r') as f:
                                line_count = sum(1 for _ in f)
                            size_kb = file_path.stat().st_size / 1024
                            print(f"   ✅ {filename}: {line_count:,} líneas, {size_kb:.1f} KB")
                        except:
                            size_kb = file_path.stat().st_size / 1024
                            print(f"   ✅ {filename}: {size_kb:.1f} KB")
                else:
                    print(f"   ❌ {filename}: No encontrado")
        else:
            print(f"⚠️ Ruta de archivos COG no especificada o no existe")
            print(f"💡 Usa --cog-path para especificar la ubicación de los archivos COG2024")
    
    def analyze_specific_protein(self, protein_id: str = None, sequence_id: int = None):
        """Analiza una proteína específica y rastrea su origen"""
        print(f"\n🧬 RASTREANDO PROTEÍNA ESPECÍFICA")
        print("=" * 50)
        
        if protein_id:
            cursor = self.conn.execute("""
                SELECT id, protein_id, sequence, functional_region, cog_id, 
                       category_id, organism, footprint, footprint_length,
                       membership_class, bit_score, e_value, assembly_id
                FROM cog_entries 
                WHERE protein_id = ?
            """, (protein_id,))
        elif sequence_id is not None:
            cursor = self.conn.execute("""
                SELECT id, protein_id, sequence, functional_region, cog_id, 
                       category_id, organism, footprint, footprint_length,
                       membership_class, bit_score, e_value, assembly_id
                FROM cog_entries 
                WHERE id = ?
            """, (sequence_id,))
        else:
            print("❌ Debe especificar protein_id o sequence_id")
            return
        
        result = cursor.fetchone()
        
        if not result:
            print("❌ Proteína no encontrada")
            return
        
        (seq_id, prot_id, sequence, func_region, cog_id, category, 
         organism, footprint, footprint_len, membership_class, 
         bit_score, e_value, assembly_id) = result
        
        print(f"✅ Proteína encontrada:")
        print(f"   🆔 ID en RAG: {seq_id}")
        print(f"   🧬 Protein ID: {prot_id}")
        print(f"   🏷️ Categoría COG: {category}")
        print(f"   🔬 COG ID: {cog_id}")
        print(f"   🦠 Organismo: {organism}")
        print(f"   📊 Assembly ID: {assembly_id}")
        
        print(f"\n📏 Información de secuencia:")
        print(f"   📐 Secuencia completa: {len(sequence)} aminoácidos")
        print(f"   🎯 Región funcional: {len(func_region)} aminoácidos")
        print(f"   📍 Footprint: {footprint}")
        print(f"   📏 Longitud footprint: {footprint_len}")
        
        print(f"\n🔍 Información de calidad:")
        print(f"   ⭐ Membership class: {membership_class}")
        print(f"   📊 Bit score: {bit_score}")
        print(f"   📈 E-value: {e_value}")
        
        print(f"\n🧬 Secuencias:")
        print(f"   Completa: {sequence[:50]}...")
        print(f"   Funcional: {func_region[:50]}...")
        
        # Verificar si las secuencias son diferentes
        if sequence != func_region:
            print(f"   ✂️ Footprint extraído: Secuencia funcional es diferente de la completa")
        else:
            print(f"   📝 Sin footprint: Secuencia funcional = secuencia completa")
        
        return {
            'id': seq_id,
            'protein_id': prot_id,
            'sequence': sequence,
            'functional_region': func_region,
            'cog_id': cog_id,
            'category': category,
            'organism': organism,
            'footprint': footprint,
            'quality_info': {
                'membership_class': membership_class,
                'bit_score': bit_score,
                'e_value': e_value
            }
        }
    
    def show_embedding_creation_pipeline(self):
        """Muestra el pipeline de creación de embeddings"""
        print(f"\n⚙️ PIPELINE DE CREACIÓN DE EMBEDDINGS")
        print("=" * 60)
        
        pipeline_steps = [
            {
                'step': 1,
                'title': 'Carga de definiciones COG',
                'file': 'cog-24.def.tab',
                'description': 'Mapeo COG_ID → Categoría funcional',
                'example': 'COG0001 → J (Translation, ribosomal...)'
            },
            {
                'step': 2,
                'title': 'Carga de asignaciones proteína-COG',
                'file': 'cog-24.cog.csv',
                'description': 'Mapeo Protein_ID → COG_ID + footprint',
                'example': 'WP_000001.1 → COG0001, footprint: 45-380'
            },
            {
                'step': 3,
                'title': 'Procesamiento de secuencias',
                'file': 'COGorg24.faa.gz',
                'description': 'Lectura de secuencias FASTA + extracción de footprints',
                'example': 'MAKT... → LKDFIF... (región 45-380)'
            },
            {
                'step': 4,
                'title': 'Filtrado de calidad',
                'file': 'Filtros aplicados',
                'description': 'membership_class ≤ 2, footprint_length ≥ 30',
                'example': f'{self.config.get("total_entries", "N/A")} secuencias finales'
            },
            {
                'step': 5,
                'title': 'Generación de embeddings',
                'file': f'Modelo: {self.config.get("model_name", "ESM-2")}',
                'description': 'Conversión de secuencias funcionales → vectores 320D',
                'example': 'LKDFIF... → [0.123, -0.456, 0.789, ...]'
            },
            {
                'step': 6,
                'title': 'Construcción del índice FAISS',
                'file': 'cog_rag.index',
                'description': 'Índice optimizado para búsqueda de similitud',
                'example': 'Búsqueda de vecinos más cercanos en milisegundos'
            }
        ]
        
        for step_info in pipeline_steps:
            print(f"🔄 Paso {step_info['step']}: {step_info['title']}")
            print(f"   📁 Archivo: {step_info['file']}")
            print(f"   📝 Proceso: {step_info['description']}")
            print(f"   💡 Ejemplo: {step_info['example']}")
            print()
    
    def verify_embedding_consistency(self, n_samples: int = 10):
        """Verifica la consistencia entre datos y embeddings"""
        print(f"\n🔍 VERIFICACIÓN DE CONSISTENCIA")
        print("=" * 50)
        
        # Verificar que el número de embeddings coincide con la base de datos
        if self.embeddings_path.exists():
            embeddings = np.load(self.embeddings_path)
            print(f"📊 Embeddings en archivo: {embeddings.shape[0]:,}")
        else:
            print("❌ Archivo de embeddings no encontrado")
            return
        
        # Contar entradas en la base de datos
        cursor = self.conn.execute("SELECT COUNT(*) FROM cog_entries")
        db_count = cursor.fetchone()[0]
        print(f"📊 Entradas en base de datos: {db_count:,}")
        
        if embeddings.shape[0] == db_count:
            print("✅ Consistencia: Número de embeddings = Número de entradas en DB")
        else:
            print("❌ Inconsistencia detectada!")
            return
        
        # Verificar algunas muestras aleatorias
        print(f"\n🧪 Verificando {n_samples} muestras aleatorias:")
        
        sample_ids = np.random.choice(db_count, min(n_samples, db_count), replace=False)
        
        for i, sample_id in enumerate(sample_ids, 1):
            cursor = self.conn.execute("""
                SELECT protein_id, functional_region, category_id 
                FROM cog_entries WHERE id = ?
            """, (int(sample_id),))
            
            result = cursor.fetchone()
            if result:
                protein_id, func_region, category = result
                embedding = embeddings[sample_id]
                
                print(f"   {i:2d}. ID:{sample_id} | {protein_id[:15]}... | "
                      f"Cat:{category} | Func:{len(func_region)}aa | "
                      f"Embed:{embedding.shape}")
            else:
                print(f"   {i:2d}. ID:{sample_id} | ❌ No encontrado en DB")
    
    def generate_data_lineage_report(self, output_file: str = "data_lineage_report.txt"):
        """Genera reporte completo de linaje de datos"""
        print(f"\n📄 Generando reporte de linaje de datos...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("REPORTE DE LINAJE DE DATOS - RAG COG\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("1. ARCHIVOS FUENTE:\n")
            f.write("-" * 20 + "\n")
            f.write("- cog-24.def.tab: Definiciones de COGs\n")
            f.write("- cog-24.cog.csv: Mapeo proteína→COG + footprints\n")
            f.write("- COGorg24.faa.gz: Secuencias proteicas (7.7M)\n")
            f.write("- cog-24.org.csv: Información de organismos\n\n")
            
            f.write("2. PROCESO DE TRANSFORMACIÓN:\n")
            f.write("-" * 30 + "\n")
            f.write("a) Carga de definiciones COG\n")
            f.write("b) Mapeo proteína→COG con footprints\n")
            f.write("c) Extracción de regiones funcionales\n")
            f.write("d) Filtrado por calidad (membership_class ≤ 2)\n")
            f.write("e) Generación de embeddings con ESM-2\n")
            f.write("f) Construcción de índice FAISS\n\n")
            
            f.write("3. DATOS FINALES:\n")
            f.write("-" * 15 + "\n")
            if self.config:
                f.write(f"- Total entradas: {self.config.get('total_entries', 'N/A'):,}\n")
                f.write(f"- Dimensión embeddings: {self.config.get('embedding_dim', 'N/A')}\n")
                f.write(f"- Modelo usado: {self.config.get('model_name', 'N/A')}\n")
                f.write(f"- Fecha creación: {self.config.get('build_date', 'N/A')}\n")
            
            # Estadísticas por categoría
            cursor = self.conn.execute("""
                SELECT category_id, COUNT(*) as count 
                FROM cog_entries GROUP BY category_id ORDER BY count DESC
            """)
            
            f.write(f"\n4. DISTRIBUCIÓN POR CATEGORÍA:\n")
            f.write("-" * 30 + "\n")
            for category, count in cursor.fetchall():
                f.write(f"- {category}: {count:,} secuencias\n")
        
        print(f"✅ Reporte guardado: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Rastrear datos usados para crear embeddings")
    parser.add_argument('--rag-path', type=str, default='RAG_COG',
                       help='Ruta al directorio RAG')
    parser.add_argument('--cog-path', type=str,
                       help='Ruta a los archivos COG2024 originales')
    parser.add_argument('--protein-id', type=str,
                       help='ID de proteína específica a analizar')
    parser.add_argument('--sequence-id', type=int,
                       help='ID de secuencia específica a analizar')
    
    args = parser.parse_args()
    
    print("🔍 RASTREADOR DE DATOS PARA EMBEDDINGS COG")
    print("=" * 60)
    
    try:
        tracer = EmbeddingDataTracer(args.rag_path, args.cog_path)
        
        # Análisis general
        tracer.analyze_rag_creation_process()
        tracer.trace_data_sources()
        tracer.show_embedding_creation_pipeline()
        tracer.verify_embedding_consistency()
        
        # Análisis específico
        if args.protein_id or args.sequence_id:
            tracer.analyze_specific_protein(args.protein_id, args.sequence_id)
        
        # Generar reporte
        tracer.generate_data_lineage_report()
        
        print(f"\n🎉 Análisis completado!")
        print(f"💡 Usa --protein-id o --sequence-id para análisis específico")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    main()
