#!/usr/bin/env python3
"""
Script COG Mejorado: Procesador con datasets zero-shot y one-shot
Genera datasets de entrenamiento con footprints guardados y columnas específicas
"""

import pandas as pd
import numpy as np
import gzip
import json
import argparse
import logging
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass
from datetime import datetime
from Bio import SeqIO
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class COGEntry:
    """Clase para almacenar información completa de una entrada COG"""
    protein_id: str
    sequence: str
    functional_region: str
    cog_id: str
    category_id: str
    category_description: str
    functional_group: str
    footprint: str
    footprint_length: int
    membership_class: int
    bit_score: float
    e_value: float
    organism: str
    assembly_id: str
    sequence_length: int
    functional_length: int
    gene_name: str
    pathway: str

class COGDataProcessor:
    def __init__(self, data_dir: str = "./cog_data", verbose: bool = False):
        """Inicializar procesador de datos COG"""
        self.data_dir = Path(data_dir)
        self.verbose = verbose
        
        if verbose:
            logger.setLevel(logging.INFO)
        
        # Categorías COG completas
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
        
        # Grupos funcionales
        self.functional_groups = {
            'J': 'INFORMATION STORAGE AND PROCESSING', 'A': 'INFORMATION STORAGE AND PROCESSING',
            'K': 'INFORMATION STORAGE AND PROCESSING', 'L': 'INFORMATION STORAGE AND PROCESSING',
            'B': 'INFORMATION STORAGE AND PROCESSING', 'D': 'CELLULAR PROCESSES AND SIGNALING',
            'O': 'CELLULAR PROCESSES AND SIGNALING', 'M': 'CELLULAR PROCESSES AND SIGNALING',
            'N': 'CELLULAR PROCESSES AND SIGNALING', 'P': 'CELLULAR PROCESSES AND SIGNALING',
            'T': 'CELLULAR PROCESSES AND SIGNALING', 'U': 'CELLULAR PROCESSES AND SIGNALING',
            'V': 'CELLULAR PROCESSES AND SIGNALING', 'W': 'CELLULAR PROCESSES AND SIGNALING',
            'C': 'METABOLISM', 'G': 'METABOLISM', 'E': 'METABOLISM', 'F': 'METABOLISM',
            'H': 'METABOLISM', 'I': 'METABOLISM', 'Q': 'METABOLISM',
            'R': 'POORLY CHARACTERIZED', 'S': 'POORLY CHARACTERIZED'
        }
        
        # Almacenamiento de datos
        self.cog_definitions = {}
        self.protein_sequences = {}
        self.protein_to_cog = {}
        self.organism_info = {}
        self.processed_entries = []
        
        if verbose:
            print(f"🧬 Inicializando procesador COG - Directorio: {self.data_dir}")
    
    def extract_footprint_sequence(self, sequence: str, footprint: str) -> str:
        """Extrae la región funcional usando coordenadas del footprint"""
        if not footprint or footprint == "":
            return sequence
        
        try:
            functional_sequence = ""
            
            if '=' in footprint:
                # Footprint segmentado (ej: "1-100=201-300")
                segments = footprint.split('=')
                for segment in segments:
                    if '-' in segment:
                        start, end = map(int, segment.split('-'))
                        functional_sequence += sequence[start-1:end]
            else:
                # Footprint simple (ej: "45-380")
                if '-' in footprint:
                    start, end = map(int, footprint.split('-'))
                    functional_sequence = sequence[start-1:end]
                else:
                    functional_sequence = sequence
            
            return functional_sequence if functional_sequence else sequence
            
        except Exception:
            return sequence
    
    def load_cog_data(self, max_membership_class: int = 2, min_footprint_length: int = 30,
                     min_bit_score: float = 50.0, max_e_value: float = 1e-5):
        """Cargar todos los archivos COG con filtros de calidad"""
        
        if self.verbose:
            print("📖 Cargando datos COG...")
            print(f"   Filtros: membership≤{max_membership_class}, footprint≥{min_footprint_length}, bit_score≥{min_bit_score}")
        
        try:
            self._load_cog_definitions()
            self._load_organism_info()
            self._load_protein_cog_mappings(max_membership_class, min_footprint_length, 
                                          min_bit_score, max_e_value)
            self._load_protein_sequences()
            self._create_cog_entries()
            
            if self.verbose:
                print(f"✅ Datos cargados: {len(self.processed_entries):,} entradas válidas")
            
        except Exception as e:
            logger.error(f"Error cargando datos COG: {e}")
            raise
    
    def _load_cog_definitions(self):
        """Cargar definiciones COG desde cog-24.def.tab"""
        def_file = self.data_dir / "cog-24.def.tab"
        if not def_file.exists():
            raise FileNotFoundError(f"No se encontró {def_file}")
        
        with open(def_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        cog_id = parts[0]
                        functional_cat = parts[1]
                        cog_name = parts[2]
                        gene_name = parts[3] if len(parts) > 3 else ""
                        pathway = parts[4] if len(parts) > 4 else ""
                        
                        main_category = functional_cat[0] if functional_cat else 'S'
                        
                        self.cog_definitions[cog_id] = {
                            'category': main_category,
                            'name': cog_name,
                            'gene_name': gene_name,
                            'pathway': pathway,
                            'full_categories': functional_cat
                        }
    
    def _load_organism_info(self):
        """Cargar información de organismos"""
        org_file = self.data_dir / "cog-24.org.csv"
        if not org_file.exists():
            return
        
        with open(org_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 4:
                    assembly_id, organism_name, tax_id, tax_category = row[:4]
                    self.organism_info[assembly_id] = {
                        'name': organism_name,
                        'tax_id': tax_id,
                        'tax_category': tax_category
                    }
    
    def _load_protein_cog_mappings(self, max_membership_class, min_footprint_length, 
                                  min_bit_score, max_e_value):
        """Cargar asignaciones proteína-COG con filtros de calidad"""
        cog_file = self.data_dir / "cog-24.cog.csv"
        if not cog_file.exists():
            raise FileNotFoundError(f"No se encontró {cog_file}")
        
        valid = 0
        total = 0
        
        iterator = open(cog_file, 'r', encoding='utf-8')
        if self.verbose:
            # Contar líneas totales para progreso
            with open(cog_file, 'r') as f:
                total_lines = sum(1 for _ in f)
            iterator = tqdm(open(cog_file, 'r', encoding='utf-8'), 
                          total=total_lines, desc="📖 Cargando mappings", 
                          disable=not self.verbose)
        
        with iterator as f:
            reader = csv.reader(f)
            for row in reader:
                total += 1
                
                if len(row) >= 13:
                    gene_id, assembly_id, protein_id = row[0], row[1], row[2]
                    protein_length = int(row[3]) if row[3].isdigit() else 0
                    footprint = row[4]
                    footprint_length = int(row[5]) if row[5].isdigit() else 0
                    cog_id = row[6]
                    membership_class = int(row[8]) if row[8].isdigit() else 3
                    bit_score = float(row[9]) if row[9] and row[9] != '' else 0.0
                    e_value = float(row[10]) if row[10] and row[10] != '' else 1.0
                    
                    # Aplicar filtros
                    if (membership_class <= max_membership_class and
                        footprint_length >= min_footprint_length and
                        bit_score >= min_bit_score and
                        e_value <= max_e_value and
                        cog_id in self.cog_definitions):
                        
                        cog_info = self.cog_definitions[cog_id]
                        if cog_info['category'] in self.cog_categories:
                            self.protein_to_cog[protein_id] = {
                                'cog_id': cog_id, 'footprint': footprint,
                                'footprint_length': footprint_length,
                                'membership_class': membership_class,
                                'bit_score': bit_score, 'e_value': e_value,
                                'assembly_id': assembly_id, 'protein_length': protein_length,
                                'gene_id': gene_id
                            }
                            valid += 1
        
        if self.verbose:
            print(f"✅ Mappings válidos: {valid:,} de {total:,} ({valid/total:.1%})")
    
    def _load_protein_sequences(self):
        """Cargar secuencias de proteínas (solo las que tienen COG válido)"""
        faa_file = self.data_dir / "COGorg24.faa.gz"
        if not faa_file.exists():
            raise FileNotFoundError(f"No se encontró {faa_file}")
        
        target_proteins = set(self.protein_to_cog.keys())
        found = 0
        
        iterator = SeqIO.parse(gzip.open(faa_file, 'rt'), "fasta")
        if self.verbose:
            iterator = tqdm(iterator, desc="🧬 Cargando secuencias", 
                          unit="seq", disable=not self.verbose)
        
        for record in iterator:
            protein_id = record.id
            
            if protein_id in target_proteins:
                self.protein_sequences[protein_id] = str(record.seq)
                found += 1
                
                if found >= len(target_proteins):
                    break
        
        # Filtrar protein_to_cog para solo incluir proteínas con secuencia
        proteins_with_sequence = set(self.protein_sequences.keys())
        self.protein_to_cog = {
            pid: info for pid, info in self.protein_to_cog.items() 
            if pid in proteins_with_sequence
        }
        
        if self.verbose:
            print(f"✅ Secuencias encontradas: {len(self.protein_sequences):,}")
    
    def _create_cog_entries(self):
        """Crear entradas COG procesadas con regiones funcionales"""
        self.processed_entries = []
        
        iterator = self.protein_to_cog.items()
        if self.verbose:
            iterator = tqdm(iterator, desc="✨ Procesando entradas", 
                          disable=not self.verbose)
        
        for protein_id, cog_mapping in iterator:
            if protein_id in self.protein_sequences:
                sequence = self.protein_sequences[protein_id]
                cog_id = cog_mapping['cog_id']
                cog_info = self.cog_definitions[cog_id]
                
                functional_region = self.extract_footprint_sequence(
                    sequence, cog_mapping['footprint']
                )
                
                assembly_id = cog_mapping['assembly_id']
                organism_name = "Unknown"
                if assembly_id in self.organism_info:
                    organism_name = self.organism_info[assembly_id]['name']
                
                entry = COGEntry(
                    protein_id=protein_id, sequence=sequence,
                    functional_region=functional_region, cog_id=cog_id,
                    category_id=cog_info['category'],
                    category_description=self.cog_categories[cog_info['category']],
                    functional_group=self.functional_groups[cog_info['category']],
                    footprint=cog_mapping['footprint'],
                    footprint_length=cog_mapping['footprint_length'],
                    membership_class=cog_mapping['membership_class'],
                    bit_score=cog_mapping['bit_score'], e_value=cog_mapping['e_value'],
                    organism=organism_name, assembly_id=assembly_id,
                    sequence_length=len(sequence), functional_length=len(functional_region),
                    gene_name=cog_info['gene_name'], pathway=cog_info['pathway']
                )
                
                self.processed_entries.append(entry)
    
    def generate_datasets(self, num_train_sequences: int = 8000, num_test_sequences: int = 2000, 
                         min_length: int = 50, max_length: int = 800, balance_categories: bool = True,
                         use_functional_regions: bool = True, quality_threshold: int = 2):
        """Generar datasets de entrenamiento y test"""
        
        total_sequences = num_train_sequences + num_test_sequences
        if self.verbose:
            print(f"📚 Generando datasets con {total_sequences} secuencias totales...")
            print(f"   🏋️ Entrenamiento: {num_train_sequences} secuencias")
            print(f"   🧪 Test: {num_test_sequences} secuencias")
        
        if not self.processed_entries:
            raise ValueError("No hay entradas procesadas. Ejecuta load_cog_data() primero.")
        
        # Filtrar entradas válidas
        valid_entries = []
        for entry in self.processed_entries:
            target_seq = entry.functional_region if use_functional_regions else entry.sequence
            
            if (min_length <= len(target_seq) <= max_length and 
                entry.membership_class <= quality_threshold):
                valid_entries.append(entry)
        
        if not valid_entries:
            raise ValueError("No se encontraron entradas válidas")
        
        if self.verbose:
            print(f"📊 Entradas válidas encontradas: {len(valid_entries):,}")
        
        # Verificar que tenemos suficientes datos
        if len(valid_entries) < total_sequences:
            logger.warning(f"⚠️ Solo {len(valid_entries)} entradas disponibles, pero se necesitan {total_sequences}")
            ratio = len(valid_entries) / total_sequences
            num_train_sequences = int(num_train_sequences * ratio)
            num_test_sequences = len(valid_entries) - num_train_sequences
            print(f"📊 Ajustado - Entrenamiento: {num_train_sequences}, Test: {num_test_sequences}")
        
        # Dividir entradas por categoría
        if balance_categories:
            train_entries, test_entries = self._split_balanced_train_test(
                valid_entries, num_train_sequences, num_test_sequences
            )
        else:
            random.shuffle(valid_entries)
            selected = valid_entries[:total_sequences]
            train_entries = selected[:num_train_sequences]
            test_entries = selected[num_train_sequences:]
        
        # Determinar sufijos de archivos
        sequence_type = "functional" if use_functional_regions else "complete"
        
        # Generar archivos CSV con las columnas específicas
        print(f"\n🎯 Generando datasets...")
        train_file = self.generate_dataset_csv(
            train_entries, use_functional_regions, f"cog_train_{sequence_type}"
        )
        test_file = self.generate_dataset_csv(
            test_entries, use_functional_regions, f"cog_test_{sequence_type}"
        )
        
        return {
            'train': train_file,
            'test': test_file
        }
    
    def _split_balanced_train_test(self, entries: List[COGEntry], num_train: int, num_test: int) -> Tuple[List[COGEntry], List[COGEntry]]:
        """Dividir entradas en train/test manteniendo balance por categorías"""
        
        category_groups = defaultdict(list)
        
        # Agrupar por categoría y ordenar por calidad
        for entry in entries:
            category_groups[entry.category_id].append(entry)
        
        for category in category_groups:
            category_groups[category].sort(key=lambda x: (x.membership_class, -x.bit_score))
        
        # Calcular distribución train/test por categoría
        train_entries = []
        test_entries = []
        
        num_categories = 23
        train_per_category = num_train // num_categories
        test_per_category = num_test // num_categories
        train_remainder = num_train % num_categories
        test_remainder = num_test % num_categories
        
        for i, category in enumerate(sorted(self.cog_categories.keys())):
            if category in category_groups and category_groups[category]:
                available = category_groups[category]
                
                train_needed = train_per_category + (1 if i < train_remainder else 0)
                test_needed = test_per_category + (1 if i < test_remainder else 0)
                total_needed = train_needed + test_needed
                
                if len(available) >= total_needed:
                    train_entries.extend(available[:train_needed])
                    test_entries.extend(available[train_needed:train_needed + test_needed])
                else:
                    train_ratio = train_needed / total_needed
                    actual_train = int(len(available) * train_ratio)
                    actual_test = len(available) - actual_train
                    
                    train_entries.extend(available[:actual_train])
                    test_entries.extend(available[actual_train:actual_train + actual_test])
        
        # Mezclar para randomizar orden
        random.shuffle(train_entries)
        random.shuffle(test_entries)
        
        return train_entries[:num_train], test_entries[:num_test]
    
    def generate_dataset_csv(self, entries: List[COGEntry], use_functional_regions: bool = True,
                           output_file: str = "cog_dataset") -> str:
        """Generar archivo CSV con las columnas específicas requeridas"""
        
        dataset = []
        iterator = entries
        if self.verbose:
            iterator = tqdm(entries, desc="📝 Generando CSV", disable=not self.verbose)
        
        for entry in iterator:
            target_seq = entry.functional_region if use_functional_regions else entry.sequence
            
            # Crear entrada con las columnas específicas requeridas
            row = {
                'protein_id': entry.protein_id,
                'sequence': target_seq,
                'category': entry.category_id,
                'category_name': entry.category_description,
                'functional_group': entry.functional_group,
                'cog_id': entry.cog_id,
                'organism': entry.organism,
                'gene_name': entry.gene_name,
                'sequence_length': len(target_seq),
                'footprint': entry.footprint
            }
            
            dataset.append(row)
        
        # Guardar dataset
        df = pd.DataFrame(dataset)
        output_file_final = f"{output_file}.csv"
        df.to_csv(output_file_final, index=False)
        
        if self.verbose:
            self._show_statistics(df)
        
        print(f"✅ Dataset guardado: {output_file_final} ({len(df):,} muestras)")
        return output_file_final
    
    def _show_statistics(self, df: pd.DataFrame):
        """Mostrar estadísticas resumidas del dataset"""
        if not self.verbose:
            return
            
        print(f"\n📊 ESTADÍSTICAS")
        print(f"Total secuencias: {len(df):,}")
        
        # Distribución por categorías (top 5)
        category_counts = df['category'].value_counts().sort_index()
        print(f"Categorías representadas: {len(category_counts)}")
        print("Top 5 categorías:")
        for category, count in category_counts.head(5).items():
            pct = (count / len(df)) * 100
            desc = self.cog_categories.get(category, 'Unknown')[:30]
            print(f"  {category}: {count:,} ({pct:.1f}%) - {desc}...")

def main():
    """Función principal con argumentos simplificados"""
    parser = argparse.ArgumentParser(
        description='Procesador COG simplificado',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Argumentos principales
    parser.add_argument('--data-dir', default='./cog_data', help='Directorio con datos COG')
    parser.add_argument('--train-sequences', type=int, default=8000, help='Secuencias de entrenamiento')
    parser.add_argument('--test-sequences', type=int, default=2000, help='Secuencias de test')
    
    # Filtros de calidad
    parser.add_argument('--max-membership', type=int, default=2, help='Clase máxima membresía (0-3)')
    parser.add_argument('--min-bit-score', type=float, default=50.0, help='Bit score mínimo')
    parser.add_argument('--max-e-value', type=float, default=1e-5, help='E-value máximo')
    parser.add_argument('--min-footprint', type=int, default=30, help='Longitud mínima footprint')
    
    # Opciones de procesamiento
    parser.add_argument('--min-length', type=int, default=50, help='Longitud mínima secuencia')
    parser.add_argument('--max-length', type=int, default=800, help='Longitud máxima secuencia')
    parser.add_argument('--use-functional', action='store_true', help='Usar regiones funcionales')
    parser.add_argument('--no-balance', action='store_true', help='No balancear categorías')
    parser.add_argument('--verbose', '-v', action='store_true', help='Output detallado')
    
    args = parser.parse_args()
    
    # Inicializar procesador
    processor = COGDataProcessor(args.data_dir, verbose=args.verbose)
    
    try:
        if args.verbose:
            print("🧬 PROCESADOR COG SIMPLIFICADO")
            print("=" * 50)
        
        # Cargar datos COG con filtros de calidad
        processor.load_cog_data(
            max_membership_class=args.max_membership,
            min_footprint_length=args.min_footprint,
            min_bit_score=args.min_bit_score,
            max_e_value=args.max_e_value
        )
        
        # Generar datasets train/test
        dataset_files = processor.generate_datasets(
            num_train_sequences=args.train_sequences,
            num_test_sequences=args.test_sequences,
            min_length=args.min_length,
            max_length=args.max_length,
            balance_categories=not args.no_balance,
            use_functional_regions=args.use_functional,
            quality_threshold=args.max_membership
        )
        
        if args.verbose:
            print(f"\n🎉 PROCESAMIENTO COMPLETADO")
            print(f"🧬 Tipo de secuencias: {'Regiones funcionales' if args.use_functional else 'Secuencias completas'}")
            
            # Mostrar archivos generados
            print(f"\n📁 ARCHIVOS GENERADOS:")
            for dataset_type, filename in dataset_files.items():
                emoji = "🏋️" if dataset_type == "train" else "🧪"
                print(f"  {emoji} {dataset_type.title()}: {filename}")
            
            print(f"\n💡 COLUMNAS EN LOS ARCHIVOS:")
            print(f"  - protein_id: ID único de la proteína")
            print(f"  - sequence: Secuencia de aminoácidos")
            print(f"  - category: Categoría COG (letra)")
            print(f"  - category_name: Descripción de la categoría")
            print(f"  - functional_group: Grupo funcional principal")
            print(f"  - cog_id: ID del COG específico")
            print(f"  - organism: Nombre del organismo")
            print(f"  - gene_name: Nombre del gen")
            print(f"  - sequence_length: Longitud de la secuencia")
            print(f"  - footprint: Coordenadas de la región funcional")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            raise

if __name__ == "__main__":
    main()