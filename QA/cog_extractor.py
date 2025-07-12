import pandas as pd
import gzip
from pathlib import Path
import csv
from Bio import SeqIO
import argparse
import sys

def extract_cog_categories(base_path, output_file="cog_categories.csv", 
                          total_samples=None):
    """
    Extrae secuencias y categorías COG específicas para usar con Ollama
    
    Args:
        base_path: Ruta base donde están los archivos COG
        output_file: Nombre del archivo CSV de salida
        total_samples: Número TOTAL de muestras a distribuir equitativamente entre categorías (None = todas)
    """
    
    base_path = Path(base_path)
    
    print("🔍 Iniciando extracción de categorías COG...")
    print(f"📁 Directorio: {base_path}")
    print(f"📄 Archivo salida: {output_file}")
    if total_samples:
        print(f"⚖️ Modo: Balanceado ({total_samples} muestras totales distribuidas equitativamente)")
    else:
        print(f"📊 Modo: Completo (todas las secuencias disponibles)")
    
    # 1. Definir categorías COG estándar
    print("\n📖 Definiendo categorías COG estándar...")
    standard_categories = [
        ('J', 1, '#FF6B6B', 'Translation, ribosomal structure and biogenesis'),
        ('A', 1, '#4ECDC4', 'RNA processing and modification'),
        ('K', 1, '#45B7D1', 'Transcription'),
        ('L', 1, '#96CEB4', 'Replication, recombination and repair'),
        ('B', 1, '#FECA57', 'Chromatin structure and dynamics'),
        ('D', 2, '#FF9FF3', 'Cell cycle control, cell division'),
        ('O', 2, '#54A0FF', 'Molecular chaperones and related functions'),
        ('M', 2, '#5F27CD', 'Cell wall/membrane/envelope biogenesis'),
        ('N', 2, '#00D2D3', 'Cell motility'),
        ('P', 2, '#FF6348', 'Inorganic ion transport and metabolism'),
        ('T', 2, '#2ED573', 'Signal transduction mechanisms'),
        ('U', 2, '#FFD93D', 'Intracellular trafficking, secretion'),
        ('V', 2, '#6C5CE7', 'Defense mechanisms'),
        ('W', 2, '#FD79A8', 'Extracellular structures'),
        ('C', 3, '#74B9FF', 'Energy production and conversion'),
        ('G', 3, '#00B894', 'Carbohydrate transport and metabolism'),
        ('E', 3, '#FDCB6E', 'Amino acid transport and metabolism'),
        ('F', 3, '#E17055', 'Nucleotide transport and metabolism'),
        ('H', 3, '#81ECEC', 'Coenzyme transport and metabolism'),
        ('I', 3, '#FD79A8', 'Lipid transport and metabolism'),
        ('Q', 3, '#6C5CE7', 'Secondary metabolites biosynthesis'),
        ('R', 4, '#A29BFE', 'General function prediction only'),
        ('S', 4, '#636E72', 'Function unknown')
    ]
    
    # Mapeo de grupos funcionales para referencia
    functional_groups = {
        1: 'INFORMATION STORAGE AND PROCESSING',
        2: 'CELLULAR PROCESSES AND SIGNALING',
        3: 'METABOLISM',
        4: 'POORLY CHARACTERIZED'
    }
    
    # Crear mapeos
    categories = {}
    category_groups = {}
    category_to_group = {}
    
    for cat_id, group_id, color, description in standard_categories:
        categories[cat_id] = description
        category_groups[cat_id] = functional_groups[group_id]
        category_to_group[cat_id] = functional_groups[group_id]
    
    print(f"✅ Definidas {len(categories)} categorías COG:")
    for group_id, group_name in functional_groups.items():
        cats_in_group = [cat for cat, grp_id, _, _ in standard_categories if grp_id == group_id]
        print(f"   {group_name}: {', '.join(cats_in_group)}")
    
    # 2. Cargar definiciones COG (COG ID -> categoría)
    print("\n📖 Leyendo definiciones COG...")
    def_file = base_path / "cog-24.def.tab"
    
    if not def_file.exists():
        print(f"❌ Error: No se encontró el archivo {def_file}")
        return None
    
    cog_to_category = {}
    with open(def_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    cog_id = parts[0]
                    functional_cat = parts[1]  # Puede tener múltiples letras
                    cog_name = parts[2]
                    # Tomar la primera categoría (más importante)
                    main_category = functional_cat[0] if functional_cat else 'X'
                    cog_to_category[cog_id] = {
                        'category': main_category,
                        'name': cog_name,
                        'full_categories': functional_cat
                    }
    
    print(f"✅ Cargados {len(cog_to_category)} COGs")
    
    # 3. Cargar asignaciones proteína -> COG
    print("📖 Leyendo asignaciones proteína-COG...")
    cog_file = base_path / "cog-24.cog.csv"
    
    if not cog_file.exists():
        print(f"❌ Error: No se encontró el archivo {cog_file}")
        return None
    
    protein_to_cog = {}
    with open(cog_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 7:
                protein_id = row[2]  # Protein ID
                cog_id = row[6]      # COG ID
                
                # Si una proteína tiene múltiples COGs, tomar el primero
                if protein_id not in protein_to_cog:
                    protein_to_cog[protein_id] = cog_id
    
    print(f"✅ Cargadas {len(protein_to_cog)} asignaciones proteína-COG")
    
    # 4. Procesar secuencias FASTA de manera inteligente
    print("🧬 Procesando secuencias FASTA...")
    fasta_file = base_path / "COGorg24.faa.gz"
    
    if not fasta_file.exists():
        print(f"❌ Error: No se encontró el archivo {fasta_file}")
        return None
    
    # Configurar límites inteligentes
    if total_samples:
        # Calcular cuántas muestras necesitamos por categoría
        available_categories = list(category_to_group.keys())
        num_categories = len(available_categories)
        base_samples_per_category = total_samples // num_categories
        extra_samples = total_samples % num_categories
        
        print(f"🎯 Modo inteligente: buscando {total_samples} muestras")
        print(f"📊 Necesitamos ~{base_samples_per_category}-{base_samples_per_category+1} muestras por categoría")
        
        # Crear contadores por categoría
        category_collections = {cat_id: [] for cat_id in available_categories}
        category_targets = {}
        
        # Asignar objetivos por categoría
        for i, cat_id in enumerate(sorted(available_categories)):
            target = base_samples_per_category
            if i < extra_samples:
                target += 1
            category_targets[cat_id] = target
        
        # Estadísticas de progreso
        total_needed = sum(category_targets.values())
        total_collected = 0
        sequences_read = 0
        
    else:
        # Modo completo: leer todas las secuencias
        print("📚 Modo completo: procesando todas las secuencias")
        sequences_data = []
    
    processed = 0
    skipped = 0
    
    try:
        with gzip.open(fasta_file, 'rt') as f:
            for record in SeqIO.parse(f, "fasta"):
                sequences_read = processed + skipped + 1
                
                protein_id = record.id
                sequence = str(record.seq)
                
                # Buscar COG asociado
                if protein_id in protein_to_cog:
                    cog_id = protein_to_cog[protein_id]
                    
                    if cog_id in cog_to_category:
                        cog_info = cog_to_category[cog_id]
                        category_id = cog_info['category']
                        
                        if category_id in category_to_group:
                            functional_group = category_to_group[category_id]
                            
                            sequence_entry = {
                                'protein_id': protein_id,
                                'sequence': sequence,
                                'functional_group': functional_group,
                                'cog_id': cog_id,
                                'category_id': category_id,
                                'sequence_length': len(sequence)
                            }
                            
                            if total_samples:
                                # Modo inteligente: verificar si necesitamos esta categoría
                                if category_id in category_collections:
                                    current_count = len(category_collections[category_id])
                                    target_count = category_targets[category_id]
                                    
                                    if current_count < target_count:
                                        category_collections[category_id].append(sequence_entry)
                                        total_collected += 1
                                        processed += 1
                                        
                                        if processed % 1000 == 0:
                                            categories_complete = sum(1 for cat_id in category_collections 
                                                                    if len(category_collections[cat_id]) >= category_targets[cat_id])
                                            print(f"⏳ Recolectadas {total_collected:,}/{total_needed:,} muestras | "
                                                  f"Categorías completas: {categories_complete}/{len(category_targets)} | "
                                                  f"Secuencias leídas: {sequences_read:,}")
                                        
                                        # Verificar si todas las categorías están completas
                                        if total_collected >= total_needed:
                                            print(f"🎉 ¡Completado! Recolectadas todas las {total_needed:,} muestras necesarias")
                                            break
                                    else:
                                        skipped += 1
                                else:
                                    skipped += 1
                            else:
                                # Modo completo: agregar todas las secuencias
                                sequences_data.append(sequence_entry)
                                processed += 1
                                
                                if processed % 10000 == 0:
                                    print(f"⏳ Procesadas {processed:,} secuencias...")
                        else:
                            skipped += 1
                    else:
                        skipped += 1
                else:
                    skipped += 1
                    
    except Exception as e:
        print(f"❌ Error al leer archivo FASTA: {e}")
        return None
    
    # Consolidar resultados según el modo
    if total_samples:
        # Combinar todas las colecciones de categorías
        sequences_data = []
        for cat_id, entries in category_collections.items():
            sequences_data.extend(entries)
        
        print(f"✅ Recolección inteligente completada:")
        print(f"   📖 Secuencias leídas del archivo: {sequences_read:,}")
        print(f"   ✅ Secuencias válidas recolectadas: {len(sequences_data):,}")
        print(f"   ⏭️ Secuencias omitidas: {skipped:,}")
        
        # Mostrar resumen por categoría
        print(f"\n📊 Muestras recolectadas por categoría:")
        for cat_id in sorted(category_targets.keys()):
            collected = len(category_collections[cat_id])
            target = category_targets[cat_id]
            status = "✅" if collected >= target else "⚠️"
            print(f"   {status} {cat_id}: {collected:,}/{target:,}")
    else:
        print(f"✅ Procesamiento completo:")
        print(f"   📖 Secuencias leídas: {sequences_read:,}")
        print(f"   ✅ Secuencias válidas: {processed:,}")
        print(f"   ⏭️ Secuencias omitidas: {skipped:,}")
    
    if not sequences_data:
        print("❌ No se encontraron secuencias válidas")
        return None
    
    # 5. Crear DataFrame
    print("📊 Creando dataset...")
    df = pd.DataFrame(sequences_data)
    
    # Estadísticas
    print(f"\n📈 Estadísticas del dataset:")
    print(f"Total de secuencias: {len(df):,}")
    print(f"Categorías COG: {df['category_id'].nunique()}")
    print(f"Longitud promedio de secuencias: {df['sequence_length'].mean():.1f}")
    
    if not total_samples:  # Solo mostrar distribución detallada en modo completo
        print(f"\n🏷️ Distribución por categorías COG (Top 10):")
        category_counts = df.groupby('category_id').size().sort_values(ascending=False)
        for cat_id, count in category_counts.head(10).items():
            cat_desc = categories[cat_id]
            group = category_groups.get(cat_id, 'Unknown')
            percentage = (count / len(df)) * 100
            print(f"  {cat_id}: {cat_desc[:40]}... [{group}] ({count:,}, {percentage:.1f}%)")
        
        print(f"\n🔬 Resumen por grupos funcionales:")
        group_counts = df.groupby('functional_group').size().sort_values(ascending=False)
        for group, count in group_counts.items():
            percentage = (count / len(df)) * 100
            cats_in_group = df[df['functional_group'] == group]['category_id'].nunique()
            print(f"  {group}: {count:,} secuencias ({percentage:.1f}%) - {cats_in_group} categorías")
    
    # 6. Procesar dataset final
    if total_samples:
        # Las secuencias ya están balanceadas de la recolección inteligente
        final_df = pd.DataFrame(sequences_data)
        # Mezclar el dataset
        final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\n📊 Dataset balanceado automáticamente:")
        print(f"✅ Total de muestras finales: {len(final_df):,}")
        
        # Verificar distribución final
        final_counts = final_df.groupby('category_id').size().sort_values()
        min_samples = final_counts.min()
        max_samples = final_counts.max()
        print(f"⚖️ Distribución: {min_samples}-{max_samples} muestras por categoría")
        
        # Mostrar algunas categorías como ejemplo
        print(f"\n🔍 Verificación de distribución:")
        sample_counts = final_df.groupby('category_id').size().head(5)
        for cat_id, count in sample_counts.items():
            cat_desc = categories[cat_id]
            print(f"  {cat_id}: {count:,} muestras ({cat_desc[:25]}...)")
        if len(final_counts) > 5:
            print(f"  ... y {len(final_counts)-5} categorías más")
    else:
        # En modo completo, final_df es el mismo que df
        final_df = df
    
    # 7. Guardar archivo para Ollama
    print(f"\n💾 Guardando archivo: {output_file}")
    ollama_df = final_df[['sequence', 'category_id']].copy()
    ollama_df.columns = ['sequence', 'category']
    ollama_df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"✅ Archivo guardado con formato: sequence,category")
    print(f"📝 Ejemplo: MLSNKYLK...,K")
    
    # 8. Crear archivo de metadatos
    metadata_file = output_file.replace('.csv', '_metadata.txt')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        f.write("COG Categories Dataset\n")
        f.write("=====================\n\n")
        f.write(f"Dataset type: {'Balanced' if total_samples else 'Complete'}\n")
        f.write(f"Total sequences: {len(final_df):,}\n")
        f.write(f"COG categories: {final_df['category_id'].nunique()}\n")
        f.write(f"Average sequence length: {final_df['sequence_length'].mean():.1f}\n")
        if total_samples:
            f.write(f"Total samples requested: {total_samples:,}\n")
            available_categories = final_df['category_id'].nunique()
            base_samples = total_samples // available_categories
            extra_samples = total_samples % available_categories
            f.write(f"Available categories: {available_categories}\n")
            f.write(f"Base samples per category: {base_samples:,}\n")
            if extra_samples > 0:
                f.write(f"Categories with extra sample: {extra_samples}\n")
            f.write(f"Processing method: Intelligent reading (stopped early)\n")
        
        f.write(f"\nCategory Distribution:\n")
        f.write("-" * 22 + "\n")
        
        final_counts = final_df.groupby('category_id').size().sort_values(ascending=False)
        for cat_id, count in final_counts.items():
            percentage = (count / len(final_df)) * 100
            cat_desc = categories.get(cat_id, 'Unknown')
            group = category_groups.get(cat_id, 'Unknown')
            f.write(f"{cat_id}: {cat_desc} [{group}] ({count:,} sequences, {percentage:.1f}%)\n")
        
        f.write(f"\nFunctional Groups Summary:\n")
        f.write("-" * 26 + "\n")
        final_group_counts = final_df.groupby('functional_group').size()
        for group, count in final_group_counts.items():
            percentage = (count / len(final_df)) * 100
            cats_in_group = final_df[final_df['functional_group'] == group]['category_id'].nunique()
            f.write(f"{group}: {count:,} sequences ({percentage:.1f}%) - {cats_in_group} categories\n")
    
    print(f"✅ Metadatos guardados: {metadata_file}")
    print(f"\n🎉 Proceso completado!")
    print(f"📄 Archivo principal: {output_file}")
    print(f"📄 Metadatos: {metadata_file}")
    print(f"\n🤖 Formato para Ollama: sequence,category")
    print(f"📊 Total de categorías en el dataset final: {final_df['category_id'].nunique()}")
    print(f"🏷️ Categorías incluidas: {', '.join(sorted(final_df['category_id'].unique()))}")
    
    return final_df

def main():
    parser = argparse.ArgumentParser(
        description="Extractor de categorías COG específicas para Ollama (formato: sequence,letra)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Uso básico (dataset completo):
  python cog_extractor.py /ruta/a/cog_databases

Uso con balanceado automático (SÚPER RÁPIDO):
  python cog_extractor.py /ruta/a/cog_databases --samples 500
  python cog_extractor.py /ruta/a/cog_databases --samples 2300

OPTIMIZACIÓN INTELIGENTE: 
  --samples 500 = Lee SOLO las secuencias necesarias para 500 muestras balanceadas
  Ejemplo: 500 ÷ 23 = ~22 por categoría → Para cuando tenga 22 de cada una
  ¡Mucho más rápido que leer todo el archivo!

FORMATO DE SALIDA: sequence,category (solo letra COG)
  Ejemplo: MLSNKYLKDFIF...,K

Categorías COG incluidas (~23 total):
  J,A,K,L,B: Information Storage and Processing
  D,O,M,N,P,T,U,V,W: Cellular Processes and Signaling  
  C,G,E,F,H,I,Q: Metabolism
  R,S: Poorly Characterized
        """
    )
    
    parser.add_argument(
        'base_path',
        help='Ruta al directorio que contiene los archivos COG'
    )
    
    parser.add_argument(
        '--samples', '-s',
        type=int,
        help='TOTAL de muestras a distribuir equitativamente entre todas las categorías COG (formato: sequence,letra)'
    )
    
    args = parser.parse_args()
    
    # Validar que el directorio existe
    if not Path(args.base_path).exists():
        print(f"❌ Error: El directorio {args.base_path} no existe")
        sys.exit(1)
    
    # Configurar parámetros
    total_samples = args.samples
    
    # Determinar nombre del archivo de salida
    if total_samples:
        output_file = f"cog_categories_balanced_{total_samples}.csv"
    else:
        output_file = "cog_categories_complete.csv"
    
    # Mostrar configuración
    print("🚀 Configuración:")
    if total_samples:
        print(f"   Modo: Balanceado automático con lectura inteligente")
        print(f"   Total de muestras: {total_samples:,}")
        print(f"   Distribución: Equitativa entre ~23 categorías COG")
        print(f"   Muestras por categoría: ~{total_samples // 23}")
        print(f"   Optimización: Para cuando tenga las muestras necesarias ⚡")
    else:
        print(f"   Modo: Completo")
        print(f"   Incluye: Todas las secuencias disponibles")
    print(f"   Archivo de salida: {output_file}")
    
    # Ejecutar extracción
    try:
        df = extract_cog_categories(
            base_path=args.base_path,
            output_file=output_file,
            total_samples=total_samples
        )
        
        if df is not None:
            print(f"\n✨ ¡Éxito! Dataset con {df['category_id'].nunique()} categorías COG listo para Ollama")
            print(f"📄 Archivo generado: {output_file}")
            print(f"📊 Total de secuencias: {len(df):,}")
            print(f"🔤 Formato: sequence,category (usando letras: {', '.join(sorted(df['category_id'].unique()))})")
        else:
            print(f"\n❌ Error al procesar los datos")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
