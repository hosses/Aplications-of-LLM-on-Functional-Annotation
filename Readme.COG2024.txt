Readme.txt for COG2024  

#-----------------------------------------------------------
# cog-24.cog.csv
#-----------------------------------------------------------
Comma-delimited plain text file assigning proteins to COGs
Columns:
1. Gene ID (GenBank or ad hoc)
2. NCBI Assembly ID
3. Protein ID (GenBank if conforms to [A-Za-z0-9_]+\.[0-9]+ regex; ad hoc otherwise)
4. Protein length
5. COG footprint coordinates on the protein. "201-400" means "from position 201 to position 400"; "1-100=201-300" indicates a segmented footprint, 1-100 AND 201-300
6. Length of the COG footprint on the proteins
7. COG ID
8. Reseved (currently shows COG ID)
9. COG membership class (0: footprint covers most of the protein and most of the COG profile; 1: footprint covers most of the COG profile and part of the protein; 2: footprint covers most of the protein and part of the COG profile; 3: partial match on both protein and COG profile)
10. PSI-BLAST bit score for the match between the protein and COG profile
11. PSI-BLAST e-value for the match between the protein and COG profile
12. COG profile length
13. Protein footprint coordinates on the COG profile
(fields 10-13 may be empty)
Each line corresponds to one instance of a COG in a protein-coding gene. Multidomain proteins are represented by multiple lines. Protein IDs can be shared between multiple genes. A combination of a gene ID and assembly ID uniquely identifies a location in the genome. The order of entries is arbitrary.

#-----------------------------------------------------------
# cog-24.fun.tab
#-----------------------------------------------------------
Tab-delimited plain text file with descriptions of COG functional categories
The categories form four functional groups:
1. INFORMATION STORAGE AND PROCESSING
2. CELLULAR PROCESSES AND SIGNALING
3. METABOLISM
4. POORLY CHARACTERIZED
Columns:
1. Functional category ID (one letter)
2. Functional group (1-4, as above)
3. Hexadecimal RGB color associated with the functional category
4. Functional category description
Each line corresponds to one functional category. The order of the categories is meaningful (reflects a hierarchy of functions; determines the order of display)

#-----------------------------------------------------------
# cog-24.def.tab
#-----------------------------------------------------------
Tab-delimited plain text file with COG descriptions
Columns:
1. COG ID
2. COG functional category (could include multiple letters in the order of importance)
3. COG name
4. Gene name associated with the COG (optional)
5. Functional pathway associated with the COG (optional)
6. PubMed ID, associated with the COG (multiple entries are semicolon-separated; optional)
7. PDB ID of the structure associated with the COG (multiple entries are semicolon-separated; optional)
Each line corresponds to one COG. The order of the COGs is arbitrary (displayed in the lexicographic order)

#-----------------------------------------------------------
# cog-24.mapping.tab
#-----------------------------------------------------------
Tab-delimited plain text file listing representative UniProt entries for each COG
Columns:
1. COG ID
2. COG functional category (could include multiple letters in the order of importance)
3. Gene name associated with the COG (optional)
4. COG name (long names are cut to 42 characters)
5. Representative UniProt ID
6. Representative UniProt protein name
7. Length of the representative protein or domain (amino acid residues)
Each line corresponds to one COG. The order of the COGs is arbitrary (displayed in the lexicographic order)

#-----------------------------------------------------------
# cog-24.org.csv
#-----------------------------------------------------------
Comma-delimited plain text describing genome assemblies
Columns:
1. NCBI Assembly ID
2. Organism (genome) name
3. NCBI Tax ID of the assembly
4. Taxonomic category used in COGs
Each line corresponds to one genome assembly. The order of the assemblies is meaningful (roughly corresponds to the taxonomic order; determines the order of display)

#-----------------------------------------------------------
# cog-24.tax.csv
#-----------------------------------------------------------
Comma-delimited plain text describing taxonomic categories
Columns:
1. Taxonomic category in COGs
2. Parent taxonomic category (self, if top of the hierarchy)
3. NCBI Tax ID of the assembly
Each line corresponds to one genome taxonomic category. The order of the taxonomic category is meaningful (determines the order of display)

#-----------------------------------------------------------
# cog-24.pathways.tab
#-----------------------------------------------------------
Tab-delimited plain text listing COG pathways and functional systems, COGs assigned to them, and the Enzyme Commission EC numbers, where available 
Columns:
1. COG pathway or functional system
2. COG ID
3. COG functional category
4. Gene name associated with the COG
5. COG name
6. Enzyme Commission EC number(s) (optional) 
Each line corresponds to one COG. The pathways and functional systems are split into groups: those listed in the 2020 version (in alphabetical order) and those added in the 2024 release (starting from line 834, in alphabetical order)

#-----------------------------------------------------------
# COGorg24.faa.gz
#-----------------------------------------------------------
A gzip-compressed text file containing amino acid sequences of all 7,698,004 proteins encoded in the 2,296 genomes included in COGs (see the list in cog-24.org.csv).
The sequences are in FASTA format and are listed in the order of their IDs. COGs cover ~73% of these sequences; some proteins are encoded in two or more genomes.

#-----------------------------------------------------------
# COGorg24.gene.tab.gz
#-----------------------------------------------------------
A gzip-compressed text file containing information on all protein-coding genes encoded in the 2,296 genomes included in COGs (see the list in cog-24.org.csv).
Tab-delimited plain text listing all genes:
Columns:
1. Gene ID (NCBI locus_tag; not guaranteed to be unique)
2. Gene ccordinates on the genome partition (start..end), as annotated in GenBank
3. Gene direction ("+" or "-")
4. NCBI Assembly ID
5. GenBank partiton ID (accession number)
6. GenBank protein ID
NB: 3,047 gene IDs are associated with two or more distinct genes.
NB: 29,316,134 genes, 7,698,004 protein IDs
