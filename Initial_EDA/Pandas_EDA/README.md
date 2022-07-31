## Pandas EDA 

## Dependencies:

- [x] Install SQLite Viewer app 
- [x] Download SQLite db files
- [x] conda install -c anaconda sqlite
- [x] conda install -c anaconda pandas

## Overview:

Dataset contains 21 WSI images. The images are processed and broken down to create the various datasets. Each dataset has different features stored in SQLite tables. Each of the tables contain different sets of "features" pertaining to the table and correlating to the WSIs. The purpose of the notebooks is to convert the SQLite data from the DB Tables into Pandas dataframes for EDA.  

## Datasets:

- [x] MEL - Manually labeled
- [x] ODAEL - Augmented manually expert labelled
- [x] CODAEL - Clustering and object detection augmented manually expert labeled variant
- [x] TUPAC

## SQLite DB Table Layout:

- Annotations
- sqlite_sequence
- Annotations_coordinates
- Annotations_label
- Classes
- Log
- Persons
- Slides

## Findings:

### COADEL:

- 5 Classes:
	1. Mitotic cell look-alike
	2. Mitotic Figure
	4. Tumor Regio
	5. Exclude from Tumor Region

### ODAEL:

- 5 Classes:
	1. Mitotic cell look-alike
	2. Mitotic Figure
	4. Tumor Region
	5. Exclude from Tumor Region

### MEL:

- 3 Classes:
	1. Mitotic cell look-alike
	2. Mitotic Figure
