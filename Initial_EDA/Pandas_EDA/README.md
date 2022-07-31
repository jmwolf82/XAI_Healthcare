### Pandas EDA 

Dependencies:

- [x] Install SQLite Viewer app 
- [x] Download SQLite db files
- [x] conda install -c anaconda sqlite

Overview:

Dataset contains 21 WSI images. The images are processed and broken down to create the various datasets. Each dataset has different features stored in SQLite tables. Each of the tables contain different sets of "features" pertaining to the table and correlating to the WSIs. 

SQLite DB Table Layout:

- Annotations
- sqlite_sequence
- Annotations_coordinates
- Annotations_label
- Classes
- Log
- Persons
- Slides

 
