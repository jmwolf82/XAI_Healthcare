## XAI_Healthcare

### Project: Automating Explainability in Healthcare

Medical patients have a legally protected “right to explain” regarding decisions
made about their medical care. The “black box” prediction nature of ML models
doesn’t adhere to such Explainability leaving patients less likely to trust decisions
made based on a ML algorithm. By incorporating Explainable AI (XAI) techniques in
medical ML diagnostic applications, the patients’ rights remain protected while
helping care providers explain the diagnosis provided.

### Team: Bender

Members:
 - Arvinder Singh
 - Valentina Montesano
 - Jordan Wolf

### Data Lineage

Paper: https://www.nature.com/articles/s41597-020-00756-z  
Dataset:  https://github.com/DeepPathology/MITOS_WSI_CMC

### Data Records

The dataset, consisting of 21 anonymized WSIs in Aperio SVS file format, is publicly available on figshare20. Alongside, we provide cell annotations according to both classes in a SQLite3 database. For each annotation, this database provides:

- The WSI of the annotation

- The absolute center coordinates (x,y) on the WSI

- The class labels, assigned by all experts and the final agreed class. Each annotation label is included in this, resulting in at least two labels (in the case of initial agreement and no further modifications), one by each expert. The unique numeric identifier of each label furthermore represents the order in which the labels were added to the database.

### Database Folder

- [x] Upload sqlite database files that contain annotations
- [ ] Investigate annotations

### References:
- Aubreville, M., Bertram, C.A., Donovan, T.A. et al. A completely annotated whole slide image dataset of canine breast cancer to aid human breast cancer research. Sci Data 7, 417 (2020). https://doi.org/10.1038/s41597-020-00756-z
- M. Aubreville, C. Bertram, R. Klopfleisch and A. Maier (2018) SlideRunner - A Tool for Massive Cell Annotations in Whole Slide Images. In: Bildverarbeitung für die Medizin 2018. Springer Vieweg, Berlin, Heidelberg, 2018. pp. 309-314. link arXiv:1802.02347

### Open Issues/Questions/Concerns:

- [ ] Contact authors of dataset
- [ ] Investigate how expert text annotations on dataset images were implemented
- [ ] Modify expert text annotations on images for easier readability
- [ ] Is this the dataset complete?
- [ ] Are there other datasets that could be used for testing?
- [ ] Is the dataset balanced?
- [ ] What level of Explainability can be best implemented?
