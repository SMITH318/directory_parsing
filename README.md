# Directory Parsing

A pipeline for OCRing and turning historical directories into data (currently focused on the 1918 AMD).

Heavily modified from https://github.com/XLabCU/newspaper_utilities.git beyond the initial preprocessing step.

## Pipeline Overview
![An AI-generated image showing the directory parsing pipeline](<Directory Parsing Update - 2026.04.16.png>)
*An AI-generated visualization of the directory parsing pipeline*

1. **Preprocessing**: Create JPGs for each column of the scanned directory pages 
from the source PDFs.
2. **OCR**: Use the Gemini Batch API to OCR the JPGs and extract text. Then 
sanity check the extracted text to ensure it contains numbers of row, etc.
3. **Combine and Sort**: Combine the OCRed text from all columns (if needed) and 
sort it by row number to reconstruct the original directory order.
4. **Clean OCR Text**: Clean the OCRed text based on low confidence scores and 
unexpected characters usually manual review.
5. **Entry Grouping and Classification**: Group the cleaned individual lines of 
text into directory entries and classify them (e.g. state, city, doctor, etc.) 
using the Gemini Batch API. Then perform sanity checks looking for missing text, 
unusual numbers or rates of states, cities, unclassified lines, etc.
6. **Prepare for Text Review**: Restructure output to prepare for manual review 
of any changes made to the text during cleaning and classification.
7. **Review Text Changes**: Manually review any changes made to the text during 
cleaning and classification, and make adjustments as needed to ensure accuracy.
8. **Split and ID Entries**: Divide the cleaned and classified text by 
entry type, and assign unique IDs to each entry for tracking, analysis, and
cross-referencing. Doctors are linked to their cities based on their order
in the directory listings, and cities are linked to their states.
9. **Entry Processing**: Process the cleaned, classified, and ID'd entries to 
extract structured data using Gemini Batch API, such as names, addresses, 
occupations, etc. City and doctor entries are processed separately to extract 
specific information relevant to those categories. Then perform sanity checks looking for dropped or invented entries, invalid IDs, etc.
10. **Sort and Output**: Sort the processed entries by their unique IDs to maintain the original directory order, and output the final structured data for analysis.
11. **Reformatting for Analysis**: Restructure the output data into a format suitable for analysis, such as CSV or JSON.

## License

MIT License - See LICENSE file for details

## Citation

This pipeline draws inspiration and code from [Newspaper Utilities](https://github.com/XLabCU/newspaper_utilities.git). 
If you use this pipeline in your research, please cite _both_:

```
Smith, Sean. (2026). Directory Parsing: A pipeline for OCRing and turning historical directories into data.
https://github.com/SMITH318/directory_parsing/

XLabCU. (2026). Newspaper Utilities: Configurable Pipeline for Historical Newspaper Analysis.
https://github.com/XLabCU/newspaper_utilities
```

## Acknowledgments

- Original Whitechapel in Shawville research project
- BANQ for historical newspaper archives
- spaCy, NetworkX, D3.js, and other open-source libraries

![](dash.png)
