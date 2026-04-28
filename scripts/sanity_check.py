"""
Sanity checker script for comparing CSV, JSON, JSONL files, and image/OCR data.
Checks for:
  - Consistent entry counts across files
  - Missing IDs (gaps in ID sequence)
  - Duplicate IDs
  - ID consistency across related files
  - Image and OCR text line consistency
  - Text line count vs image height ratios
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Set, Optional
from PIL import Image
import sys
import difflib

sys.stdout.reconfigure(encoding='utf-8')

COLUMNS_PER_PAGE = 3

class SanityChecker:
    """Check consistency and integrity of data files."""
    
    def __init__(self, id_column: str = "entry_id"):
        """
        Initialize the checker.
        
        Args:
            id_column: Name of the ID column/field to check. Defaults to 'entry_id'.
        """
        self.id_column = id_column
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.metadata: Optional[Dict] = None
        self.ocr_data: Optional[pd.DataFrame] = None
        self.classified_entries: Optional[pd.DataFrame] = None
        self.parsed_docs: Optional[pd.DataFrame] = None
        self.parsed_cities: Optional[pd.DataFrame] = None
        self.formatted_entries: Optional[pd.DataFrame] = None
    
    def load_metadata(self, metadata_path: Path):
        """Load metadata JSON file with image snippet information."""
        try:
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        except Exception as e:
            self.errors.append(f"Error loading metadata {metadata_path.name}: {str(e)}")
    
    def load_ocr_output(self, ocr_path: Path):
        """Load OCR output JSONL file."""
        try:
            self.ocr_data = pd.read_json(ocr_path, lines=True)
        except Exception as e:
            self.errors.append(f"Error loading OCR output {ocr_path.name}: {str(e)}")
        
    def load_classified_entries_output(self, classified_entries_path: Path):
        try:
            self.classified_entries = pd.read_csv(classified_entries_path, encoding="utf-8")
        except Exception as e:
            self.errors.append(f"Error loading classified entries output {classified_entries_path.name}: {str(e)}")
        
    def load_parsed_entries_output(self, parsed_docs_path: Path, parsed_cities_path: Path):
        try:
            self.parsed_docs = pd.read_csv(parsed_docs_path, encoding="utf-8")
            self.parsed_cities = pd.read_csv(parsed_cities_path, encoding="utf-8")
        except Exception as e:
            self.errors.append(f"Error loading parsed entries output: {str(e)}")

    def load_reformatted_output(self, formatted_path: Path):
        try:
            self.formatted_entries = pd.read_csv(formatted_path, encoding="utf-8")
        except Exception as e:
            self.errors.append(f"Error loading formatted entries output {formatted_path.name}: {str(e)}")

    def compare_texts(self, text1, text2):
        matcher = difflib.SequenceMatcher(None, text1, text2)

        for change_type, i1, i2, j1, j2 in matcher.get_opcodes():
            if change_type == 'equal':
                continue

            affected_text_in = text1[i1:i2]
            affected_text_out = text2[j1:j2]

            # Get 10 characters of surrounding context
            context_before_in = text1[max(0, i1 - 10):i1]
            context_after_in = text1[i2:min(len(text1), i2 + 10)]
            context_before_out = text2[max(0, j1 - 10):j1]
            context_after_out = text2[j2:min(len(text2), j2 + 10)]

            # Calculate max length for padding
            len_in = len(affected_text_in)
            len_out = len(affected_text_out)
            max_len = max(len_in, len_out)

            # Pad the strings for consistent display
            if change_type != "delete":
                affected_text_in = f"{affected_text_in:<{max_len}}"
                affected_text_out = f"{affected_text_out:<{max_len}}"

            print(f"    Change Type: {change_type}")
            # Highlight changes using brackets and padded text
            print(f"    text_in:  '{context_before_in}{{{{{affected_text_in}}}}}{context_after_in}'")
            print(f"    text_out: '{context_before_out}{{{{{affected_text_out}}}}}{context_after_out}'")
            print("--------------------------------------------------")
            self.text_changes += 1
    
    def analyze_image_ocr_consistency(self, skip_image_loading: bool = False) -> Dict[str, Any]:
        """
        Analyze consistency between images and OCR text lines. Verify that OCR data exists
        for all images listed in the metadata, that there each page has COLUMNS_PER_PAGE 
        columns, that the number of lines for each column on a page are similar, and
        that the pixels per line are similar across for all columns.
        
        Args:
            skip_image_loading: If True, skip loading PIL images (faster, doesn't provide image dimensions)
        """
        if not self.metadata or self.ocr_data is None:
            return {"error": "Metadata and OCR data must be loaded first"}
        
        # Get base directory - use current working directory
        base_dir = Path.cwd()

        all_pubs_pages_cols_OCRed = True
        if len(self.metadata) != len(self.ocr_data["pub"].unique()):
            all_pubs_pages_cols_OCRed = False
            print(
                f"\n❌ number of metadata pubs {len(self.metadata)},"
                f"doesn't match number of unique OCRed pubs {len(self.ocr_data["pub"].unique())}"
            )

        pixels_per_line_by_page_col = {} # (doc, page, col) -> int
        
        for pub_data in self.metadata:
            pub_id = pub_data["pub_id"]
            pub_ocr = self.ocr_data[self.ocr_data["pub"] == pub_id]

            if len(pub_data["pages"]) != len(pub_ocr["page"].unique()):
                all_pubs_pages_cols_OCRed = False
                print(
                    f"\n❌ in {pub_id}, number of metadata pages {len(pub_data["pages"])},"
                    f"doesn't match number of unique OCRed pages {len(pub_ocr["page"].unique())}"
                )
            
            for page_data in pub_data["pages"]:
                page_num = page_data["page_num"]
                snippets = page_data["snippets"]
                
                # Get OCR lines for this page
                page_ocr = pub_ocr[pub_ocr["page"] == page_num]
                
                # Count OCR lines per column
                if len(snippets) != COLUMNS_PER_PAGE:
                    print(
                        f"\n❌ {pub_id} page {page_num}: unexpected number of columns,",
                        f"found {len(snippets)}"
                    )

                if len(snippets) != len(page_ocr["col"].unique()):
                    all_pubs_pages_cols_OCRed = False
                    print(
                        f"\n❌ in {pub_id}.{page_data["page_num"]}, number of metadata columns {len(snippets)},"
                        f"doesn't match number of unique OCRed pages {len(page_ocr["col"].unique())}"
                    )
            
                column_line_counts = {}
                for snippet in snippets:
                    col = snippet["column"]
                    col_ocr = page_ocr[page_ocr["col"] == col]
                    column_line_counts[col] = len(col_ocr)
                
                # Check for column line count consistency for page
                if column_line_counts and len(column_line_counts) > 1:
                    line_counts = list(column_line_counts.values())
                    avg_lines = sum(line_counts) / len(line_counts)
                    max_deviation = max(abs(count - avg_lines) for count in line_counts)
                    # deviation_percent = (max_deviation / avg_lines * 100) if avg_lines > 0 else 0
                    
                    # Flag if any column differs by more than 1% from average
                    if max_deviation>8:
                        print(
                            # f"{pub_id} page {page_num}: Column variance {deviation_percent:.1f}% "
                            f"\n❌ {pub_id} page {page_num}: Column lines differ by more than 8: {max_deviation:.1f} "
                            f"(cols: {column_line_counts})"
                        )
                
                # Analyze image snippets (load images only if requested)
                if not skip_image_loading:
                    for snippet in snippets:
                        img_path = Path(snippet["path"])
                        full_path = base_dir / img_path
                        col = snippet["column"]
                        
                        try:
                            # Try to load image to get dimensions
                            if full_path.exists():
                                with Image.open(full_path) as img:
                                    width, height = img.size
                                    px_per_lines = height // column_line_counts[col]
                                    pixels_per_line_by_page_col[(pub_id, page_num, col)] = px_per_lines
                            else:
                                print(f"\n❌ Image file not found: {full_path}")
                        except Exception as e:
                            print(f"\n❌ Could not read image: {str(e)}")

        # # Calculate overall average lines per image
        pix_per_line = list(pixels_per_line_by_page_col.values())
        avg_pix_per_lines = sum(pix_per_line) / len(pix_per_line)
        deviations = {id: pixels for id, pixels in pixels_per_line_by_page_col.items() if abs(pixels - avg_pix_per_lines) > (avg_pix_per_lines*.3)}
        if deviations:
            print(
                # f"{pub_id} page {page_num}: Column variance {deviation_percent:.1f}% "
                f"\n⚠️ {len(deviations)} columns' pixels per line vary more than .3x from average ({avg_pix_per_lines}):"
                f"\n\t{"\n\t".join(f"{k}={v}" for k, v in deviations.items())}"
            )
        
        if all_pubs_pages_cols_OCRed:
            print("✅ All publications, pages, and columns exist in OCR")
        # return results

    def analyze_classification_consistency(self) -> Dict[str, Any]:
        """
        Analyze consistency between raw OCR text lines and grouped and classified entries. 
        Verify that entries exist for all pages, rows, and columns, that all text has 
        been preserved in order, that bounding boxes match grouped lines, that each document 
        has only one, starting state, that the relative proportion of doc to city entries
        matches across publications, and that there are only a few, correct UNKNOWNs. 
        """
        OCR_COLS = ["pub", "page", "col"]
        CLASSIFIED_COLS = ["publication", "page_number", "column"]

        # Check entries exist for all pages, rows, and columns
        ocr_cols = self.ocr_data[OCR_COLS].drop_duplicates()
        classified_cols = self.classified_entries[CLASSIFIED_COLS].drop_duplicates()
        if len(ocr_cols) != len(classified_cols):
            print(f"\n❌  Inconsistent publication, column counts: "
                  f"OCR has {len(ocr_cols)}, classified has {len(classified_cols)}")
        unmatched_ocr_cols = ocr_cols.merge(
            classified_cols, 
            left_on=OCR_COLS, 
            right_on=CLASSIFIED_COLS, 
            how="outer", 
            indicator=True
        ).query("_merge != 'both'")
        if not unmatched_ocr_cols.empty:
            unmatched_ocr_cols.loc[unmatched_ocr_cols['_merge'] == "left_only", 'reason'] = "missing from classified"
            unmatched_ocr_cols.loc[unmatched_ocr_cols['_merge'] == "right_only", 'reason'] = "added in classified"
            print(f"\n❌  OCR columns with no classified entries:\n{unmatched_ocr_cols.drop('_merge', axis=1)}\n")
        else:
            print("✅ All OCR columns have classified entries")
        
        # Check all text has been preserved in order
        ocr_data = self.ocr_data.copy()
        ocr_data['text'] = ocr_data['text'].str.replace(r"[\s-]", "", regex=True).str.strip()
        ocr_cols_text = ocr_data.groupby(OCR_COLS)['text'].agg("".join).reset_index()
        classified_entries = self.classified_entries.copy()
        classified_entries['full_text'] = classified_entries['full_text'].str.replace(r"[\s-]", "", regex=True).str.strip()
        classified_cols_text = classified_entries.groupby(CLASSIFIED_COLS)['full_text'].agg("".join).reset_index()
        matched_cols_text = ocr_cols_text.merge(
            classified_cols_text, 
            left_on=OCR_COLS, 
            right_on=CLASSIFIED_COLS, 
            how="inner"
        ).rename(columns={"text": "ocr_text", "full_text": "classified_text"})
        self.text_changes = 0
        matched_cols_text.apply(
            lambda row: (
                print(
                    f"⚠️ warning: text changed in {row['pub']}.{row['page']}.{row['col']}! "
                    f"(length in {len(row['ocr_text'])} vs. out {len(row['classified_text'])})"
                ),
                self.compare_texts(row["ocr_text"], row["classified_text"])
            ) if row["ocr_text"] != row["classified_text"] else None, 
            axis=1
        )
        if self.text_changes == 0:
            print("✅ All OCR text preserved in classified entries")
        else:
            print(f"⚠️ {self.text_changes} text changes detected between OCR and classified entries")

        # TODO: Check bounding boxes match grouped lines

        # Check each document has only one, starting state
        drop_num_phys = self.classified_entries[
            (self.classified_entries["entryType"] != "UNKNOWN") |
            (~self.classified_entries["full_text"].str.startswith("NUMBER OF"))
        ]
        type_counts_by_pub = drop_num_phys.groupby("publication", as_index=False)["entryType"].value_counts()
        bad_state_counts = type_counts_by_pub[
            (type_counts_by_pub["entryType"] == "STATE") & (type_counts_by_pub["count"] != 1)
        ]
        if len(bad_state_counts) > 0:
            print(f"\n❌ Unexpected number of state entries found:\n{bad_state_counts}\n")
        else:
            print("✅ All publications have exactly one STATE entry")
        # Check that there are only a few, correct UNKNOWNs. 
        bad_unknown_counts = type_counts_by_pub[
            (type_counts_by_pub["entryType"] == "UNKNOWN") & (type_counts_by_pub["count"] > 0)
        ]
        if len(bad_unknown_counts) > 0:
            print(f"\n❌ UNKNOWN entries found:\n{bad_unknown_counts}\n")
        else:
            print("✅ No UNKNOWN entries found")

        # Check the relative proportion of doc to city entries across publications
        city_doc_proportions_by_pub = self.classified_entries[
            self.classified_entries["entryType"].isin(["DOC", "CITY"])
        ].groupby("publication", as_index=False)["entryType"].value_counts(normalize=True)
        bad_doc_proportions_by_pub = city_doc_proportions_by_pub[
            (city_doc_proportions_by_pub["entryType"] == "DOC") & 
            ((city_doc_proportions_by_pub["proportion"] > 0.94) | 
             (city_doc_proportions_by_pub["proportion"] < 0.69))
        ]
        if len(bad_doc_proportions_by_pub) > 0:
            print(f"⚠️ Unusually high or low proportions of docs to cities:\n{bad_doc_proportions_by_pub}\n")
        else:
            print("✅ Proportions of doc to city entries look consistent across publications")

    def analyze_parsing_consistency(self):
        """
        Analyze consistency between classified entries and parsed entries. Verify that all 
        classified entries are represented in parsed entries and
        that parsed cities and docs are consistent with each other and with classified entries.
        """
        # Verify no repeated ids in parsed entries
        all_ids = pd.concat([self.parsed_docs["entry_id"], self.parsed_cities["entry_id"]], ignore_index=True)
        duplicated_ids = all_ids[all_ids.duplicated()]
        if not duplicated_ids.empty:
            print(f"\n❌ Duplicate IDs found in parsed entries:\n{duplicated_ids.to_string()}\n")
        else:
            print("✅ No duplicate IDs found in parsed entries")
        
        # Verify no seqentially missing ids in parsed entries
        entry_counts = self.classified_entries["entryType"].value_counts()
        missing = self.check_missing_ids(pd.concat([self.parsed_docs[['entry_id']], self.parsed_cities[['entry_id']]]), 'entry_id')
        if missing and entry_counts.get("STATE", 0) - 1 != len(missing): # STATE get IDs but aren't preserved
            print(f"\n❌ Missing {len(missing)} sequential ID numbers (expected {entry_counts.get("STATE", 0) - 1}): \n\t{missing}\n")
            # print("⚠️ Skipping deeper ID verification")
            # return
        else:
            print(f"✅ No missing sequential IDs (outside of expected {entry_counts.get("STATE", 0) - 1})")

        # Verify parsed entries have same number of cities and docs as classified entries
        if entry_counts.get("DOC", 0) != len(self.parsed_docs):
            print(
                f"\n❌ Number of DOC entries mismatch: "
                f"classified has {entry_counts.get('DOC', 0)}, parsed has {len(self.parsed_docs)}"
            )
            doc_in_entries = self.classified_entries[self.classified_entries["entryType"] == "DOC"]
            doc_in_counts = doc_in_entries.groupby(['publication', 'page_number', 'column'])['x'].count().reset_index()
            doc_out_counts = self.parsed_docs.groupby(['publication', 'page_number', 'column'])['x'].count().reset_index()
            doc_counts = doc_in_counts.merge(doc_out_counts, on = ['publication', 'page_number', 'column'], suffixes=['_in', '_out'], validate='1:1')
            doc_off_counts = doc_counts[doc_counts["x_in"] != doc_counts["x_out"]]
            print(doc_off_counts)


        else:
            print("✅ Number of DOC entries matches")

        if entry_counts.get("CITY", 0) != len(self.parsed_cities):
            print(
                f"\n❌ Number of CITY entries mismatch: "
                f"classified has {entry_counts.get('CITY', 0)}, parsed has {len(self.parsed_cities)}"
            )
        else:
            print("✅ Number of CITY entries matches")

        # Verify all classified entry IDs are in parsed entries
        if 'entry_id' in self.classified_entries.columns:
            classified_ids = set(self.classified_entries['entry_id'])
            parsed_doc_ids = set(self.parsed_docs['entry_id'])
            parsed_city_ids = set(self.parsed_cities['entry_id'])

            missing_ids = classified_ids - parsed_doc_ids - parsed_city_ids

            if missing_ids:
                print(f"\n❌ Classify entry IDs ({len(missing_ids)}) not found in parsed docs:\n{missing_ids.to_string()}\n")
            else:
                print("✅ All classified doc entry IDs are in parsed entries")
        else:
            print("⚠️ 'entry_id' column not found in classified entries, skipping ID consistency check with parsed entries")

        # Verify all city entries referenced by a doc entry
        city_with_doc = self.parsed_cities.merge(
            self.parsed_docs[["entry_id", "city_id"]], 
            left_on="entry_id", 
            right_on="city_id", 
            how="left", 
            # validate="1:m"
        )
        missing_doc_refs = city_with_doc[city_with_doc["entry_id_y"].isna()]
        if not missing_doc_refs.empty:
            missing_doc_refs = missing_doc_refs.rename(columns={"entry_id_x": "city_entry_id"})
            print(f"\n ⚠️ City entries ({len(missing_doc_refs)}) with no doc references (this is OK if city a 'See' or small):"
                  f"\n{missing_doc_refs[['city_entry_id']].to_string()}\n")
        else:
            print("✅ All city entries have valid doc references")

        # Verify all doc entries reference a valid city entry
        doc_with_city = self.parsed_docs.merge(
            self.parsed_cities[["entry_id"]], 
            left_on="city_id", 
            right_on="entry_id", 
            how="left", 
            # validate="m:1"
        )
        missing_city_refs = doc_with_city[doc_with_city["entry_id_y"].isna()]
        if not missing_city_refs.empty:
            missing_city_refs = missing_city_refs.rename(columns={"entry_id_x": "doc_entry_id"})
            print(f"\n❌ Doc entries ({len(missing_city_refs)}) with non-existent city references:\n{missing_city_refs[['doc_entry_id', 'city_id']].to_string()}\n")
        else:
            print("✅ All doc entries reference valid city entries")
        
    
    def check_duplicates(self, file_name: str) -> Tuple[int, List[str]]:
        """Check for duplicate IDs in a file."""
        file_data = self.files_data[file_name]
        ids = file_data["ids"]
        id_set = set(ids)
        
        if len(id_set) < len(ids):
            duplicates = [id for id in id_set if ids.count(id) > 1]
            return len([x for x in ids if x in duplicates]), duplicates
        
        return 0, []
    
    def check_missing_ids(self, df:pd.DataFrame, id_col: str) -> List[str]:
        """
        Check for missing IDs (gaps in numeric sequences).
        Only works for numeric IDs or numeric suffixes.
        """
        ids = df[id_col]# [id for id in df[id_col] if id]  # Filter empty IDs
        
        missing = []
        
        # Try to extract numeric parts
        numeric_ids = []
        for id_str in ids:
            # Look for trailing numbers
            i = len(id_str) - 1
            while i >= 0 and id_str[i].isdigit():
                i -= 1
            if i < len(id_str) - 1:
                try:
                    numeric_ids.append((id_str, int(id_str[i+1:])))
                except ValueError:
                    pass
        
        if numeric_ids:
            # Sort by numeric part and check for gaps
            numeric_ids.sort(key=lambda x: x[1])
            
            min_num = numeric_ids[0][1]
            max_num = numeric_ids[-1][1]
            existing_nums = set(num for _, num in numeric_ids)
            
            for num in range(min_num, max_num + 1):
                if num not in existing_nums:
                    missing.append(f"{num}")
        
        return missing
    
    def check_file(self, file_name: str) -> Dict[str, Any]:
        """Run all checks on a single file."""
        if file_name not in self.files_data:
            return {"error": f"File {file_name} not loaded"}
        
        file_data = self.files_data[file_name]
        results = {
            "file": file_name,
            "type": file_data["type"],
            "total_entries": file_data["count"],
            "empty_ids": sum(1 for id in file_data["ids"] if not id),
            "unique_ids": len(set(file_data["ids"])),
        }
        
        # Check duplicates
        dup_count, duplicates = self.check_duplicates(file_name)
        results["duplicate_count"] = dup_count
        if duplicates:
            results["duplicate_ids"] = duplicates[:10]  # Show first 10
            if len(duplicates) > 10:
                results["duplicate_ids"].append(f"... and {len(duplicates) - 10} more")
        
        # Check missing IDs
        missing = self.check_missing_ids(file_name)
        results["missing_ids"] = missing[:10] if missing else []
        if missing and len(missing) > 10:
            results["missing_ids"].append(f"... and {len(missing) - 10} more")
        
        return results
    
    def compare_files(self) -> Dict[str, Any]:
        """Compare IDs across all loaded files."""
        if len(self.files_data) < 2:
            return {"message": "Need at least 2 files to compare"}
        
        file_names = list(self.files_data.keys())
        all_id_sets = [set(self.files_data[fn]["ids"]) for fn in file_names]
        
        # Find IDs common to all files
        common_ids = all_id_sets[0]
        for id_set in all_id_sets[1:]:
            common_ids = common_ids.intersection(id_set)
        
        # Find IDs unique to each file
        unique_by_file = {}
        for i, fn in enumerate(file_names):
            unique = all_id_sets[i].copy()
            for j, other_set in enumerate(all_id_sets):
                if i != j:
                    unique = unique.difference(other_set)
            if unique:
                unique_by_file[fn] = sorted(list(unique))[:10]
        
        return {
            "files_compared": file_names,
            "common_ids_count": len(common_ids),
            "unique_by_file": unique_by_file if unique_by_file else "All IDs are common across files"
        }
    
    def generate_report(self):
        """Generate and print a comprehensive report."""
        print("\n" + "=" * 80)
        print("SANITY CHECK REPORT")
        print("=" * 80 + "\n")
        
        # Individual file checks
        # if self.files_data:
        #     print("FILE CHECKS")
        #     print("-" * 80)
        #     for file_name in self.files_data.keys():
        #         results = self.check_file(file_name)
        #         print(f"\n{file_name}:")
        #         print(f"  Type: {results['type']}")
        #         print(f"  Total entries: {results['total_entries']}")
        #         print(f"  Unique IDs: {results['unique_ids']}")
                
        #         if results['empty_ids'] > 0:
        #             print(f"  ⚠️  Empty IDs: {results['empty_ids']}")
                
        #         if results['duplicate_count'] > 0:
        #             print(f"  \n❌ Duplicates: {results['duplicate_count']} rows with duplicate IDs")
        #             print(f"      {results['duplicate_ids']}")
                
        #         if results['missing_ids']:
        #             print(f"  ⚠️  Missing IDs: {len(results['missing_ids'])} gaps detected")
        #             print(f"      {results['missing_ids']}")
                
        #         if results['empty_ids'] == 0 and results['duplicate_count'] == 0 and not results['missing_ids']:
        #             print(f"  ✓ All checks passed")
            
        #     # Cross-file comparison
        #     if len(self.files_data) > 1:
        #         print("\n" + "-" * 80)
        #         print("CROSS-FILE COMPARISON")
        #         print("-" * 80)
        #         comparison = self.compare_files()
        #         print(f"\nCommon IDs across files: {comparison['common_ids_count']}")
                
        #         if isinstance(comparison['unique_by_file'], dict):
        #             for file_name, unique_ids in comparison['unique_by_file'].items():
        #                 print(f"\nUnique to {file_name}:")
        #                 print(f"  {unique_ids}")
        #         else:
        #             print(f"\n{comparison['unique_by_file']}")
        
        # Image and OCR checks
        if self.metadata and self.ocr_data is not None:
            print("\n" + "-" * 80)
            print("IMAGE & OCR CONSISTENCY CHECKS")
            print("-" * 80)
            self.analyze_image_ocr_consistency()

        # compare 2 to 3
        if self.ocr_data is not None and self.classified_entries is not None:
            print("\n" + "-" * 80)
            print("LINE GROUPING & CLASSIFICATION CONSISTENCY CHECKS")
            print("-" * 80)
            self.analyze_classification_consistency()
        
        # compare 3 to 4
        if self.classified_entries is not None and self.parsed_docs is not None and self.parsed_cities is not None:
            print("\n" + "-" * 80)
            print("ENTRY PARSING CHECKS")
            print("-" * 80)
            self.analyze_parsing_consistency()

        # TODO: compare 4 to 5
        if self.parsed_docs is not None and self.parsed_cities is not None and self.formatted_entries is not None:
            print("\n" + "-" * 80)
            print("REFORMATTING CONSISTENCY CHECKS")
            print("-" * 80)
            self.analyze_reformatting_consistency()

        # Summary
        print("\n" + "=" * 80)
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.warnings:
            print(f"⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        # if not self.errors and not self.warnings:
        #     print("✓ All checks passed!")
        
        print("=" * 80 + "\n")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python sanity_check.py <file1> [file2] ... [OPTIONS]")
        print("\nOptions:")
        print("  --id-column COLUMN      Column name for IDs (default: entry_id)")
        print("  --metadata FILE         Path to metadata JSON file")
        print("  --ocr FILE              Path to OCR output JSONL file")
        print("\nExample:")
        print("  python sanity_check.py data.csv data.json")
        print("  python sanity_check.py entries.jsonl --id-column id")
        print("  python sanity_check.py --metadata all_metadata.json --ocr ocr_output.jsonl")
        sys.exit(1)
    
    # Parse arguments
    files = []
    id_column = "entry_id"
    metadata_file = None
    ocr_file = None
    
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--id-column" and i + 1 < len(sys.argv):
            id_column = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--metadata" and i + 1 < len(sys.argv):
            metadata_file = Path(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--ocr" and i + 1 < len(sys.argv):
            ocr_file = Path(sys.argv[i + 1])
            i += 2
        else:
            files.append(Path(sys.argv[i]))
            i += 1
    run_check(files, id_column, metadata_file, ocr_file)
    
def run_check(
        metadata_file_01: Optional[Path] = None, 
        ocr_file_02: Optional[Path] = None, 
        classified_03: Optional[Path] = None,
        parsed_docs_04: Optional[Path] = None,
        parsed_cities_04: Optional[Path] = None,
        foramtted_05: Optional[Path] = None
    ):
    
    file_missing = False
    for name, value in locals().items():
        if value:
            if not value.exists():
                file_missing = True
                print(f"\n\n❌ Error: {name} file not found: {value}")
    if file_missing:
        sys.exit(1)
    else:
        print("✅ All supplied files found, sanity checking....")

    checker = SanityChecker(id_column="id_column")
    
    # Load files
    if metadata_file_01:
        checker.load_metadata(metadata_file_01)
    if ocr_file_02:
        checker.load_ocr_output(ocr_file_02)
    if classified_03:
        checker.load_classified_entries_output(classified_03)
    if parsed_docs_04 or parsed_cities_04:
        if not (parsed_docs_04 and parsed_cities_04):
            print(
                f"\n\n❌ Error: only one step 4 output provided: "
                f"{"docs" if parsed_cities_04 else "cities"} missing"
            )
            sys.exit(1)
        checker.load_parsed_entries_output(parsed_docs_04, parsed_cities_04)
    if foramtted_05:
        checker.load_reformatted_output(foramtted_05)
    # Generate report
    checker.generate_report()


if __name__ == "__main__":
    # main()
    # run_check(
    #     metadata_file_01=Path("data/01_preprocessed/all_metadata.json"),
    #     ocr_file_02=Path("data/02_raw_batch/ocr_output_reviewed_2026.03.18.jsonl"),
    #     classified_03=Path("data/03_processed_batch/entries_segmented_2026.03.18_man_cleaned.csv"),
    # )
    run_check(
        classified_03=Path("data/03_processed_batch/entries_segmented_2026.03.18_man_cleaned.csv"),
        parsed_docs_04=Path("data/04_extracted_entries_gemini_2026.03.18/amd_1918_doc_entries_sorted_deduped.csv"),
        parsed_cities_04=Path("data/04_extracted_entries_gemini_2026.03.18/amd_1918_city_entries_sorted_deduped.csv"),
        # foramtted_05=Path("data/05_reformatted_entries_2026.03.18/amd_1918_reformatted.csv")
    )
    # run_check(
    #     classified_03=Path("data/03_processed_batch/doc_entries_2026.03.18.csv"),
    #     parsed_docs_04=Path("data/04_extracted_entries_gemini_2026.03.18/amd_1918_doc_entries.csv"),
    #     parsed_cities_04=Path("data/04_extracted_entries_gemini_2026.03.18/amd_1918_city_entries.csv"),
    #     # foramtted_05=Path("data/05_reformatted_entries_2026.03.18/amd_1918_reformatted.csv")
    # )
    