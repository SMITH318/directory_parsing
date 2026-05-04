import pandas as pd
from scipy.stats import chisquare
import os
import math

# for 149,566 docs
# to reach 95% confidence with 5% margin of error, need sample of 384
# to reach 99% confidence with 5% margin of error, need sample of 661
# to reach 95% confidence with 3% margin of error, need sample of 1060
# to reach 99% confidence with 3% margin of error, need sample of 1820

def get_diff_cols(df1, df2, col):
    return (df1[col] != df2[col]) & (df1[col].notna() | df2[col].notna())  # Count only if not both NaN

def n_rate_str(n, rate):
        return f"{n:.0f} ({rate*100:.2f}%)"

def print_sep_row(sep='-'):
    print(f'|{sep*15}|{sep*7}|{sep*15}|{sep*15}|{sep*10}|{sep*8}|{sep*16}|')

def print_header():
    print(
        f"\n|{'grouping':<15}|{'n':<7}|{'expected diffs':<15}|"
        f"{'observed diffs':<15}|{'chi^2':<10}|{'p-value':<8}|{'significant diff'}|"
    )
    print_sep_row()

def test_differences(group, n_diffs, n_cells, expected_diff_rate = 0.005):
    """Perform a chi-squared test to see if the observed differences are significantly different from expected noise."""
    # Null Hypothesis: We expect a difference rate (random noise) of expected_diff_rate (e.g., 0.5% or 1%) due to random errors.
    expected_diffs = expected_diff_rate * n_cells
    # print(f"Expected differences (noise): {expected_diffs:.0f} ({expected_diff_rate*100:.2f}%)")
    expected_matches = (1 - expected_diff_rate) * n_cells

    # Observed vs Expected
    # print(f"Observed differences: {n_diffs} ({n_diffs/n_cells*100:.2f}%)")
    obs_diff_rate = n_diffs/n_cells
    f_obs = [n_diffs, n_cells - n_diffs]
    f_exp = [expected_diffs, expected_matches]

    chi2, p_value = chisquare(f_obs=f_obs, f_exp=f_exp)

    # print(
    #     f"Chi-squared test of _cell_ differences chi2={chi2:.2f}, p-value={p_value:.4f}"
    # )

    diff_str = "lower!" if obs_diff_rate < expected_diff_rate else "higher"
    significant_diff = f"*** ({diff_str})" if p_value < 0.05 else ""
    
    print(
        f"|{group:<15}|{n_cells:>7}|{n_rate_str(expected_diffs, expected_diff_rate):>15}"
        f"|{n_rate_str(n_diffs, obs_diff_rate):>15}|{chi2:>10.5f}|{p_value:>8.4f}|{significant_diff:<16}|"
    )


CONFIDENCE_ZSCORES = [
    # (0.90, 1.645), 
    (0.95, 1.96), 
    (0.97, 2.17), 
    (0.99, 2.576)
]
def margin_of_error(z_score, sample_n, match_rate):
    return z_score * (((match_rate*(1-match_rate))/sample_n) ** 0.5) # ^0.5  == sqrt

def print_MOE_sep_row(sep='-'):
    conf_levels = "|".join([f"{sep*11}" for _ in CONFIDENCE_ZSCORES])
    print(f'|{sep*15}|{sep*8}|{sep*8}|{sep*10}|{conf_levels}|')

def print_MOE_header():
    conf_levels = "|".join([f"MOE @ {conf*100:.1f}%" for conf, _ in CONFIDENCE_ZSCORES])
    print(
        f"\n|{'grouping':<15}|{'sample n':<8}|{'n errors':<8}|{'error rate':<10}|{conf_levels}|"
    )
    print_MOE_sep_row()

def compare_MOEs(group, n_diffs, n_cells):
    obs_diff_rate = n_diffs/n_cells
    conf_moes = "|".join([f"+/-{margin_of_error(z, n_cells, obs_diff_rate)*100:>7.2}%" for _, z in CONFIDENCE_ZSCORES])
    print(
        f"|{group:<15}|{n_cells:>8}|{n_diffs:>8}|{obs_diff_rate*100:>9.2f}%|{conf_moes}|"
    )
    


# Define file paths
data_dir = "data/04_extracted_entries_gemini_2026.03.18"
file1 = os.path.join(data_dir, "sampled_entries_docs_output.csv")
file2 = os.path.join(data_dir, "sampled_entries_docs_cleaned.csv")

# Load dataframes
print("Loading CSVs...")
df_output = pd.read_csv(file1)
df_cleaned = pd.read_csv(file2)

# drop columns we're not verifying
cols_to_drop = ['publication','page_number','column','x','y','width','height']
print(f"Ignoring columns: {', '.join(cols_to_drop)}")
df_output = df_output.drop(columns=cols_to_drop)
df_cleaned = df_cleaned.drop(columns=cols_to_drop)

# Compare shapes
if df_output.shape != df_cleaned.shape:
    print(f"\n** Shape mismatch: {df_output.shape} vs {df_cleaned.shape}")
else:
    print(f"Shapes match: {df_output.shape}")
print("** Docs sampled:", len(df_output))

# Find different cells
def find_differences(df1, df2, fix_easy_errors):
    
    print("\n"+"="*90)
    # Ignore easy to fix OCR errors
    if fix_easy_errors:
        print("** Ignoring easy to fix OCR errors (0. vs O.)")
        df1["schools"] = df1["schools"].str.replace("0.", "O.")
        df1["other_info"] = df1["other_info"].str.replace("0.", "O.")
    else:
        print("** Counting all differences")

    differences = pd.DataFrame(columns = df1.columns)
    diffs_by_col = pd.Series(dtype=int)
    for col in df1.columns:
        if col in df2.columns:
            differences[col] = get_diff_cols(df1, df2, col)
            diffs_by_col[col] = differences[col].sum()
        else:
            print(f"** Column {col} in raw output not found in cleaned")

    if diffs_by_col[diffs_by_col > 0].empty:
        print("No differences found!")
        exit()

    print(f"Total columns with differences: {len(diffs_by_col[diffs_by_col > 0])}")
    print(f"Perfect columns: {', '.join(diffs_by_col[diffs_by_col == 0].index.to_list())}")
    # print_header()
    # print("\nTesting _overall_ cell differences against expected noise levels...")
    # test_differences("all cells", diffs_by_col.sum(), len(df_output) * len(df_output.columns), expected_diff_rate=0.005 if FIX_EASY_ERRORS else 0.0075)

    # print("\nTesting _row_ differences against expected noise levels...")
    diffs_by_row = differences.any(axis=1)
    # test_differences("rows", diffs_by_row.sum(), len(df_output), expected_diff_rate=0.05 if FIX_EASY_ERRORS else 0.1)

    # Test if columns' differences are above expected noise
    # print_sep_row('=')
    # for col, n in diffs_by_col[diffs_by_col > 0].items():
        # test_differences(col, n, len(df_output), expected_diff_rate=0.005)  

    print_MOE_header()
    compare_MOEs("all cells", diffs_by_col.sum(), len(df_output.columns) * len(df_output))
    compare_MOEs("rows", diffs_by_row.sum(), len(df_output))

    # Print columns' MOEs
    print_MOE_sep_row('=')
    for col, n in diffs_by_col[diffs_by_col > 0].items(): 
        compare_MOEs(col, n, len(df_output))

    # Print detailed column differences
    print("\nDetailed column differences (first 10):")
    print("\tfound vs. expected")
    for col, n in diffs_by_col[diffs_by_col > 0].items():
        print(f"{col} differences: {n} ({n/len(df_output)*100:.2f}%)")
        diff_rows = get_diff_cols(df_output, df_cleaned, col)
        for i in range(min(10, n)): # Show first differences
            found = df_output[col][diff_rows].iloc[i]
            expected = df_cleaned[col][diff_rows].iloc[i]
            print(
                f"\t'{'' if isinstance(found, float) and math.isnan(found) else found}' "
                f"vs. '{'' if isinstance(expected, float) and math.isnan(expected) else expected}'"
            )

find_differences(df_output, df_cleaned, fix_easy_errors=False)
find_differences(df_output, df_cleaned, fix_easy_errors=True)
