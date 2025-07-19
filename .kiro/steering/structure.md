# Project Structure

## File Organization

```
├── hsmdo.py                      # Main analysis script
├── cleaned_data.csv              # Input transaction data
├── hsmdo_customer_segments.csv   # Output segmentation results
├── hsmdo_model                   # Compiled STAN model (binary)
└── .kiro/
    └── steering/                 # AI assistant guidance files
```

## Core Files

### `hsmdo.py`
Main Python script containing the complete HSMDO analysis pipeline:
- **Data Preparation**: `prepare_data_for_hsmdo()` function
- **Model Training**: `run_hsmdo_model()` function  
- **Segmentation**: `profile_and_segment_customers()` function
- **Configuration**: Parameters at top of main block

### `cleaned_data.csv`
Input dataset with transaction records:
- Required columns: `order_date`, `name`, `order_id`
- Optional columns: `phone`, `order_sum`, `client_type`
- Date format: DD.MM.YYYY
- Customer identifier: `name` field
- Encoding: UTF-8

### `hsmdo_customer_segments.csv`
Output file with customer analysis results:
- `customer_id`: Customer identifier
- `lambda_mean`: Average purchase rate
- `pzf_mean`: Probability of zero future purchases
- `segment`: Assigned customer segment (in Russian)

## Code Structure

### Main Pipeline (3 Steps)
1. **Data Preparation** (`prepare_data_for_hsmdo`)
   - Converts transactions to monthly purchase matrix
   - Calculates recency metrics
   - Filters customers with no purchases
   - Splits calibration/holdout periods

2. **Model Training** (`run_hsmdo_model`)
   - Compiles STAN model code
   - Runs MCMC sampling
   - Returns posterior samples

3. **Customer Profiling** (`profile_and_segment_customers`)
   - Extracts key metrics from posterior
   - Applies segmentation rules
   - Returns final customer profiles

### Configuration Section
Located in `if __name__ == '__main__':` block:
- `HOLDOUT_MONTHS`: Validation period length
- `ANALYSIS_DATE`: Analysis cutoff date
- Data loading logic (real vs demo data)

## Data Flow
1. Raw transactions → Monthly aggregation
2. Monthly data → STAN model format
3. STAN sampling → Posterior distributions
4. Posterior means → Customer segments
5. Segments → CSV export

## Temporary Files
- `hsmdo_model.stan`: Generated during execution, cleaned up automatically
- `hsmdo_model`: Compiled STAN model, persists for performance