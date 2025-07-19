# Technology Stack

## Core Technologies
- **Python 3.x**: Main programming language
- **STAN**: Bayesian statistical modeling framework via CmdStanPy
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing

## Key Dependencies
- `cmdstanpy`: Python interface to STAN for Bayesian modeling
- `pandas`: DataFrame operations and CSV processing
- `numpy`: Mathematical operations and array handling

## Data Processing
- **Input Format**: CSV files with columns: `order_date`, `name`, `order_id`, `phone`, `order_sum`, `client_type`
- **Output Format**: CSV with customer segments and metrics
- **Date Format**: DD.MM.YYYY (day-first format)
- **Encoding**: UTF-8 with BOM for Russian text support

## Model Architecture
- **HSMDO Model**: Hierarchical Shifted Multiplicative Dirichlet-Multinomial
- **MCMC Sampling**: 4 chains, 1000 warmup + 2000 sampling iterations
- **Key Metrics**: 
  - Lambda (λ): Purchase rate parameter
  - PZF: Probability of Zero Future purchases
  - Recency (t): Last purchase month in calibration period

## Common Commands

### Setup
```bash
# Install STAN (first time only)
python -c "from cmdstanpy import install_cmdstan; install_cmdstan()"

# Install dependencies
pip install cmdstanpy pandas numpy
```

### Running the Model
```bash
# Run full analysis
python hsmdo.py

# The script will:
# 1. Load data from cleaned_data.csv
# 2. Prepare data for HSMDO model
# 3. Run MCMC sampling (can take hours)
# 4. Generate customer segments
# 5. Save results to hsmdo_customer_segments.csv
```

### Configuration
- `HOLDOUT_MONTHS = 6`: Months reserved for validation
- `ANALYSIS_DATE = ''`: Leave empty to use latest date in data
- Modify data loading section to switch between real and demo data

## Performance Notes
- MCMC sampling is computationally intensive and can take several hours
- Model compilation happens on first run and is cached
- Memory usage scales with number of customers and time periods
- Progress is shown during sampling