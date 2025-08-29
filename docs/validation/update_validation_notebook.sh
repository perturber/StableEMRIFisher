#!/bin/bash
# Script to update the pre-executed validation notebook

echo "📊 Executing check_fisher_against_mcmc.ipynb..."
jupyter nbconvert --to notebook --execute check_fisher_against_mcmc.ipynb --output check_fisher_against_mcmc_executed.ipynb

