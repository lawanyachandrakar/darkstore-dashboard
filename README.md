# 📦 Dark Store Control Room

Streamlit dashboard for the IIM Ranchi IPM "Introduction to Probability Theory" WAI project.
Models a quick-commerce dark store using Poisson (orders/min), Normal and Z-scores (delivery time vs 10-min promise),
Hypergeometric (inbound quality inspection) and Binomial (late orders in a batch), linked by total probability and
Bayes' theorem, and verified by Monte Carlo simulation.

**Data:** hypothetical / synthetic parameters (adjustable in the sidebar).
**AI tools used:** Claude (code generation, model formulation); outputs independently verified by simulation.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
