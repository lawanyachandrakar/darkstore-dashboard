"""
Dark Store Control Room
-----------------------
Quick-commerce dark store: managing uncertainty with probability distributions.

Distributions used (all within the Introduction to Probability Theory course):
  * Poisson        -> orders arriving per minute (demand uncertainty)
  * Normal / Z     -> delivery time and the 10-minute promise (SLA)
  * Hypergeometric -> inbound quality inspection (sampling without replacement)
  * Binomial       -> number of late orders among a batch of orders
  * Total probability + Bayes' theorem -> linking overload and late deliveries

NOTE: All default parameters are HYPOTHETICAL (synthetic) and must be disclosed
as such in the report. Change them in the sidebar to test other scenarios.

Run locally:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Dark Store Control Room", page_icon="📦", layout="wide")

BLUE, RED, GREEN, GREY = "#2E75B6", "#D9534F", "#2E9E6B", "#8A8F98"
FILL = {GREEN: "rgba(46,158,107,0.30)", RED: "rgba(217,83,79,0.30)"}  # translucent fills


def pct(x: float) -> str:
    """Format a probability as a percentage."""
    return f"{x:.2%}"


# ----------------------------------------------------------------------------
# Core probability model
# ----------------------------------------------------------------------------
def capacity_per_minute(pickers: int, rate: float) -> int:
    """Max whole orders the picking team can process in one minute."""
    return int(np.floor(pickers * rate))


def risk_model(lam, pickers, rate, mu, sd, delay, sd_over, sla):
    """
    Model logic
    1) Orders per minute X ~ Poisson(lam).  Store is OVERLOADED in a minute if X > capacity c.
    2) Delivery time T ~ Normal(mu, sd) in normal operation,
                         Normal(mu + delay, sd_over) when overloaded.
    3) Total probability:  P(late) = P(O)P(T>SLA|O) + P(not O)P(T>SLA|not O)
    4) Bayes:              P(O | late) = P(O)P(T>SLA|O) / P(late)
    """
    c = capacity_per_minute(pickers, rate)
    p_over = stats.poisson.sf(c, lam)                      # P(X > c)
    p_late_normal = stats.norm.sf(sla, mu, sd)             # P(T > SLA | normal)
    p_late_over = stats.norm.sf(sla, mu + delay, sd_over)  # P(T > SLA | overloaded)
    p_late = p_over * p_late_over + (1 - p_over) * p_late_normal
    p_over_given_late = (p_over * p_late_over / p_late) if p_late > 0 else 0.0
    return {
        "c": c,
        "p_over": p_over,
        "p_late_normal": p_late_normal,
        "p_late_over": p_late_over,
        "p_late": p_late,
        "p_over_given_late": p_over_given_late,
    }


def staffing_table(lam, rate, mu, sd, delay, sd_over, sla, wage, penalty, max_pickers=40):
    """Evaluate every staffing level from 1..max_pickers."""
    rows = []
    for p in range(1, max_pickers + 1):
        m = risk_model(lam, p, rate, mu, sd, delay, sd_over, sla)
        late_per_hour = lam * 60 * m["p_late"]
        cost = p * wage + late_per_hour * penalty
        rows.append(
            {
                "Pickers": p,
                "Capacity/min": m["c"],
                "P(overload)": m["p_over"],
                "P(late)": m["p_late"],
                "Late orders/hour": late_per_hour,
                "Staff cost (Rs/hr)": p * wage,
                "Penalty cost (Rs/hr)": late_per_hour * penalty,
                "Total cost (Rs/hr)": cost,
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def simulate(lam, c, mu, sd, delay, sd_over, sla, n_sims, seed):
    """Monte Carlo check of the analytical results (independent verification)."""
    rng = np.random.default_rng(seed)
    orders = rng.poisson(lam, n_sims)
    overloaded = orders > c
    times = np.where(
        overloaded,
        rng.normal(mu + delay, sd_over, n_sims),
        rng.normal(mu, sd, n_sims),
    )
    late = times > sla
    p_over = overloaded.mean()
    p_late = late.mean()
    p_over_given_late = overloaded[late].mean() if late.any() else 0.0
    return orders.mean(), p_over, p_late, p_over_given_late


# ----------------------------------------------------------------------------
# Sidebar: all assumptions in one place
# ----------------------------------------------------------------------------
st.sidebar.title("⚙️ Assumptions")
st.sidebar.caption("Hypothetical / synthetic parameters - disclose in report.")

st.sidebar.subheader("1. Demand (Poisson)")
lam = st.sidebar.slider("Avg orders per minute (λ) at peak", 0.5, 10.0, 3.0, 0.1)

st.sidebar.subheader("2. Picking capacity")
pickers = st.sidebar.slider("Number of pickers on shift", 1, 30, 8)
rate = st.sidebar.slider("Orders one picker handles per minute", 0.20, 1.00, 0.50, 0.05)

st.sidebar.subheader("3. Delivery time (Normal)")
sla = st.sidebar.slider("Promised delivery time / SLA (min)", 5.0, 20.0, 10.0, 0.5)
mu = st.sidebar.slider("Mean delivery time - normal ops (μ)", 3.0, 15.0, 7.5, 0.1)
sd = st.sidebar.slider("Std. dev. - normal ops (σ)", 0.5, 5.0, 1.5, 0.1)
delay = st.sidebar.slider("Extra delay when overloaded (min)", 0.0, 8.0, 3.0, 0.5)
sd_over = st.sidebar.slider("Std. dev. when overloaded", 0.5, 6.0, 2.0, 0.1)

st.sidebar.subheader("4. Economics")
wage = st.sidebar.number_input("Cost per picker per hour (Rs)", 50, 500, 120, 10)
penalty = st.sidebar.number_input("Cost of one late order (Rs)", 10, 500, 50, 5)
target = st.sidebar.slider("Target max late-delivery rate", 0.01, 0.20, 0.05, 0.01)

st.sidebar.subheader("5. Inbound quality check (Hypergeometric)")
N_batch = st.sidebar.number_input("Batch size (N units)", 20, 2000, 200, 10)
K_def = st.sidebar.number_input("Defective units in batch (K)", 0, int(N_batch), 12, 1)
n_sample = st.sidebar.number_input("Sample inspected (n)", 1, int(N_batch), 10, 1)
c_accept = st.sidebar.number_input("Accept batch if defects found ≤", 0, int(n_sample), 1, 1)

st.sidebar.subheader("6. Order batch (Binomial)")
n_orders = st.sidebar.number_input("Orders in a rider's batch (n)", 2, 100, 20, 1)
k_thr = st.sidebar.number_input("Alert if late orders ≥", 1, int(n_orders), 3, 1)

st.sidebar.subheader("7. Verification")
n_sims = st.sidebar.select_slider("Monte Carlo minutes simulated", [10_000, 50_000, 100_000, 500_000], 100_000)
seed = st.sidebar.number_input("Random seed", 0, 99999, 42, 1)

# ----------------------------------------------------------------------------
# Compute everything once
# ----------------------------------------------------------------------------
m = risk_model(lam, pickers, rate, mu, sd, delay, sd_over, sla)
late_per_hour = lam * 60 * m["p_late"]
total_cost = pickers * wage + late_per_hour * penalty
df_staff = staffing_table(lam, rate, mu, sd, delay, sd_over, sla, wage, penalty)
meets = df_staff[df_staff["P(late)"] <= target]
rec_service = int(meets["Pickers"].iloc[0]) if not meets.empty else None
rec_cost = int(df_staff.loc[df_staff["Total cost (Rs/hr)"].idxmin(), "Pickers"])

# ----------------------------------------------------------------------------
# Header + KPIs
# ----------------------------------------------------------------------------
st.title("📦 Dark Store Control Room")
st.caption(
    "Managing demand and delivery uncertainty in a 10-minute quick-commerce dark store "
    "using Poisson, Normal, Hypergeometric and Binomial models."
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Capacity (orders/min)", m["c"], help="floor(pickers × rate)")
k2.metric("P(store overloaded)", pct(m["p_over"]), help="Poisson: P(X > capacity)")
k3.metric("P(delivery late)", pct(m["p_late"]), help=f"Target: {pct(target)} or lower")
k4.metric("Total cost (Rs/hr)", f"{total_cost:,.0f}")

if m["p_late"] <= target:
    st.success(f"✅ Current staffing of {pickers} pickers meets the {pct(target)} late-delivery target.")
else:
    st.warning(
        f"⚠️ Current staffing of {pickers} pickers misses the {pct(target)} target. "
        + (f"Minimum pickers needed: **{rec_service}**." if rec_service else "No staffing level up to 40 meets it.")
    )

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["1️⃣ Orders (Poisson)", "2️⃣ Delivery time (Normal)", "3️⃣ Combined risk & staffing",
     "4️⃣ Quality check (Hypergeometric)", "5️⃣ Late-order batch (Binomial)", "6️⃣ Verification"]
)

# ----------------------------------------------------------------------------
# Tab 1: Poisson
# ----------------------------------------------------------------------------
with tab1:
    st.subheader("Orders arriving per minute ~ Poisson(λ)")
    st.markdown(
        "**Why Poisson?** Orders arrive randomly, one at a time, independently, at a roughly constant "
        "average rate during the peak window, and we count events per fixed interval."
    )
    xmax = int(max(12, np.ceil(lam + 5 * np.sqrt(lam))))
    xs = np.arange(0, xmax + 1)
    ys = stats.poisson.pmf(xs, lam)
    fig = go.Figure(go.Bar(x=xs, y=ys, marker_color=[RED if x > m["c"] else BLUE for x in xs]))
    fig.update_layout(
        title=f"P(X = x) with λ = {lam}. Red = overload (more than {m['c']} orders/min)",
        xaxis_title="Orders in one minute (x)", yaxis_title="Probability", height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    a, b, c3 = st.columns(3)
    a.metric("Mean = Variance = λ", f"{lam:.2f}")
    b.metric(f"P(X ≤ {m['c']})", pct(stats.poisson.cdf(m["c"], lam)))
    c3.metric(f"P(X > {m['c']})  (overload)", pct(m["p_over"]))

    st.markdown("**Probability calculator**")
    x_q = st.number_input("Orders in a minute (x)", 0, 50, int(round(lam)), 1, key="pq")
    q1, q2, q3 = st.columns(3)
    q1.metric(f"P(X = {x_q})", pct(stats.poisson.pmf(x_q, lam)))
    q2.metric(f"P(X ≥ {x_q})", pct(stats.poisson.sf(x_q - 1, lam)))
    q3.metric(f"P(X ≤ {x_q})", pct(stats.poisson.cdf(x_q, lam)))
    st.caption("Formula: P(X = x) = e^(−λ) λ^x / x!")

# ----------------------------------------------------------------------------
# Tab 2: Normal
# ----------------------------------------------------------------------------
with tab2:
    st.subheader("Delivery time ~ Normal(μ, σ)")
    st.markdown(
        "**Why Normal?** Delivery time is continuous and is the sum of many small independent delays "
        "(picking, packing, riding, handover), so it is symmetric and bell-shaped around a mean."
    )
    mu_o = mu + delay
    lo = min(mu, mu_o) - 4 * max(sd, sd_over)
    hi = max(mu, mu_o) + 4 * max(sd, sd_over)
    xs = np.linspace(max(lo, 0), hi, 500)

    fig = go.Figure()
    for label, mean, s, col in [("Normal operations", mu, sd, GREEN), ("Overloaded", mu_o, sd_over, RED)]:
        fig.add_trace(go.Scatter(x=xs, y=stats.norm.pdf(xs, mean, s), name=label, line=dict(color=col, width=3)))
        xl = xs[xs >= sla]
        fig.add_trace(go.Scatter(x=xl, y=stats.norm.pdf(xl, mean, s), fill="tozeroy", mode="none",
                                 name=f"Late area ({label})", fillcolor=FILL[col]))
    fig.add_vline(x=sla, line_dash="dash", line_color="black", annotation_text=f"SLA = {sla} min")
    fig.update_layout(xaxis_title="Delivery time (minutes)", yaxis_title="Density", height=420)
    st.plotly_chart(fig, use_container_width=True)

    z_n = (sla - mu) / sd
    z_o = (sla - mu_o) / sd_over
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Normal operations**")
        st.metric("Z-score of the SLA", f"{z_n:.2f}", help="Z = (SLA − μ)/σ")
        st.metric("P(on time) = P(T ≤ SLA)", pct(stats.norm.cdf(z_n)))
        st.metric("P(late) = P(T > SLA)", pct(m["p_late_normal"]))
    with d2:
        st.markdown("**Overloaded**")
        st.metric("Z-score of the SLA", f"{z_o:.2f}")
        st.metric("P(on time) = P(T ≤ SLA)", pct(stats.norm.cdf(z_o)))
        st.metric("P(late) = P(T > SLA)", pct(m["p_late_over"]))

    st.markdown("**Probability calculator (normal operations)**")
    cc1, cc2 = st.columns(2)
    t1 = cc1.number_input("From (min)", 0.0, 40.0, float(max(mu - sd, 0)), 0.5)
    t2 = cc2.number_input("To (min)", 0.0, 40.0, float(mu + sd), 0.5)
    if t2 > t1:
        st.info(f"P({t1} ≤ T ≤ {t2}) = **{pct(stats.norm.cdf(t2, mu, sd) - stats.norm.cdf(t1, mu, sd))}**")
    pctl = st.slider("Percentile of orders to guarantee", 50, 99, 95)
    z_p = stats.norm.ppf(pctl / 100)
    st.info(f"{pctl}% of orders arrive within **{mu + z_p * sd:.2f} min** (Z = {z_p:.3f}; T = μ + Zσ).")

# ----------------------------------------------------------------------------
# Tab 3: Combined risk and staffing
# ----------------------------------------------------------------------------
with tab3:
    st.subheader("Linking demand (Poisson) and delivery time (Normal)")
    st.markdown(
        "- **Total probability:** P(late) = P(O)·P(late | O) + P(Oᶜ)·P(late | Oᶜ)\n"
        "- **Bayes' theorem:** P(O | late) = P(O)·P(late | O) / P(late)\n\n"
        "where O = store is overloaded in that minute."
    )
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("P(O)", pct(m["p_over"]))
    r2.metric("P(late | O)", pct(m["p_late_over"]))
    r3.metric("P(late | not O)", pct(m["p_late_normal"]))
    r4.metric("P(late)", pct(m["p_late"]))
    st.info(f"If a delivery is late, the probability the store was overloaded at that time is "
            f"**{pct(m['p_over_given_late'])}** (Bayes' theorem).")

    colA, colB = st.columns(2)
    with colA:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_staff["Pickers"], y=df_staff["P(late)"], mode="lines+markers",
                                 line=dict(color=RED), name="P(late)"))
        fig.add_hline(y=target, line_dash="dash", line_color=GREY, annotation_text="target")
        fig.add_vline(x=pickers, line_dash="dot", line_color=BLUE, annotation_text="current")
        fig.update_layout(title="Late-delivery probability vs pickers", xaxis_title="Pickers",
                          yaxis_title="P(late)", yaxis_tickformat=".0%", height=380)
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_staff["Pickers"], y=df_staff["Staff cost (Rs/hr)"], name="Staff cost",
                                 line=dict(color=BLUE)))
        fig.add_trace(go.Scatter(x=df_staff["Pickers"], y=df_staff["Penalty cost (Rs/hr)"], name="Late-order penalty",
                                 line=dict(color=RED)))
        fig.add_trace(go.Scatter(x=df_staff["Pickers"], y=df_staff["Total cost (Rs/hr)"], name="Total cost",
                                 line=dict(color="black", width=3)))
        fig.add_vline(x=rec_cost, line_dash="dot", line_color=GREEN, annotation_text="min cost")
        fig.update_layout(title="Cost vs pickers (Rs per hour)", xaxis_title="Pickers", yaxis_title="Rs/hour", height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.success(
        f"**Recommendation:** cheapest staffing = **{rec_cost} pickers**; "
        + (f"smallest team meeting the {pct(target)} service target = **{rec_service} pickers**."
           if rec_service else f"no team size up to 40 meets the {pct(target)} target.")
    )
    st.dataframe(
        df_staff.style.format({"P(overload)": "{:.2%}", "P(late)": "{:.2%}", "Late orders/hour": "{:.1f}",
                               "Staff cost (Rs/hr)": "{:,.0f}", "Penalty cost (Rs/hr)": "{:,.0f}",
                               "Total cost (Rs/hr)": "{:,.0f}"}),
        use_container_width=True, height=300,
    )
    st.download_button("⬇️ Download staffing table (CSV)", df_staff.to_csv(index=False).encode(),
                       "staffing_table.csv", "text/csv")

# ----------------------------------------------------------------------------
# Tab 4: Hypergeometric
# ----------------------------------------------------------------------------
with tab4:
    st.subheader("Inbound perishables inspection ~ Hypergeometric(N, K, n)")
    st.markdown(
        "**Why Hypergeometric?** We inspect a sample **without replacement** from a **small, finite batch** with "
        "a known number of defective units, so trials are NOT independent and Binomial would be inaccurate."
    )
    N_i, K_i, n_i, c_i = int(N_batch), int(K_def), int(n_sample), int(c_accept)
    hg = stats.hypergeom(M=N_i, n=K_i, N=n_i)
    ks = np.arange(0, min(n_i, K_i) + 1)
    fig = go.Figure(go.Bar(x=ks, y=hg.pmf(ks), marker_color=[GREEN if k <= c_i else RED for k in ks]))
    fig.update_layout(title=f"Defects found in a sample of {n_i} (green = batch accepted)",
                      xaxis_title="Defective units in sample (k)", yaxis_title="Probability", height=360)
    st.plotly_chart(fig, use_container_width=True)

    p_acc = hg.cdf(c_i)
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("P(accept batch)", pct(p_acc), help=f"P(k ≤ {c_i})")
    h2.metric("P(at least 1 defect found)", pct(1 - hg.pmf(0)))
    h3.metric("Expected defects in sample", f"{hg.mean():.2f}", help="n·K/N")
    h4.metric("Std. dev.", f"{hg.std():.2f}")
    st.caption(
        f"Defect rate in batch = {K_i / N_i:.1%}. Probability a **bad** batch still gets accepted "
        f"(consumer's risk) = {pct(p_acc)}. Formula: P(k) = C(K,k)·C(N−K,n−k) / C(N,n)."
    )
    b_approx = stats.binom(n_i, K_i / N_i)
    st.write(f"For comparison, a (wrong) Binomial approximation gives P(accept) = {pct(b_approx.cdf(c_i))}; "
             f"the difference arises from sampling without replacement.")

# ----------------------------------------------------------------------------
# Tab 5: Binomial
# ----------------------------------------------------------------------------
with tab5:
    st.subheader("Late orders in a batch ~ Binomial(n, p)")
    st.markdown(
        f"**Why Binomial?** Each of the n orders is a trial with two outcomes (late / on time), "
        f"independent, with the same probability p = P(late) = **{pct(m['p_late'])}** from the combined model."
    )
    n_o, k_o = int(n_orders), int(k_thr)
    bn = stats.binom(n_o, m["p_late"])
    ks = np.arange(0, n_o + 1)
    fig = go.Figure(go.Bar(x=ks, y=bn.pmf(ks), marker_color=[RED if k >= k_o else BLUE for k in ks]))
    fig.update_layout(title=f"Late orders out of {n_o} (red = alert zone, ≥ {k_o})",
                      xaxis_title="Number of late orders", yaxis_title="Probability", height=360)
    st.plotly_chart(fig, use_container_width=True)
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Expected late orders (np)", f"{bn.mean():.2f}")
    e2.metric("Std. dev. √(npq)", f"{bn.std():.2f}")
    e3.metric(f"P(X ≥ {k_o})", pct(bn.sf(k_o - 1)))
    e4.metric("P(no late order)", pct(bn.pmf(0)))

# ----------------------------------------------------------------------------
# Tab 6: Verification
# ----------------------------------------------------------------------------
with tab6:
    st.subheader("Independent verification by Monte Carlo simulation")
    st.markdown(
        "AI-generated formulas must be verified. Here we simulate thousands of random minutes and compare "
        "the simulated frequencies with the analytical probabilities. They should agree closely."
    )
    s_mean, s_over, s_late, s_bayes = simulate(lam, m["c"], mu, sd, delay, sd_over, sla, int(n_sims), int(seed))
    rng = np.random.default_rng(int(seed))
    sim_acc = (rng.hypergeometric(K_i, N_i - K_i, n_i, int(n_sims)) <= c_i).mean()

    ver = pd.DataFrame(
        {
            "Quantity": ["Mean orders/min", "P(overload)", "P(late)", "P(overloaded | late)  [Bayes]",
                         "P(batch accepted)  [Hypergeometric]"],
            "Analytical": [lam, m["p_over"], m["p_late"], m["p_over_given_late"], p_acc],
            "Simulated": [s_mean, s_over, s_late, s_bayes, sim_acc],
        }
    )
    ver["Abs. difference"] = (ver["Analytical"] - ver["Simulated"]).abs()
    st.dataframe(ver.style.format({"Analytical": "{:.4f}", "Simulated": "{:.4f}", "Abs. difference": "{:.4f}"}),
                 use_container_width=True)
    st.caption(f"Based on {int(n_sims):,} simulated minutes (seed {int(seed)}). "
               "Differences shrink as the number of simulations grows.")

st.divider()
st.caption(
    "Course: Introduction to Probability Theory, IIM Ranchi (IPM). Data are hypothetical and AI-assisted; "
    "all outputs were verified by Monte Carlo simulation. Built with Streamlit, SciPy, NumPy, Pandas, Plotly."
)
