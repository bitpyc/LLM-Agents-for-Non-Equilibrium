import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from openai import AsyncOpenAI
import pandas as pd
import asyncio
from datetime import datetime
import os

# np.random.seed(42)

# ---------------- Argument Parsing ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Model name")
parser.add_argument("--J", type=float, required=True, help="Network effect parameter J_bar")
parser.add_argument("--rho", type=float, required=True, help="Update rate rho")
parser.add_argument("--Nt", type=int, required=True, help="Total number of simulation steps")
parser.add_argument(
    "--base_url",
    type=str,
    default="https://api.openai.com/v1",
    help="Base URL for the OpenAI-compatible API endpoint (default: https://api.openai.com/v1)",
)
args = parser.parse_args()

# total completion_tokens and prompt_tokens
c_token = 0
p_token = 0

# ---------------- Fixed Parameters ----------------
J_bar = args.J
rho = args.rho
mu = 0.50
sigma = 1
M = 10   # number of sellers
N = 5000 # number of buyers

# Async client (API key should be provided via environment variable)
async_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=args.base_url)

# Create a time-stamped log folder
time_log = datetime.now().strftime("%Y%m%d_%H%M%S")
model_save_name = args.model.replace('/', '_')
log_dir = f"LLM_seller_{model_save_name}_{time_log}"
os.makedirs(log_dir, exist_ok=True)

LOG_FILE = os.path.join(log_dir, f"sellerV2_{model_save_name}_{J_bar}_{rho}.jsonl")


# ---------------- LLM Utility Functions ----------------
async def predict_one_choice(system_prompt: str, prompt: str, model: str) -> str:
    """Call the LLM once and return the JSON-formatted response as a string."""
    global c_token, p_token
    response = await async_client.chat.completions.create(
        model=model,
        reasoning_effort="minimal",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=2048,
    )

    if response.usage is not None:
        c_token += int(response.usage.completion_tokens)
        p_token += int(response.usage.prompt_tokens)

    return response.choices[0].message.content.strip()


# ---------------- Buyer Dynamics (Rule-based) ----------------
def make_choice(x, q, p, u, J, N, M, delays):
    """
    Buyers make choices based on utilities in a rule-based way.
    x: current choices of buyers (length N, each entry in {-1, 0, ..., M-1})
    q: current market shares (length M)
    p: current prices (length M)
    u: inherent preferences (N x M)
    delays: 1 if buyer updates, 0 otherwise
    """
    changed = 0
    conv = np.ones((N, 1))

    # Broadcast q and p (1 x M) to match u (N x M)
    q_col = q.reshape(1, -1)
    p_col = p.reshape(1, -1)

    # Utility: U = u + J * q - p
    U = u + J * conv * q_col - conv * p_col

    q_out = np.zeros(M)
    x_out = np.zeros(N, dtype=int)

    for j in range(N):
        if delays[j] > 0.5:
            # Add "no purchase" option with utility 0
            utemp = np.concatenate(([0], U[j, :]))
            xtemp = np.argmax(utemp)
            xtemp -= 1  # shift back to seller index (-1 means no purchase)
        else:
            xtemp = x[j]

        if xtemp >= 0:
            q_out[xtemp] += 1.0 / N

        if xtemp != x[j]:
            changed += 1

        x_out[j] = xtemp

    return x_out, q_out, changed


def buyer_round(x, q, p, u, J, N, M, delays):
    """
    Iterate rule-based buyer decisions until convergence (no changes).
    """
    q_new = q.copy()
    changed = 1

    while changed > 0:
        q_old = q_new.copy()
        x, q_new, changed = make_choice(x, q_old, p, u, J, N, M, delays)

    return x, q_new


# ---------------- Seller Dynamics (LLM-driven or Rule-based) ----------------
async def llm_seller_round(j, x, q, p, u, J, N, M, delays, a, b, sellernoise, sellerprob):
    """
    LLM-driven seller j updates its price based on sampled buyers' reservation utilities,
    then buyers react via the rule-based mechanism.
    """
    count_buyer = np.random.rand(N) < sellerprob
    II = np.where(count_buyer)[0]
    N2 = int(np.sum(count_buyer))
    if N2 == 0:
        # No sampled buyers, keep everything unchanged
        return x, q, p

    utemp = u[II, :]
    conv = np.ones((N2, 1))

    # U: effective utility matrix (N2 x M)
    U = utemp + J * conv * q.T - conv * p.T

    # Compute reservation utility ueff for seller j
    I = [k for k in range(M) if abs(k - j) > 0.5]
    ueff = np.zeros(N2)
    for i in range(N2):
        ueff[i] = U[i, j] - np.max(np.concatenate((U[i, I], [0])))

    # Basic statistics for price premium
    mean_ueff = float(np.mean(ueff))
    median_ueff = float(np.median(ueff))
    std_ueff = float(np.std(ueff))
    min_ueff = float(np.min(ueff))
    max_ueff = float(np.max(ueff))
    p25_ueff, p75_ueff = np.percentile(ueff, [25, 75])

    seller_info = []
    for idx in range(M):
        if idx == j:
            continue
        seller_info.append(
            f'{{"price": {p[idx]:.2f}, "market_share": {q[idx]:.2f}}}'
        )
    seller_info_str = "\n".join(seller_info)

    # Histogram of price premiums
    num_bins = 20
    counts, bin_edges = np.histogram(ueff, bins=num_bins)
    histogram_lines = []
    for i in range(num_bins):
        bin_range = f"[{bin_edges[i]:.2f} – {bin_edges[i+1]:.2f})"
        histogram_lines.append(f"{bin_range}: {int(counts[i])}")
    histogram_str = "\n".join(histogram_lines)

    prompt = f"""
Based on the latest market research, there are {N2} buyers actively considering a purchase. 
Your price in the last round was {p[j]:.2f}, which secured you a market share of {q[j]:.2%}.

Our research shows what each buyer is willing to pay for your product over your current price, compared to their next-best alternative. We call this the "Price Premium".

Here are the key statistics for this Price Premium:
– Mean: {mean_ueff:.2f}
– Median: {median_ueff:.2f}
– Standard Deviation: {std_ueff:.2f}
– Minimum (largest required discount): {min_ueff:.2f}
– Maximum (highest price increase potential): {max_ueff:.2f}
– Interquartile Range (25th–75th percentile): [{p25_ueff:.2f}, {p75_ueff:.2f}]

The detailed distribution of Price Premium across the market is shown below:
{histogram_str}

Note that a positive value indicates a loyal customer with room for a price increase, 
while a negative value represents the minimum discount required to win that customer's business.

In addition, you are currently competing with {M - 1} other sellers. 
Their prices and resulting market shares from the last round are listed below:
[
{seller_info_str}
]

Your goal is to choose a single price that maximizes your expected net profit.

Please analyze the situation and determine your optimal price. 
We need your decision and a brief rationale for your strategy, provided in the following JSON format:
{{
  "reason": "...",
  "final_price": ...
}}

Only return the JSON output and no other explanations.
"""

    raw_response = await predict_one_choice(
        system_prompt="You are an economic simulation decision agent.",
        prompt=prompt,
        model=args.model,
    )

    # Parse JSON result
    try:
        cleaned = raw_response.strip().strip('```').strip('json').strip()
        result = json.loads(cleaned)
        reason = result.get("reason", "")
        new_price = float(result.get("final_price", p[j]))
    except Exception as e:
        print("Failed to parse JSON from LLM response. Raw output:", raw_response)
        reason = "Fallback to previous price due to parsing error."
        new_price = p[j]

    # Log LLM decision
    log_entry = {
        "time": datetime.now().isoformat(),
        "seller": j,
        "prompt": prompt,
        "raw_response": raw_response,
        "parsed_reason": reason,
        "new_price": new_price,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    # Update price and let buyers respond via rule-based choice
    p_new = p.copy()
    p_new[j] = max(0.0, new_price)
    x_new, q_new, _ = make_choice(x, q, p_new, u, J, N, M, delays)

    return x_new, q_new, p_new


# ---------------- Main Experiment Entry ----------------
async def main():
    J = J_bar * M / np.sqrt(np.log(M / 2))
    print(f"Running model={args.model}, J_bar={J_bar}, rho={rho}, effective J={J}")

    b = 0.0
    a = 0.8 / M
    sellernoise = 0.0
    sellerprob = 1.0
    waitprob = 1 - rho * np.log(M) / M

    # Inherent preferences
    u = np.random.normal(mu, sigma, (N, M))
    z = u.copy()

    Nt = args.Nt
    x = np.zeros((N, Nt), dtype=int)
    q = np.zeros((M, Nt))
    p = np.zeros((M, Nt))
    profit = np.zeros((M, Nt))
    tt = np.arange(Nt)

    # Initialization
    p[:, 0] = np.zeros(M) + b
    qinit = np.zeros(M) + 1.0 / M
    x[:, 0], q[:, 0] = buyer_round(x[:, 0], qinit, p[:, 0], u, J, N, M, np.ones(N))
    profit[:, 0] = q[:, 0] * p[:, 0] - b * np.minimum(a, q[:, 0])

    j_n = []

    for t in range(1, Nt):
        jnext = np.random.randint(0, M)
        j_n.append(jnext)

        # Some buyers update their seller choice
        delays = (np.random.rand(N) > waitprob).astype(int)

        # LLM-driven seller updates, buyers respond via rule-based dynamics
        x[:, t], q[:, t], p[:, t] = await llm_seller_round(
            jnext, x[:, t - 1], q[:, t - 1], p[:, t - 1],
            u, J, N, M, delays, a, b, sellernoise, sellerprob,
        )
        profit[:, t] = q[:, t] * p[:, t]

    print("Simulation Done")

    # Plotting
    cmap = plt.cm.get_cmap('hsv', M)
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    for i in range(M):
        plt.plot(tt, q[i, :], color=cmap(i))
    plt.title('Market Shares')

    plt.subplot(2, 1, 2)
    for i in range(M):
        plt.plot(tt, p[i, :], color=cmap(i))
    plt.title('Prices')

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"{J_bar}-{rho}.png"))

    j_n = np.array(j_n, dtype=int)
    np.savez(os.path.join(log_dir, f"{J_bar}-{rho}.npz"), q=q, p=p, x=x, u=u, j_n=j_n)


if __name__ == "__main__":
    asyncio.run(main())
    # print(f"c_tokens = {c_token}")
    # print(f"p_tokens = {p_token}")
