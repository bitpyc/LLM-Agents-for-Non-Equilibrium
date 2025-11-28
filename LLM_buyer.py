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

# ---------------- Argument Parsing ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Model name")
parser.add_argument("--J", type=float, required=True, help="Network effect parameter J_bar")
parser.add_argument("--rho", type=float, required=True, help="Update rate rho")
parser.add_argument("--ratio", type=float, required=True, help="Proportion of LLM-driven buyers (0–1)")
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

# Async client (API key should be provided via environment variable)
async_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=args.base_url)

# Create a time-stamped folder
time_log = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"RQ3_ratio-{args.ratio}_{time_log}"
os.makedirs(log_dir, exist_ok=True)
model_save_name = args.model.replace('/', '_')

LOG_FILE = os.path.join(log_dir, f"sellerV2_{model_save_name}_{J_bar}_{rho}.jsonl")
B_LOG_FILE = os.path.join(log_dir, f"buyer_{model_save_name}_{J_bar}_{rho}.jsonl")

M = 10
N = 100


# ---------------- Helper Functions ----------------
def assign_preference_level(u):
    """
    Map continuous preference scores to discrete verbal categories.
    """
    thresholds = [-np.inf, mu - 1.5 * sigma, mu - 0.5 * sigma, mu + 0.5 * sigma, mu + 1.5 * sigma, np.inf]
    levels = np.array(["terrible", "bad", "neutral", "good", "excellent"])
    indices = np.digitize(u, thresholds) - 1
    indices = np.clip(indices, 0, len(levels) - 1)
    return levels[indices]


def init_buyer_impressions(llm_buyers, u):
    """
    Initialize impressions for LLM-driven buyers.
    buyer_impressions is a dict:
        buyer_impressions = {
            buyer_id: {
                seller_id: "neutral" / "good" / ...
            }
        }
    Here we initialize impressions based on inherent preferences.
    """
    u_w = assign_preference_level(u)
    buyer_impressions = {b: {j: u_w[b, j] for j in range(M)} for b in llm_buyers}
    return buyer_impressions


async def predict_one_choice(system_prompt, prompt, model="deepseek-chat"):
    """Single LLM call returning a JSON-formatted string."""
    global c_token, p_token
    response = await async_client.chat.completions.create(
        model=model,
        reasoning_effort="minimal",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=1024,
    )
    if response.usage is not None:
        c_token += int(response.usage.completion_tokens)
        p_token += int(response.usage.prompt_tokens)
    return response.choices[0].message.content.strip()


async def batch_predict(system_prompt, prompts, model="deepseek-chat"):
    """Call the LLM in parallel for a batch of prompts."""
    tasks = [predict_one_choice(system_prompt, prompt, model) for prompt in prompts]
    return await asyncio.gather(*tasks)


def init_llm_buyers(N, ratio=0.1, seed=42):
    """
    Select a fixed subset of buyers that will be LLM-driven.
    """
    rng = np.random.default_rng(seed)
    llm_buyers = rng.choice(N, size=int(N * ratio), replace=False)
    return set(llm_buyers)


async def llm_buyer_choice_parallel(x, q, p, u, J, N, M, delays, llm_buyers, buyer_impressions, model="deepseek-chat"):
    """
    Buyer decision module:
      - Buyers in llm_buyers use an LLM-based decision process.
      - Other buyers follow the rule-based utility-maximizing choice.
      - buyer_impressions propagates over time for LLM buyers.
    """
    changed = 0
    update_indices = np.where(delays > 0.5)[0]
    if len(update_indices) == 0:
        return x, np.zeros(M), changed

    p_prompt = np.around(p, 2)
    q_prompt = np.around(q, 2)

    x_new = x.copy()
    q_out = np.zeros(M)

    # 1) Rule-based buyers (non-LLM)
    rule_indices = [i for i in range(N) if i not in llm_buyers]
    if len(rule_indices) > 0:
        xtemp, _, _ = make_choice(x[rule_indices], q, p, u[rule_indices, :], J, len(rule_indices), M, delays)
        for i, b in enumerate(rule_indices):
            x_new[b] = xtemp[i]
            if xtemp[i] >= 0:
                q_out[xtemp[i]] += 1.0 / N

    # 2) LLM-driven buyers
    llm_indices = [i for i in llm_buyers]
    if len(llm_indices) > 0:
        # 2a) Update impressions of the current seller
        update_prompts = []
        for i in llm_indices:
            cur_seller = x[i] if x[i] >= 0 else None
            if cur_seller is None:
                continue

            cur_price = p_prompt[cur_seller]
            cur_share = q_prompt[cur_seller]
            cur_impression = buyer_impressions[i][cur_seller]
            other_sellers = [
                {"seller": f"Seller_{j}", "price": float(p_prompt[j]), "market_share": float(q_prompt[j])}
                for j in range(M)
                if j != x[i]
            ]

            prompt = f"""
You are a buyer in a competitive market with multiple sellers.

Your current seller is Seller_{cur_seller}. In the previous round, they offered you a price of {cur_price}, and held a market share of {cur_share}.
You currently perceive this seller as {cur_impression}.

Other sellers in the market are:
{other_sellers}

Each seller has a price and a market share from the last round. 
Use this information to decide how your impression of your current seller should change.
For example, if your seller's price is higher than others and their market share is falling, your impression may worsen.
If their price is competitive and their market share is stable or growing, your impression may improve.

Please update ONLY the impression of your current seller in the following JSON format:

{{ "updated_impression": "..." }}
"""
            update_prompts.append(prompt)

        update_results = await batch_predict(
            "You are a boundedly rational buyer agent operating in a competitive market with multiple sellers.",
            update_prompts,
            model,
        )

        # 2b) Decision-making based on updated impressions
        decision_prompts = []
        impression_prompts = []

        for idx, res in zip(llm_indices, update_results):
            cur_seller = x[idx] if x[idx] >= 0 else None
            try:
                returned_impression = json.loads(res).get("updated_impression", "neutral")
            except Exception:
                returned_impression = "neutral"

            # Update impression for the current seller; keep others unchanged
            if cur_seller is not None and cur_seller >= 0:
                buyer_impressions[idx][cur_seller] = returned_impression

            impression_prompts.append(returned_impression)

            sellers_list = [
                {
                    "seller": f"Seller_{j}",
                    "price": float(p_prompt[j]),
                    "market_share": float(q_prompt[j]),
                    "your_impression": buyer_impressions[idx][j],
                }
                for j in range(M)
            ]

            cur_price = p_prompt[cur_seller] if cur_seller is not None and cur_seller >= 0 else "N/A"
            cur_share = q_prompt[cur_seller] if cur_seller is not None and cur_seller >= 0 else 0.0

            prompt = f"""
You are a buyer in a competitive market with multiple sellers.

Your current seller is Seller_{cur_seller}, offering a price of {cur_price}, and capturing a market share of {cur_share}.
You currently perceive this seller as {returned_impression}.

There are {M} sellers in the market. 
For each seller, you are given the price, market share, and your subjective impression:

{sellers_list}

Your task is to evaluate all available sellers and decide which one you would prefer to purchase from, 
taking into account both price and your personal impression of each seller.

Please provide your decision and a brief rationale in the following JSON format:

{{
  "reason": "...",
  "choice": "Seller_k"
}}
"""
            decision_prompts.append(prompt)

        decision_results = await batch_predict(
            "You are a boundedly rational buyer agent operating in a competitive market with multiple sellers.",
            decision_prompts,
            model,
        )

        log_entry = {}
        for idx, res, imp in zip(llm_indices, decision_results, impression_prompts):
            try:
                cleaned = res.strip().strip('```').strip('json').strip()
                result_json = json.loads(cleaned)
                seller_choice = result_json.get("choice", "None")
                reason = result_json.get("reason", "None")
                xtemp = int(seller_choice.split("_")[-1]) if "Seller" in seller_choice else -1
            except Exception:
                xtemp = -1
                reason = f"Fallback to previous choice, raw_response={res}"

            log_entry[f"buyer_{idx}"] = {
                "time": datetime.now().isoformat(),
                "current_choice": str(x[idx]),
                "impression_used": imp,
                "raw_response": res,
                "reason": reason,
                "new_choice": xtemp,
            }

            if xtemp >= 0:
                q_out[xtemp] += 1.0 / N
            if xtemp != x[idx]:
                changed += 1
            x_new[idx] = xtemp

        with open(B_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    return x_new, q_out, changed


def make_choice(x, q, p, u, J, N, M, delays):
    """
    Rule-based buyer choice based on utility maximization.
    """
    changed = 0
    conv = np.ones((N, 1))

    q_col = q.reshape(1, -1)
    p_col = p.reshape(1, -1)

    U = u + J * conv * q_col - conv * p_col

    q_out = np.zeros(M)
    x_out = np.zeros(N, dtype=int)

    for j in range(N):
        if delays[j] > 0.5:
            utemp = np.concatenate(([0], U[j, :]))
            xtemp = np.argmax(utemp) - 1
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
    Run rule-based buyer decision updates until convergence.
    """
    q_new = q.copy()
    changed = 1

    while changed > 0:
        q_old = q_new.copy()
        x, q_new, changed = make_choice(x, q_old, p, u, J, N, M, delays)

    return x, q_new


async def seller_round(j, x, q, p, u, J, N, M, delays, a, b, sellernoise, sellerprob, llm_customers, buyer_impressions):
    """
    Rule-based seller updates price using reservation utility,
    then LLM-driven buyers (and rule-based buyers) respond.
    """
    count_buyer = np.random.rand(N) < sellerprob
    II = np.where(count_buyer)[0]
    N2 = int(np.sum(count_buyer))
    utemp = u[II, :]

    conv = np.ones((N2, 1))
    U = utemp + J * conv * q.T - conv * p.T
    ueff = np.zeros(N2)

    # choose other sellers
    I = [k for k in range(M) if abs(k - j) > 0.5]

    for i in range(N2):
        ueff[i] = U[i, j] - np.max(np.concatenate((U[i, I], [0])))

    ueff = np.sort(ueff)
    ueff2 = np.zeros(N2 + 1)
    for k in range(1, N2):
        ueff2[k] = 0.5 * (ueff[k] + ueff[k - 1])

    ueff[0] -= 0.00001
    ueff = np.append(ueff, ueff[-1] + 0.00001)
    qeff = np.linspace(1, 0, N2 + 1)
    pieff = qeff * (p[j] + ueff) - b * np.minimum(qeff, a)
    optindex = np.argmax(pieff)

    p_new = p.copy()
    p_new[j] = max(0.0, p[j] + ueff[optindex] + sellernoise * np.random.normal(0, 1))

    x_new, q_new, _ = await llm_buyer_choice_parallel(x, q, p_new, u, J, N, M, delays, llm_customers, buyer_impressions, model=args.model)

    return x_new, q_new, p_new


# ---------------- Main Experiment Entry ----------------
async def main():
    J = J_bar * M / np.sqrt(np.log(M / 2))
    print(f"Running model={args.model}, J_bar={J_bar}, rho={rho}, ratio={args.ratio}, effective J={J}")

    b = 0.0
    a = 0.8 / M
    sellernoise = 0.0
    sellerprob = 1.0
    waitprob = 1 - rho * np.log(M) / M

    u = np.random.normal(mu, sigma, (N, M))

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

    llm_customers = init_llm_buyers(N, args.ratio)
    buyer_impressions = init_buyer_impressions(llm_customers, u)
    j_n = []

    for t in range(1, Nt):

        jnext = np.random.randint(0, M)
        j_n.append(jnext)
        delays = (np.random.rand(N) > waitprob).astype(int)

        x[:, t], q[:, t], p[:, t] = await seller_round(
            jnext, x[:, t - 1], q[:, t - 1], p[:, t - 1],
            u, J, N, M, delays, a, b, sellernoise, sellerprob,
            llm_customers, buyer_impressions
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
    plt.savefig(os.path.join(log_dir, "{}-{}-{}.png".format(model_save_name, J_bar, rho)))

    # Save simulation details
    np.savez(os.path.join(log_dir, "{}-{}.npz".format(J_bar, rho)), q=q, p=p, x=x, u=u, j_n=j_n)


if __name__ == "__main__":
    asyncio.run(main())
    # print(f"c_tokens = {c_token}")
    # print(f"p_tokens = {p_token}")
