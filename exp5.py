import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import json
from openai import OpenAI, AsyncOpenAI
import pandas as pd
import asyncio
from datetime import datetime
import os
import re

# ---------------- 参数解析 ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="模型名称")
parser.add_argument("--J", type=float, required=True, help="J 参数")
parser.add_argument("--rho", type=float, required=True, help="rho 参数")
parser.add_argument("--ratio", type=float, required=True, help="LLM买家比例")
parser.add_argument("--Nt", type=int, required=True, help="总模拟步数")
parser.add_argument("--fig", type=int, required=True, help="哪个图像")
args = parser.parse_args()
# ---------------- 固定参数 ----------------
c_token = 0
p_token = 0
J_bar = args.J
rho = args.rho
mu = 0.50
sigma = 1
LLM_Seller = True
LLM_Buyer = True
Real_Data = False
time_log = datetime.now().isoformat()
async_client = AsyncOpenAI(api_key="sk-UrncoIcMYsnMGJZwQy0VOxmQC2OZCrLkLPCzL5eSMnI1cRGz", base_url="http://35.220.164.252:3888/v1/")

# async_client = AsyncOpenAI(api_key="sk-ffc09fc0b4434f318abbeaa5b2fb6684", base_url="https://api.deepseek.com")
# 创建以时间命名的文件夹
time_log = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"RQ3_ratio-{args.ratio}_{time_log}"
os.makedirs(log_dir, exist_ok=True)
model_save_name = args.model.replace('/', '_')

LOG_FILE = os.path.join(log_dir, f"sellerV2_{model_save_name}_{J_bar}_{rho}.jsonl")
B_LOG_FILE = os.path.join(log_dir, f"buyer_{model_save_name}_{J_bar}_{rho}.jsonl")

M = 10
N = 100

c_token = 0
p_token = 0


# ---------------- 公共函数 ----------------
def assign_preference_level(u):
    thresholds = [-np.inf, mu - 1.5*sigma, mu - 0.5*sigma,
                  mu + 0.5*sigma, mu + 1.5*sigma, np.inf]
    levels = np.array(["terrible", "bad", "neutral", "good", "excellent"])
    indices = np.digitize(u, thresholds) - 1
    indices = np.clip(indices, 0, len(levels)-1)
    return levels[indices]


def init_buyer_impressions(llm_buyers, u, init_value="neutral"):
    """
    初始化 LLM 买家的印象字典
    buyer_impressions = {
        buyer_id: {
            seller_id: "neutral"  # 每个卖家对应的印象
        }
    }
    """
    u_w = assign_preference_level(u)
    buyer_impressions = {
        b: {j: u_w[b, j] for j in range(M)} for b in llm_buyers
    }
    return buyer_impressions


async def predict_one_choice(system_prompt, prompt, model="deepseek-chat"):
    global c_token, p_token
    response = await async_client.chat.completions.create(
        model=model,
        reasoning_effort="minimal",
        response_format={"type": "json_object"},
        messages=[{"role":"system","content":system_prompt},
                  {"role":"user","content":prompt}],
        max_completion_tokens=1024,
    )
    c_token += int(response.usage.completion_tokens)
    p_token += int(response.usage.prompt_tokens)
    return response.choices[0].message.content.strip()


async def batch_predict(system_prompt, prompts, model="deepseek-chat"):
    """并行调用 LLM"""
    tasks = [predict_one_choice(system_prompt, prompt, model) for prompt in prompts]
    return await asyncio.gather(*tasks)



def init_llm_buyers(N, ratio=0.1, seed=42):
    """在实验初始化时确定固定的 LLM 买家集合"""
    rng = np.random.default_rng(seed)
    llm_buyers = rng.choice(N, size=int(N * ratio), replace=False)
    return set(llm_buyers)


async def llm_buyer_choice_parallel(x, q, p, u, J, N, M, delays, llm_buyers, buyer_impressions, model="deepseek-chat"):
    """
    买家决策：
    - 固定的 llm_buyers 用大模型决策（仅更新当前卖家的印象，其余卖家基于 u）
    - 其余买家用规则决策
    - 印象通过 buyer_impressions 传递
    """
    changed = 0
    update_indices = np.where(delays > 0.5)[0]
    if len(update_indices) == 0:
        return x, q, changed

    p_prompt = np.around(p, 2)
    q_prompt = np.around(q, 2)
    u_prompt = np.around(u, 2)

    x_new = x.copy()
    q_out = np.zeros(M)

    # --- Step 1: 规则买家 ---
    rule_indices = [i for i in range(N) if i not in llm_buyers]
    xtemp, _, _ = make_choice(x[rule_indices], q, p, u[rule_indices, :], J, len(rule_indices), M, delays)
    for i, b in enumerate(rule_indices):
        x_new[b] = xtemp[i]
        if xtemp[i] >= 0:
            q_out[xtemp[i]] += 1.0 / N

    # --- Step 2: LLM 买家 ---
    llm_indices = [i for i in llm_buyers]
    if len(llm_indices) > 0:
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
                for j in range(M) if j != x[i]
            ]

            prompt = f"""
You are currently a buyer in a competitive market with multiple sellers. 

Your current seller is Seller_{cur_seller}. In the previous round, they offered you a price of {cur_price}, and captured a market share of {cur_share}.
You currently perceive this seller as {cur_impression}.

Other sellers in the market are:
{other_sellers}

Each seller has a price and a market share from the last round. Use this information to decide how your impression of your current seller should change. For example, if your seller’s price is higher than others but their market share is falling, your impression may worsen. If their price is competitive and their market share is stable or growing, your impression may improve.

Please update ONLY impression of your current seller in the following JSON format:

{{ "updated_impression": "..." }}
"""
            update_prompts.append(prompt)

        # 批量更新当前卖家印象
        update_results = await batch_predict(
            "You are a boundedly rational buyer agent operating in a competitive market with multiple sellers.",
            update_prompts,
            model
        )

        # --- 决策 ---
        decision_prompts = []
        impression_prompts = []
        # 更新 buyer_impressions
        for idx, res in zip(llm_indices, update_results):
            try:
                returned_impression = json.loads(res)
                if x[idx] >= 0:
                    updated_impressions = [{"seller": f"Seller_{j}", "updated_impression": buyer_impressions[idx][j] if j != x[idx] else returned_impression} for j in range(M)]
                else:
                    updated_impressions = [{"seller": f"Seller_{j}", "updated_impression": buyer_impressions[idx][j]}
                                           for j in range(M)]
            except:
                returned_impression = "None"
                updated_impressions = [{"seller": f"Seller_{j}", "updated_impression": buyer_impressions[idx][j]} for j in range(M)]
            impression_prompts.append(returned_impression)
            # 写回字典
            for j in range(M):
                buyer_impressions[idx][j] = updated_impressions[j]["updated_impression"]

            # 构造 sellers 列表：当前卖家用新印象，其余卖家用 u-based 固有偏好
            sellers_list = [
                {
                    "seller": f"Seller_{j}",
                    "price": float(p_prompt[j]),
                    "market_share": float(q_prompt[j]),
                    "your_impression": buyer_impressions[idx][j]
                }
                for j in range(M)
            ]

            cur_price = p_prompt[cur_seller] if cur_seller is not None else "N/A"
            cur_share = q_prompt[cur_seller] if cur_seller is not None else 0.0

            prompt = f"""
You are currently a buyer in a competitive market with multiple sellers. 

Your current seller is Seller_{cur_seller}, offering a price of {cur_price}, and capturing a market share of {cur_share}.
You currently perceive this seller as {updated_impressions}.

There are {M} sellers in the market. For each seller, their price, market share, and your subjective impression are provided below:

{sellers_list}

Your task is to evaluate all available sellers and decide which one you would prefer to purchase from, taking into account both price and your personal impression of each seller.

Please provide your decision and a brief rationale for your choice in the following JSON format:

{{
"reason": "...",
"choice": "Seller_k"
}}
"""
            decision_prompts.append(prompt)

        # 批量决策
        decision_results = await batch_predict(
            "You are a boundedly rational buyer agent operating in a competitive market with multiple sellers.",
            decision_prompts,
            model
        )

        log_entry = {}
        # 更新 LLM 买家选择
        for idx, res, i_prompt in zip(llm_indices, decision_results, impression_prompts):
            try:
                res = res.strip().strip('```').strip('json').strip()
                result_json = json.loads(res)
                seller_choice = result_json.get("choice", "None")
                reason = result_json.get("reason", "None")
                xtemp = int(seller_choice.split("_")[-1]) if "Seller" in seller_choice else -1
            except:
                xtemp = -1
                reason = "fallback to previous price, raw_response={}".format(res)

            log_entry[f'buyer_{idx}'] = {
                "time": datetime.now().isoformat(),
                "current_choice":str(x[idx]),
                "prompt": i_prompt,
                "reason": reason,
                "new_choice": xtemp
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
    """Buyers make choices based on utilities"""
    changed = 0
    conv = np.ones((N, 1))  # Nx1 column vector

    # Reshape q and p to be column vectors for broadcasting
    q_col = q.reshape(1, -1)  # 1xM
    p_col = p.reshape(1, -1)  # 1xM

    # Calculate utilities (u is NxM)
    U = u + J * conv * q_col - conv * p_col

    q_out = np.zeros(M)
    x_out = np.zeros(N, dtype=int)  # Changed from M to N - buyers choose sellers

    for j in range(N):
        if delays[j] > 0.5:
            # Include option 0 (no purchase)
            utemp = np.concatenate(([0], U[j, :]))
            xtemp = np.argmax(utemp)
            # Convert to seller index (0=no purchase, 1=M are sellers)
            xtemp = xtemp - 1
        else:
            xtemp = x[j] # Keep previous choice

        if xtemp >= 0:
            q_out[xtemp] += 1.0 / N

        if xtemp != x[j]:  # Changed from abs difference check
            changed += 1

        x_out[j] = xtemp

    return x_out, q_out, changed


def buyer_round(x, q, p, u, J, N, M, delays):
    """Run buyer decision making until convergence"""
    q_new = q.copy()
    changed = 1

    while changed > 0:
        q_old = q_new.copy()
        x, q_new, changed = make_choice(x, q_old, p, u, J, N, M, delays)

    return x, q_new


async def seller_round(j, x, q, p, u, J, N, M, delays, a, b, sellernoise, sellerprob, llm_customers, buyer_impressions):
    """Seller updates price and buyers respond"""
    count_buyer = np.random.rand(N) < sellerprob * np.ones(N)
    II = np.where(count_buyer)[0]
    N2 = np.sum(count_buyer)
    utemp = u[II, :]

    conv = np.ones((N2, 1))
    U = utemp + J * conv * q.T - conv * p.T
    ueff = np.zeros(N2)

    # choose other sellers
    I = []
    for k in range(M):
        if np.abs(k - j) > 0.5:
            I.append(k)

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
    p_new[j] = max(0, p[j] + ueff[optindex] + sellernoise * np.random.normal(0, 1))
    x_new, q_new, _ = await llm_buyer_choice_parallel(x, q, p_new, u, J, N, M, delays, llm_customers, buyer_impressions)

    return x_new, q_new, p_new


# ---------------- 实验入口 ----------------
async def main():
    J = J_bar * M / np.sqrt(np.log(M/2))
    print(f"Running model={args.model}, J_bar={J_bar}, rho={rho}, effective J={J}")

    b = 0.0
    a = 0.8 / M
    sellernoise = 0.0
    sellerprob = 1
    waitprob = 1 - rho * np.log(M) / M

    u = np.random.normal(mu, sigma, (N, M))

    Nt = args.Nt
    x = np.zeros((N, Nt), dtype=int)
    q = np.zeros((M, Nt))
    p = np.zeros((M, Nt))
    profit = np.zeros((M, Nt))
    tt = np.arange(Nt)

    p[:, 0] = np.zeros(M) + b
    qinit = np.zeros(M) + 1.0 / M
    x[:, 0], q[:, 0] = buyer_round(x[:, 0], qinit, p[:, 0], u, J, N, M, np.ones(N))
    profit[:, 0] = q[:, 0] * p[:, 0] - b * np.minimum(a, q[:, 0])
    llm_customers = init_llm_buyers(N, args.ratio)
    buyer_impressions = init_buyer_impressions(llm_customers, u)
    j_n = []

    for t in range(1, Nt):
        if t % 10 == 0:
            print(f"round {t}...")
        jnext = np.random.randint(0, M)
        j_n.append(jnext)
        delays = (np.random.rand(N) > waitprob).astype(int)
        x[:, t], q[:, t], p[:, t] = await seller_round(
            jnext, x[:, t - 1], q[:, t - 1], p[:, t - 1], u, J, N, M, delays, a, b, sellernoise, sellerprob, llm_customers, buyer_impressions
        )
        profit[:, t] = q[:, t] * p[:, t]

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
    plt.savefig(os.path.join(log_dir, '{}-{}-{}-{}-{}.png'.format(model_save_name, J_bar, rho, LLM_Seller, LLM_Buyer)))
    # save simulation details
    np.savez(os.path.join(log_dir, '{}-{}.npz'.format(J_bar, rho)), q=q, p=p, x=x, u=u, j_n=j_n)

if __name__ == "__main__":
    asyncio.run(main())
    print("c_tokens = {}".format(c_token))
    print("p_tokens = {}".format(p_token))
