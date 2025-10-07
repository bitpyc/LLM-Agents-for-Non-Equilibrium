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

np.random.seed(42)
# ---------------- 参数解析 ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="模型名称")
parser.add_argument("--J", type=float, required=True, help="J 参数")
parser.add_argument("--rho", type=float, required=True, help="rho 参数")
parser.add_argument("--prob", type=float, required=True, help="可见买家的比例")
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
time_log = datetime.now().isoformat()
# async_client = AsyncOpenAI(api_key="sk-UrncoIcMYsnMGJZwQy0VOxmQC2OZCrLkLPCzL5eSMnI1cRGz", base_url="http://35.220.164.252:3888/v1/")

async_client = AsyncOpenAI(api_key="sk-ffc09fc0b4434f318abbeaa5b2fb6684", base_url="https://api.deepseek.com")
# 创建以时间命名的文件夹
time_log = datetime.now().strftime("%Y%m%d_%H%M%S")
model_save_name = args.model.replace('/', '_')
log_dir = f"RQ2_{model_save_name}_{args.prob}_fig{args.fig}_{time_log}"
os.makedirs(log_dir, exist_ok=True)

LOG_FILE = os.path.join(log_dir, f"sellerV2_{model_save_name}_{J_bar}_{rho}.jsonl")

M = 10
N = 5000

c_token = 0
p_token = 0

async def predict_one_choice(system_prompt, prompt, model="deepseek-chat"):
    global c_token, p_token
    response = await async_client.chat.completions.create(
        model=model,
        reasoning_effort="minimal",
        messages=[{"role":"system","content":system_prompt},
                  {"role":"user","content":prompt}],
        max_completion_tokens=1024,
    )
    c_token += int(response.usage.completion_tokens)
    p_token += int(response.usage.prompt_tokens)
    return response.choices[0].message.content.strip()


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
    x_out = np.zeros(N)  # Changed from M to N - buyers choose sellers

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


async def llm_seller_round(j, x, q, p, u, J, N, M, delays, a, b, sellernoise, sellerprob):
    count_buyer = np.random.rand(N) < sellerprob
    II = np.where(count_buyer)[0]
    N2 = np.sum(count_buyer)
    if N2 == 0:
        return x, q, p  # 没有采样到买家就直接返回

    utemp = u[II, :]
    conv = np.ones((N2, 1))
    U = utemp + J * conv * q.T - conv * p.T

    # ====== 计算 ueff ======
    I = []
    for k in range(M):
        if np.abs(k - j) > 0.5:  # 不包括自己
            I.append(k)

    ueff = np.zeros(N2)
    for i in range(N2):
        ueff[i] = U[i, j] - np.max(np.concatenate((U[i, I], [0])))

    # ====== 统计量 ======
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
        seller_info.append(f'''{{"price": {p[idx]:.2f}, "market_share": {q[idx]:.2f}}}\n''')

    # ====== 直方图（20等宽区间） ======
    num_bins = 20
    counts, bin_edges = np.histogram(ueff, bins=num_bins)
    histogram = []
    for i in range(num_bins):
        bin_range = f"[{bin_edges[i]:.2f} – {bin_edges[i + 1]:.2f})"
        histogram.append((bin_range, int(counts[i])))
    histogram_str = "\n".join([f"{rng}: {cnt}" for rng, cnt in histogram])

    # ====== Prompt ======
    prompt = f"""
    Based on the latest market research, there are {N2} buyers actively considering a purchase. Your price in the last round was {p[j]:.2f}, which secured you a market share of {q[j]:.2%}.

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

    Note that a positive value indicates a loyal customer with room for a price increase, while a negative value represents the minimum discount required to win that customer's business.

    In addition, you are currently competing with {{num_active_sellers}} other sellers. Their prices and resulting market shares from the last round are listed below:
    [
    {seller_info}
    ]

    Your goal is to choose a single price that maximizes your expected net profit.

    Please only output the final price as a number, no other explanations.
    """

    raw_response = await predict_one_choice(
        system_prompt="You are an economic simulation decision agent.",
        prompt=prompt,
        model=args.model
    )

    # ====== 解析 JSON ======
    try:
        new_price = float(raw_response)
    except Exception as e:
        print("JSON解析失败，原始输出:", raw_response)
        new_price = p[j]

    # ====== 日志 ======
    log_entry = {
        "time": datetime.now().isoformat(),
        "seller": j,
        "prompt": prompt,
        "new_price": new_price
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    # ====== 更新价格 ======
    p_new = p.copy()
    p_new[j] = max(0, new_price)
    x_new, q_new, _ = make_choice(x, q, p_new, u, J, N, M, delays)

    return x_new, q_new, p_new


async def seller_round(j, x, q, p, u, J, N, M, delays, a, b, sellernoise, sellerprob):
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
    x_new, q_new, _ = make_choice(x, q, p_new, u, J, N, M, delays)

    return x_new, q_new, p_new


def make_correlator(Q):
    """Calculate correlation function"""
    t2 = np.linspace(-(len(Q) - 1), (len(Q) - 1), 2 * len(Q) - 1)
    Q2 = np.zeros(len(t2))

    for i in range(len(Q)):
        for j in range(len(Q)):
            Q2[len(Q) + i - j - 1] += max(0, Q[i] - Q[j]) ** 2

    Q2 = Q2 / ((len(Q2) + 1) / 2 - np.abs(t2))

    return Q2, t2


# ---------------- 实验入口 ----------------
async def main():
    J = J_bar * M / np.sqrt(np.log(M/2))
    print(f"Running model={args.model}, J_bar={J_bar}, rho={rho}, effective J={J}")

    b = 0.0
    a = 0.8 / M
    sellernoise = 0.0
    sellerprob = args.prob
    waitprob = 1 - rho * np.log(M) / M

    u = np.random.normal(mu, sigma, (N, M))
    z = u.copy()

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
    j_n = []

    for t in range(1, Nt):
        if t % 10 == 0:
            print(f"round {t}...")
        jnext = np.random.randint(0, M)
        j_n.append(jnext)
        delays = (np.random.rand(N) > waitprob).astype(int)
        x[:, t], q[:, t], p[:, t] = await llm_seller_round(
            jnext, x[:, t - 1], q[:, t - 1], p[:, t - 1], u, J, N, M, delays, a, b, sellernoise, sellerprob
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
    plt.savefig(os.path.join(log_dir, '{}-{}.png'.format(J_bar, rho)))
    j_n = np.array(j_n, dtype=int)
    # save simulation details
    np.savez(os.path.join(log_dir, '{}-{}.npz'.format(J_bar, rho)), q=q, p=p, x=x, u=u, j_n=j_n)

if __name__ == "__main__":
    if not os.path.exists(os.path.join(f"RQ2_deepseek-chat_{args.prob}_fig3", f"{J_bar}-{rho}.npz")):
        asyncio.run(main())
        print("c_tokens = {}".format(c_token))
        print("p_tokens = {}".format(p_token))