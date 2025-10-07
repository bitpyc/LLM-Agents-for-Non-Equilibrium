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
parser.add_argument("--J", type=float, required=True, help="J 参数")
parser.add_argument("--rho", type=float, required=True, help="rho 参数")
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
LLM_Buyer = False
Real_Data = False
time_log = datetime.now().isoformat()
# 创建以时间命名的文件夹
time_log = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"exp0_fig{args.fig}_{time_log}"
os.makedirs(log_dir, exist_ok=True)

LOG_FILE = os.path.join(log_dir, f"sellerV2_{J_bar}_{rho}.jsonl")

M = 10
N = 5000

c_token = 0
p_token = 0


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


# ---------------- 实验入口 ----------------
async def main():
    J = J_bar * M / np.sqrt(np.log(M/2))
    print(f"Running J_bar={J_bar}, rho={rho}, effective J={J}")

    b = 0.0
    a = 0.8 / M
    sellernoise = 0.0
    sellerprob = 1
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
        x[:, t], q[:, t], p[:, t] = await seller_round(
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
    asyncio.run(main())
