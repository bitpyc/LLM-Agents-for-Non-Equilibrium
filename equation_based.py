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

# ---------------- Argument Parsing ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--J", type=float, required=True, help="Network effect parameter J")
parser.add_argument("--rho", type=float, required=True, help="Update rate rho")
parser.add_argument("--Nt", type=int, required=True, help="Total number of simulation steps")
args = parser.parse_args()

# ---------------- Fixed Parameters ----------------
J_bar = args.J
rho = args.rho
mu = 0.50
sigma = 1

LLM_Seller = True
LLM_Buyer = False
Real_Data = False

# Create time-stamped log folder
time_log = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"Equation_{time_log}"
os.makedirs(log_dir, exist_ok=True)

LOG_FILE = os.path.join(log_dir, f"sellerV2_{J_bar}_{rho}.jsonl")

M = 10   # number of sellers
N = 5000 # number of buyers


def make_choice(x, q, p, u, J, N, M, delays):
    """Buyers make choices based on utilities"""
    changed = 0
    conv = np.ones((N, 1))

    # Reshape q and p for broadcasting
    q_col = q.reshape(1, -1)
    p_col = p.reshape(1, -1)

    # Utility: U = u + J*q - p
    U = u + J * conv * q_col - conv * p_col

    q_out = np.zeros(M)
    x_out = np.zeros(N)

    for j in range(N):
        if delays[j] > 0.5:
            # Include "no purchase option" as index 0
            utemp = np.concatenate(([0], U[j, :]))
            xtemp = np.argmax(utemp)
            xtemp -= 1   # convert back to seller index
        else:
            xtemp = x[j]

        if xtemp >= 0:
            q_out[xtemp] += 1.0 / N

        if xtemp != x[j]:
            changed += 1

        x_out[j] = xtemp

    return x_out, q_out, changed


def buyer_round(x, q, p, u, J, N, M, delays):
    """Iterate buyer decisions until convergence"""
    q_new = q.copy()
    changed = 1

    while changed > 0:
        q_old = q_new.copy()
        x, q_new, changed = make_choice(x, q_old, p, u, J, N, M, delays)

    return x, q_new


async def seller_round(j, x, q, p, u, J, N, M, delays, a, b, sellernoise, sellerprob):
    """Seller j updates price; then buyers respond"""
    count_buyer = np.random.rand(N) < sellerprob
    II = np.where(count_buyer)[0]
    N2 = np.sum(count_buyer)
    utemp = u[II, :]

    conv = np.ones((N2, 1))
    U = utemp + J * conv * q.T - conv * p.T
    ueff = np.zeros(N2)

    # Choose other sellers
    I = []
    for k in range(M):
        if abs(k - j) > 0.5:
            I.append(k)

    # Compute reservation utilities
    for i in range(N2):
        ueff[i] = U[i, j] - np.max(np.concatenate((U[i, I], [0])))

    ueff = np.sort(ueff)
    ueff2 = np.zeros(N2 + 1)
    for k in range(1, N2):
        ueff2[k] = 0.5 * (ueff[k] + ueff[k - 1])

    ueff[0] -= 0.00001
    ueff = np.append(ueff, ueff[-1] + 0.00001)
    qeff = np.linspace(1, 0, N2 + 1)

    # profit curve
    pieff = qeff * (p[j] + ueff) - b * np.minimum(qeff, a)
    optindex = np.argmax(pieff)

    p_new = p.copy()
    p_new[j] = max(0, p[j] + ueff[optindex] + sellernoise * np.random.normal(0, 1))

    x_new, q_new, _ = make_choice(x, q, p_new, u, J, N, M, delays)

    return x_new, q_new, p_new


# ---------------- Simulation Entry ----------------
async def main():
    J = J_bar * M / np.sqrt(np.log(M / 2))
    print(f"Running: J_bar={J_bar}, rho={rho}, effective J={J}")

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
    qinit = np.ones(M) / M

    x[:, 0], q[:, 0] = buyer_round(x[:, 0], qinit, p[:, 0], u, J, N, M, np.ones(N))
    profit[:, 0] = q[:, 0] * p[:, 0] - b * np.minimum(a, q[:, 0])

    j_n = []

    for t in range(1, Nt):

        jnext = np.random.randint(0, M)
        j_n.append(jnext)

        delays = (np.random.rand(N) > waitprob).astype(int)

        x[:, t], q[:, t], p[:, t] = await seller_round(
            jnext, x[:, t - 1], q[:, t - 1], p[:, t - 1],
            u, J, N, M, delays, a, b, sellernoise, sellerprob
        )
        profit[:, t] = q[:, t] * p[:, t]

    print("Simulation Done")

    # -------- Plot Results --------
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

    # Save simulation data
    np.savez(os.path.join(log_dir, f"{J_bar}-{rho}.npz"), q=q, p=p, x=x, u=u, j_n=j_n)


if __name__ == "__main__":
    asyncio.run(main())