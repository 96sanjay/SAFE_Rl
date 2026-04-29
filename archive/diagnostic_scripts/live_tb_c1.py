import os, time, glob, csv
from torch.utils.tensorboard import SummaryWriter
COLUMNS = ["Metrics/EpRet","Metrics/EpCost","Metrics/LagrangeMultiplier",
           "Loss/Loss_pi","Loss/Loss_reward_critic","Loss/Loss_cost_critic","Train/Entropy","Train/KL"]
TB_DIR = "runs/tb_c1_final"
RUNS = {
    "A_pure": "runs/c1_A_pure/TRPOLag-{CityLearnSafety-SoC-v0}/seed-*/progress.csv",
    "B_shaped": "runs/c1_B_shaped/TRPOLag-{CityLearnSafety-SoC-v0}/seed-*/progress.csv",
    "C_forecast": "runs/c1_C_forecast/TRPOLag-{CityLearnSafety-Forecast-v0}/seed-*/progress.csv",
}
def find_latest(p):
    m = sorted(glob.glob(p)); return m[-1] if m else None
def tail(path, w, last):
    if not path or not os.path.exists(path): return last
    with open(path) as f: rows = list(csv.DictReader(f))
    if len(rows) <= last: return last
    for i in range(last, len(rows)):
        r = rows[i]
        try: ep = int(float(r.get("Train/Epoch", i)))
        except: ep = i
        for c in COLUMNS:
            v = r.get(c)
            if v and v != "":
                try: w.add_scalar(c, float(v), ep)
                except: pass
    w.flush(); return len(rows)
def main():
    os.makedirs(TB_DIR, exist_ok=True)
    ws = {n: SummaryWriter(os.path.join(TB_DIR, n)) for n in RUNS}
    cs = {n: find_latest(p) for n, p in RUNS.items()}
    ls = {n: 0 for n in RUNS}
    for n in RUNS: print(f"[TB] {n}: {cs[n]}")
    print(f"\ntensorboard --logdir {TB_DIR} --port 6006 --bind_all\n")
    try:
        while True:
            for n, p in RUNS.items():
                if not cs[n]: cs[n] = find_latest(p)
                if cs[n]:
                    new = tail(cs[n], ws[n], ls[n])
                    if new > ls[n]: print(f"  [{n}] epoch {new}")
                    ls[n] = new
            time.sleep(30)
    except KeyboardInterrupt: print("\nStopped.")
    finally:
        for w in ws.values(): w.close()
if __name__ == "__main__": main()
