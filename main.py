import sqlite3
import requests
import math
import re
import itertools
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime, timezone
import numpy as np
from collections import Counter
import logging

# ================= CONFIG =================
DB = "tx_predict.db"
SOURCE_API = "https://ahihidonguoccut-2b5i.onrender.com/mohobomaycai"
TELE_ID = "Tele@HoVanThien_Pro"
PATTERN_LENGTH = 20
HISTORY_LIMIT = 500

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TX Predictor API - Ensemble",
    description="Tai/Xiu predictor with advanced analysers",
    version="3.0"
)

# ================= DATABASE =================
def init_db():
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session INTEGER,
        d1 INTEGER, d2 INTEGER, d3 INTEGER,
        total INTEGER,
        result TEXT,
        ts TEXT
    )
    """)
    conn.commit()
    conn.close()

def save_round(session: int, d: list, total: int, result: str, ts: str):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute(
        "INSERT INTO history (session,d1,d2,d3,total,result,ts) VALUES (?,?,?,?,?,?,?)",
        (session, d[0], d[1], d[2], total, result, ts)
    )
    conn.commit()
    conn.close()

def get_last_n(n: int = 200) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT session,d1,d2,d3,total,result,ts FROM history ORDER BY id DESC LIMIT ?", (n,))
    rows = c.fetchall()
    conn.close()
    return [{
        "session": r[0],
        "dices": [r[1], r[2], r[3]],
        "total": r[4],
        "result": r[5],
        "ts": r[6]
    } for r in rows][::-1]

def get_pattern_last_k(k: int = 20) -> str:
    rows = get_last_n(k)
    return "".join(["T" if r["result"].upper().startswith("T") else "X" for r in rows])

# ================= PATTERN DETECTOR =================
def detect_pattern(history: List[Dict], k: int = 20) -> Dict[str, Any]:
    pattern = get_pattern_last_k(k)
    patterns = {
        "1-1": r"TX|XT",
        "2-2": r"TTXX|XXTT",
        "3-1": r"TTTX|XXXT",
        "1-3": r"TXXX|XTTT"
    }
    matched, follow_counts = [], {"T": 0, "X": 0}
    seq = "".join(["T" if r["result"].upper().startswith("T") else "X" for r in history])
    for name, regex in patterns.items():
        for match in re.finditer(regex, seq):
            if match.end() < len(seq):
                nxt = seq[match.end()]
                follow_counts[nxt] += 1
                if match.end() == len(seq) - 1:
                    matched.append(name)
    denom = sum(follow_counts.values())
    p_tai = follow_counts["T"] / denom if denom > 0 else 0.5
    return {"current_pattern": pattern, "matched_patterns": matched, "follow_counts": follow_counts, "p_tai": p_tai}

# ================= ANALYSERS =================
def analyser_freq_window(history, window=30):
    if not history: return {"name": f"freq_{window}", "p_tai": 0.5, "explain": "No data"}
    last = history[-window:]
    p = sum(1 for r in last if r["result"].startswith("T")) / len(last)
    return {"name": f"freq_{window}", "p_tai": p, "explain": f"{p:.3f}"}

def analyser_markov1(history):
    if len(history) < 2: return {"name": "markov1", "p_tai": 0.5, "explain": "Insufficient"}
    trans = {"T->T":0,"T->X":0,"X->T":0,"X->X":0}
    for i in range(1,len(history)):
        a="T" if history[i-1]["result"].startswith("T") else "X"
        b="T" if history[i]["result"].startswith("T") else "X"
        trans[f"{a}->{b}"]+=1
    last="T" if history[-1]["result"].startswith("T") else "X"
    denom=trans[f"{last}->T"]+trans[f"{last}->X"]
    p=trans[f"{last}->T"]/denom if denom else 0.5
    return {"name":"markov1","p_tai":p,"explain":f"{p:.3f}"}

def analyser_markov2(history):
    if len(history) < 3: return {"name":"markov2","p_tai":0.5,"explain":"Insufficient"}
    trans=Counter()
    for i in range(2,len(history)):
        a=("T" if history[i-2]["result"].startswith("T") else "X") + \
          ("T" if history[i-1]["result"].startswith("T") else "X")
        b="T" if history[i]["result"].startswith("T") else "X"
        trans[(a,b)]+=1
    last=("T" if history[-2]["result"].startswith("T") else "X") + \
         ("T" if history[-1]["result"].startswith("T") else "X")
    denom=sum(trans[(last,b)] for b in "TX")
    p=trans[(last,"T")]/denom if denom else 0.5
    return {"name":"markov2","p_tai":p,"explain":f"{p:.3f}"}

def analyser_beta(history):
    alpha=1+sum(1 for r in history if r["result"].startswith("T"))
    beta=1+sum(1 for r in history if r["result"].startswith("X"))
    p=alpha/(alpha+beta)
    return {"name":"beta","p_tai":p,"explain":f"{p:.3f}"}

def analyser_streak(history):
    if not history: return {"name":"streak","p_tai":0.5,"explain":"No data"}
    last="T" if history[-1]["result"].startswith("T") else "X"
    streak=0
    for r in reversed(history):
        if (r["result"].startswith("T") and last=="T") or (r["result"].startswith("X") and last=="X"):
            streak+=1
        else: break
    if streak>=3: p=0.65 if last=="T" else 0.35
    else: p=0.5
    return {"name":"streak","p_tai":p,"explain":f"streak={streak}"}

def analyser_entropy(history,window=50):
    if not history: return {"name":"entropy","p_tai":0.5,"explain":"No data"}
    seq="".join(["T" if r["result"].startswith("T") else "X" for r in history[-window:]])
    counts=Counter(seq)
    probs=[counts[c]/len(seq) for c in counts]
    H=-sum(p*math.log2(p) for p in probs if p>0)
    return {"name":"entropy","p_tai":0.5+(0.5-H/1.0)*0.1,"explain":f"H={H:.2f}"}

def analyser_fourier(history):
    if not history: return {"name":"fourier","p_tai":0.5,"explain":"No data"}
    seq=np.array([1 if r["result"].startswith("T") else -1 for r in history])
    fft=np.fft.fft(seq)
    mag=np.abs(fft)
    dom=np.argmax(mag[1:len(mag)//2])+1
    phase=np.angle(fft[dom])
    p=0.5+0.1*math.sin(phase)
    return {"name":"fourier","p_tai":max(0,min(1,p)),"explain":f"dom={dom},phase={phase:.2f}"}

def analyser_sliding(history,k=5):
    if len(history)<k+1: return {"name":"sliding","p_tai":0.5,"explain":"Insufficient"}
    seq="".join(["T" if r["result"].startswith("T") else "X" for r in history])
    pattern=seq[-k:]
    follow=Counter()
    for i in range(len(seq)-k):
        if seq[i:i+k]==pattern: follow[seq[i+k]]+=1
    denom=sum(follow.values())
    p=follow["T"]/denom if denom else 0.5
    return {"name":"sliding","p_tai":p,"explain":f"{p:.3f}"}

def analyser_face_freq(history):
    if not history: return {"name":"face_freq","p_tai":0.5,"explain":"No data"}
    dices=[d for r in history for d in r["dices"]]
    counts=Counter(dices)
    hi=max(counts,key=counts.get)
    lo=min(counts,key=counts.get)
    p=0.55 if hi in [4,5,6] else 0.45
    return {"name":"face_freq","p_tai":p,"explain":f"hi={hi},lo={lo}"}

def analyser_gap(history):
    if len(history)<2: return {"name":"gap","p_tai":0.5,"explain":"Insufficient"}
    gaps=[]; lastT=None
    for i,r in enumerate(history):
        if r["result"].startswith("T"):
            if lastT is not None: gaps.append(i-lastT)
            lastT=i
    if not gaps: return {"name":"gap","p_tai":0.5,"explain":"No T gaps"}
    avg=sum(gaps)/len(gaps)
    last_gap=len(history)-lastT
    p=0.65 if last_gap>=avg else 0.35
    return {"name":"gap","p_tai":p,"explain":f"avg={avg:.2f},last={last_gap}"}

def analyser_zscore(history,window=50):
    if not history: return {"name":"zscore","p_tai":0.5,"explain":"No data"}
    last=history[-window:]
    totals=[r["total"] for r in last]
    mu=np.mean(totals); sigma=np.std(totals) if np.std(totals)>0 else 1
    z=(totals[-1]-mu)/sigma
    p=0.65 if z>0 else 0.35
    return {"name":"zscore","p_tai":p,"explain":f"z={z:.2f}"}

def analyser_ml_like(history,window=40):
    if len(history)<window: return {"name":"ml_like","p_tai":0.5,"explain":"Insufficient"}
    features=[]; labels=[]
    for i in range(len(history)-window):
        seq=history[i:i+window]; label=history[i+window]["result"].startswith("T")
        mean_total=np.mean([r["total"] for r in seq])
        streak_len=0; last=seq[-1]["result"]
        for r in reversed(seq):
            if r["result"]==last: streak_len+=1
            else: break
        features.append((mean_total,streak_len)); labels.append(label)
    weights=np.mean(labels)
    mean_total,streak_len=np.mean([r["total"] for r in history[-window:]]),0
    last=history[-1]["result"]
    for r in reversed(history[-window:]):
        if r["result"]==last: streak_len+=1
        else: break
    score=0.5+0.1*(weights-0.5)+0.05*(streak_len/10)
    return {"name":"ml_like","p_tai":max(0,min(1,score)),"explain":f"{score:.3f}"}

# Ensemble
ANALYSERS=[analyser_freq_window,analyser_markov1,analyser_markov2,analyser_beta,
           analyser_streak,analyser_entropy,analyser_fourier,analyser_sliding,
           analyser_face_freq,analyser_gap,analyser_zscore,analyser_ml_like]
WEIGHTS=[1.0]*len(ANALYSERS)

def run_ensemble(history):
    results=[f(history) for f in ANALYSERS]
    norm=[w/sum(WEIGHTS) for w in WEIGHTS]
    p=sum(r["p_tai"]*norm[i] for i,r in enumerate(results))
    du_doan="TAI" if p>=0.5 else "XIU"
    conf=round(abs(p-0.5)*200,2)
    return {"p_final":p,"du_doan":du_doan,"do_tin_cay":conf,"details":results}

# ================= PARSE PAYLOAD =================
def parse_source_payload(payload: dict):
    if isinstance(payload, dict) and "list" in payload and isinstance(payload["list"], list):
        payload = payload["list"][-1]
    session = payload.get("Phien") or payload.get("session") or payload.get("id")
    dices = payload.get("dices") or [payload.get("d1"), payload.get("d2"), payload.get("d3")]
    dices = [int(x) for x in dices if x]
    total = payload.get("Tong") or payload.get("total") or sum(dices)
    result = payload.get("Ket_qua") or payload.get("result") or ("TAI" if total >= 11 else "XIU")
    return int(session), dices, int(total), str(result).upper()

# ================= RESPONSE MODEL =================
class PredictResponse(BaseModel):
    session:int; dice:str; total:int; result:str; next_session:int
    du_doan:str; do_tin_cay:float; ty_le:Dict[str,float]; pattern:str
    matched_patterns:List[str]; details:List[Dict[str,Any]]; id:str

# ================= API =================
@app.on_event("startup")
def startup(): init_db()

@app.get("/predict",response_model=PredictResponse)
def predict():
    try:
        r=requests.get(SOURCE_API,timeout=10); r.raise_for_status()
        payload=r.json()
        session,dices,total,result=parse_source_payload(payload)
    except Exception as e:
        raise HTTPException(502,f"Fetch error {e}")
    ts=datetime.now(timezone.utc).astimezone().isoformat()
    save_round(session,dices,total,result,ts)
    history=get_last_n(HISTORY_LIMIT)
    pattern_info=detect_pattern(history,PATTERN_LENGTH)
    ensemble=run_ensemble(history)
    return {
        "session":session,"dice":"-".join(map(str,dices)),"total":total,"result":result,
        "next_session":session+1,"du_doan":ensemble["du_doan"],"do_tin_cay":ensemble["do_tin_cay"],
        "ty_le":{"Tai":round(ensemble["p_final"]*100,2),"Xiu":round((1-ensemble["p_final"])*100,2)},
        "pattern":pattern_info["current_pattern"],"matched_patterns":pattern_info["matched_patterns"],
        "details":ensemble["details"],"id":TELE_ID
    }

@app.get("/history") 
def api_history(limit:int=200): return get_last_n(limit)

@app.get("/") 
def root(): return {"msg":"TX Predictor API is running","author":TELE_ID}
