import json
import pandas as pd
from collections import Counter

print("loading alerts (can take a minute)...", flush=True)
alerts = json.load(open("alerts_test.json"))
print("loading labels...", flush=True)
df = pd.read_pickle("data/cache.pkl")
illicit = set(df.loc[df["label"] == 1, "eid"].tolist())
del df

for a in alerts:
    a["gt"] = sum(1 for x in a["edges"] if x["eid"] in illicit)
alerts.sort(key=lambda a: -a["risk_score"])

top = alerts[:150]
conf = [a for a in alerts if a["gt"] > 0][:150]
force = [a for a in alerts if a["alert_id"] == "T-2758"]
by_pat = []
for p in ["chain", "fan_in", "fan_out", "cycle"]:
    by_pat += [a for a in alerts if a["patterns"] == [p]][:60]
sel, seen = [], set()
for a in top + conf + force + by_pat:
    if a["alert_id"] not in seen:
        seen.add(a["alert_id"]); sel.append(a)
sel.sort(key=lambda a: -a["risk_score"])

def summary(a):
    e, ev, pats = a["edges"], a["evidence"], a["patterns"]
    hub_out = Counter(x["from"] for x in e).most_common(1)[0][0]
    hub_in = Counter(x["to"] for x in e).most_common(1)[0][0]
    n_recv = len({x["to"] for x in e if x["from"] == hub_out})
    n_send = len({x["from"] for x in e if x["to"] == hub_in})
    times = sorted(x["time"] for x in e)
    big = max(e, key=lambda x: x["amount_usd"])
    s = [f"{len(a['accounts'])} accounts exchanged {len(e)} transactions worth "
         f"${ev['total_amount_usd']:,.0f} between {times[0].replace('T', ' ')} and "
         f"{times[-1].replace('T', ' ')}. Typology label: {a['typology'].lower()}."]
    if "cycle" in pats:
        s.append("Part of this cluster is a time-ordered loop: money left an account and "
                 "returned to it, the signature of round-tripping.")
    if "chain" in pats:
        s.append("Funds were passed from account to account within hours with most of the "
                 "amount kept, which is typical of mule accounts.")
    if "fan_out" in pats:
        s.append(f"Account {hub_out} paid {n_recv} different accounts within a short window.")
    if "fan_in" in pats:
        s.append(f"{n_send} different accounts paid account {hub_in} within a short window.")
    s.append(f"Largest single transfer: ${big['amount_usd']:,.0f} from {big['from']} to "
             f"{big['to']} at {big['time'].replace('T', ' ')}. Risk score {a['risk_score']}.")
    return " ".join(s)

rows = []
for a in sel:
    p = a["patterns"]
    ed = sorted(a["edges"], key=lambda x: x["time"])[:60]
    rows.append({
        "id": a["alert_id"], "score": a["risk_score"],
        "typ": a["typology"] if len(p) == 1 else "Mixed: " + ", ".join(p),
        "cat": p[0] if len(p) == 1 else "mixed",
        "n_acc": len(a["accounts"]), "n_tx": len(a["edges"]),
        "total": a["evidence"]["total_amount_usd"], "hours": a["evidence"]["duration_hours"],
        "summary": summary(a), "gt": a["gt"],
        "edges": [{"f": x["from"], "t": x["to"], "time": x["time"], "amt": x["amount_usd"],
                   "l": 1 if x["eid"] in illicit else 0} for x in ed],
    })
META = {"shown": len(rows), "total": len(alerts)}
print("alerts in dashboard:", len(rows), "of", len(alerts))
print("confirmed-laundering alerts included:", sum(1 for r in rows if r["gt"] > 0))
print("T-2758 included:", any(r["id"] == "T-2758" for r in rows))

METRICS = {
  "tx": [["Tuned threshold baseline", 399520, 0.0037, 0.8270, 0.0074],
         ["Plain $10k rule", 165520, 0.0045, 0.4182, 0.0090],
         ["Baseline, top N by amount (same volume)", 109895, 0.0009, 0.0539, 0.0017],
         ["MuleTrace, all alerts", 109895, 0.0022, 0.1363, 0.0044],
         ["MuleTrace, top 50 alerts", 432, 0.0972, 0.0234, 0.0377]],
  "acct": [["Tuned threshold baseline", 167811, 0.0120, 0.9128, 0.0236],
           ["MuleTrace", 43215, 0.0117, 0.2299, 0.0223]],
  "det": [["Fan-out", 95661, 0.0013], ["Fan-in", 11914, 0.0031],
          ["Chains", 3939, 0.0099], ["Cycles", 116, 0.3621]],
  "rings": [["Bipartite", 9, 17], ["Cycle", 2, 10], ["Fan-in", 2, 12], ["Fan-out", 2, 10],
            ["Gather-scatter", 2, 16], ["Random", 3, 12], ["Scatter-gather", 4, 11],
            ["Stack", 4, 7], ["All", 28, 95]],
  "limits": [
    "Synthetic benchmark (IBM AML HI-Small). About 0.18% of test transactions are labelled laundering.",
    "Whole-list F1 is below the threshold baseline: it flags 399,520 transactions for 83% recall; MuleTrace flags 109,895 for 14%.",
    "Strength: precision of ranked alerts. The top 50 alerts hold 42 labelled-laundering transactions in 432 (precision 0.097), driven mostly by the cycle detector.",
    "Ring recall is low for rings with more than 2 accounts; most rings found are 2-account rings.",
    "The fan-out detector is at or below base-rate precision.",
    "Some high-scored alerts are false positives (see the Labelled column).",
    "Typology comes from the strongest pattern in a merged cluster, so mixed clusters can carry a misleading label.",
    "Summaries are template-based narratives built from detected facts, not an LLM.",
    "The alert ranking order was chosen on the validation split."]
}

HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>MuleTrace</title>
<style>
:root{--ink:#10203a;--mut:#6b7a90;--line:#e3e9f2;--bg:#f3f6fb;--teal:#0f8b6d;--amber:#e08a1e;--red:#c0392b}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:Segoe UI,Arial,sans-serif;font-size:14px}
.page{max-width:1500px;margin:0 auto;padding:22px 30px}
header{display:flex;justify-content:space-between;align-items:center;padding-bottom:16px;border-bottom:1px solid var(--line);margin-bottom:18px}
.logo{display:flex;gap:14px;align-items:center}
.logo i{width:50px;height:50px;border-radius:12px;background:#0f1d36;color:#fff;font-style:normal;font-weight:700;font-size:24px;display:flex;align-items:center;justify-content:center}
.logo h1{margin:0;font-size:22px}.logo small{color:var(--mut)}
nav button{background:#fff;border:1px solid var(--line);border-radius:10px;padding:10px 20px;margin-left:8px;cursor:pointer;font-size:14px;font-weight:600;color:var(--ink)}
nav button.on{background:#0f1d36;color:#fff;border-color:#0f1d36}
.banner{background:#fffaea;border:1px solid #f1e3b0;border-radius:12px;padding:14px 18px;margin-bottom:18px;display:flex;justify-content:space-between;align-items:center;gap:12px}
.badge{background:#fdecc0;color:#8a5a00;font-weight:700;font-size:11px;letter-spacing:1px;padding:5px 10px;border-radius:6px;white-space:nowrap}
.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin-bottom:18px}
.kpi,.card{background:#fff;border:1px solid var(--line);border-radius:14px;padding:18px 20px;box-shadow:0 1px 3px rgba(16,32,58,.05)}
.kpi small{color:var(--mut)}.kpi b{display:block;font-size:34px;margin:6px 0 2px}
.main{display:grid;grid-template-columns:400px 1fr;gap:18px;align-items:start}
.eyebrow{font-size:11px;letter-spacing:1.5px;color:var(--mut);font-weight:700;text-transform:uppercase}
.queue h2{margin:4px 0 12px;font-size:26px}
.cat{display:flex;justify-content:space-between;padding:11px 12px;border-radius:10px;cursor:pointer;color:#3a4a63}
.cat:hover{background:#f3f6fb}.cat.on{background:#e8f0fd;color:#143a7a;font-weight:600}
.list{max-height:520px;overflow:auto;border-top:1px solid var(--line);margin-top:10px;padding-top:12px}
.item{border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin-bottom:10px;cursor:pointer}
.item:hover{border-color:#b9c9e6}.item.sel{background:#eef4ff;border-color:#8fb0ea}
.item .top{display:flex;justify-content:space-between;align-items:center}
.item .sc{color:var(--teal);font-weight:700;font-size:17px}.item small{color:var(--mut);font-weight:400}
.item .nm{font-weight:600;margin:7px 0 3px}.item .sub{color:var(--mut);font-size:12.5px}
.red{color:var(--red);font-weight:600}
.pill{padding:3px 10px;border-radius:6px;font-size:12px;font-weight:600}
.pill.cycle{background:#efe4fb;color:#5b2d91}.pill.chain{background:#fdebd2;color:#8a5200}
.pill.fan_in,.pill.fan_out{background:#dff3ea;color:#0b6b52}.pill.mixed{background:#e8edf4;color:#44546b}
.dhead{display:flex;justify-content:space-between;align-items:flex-start}
.dhead h2{margin:6px 0 0;font-size:30px}
.score{text-align:right}.score b{font-size:36px;color:var(--teal)}.score small{color:var(--mut)}
.trail{background:#f6f9fd;border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:16px 0;line-height:1.55;font-size:15px}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;padding-bottom:16px;border-bottom:1px solid var(--line)}
.stats b{display:block;font-size:22px;margin-top:5px}
.two{display:grid;grid-template-columns:1.7fr 1fr;gap:18px;margin-top:18px}
.row{display:flex;justify-content:space-between;align-items:center}
svg{width:100%;background:#fcfdff;border:1px solid var(--line);border-radius:10px;margin-top:8px}
button.sm{border:1px solid #c9d6ee;background:#eef4ff;color:#143a7a;border-radius:8px;padding:7px 12px;cursor:pointer;font-weight:600;font-size:13px;margin-right:6px}
select{border:1px solid var(--line);border-radius:8px;padding:6px}
.pbar{height:7px;background:#e6ebf3;border-radius:5px;margin:14px 0 8px;overflow:hidden}.pbar div{height:100%;background:var(--amber);width:100%}
.tl{max-height:300px;overflow:auto}
.ev{border-left:3px solid #d3dcea;padding:8px 12px;margin:6px 0;border-radius:0 8px 8px 0;cursor:pointer}
.ev.on{background:#fff6e8;border-left-color:var(--amber)}.ev.dim{opacity:.35}
.ev .a{font-weight:600}.ev .b{display:flex;justify-content:space-between;color:var(--mut);font-size:12.5px;margin-top:3px}
table{border-collapse:collapse;width:100%}
th{font-size:11px;letter-spacing:1px;text-transform:uppercase;color:var(--mut);text-align:left;padding:10px;border-bottom:1px solid var(--line)}
td{padding:11px 10px;border-bottom:1px solid #eef1f6}td.r,th.r{text-align:right}
tr.hl{background:#fff6e8}.evt tr{cursor:pointer}.evt tr:hover{background:#f6f9fd}
.tag{background:#eef1f6;color:#66758b;border-radius:6px;padding:3px 9px;font-size:12px}.tag.red{background:#fde8e5;color:var(--red)}
.foot{background:#fffaea;border:1px solid #f1e3b0;border-radius:12px;padding:14px 18px;margin-top:18px;color:#6b5a1e}
#results{display:none}.note{color:var(--mut);font-size:12.5px}
</style></head><body><div class="page">
<header><div class="logo"><i>M</i><div><h1>MuleTrace</h1><small>Financial intelligence workspace</small></div></div>
<nav><button id="b1" class="on" onclick="show('alerts')">Alerts</button><button id="b2" onclick="show('results')">Results</button></nav></header>
<div class="banner"><div><b>IBM AML HI-Small benchmark</b> (synthetic data) &middot; test split: 1,015,669 transactions &middot; <span id="samp"></span></div><span class="badge">TEST SPLIT</span></div>
<div id="alerts">
<div class="kpis">
 <div class="kpi"><small>Flags from the tuned threshold baseline</small><b>399,520</b><small>far too many to review</small></div>
 <div class="kpi"><small>Transactions in MuleTrace&rsquo;s top 50 alerts</small><b>432</b><small>reviewable by an analyst</small></div>
 <div class="kpi"><small>Of those 432, labelled laundering</small><b>42</b><small>precision 0.097, about 55x the base rate</small></div>
</div>
<div class="main">
 <div class="card queue"><div class="eyebrow">Investigation queue</div><h2>Alerts <small class="note" id="qcount"></small></h2>
  <div id="cats"></div><div class="list" id="list"></div>
  <div class="note" style="margin-top:10px">Scores rank review priority; they are not probabilities.</div></div>
 <div>
  <div class="card">
   <div class="dhead"><div><div class="eyebrow" id="crumb"></div><h2 id="title"></h2></div>
    <div class="score"><div class="eyebrow">Risk score</div><b id="score"></b><div class="note">Heuristic &middot; not a probability</div></div></div>
   <div class="trail" id="trail"></div>
   <div class="stats"><div><div class="eyebrow">Linked accounts</div><b id="s1"></b></div><div><div class="eyebrow">Supporting txns</div><b id="s2"></b></div><div><div class="eyebrow">Time span</div><b id="s3"></b></div><div><div class="eyebrow">Ground truth (evaluation only)</div><b id="s4"></b></div></div>
   <div class="two">
    <div><div class="row"><h3 style="margin:14px 0 0">Transaction graph</h3><span class="note">click an arrow row below to trace it</span></div><svg id="g" viewBox="0 0 640 420"></svg>
     <div class="note" style="margin-top:6px">Arrows show direction and order. Orange = selected transfer. Thicker = larger amount.</div></div>
    <div><div class="row"><h3 style="margin:14px 0 0">Transaction replay</h3><span class="note" id="evc"></span></div>
     <div style="margin-top:10px"><button class="sm" onclick="replay()">&#9654; Replay</button><button class="sm" onclick="showAll()">Show all</button>
      <select id="spd" onchange="setSpeed()"><option value="1">1x</option><option value="2">2x</option><option value="4">4x</option></select></div>
     <div class="pbar"><div id="bar"></div></div><div class="tl" id="tl"></div></div>
   </div>
   <h3 style="margin:22px 0 4px">Transaction evidence <span class="note" id="evn"></span></h3>
   <table class="evt"><thead><tr><th>Txn</th><th>Time</th><th>Sender &rarr; Receiver</th><th class="r">Amount</th><th>Dataset label</th></tr></thead><tbody id="ev"></tbody></table>
  </div>
 </div>
</div></div>
<div id="results"></div>
<div class="foot">Alerts indicate patterns for human review; they do not establish wrongdoing. The risk score is a hand-weighted heuristic, not a probability.</div>
</div>
<script>
const A = __ALERTS__, M = __METRICS__, META = __META__;
const $ = id => document.getElementById(id);
const fmt = n => Number(n).toLocaleString('en-US', {maximumFractionDigits: 0});
const CATS = [['all','All alerts'],['cycle','Cycle'],['chain','Rapid pass-through (chain)'],['fan_out','Fan-out'],['fan_in','Fan-in'],['mixed','Mixed patterns'],['conf','Confirmed laundering (sample)']];
let F = 'all', cur = 0, upto = 0, sel = -1, timer = null, speed = 1, playing = false;
$('samp').textContent = 'showing a curated sample of ' + META.shown + ' of ' + fmt(META.total) + ' alerts. Use the Results tab for full-test numbers.';
function show(w){
  $('alerts').style.display = w==='alerts' ? 'block' : 'none';
  $('results').style.display = w==='results' ? 'block' : 'none';
  $('b1').className = w==='alerts' ? 'on' : ''; $('b2').className = w==='results' ? 'on' : '';
}
function inCat(a,c){ if (c==='all') return true; if (c==='conf') return a.gt > 0; return a.cat === c; }
function setF(k){ F = k; const i = A.findIndex(a => inCat(a,k)); if (i >= 0) { pick(i); } else { renderQueue(); } }
function renderQueue(){
  $('cats').innerHTML = CATS.map(([k,l]) => `<div class="cat ${F===k?'on':''}" onclick="setF('${k}')"><span>${l}</span><b>${A.filter(a => inCat(a,k)).length}</b></div>`).join('');
  const list = A.map((a,i) => [a,i]).filter(([a]) => inCat(a,F));
  $('qcount').textContent = list.length + ' shown';
  $('list').innerHTML = list.map(([a,i]) => `<div class="item ${i===cur?'sel':''}" onclick="pick(${i})"><div class="top"><span class="pill ${a.cat}">${a.cat.replace('_','-')}</span><span class="sc">${a.score}<small> / 100</small></span></div><div class="nm">${a.id} &middot; ${a.typ}</div><div class="sub">${a.n_tx} transactions &middot; ${a.n_acc} accounts${a.gt>0 ? ' &middot; <span class="red">' + a.gt + ' labelled laundering</span>' : ''}</div></div>`).join('');
}
function stop(){ clearInterval(timer); playing = false; }
function pick(i){
  stop(); cur = i; sel = -1; upto = A[i].edges.length - 1;
  renderQueue(); renderDetail();
}
function renderDetail(){
  const a = A[cur];
  $('crumb').textContent = 'ALERT REVIEW / ' + a.id;
  $('title').textContent = a.typ;
  $('score').innerHTML = a.score + '<small> / 100</small>';
  $('trail').textContent = a.summary;
  $('s1').textContent = a.n_acc; $('s2').textContent = a.n_tx; $('s3').textContent = a.hours + ' h';
  $('s4').innerHTML = a.gt > 0 ? '<span class="red">' + a.gt + ' of ' + a.n_tx + '</span> labelled' : 'none labelled (false positive)';
  drawAll(); evidence();
}
function drawAll(){ drawGraph(A[cur]); drawTimeline(A[cur]); }
function evidence(){
  const a = A[cur];
  $('ev').innerHTML = a.edges.slice(0,40).map((e,k) => `<tr class="${k===sel?'hl':''}" onclick="selEdge(${k})"><td>T${String(k+1).padStart(3,'0')}</td><td>${e.time.replace('T',' ')}</td><td>${e.f.slice(-9)} &rarr; ${e.t.slice(-9)}</td><td class="r">$${fmt(e.amt)}</td><td>${e.l ? '<span class="tag red">laundering</span>' : '<span class="tag">no label</span>'}</td></tr>`).join('');
  $('evn').textContent = Math.min(40, a.edges.length) + ' of ' + a.n_tx;
}
function selEdge(k){ stop(); upto = A[cur].edges.length - 1; sel = k; drawAll(); evidence(); }
function showAll(){ stop(); sel = -1; upto = A[cur].edges.length - 1; drawAll(); evidence(); }
function ms(){ return (A[cur].edges.length > 12 ? 150 : 700) / speed; }
function step(){
  const n = A[cur].edges.length; upto++;
  if (upto >= n - 1){ upto = n - 1; stop(); }
  drawAll();
}
function replay(){ stop(); sel = -1; upto = -1; playing = true; evidence(); step(); if (playing) timer = setInterval(step, ms()); }
function setSpeed(){ speed = +$('spd').value; if (playing){ clearInterval(timer); timer = setInterval(step, ms()); } }
function drawGraph(a){
  const E = a.edges, nodes = [...new Set(E.flatMap(e => [e.f, e.t]))];
  const W = 640, H = 420, cx = W/2, cy = H/2, R = nodes.length <= 8 ? 135 : 175, pos = {};
  const a0 = nodes.length === 2 ? Math.PI : -Math.PI/2;
  nodes.forEach((n,k) => { const t = 2*Math.PI*k/nodes.length + a0; pos[n] = [cx + R*Math.cos(t), cy + R*Math.sin(t)]; });
  const rn = nodes.length <= 8 ? 16 : (nodes.length <= 14 ? 9 : 5);
  const hiIdx = playing ? upto : sel, hiNodes = new Set();
  if (hiIdx >= 0 && E[hiIdx]) { hiNodes.add(E[hiIdx].f); hiNodes.add(E[hiIdx].t); }
  const pairs = {}, has = new Set(E.map(e => e.f + '|' + e.t));
  let s = '<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0L10,5L0,10z" fill="#4d6a8f"/></marker><marker id="ao" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0L10,5L0,10z" fill="#e08a1e"/></marker></defs>';
  E.forEach((e,k) => {
    if (k > upto || e.f === e.t) return;
    const p = pos[e.f], q = pos[e.t], key = e.f + '|' + e.t, idx = (pairs[key] = (pairs[key] ?? -1) + 1);
    const off = (has.has(e.t + '|' + e.f) ? 26 : 0) + idx * 14;
    const dx = q[0]-p[0], dy = q[1]-p[1], len = Math.hypot(dx,dy) || 1, nx = -dy/len, ny = dx/len;
    const cpx = (p[0]+q[0])/2 + nx*off*2, cpy = (p[1]+q[1])/2 + ny*off*2;
    const ex = q[0]-cpx, ey = q[1]-cpy, el = Math.hypot(ex,ey) || 1, qx = q[0]-ex/el*(rn+3), qy = q[1]-ey/el*(rn+3);
    const hi = (k === hiIdx);
    const w = hi ? 3.2 : (1.1 + Math.log10(e.amt + 1) / 3).toFixed(2);
    s += `<path d="M${p[0]},${p[1]} Q${cpx},${cpy} ${qx},${qy}" fill="none" stroke="${hi ? '#e08a1e' : '#4d6a8f'}" stroke-width="${w}" opacity="${hi ? 1 : 0.75}" marker-end="url(#${hi ? 'ao' : 'ar'})"/>`;
    if (E.length <= 8) {
      const mx = .25*p[0] + .5*cpx + .25*q[0], my = .25*p[1] + .5*cpy + .25*q[1];
      s += `<text x="${mx}" y="${my}" font-size="12" font-weight="700" fill="#10203a" text-anchor="middle" stroke="#fcfdff" stroke-width="3" paint-order="stroke">${k+1} &middot; $${fmt(e.amt)}</text>`;
    }
  });
  nodes.forEach(n => {
    const h = hiNodes.has(n);
    s += `<circle cx="${pos[n][0]}" cy="${pos[n][1]}" r="${rn}" fill="${h ? '#fff3e0' : '#eaf2fb'}" stroke="${h ? '#e08a1e' : '#5b7ba6'}" stroke-width="2"/>`;
    if (nodes.length <= 14) s += `<text x="${pos[n][0]}" y="${pos[n][1] + (pos[n][1] < cy ? -rn-6 : rn+16)}" font-size="12" font-weight="600" fill="#10203a" text-anchor="middle">${n.slice(-9)}</text>`;
  });
  $('g').innerHTML = s;
}
function drawTimeline(a){
  const E = a.edges.slice(0,12), hi = playing ? upto : sel;
  $('tl').innerHTML = E.map((e,k) => `<div class="ev ${k===hi?'on':''}${(playing && k>upto)?' dim':''}" onclick="selEdge(${k})"><div class="a">${e.f.slice(-9)} &rarr; ${e.t.slice(-9)}</div><div class="b"><span>${e.time.replace('T',' ')}</span><b>$${fmt(e.amt)}</b></div></div>`).join('');
  $('evc').textContent = E.length + ' of ' + a.n_tx + ' events';
  $('bar').style.width = (100 * (upto + 1) / a.edges.length) + '%';
}
function table(head, rows, f){
  return '<table><thead><tr>' + head.map(h => '<th>' + h + '</th>').join('') + '</tr></thead><tbody>' +
         rows.map(r => '<tr>' + f(r).map(c => '<td>' + c + '</td>').join('') + '</tr>').join('') + '</tbody></table>';
}
$('results').innerHTML =
  '<div class="card" style="margin-bottom:18px"><h3>Transaction level (test split: 1,015,669 transactions, 1,798 labelled laundering)</h3>' +
  table(['System','Flagged','Precision','Recall','F1'], M.tx, r => [r[0], fmt(r[1]), r[2], r[3], r[4]]) + '</div>' +
  '<div class="card" style="margin-bottom:18px"><h3>Account level</h3>' +
  table(['System','Flagged accounts','Precision','Recall','F1'], M.acct, r => [r[0], fmt(r[1]), r[2], r[3], r[4]]) + '</div>' +
  '<div class="card" style="margin-bottom:18px"><h3>Detectors (base rate about 0.0018)</h3>' +
  table(['Detector','Transactions covered','Precision'], M.det, r => [r[0], fmt(r[1]), r[2]]) + '</div>' +
  '<div class="card" style="margin-bottom:18px"><h3>Ring recall (one alert holds at least 70% of a ring&rsquo;s accounts)</h3>' +
  table(['Ring type','Found','Total'], M.rings, r => r) + '</div>' +
  '<div class="card"><h3>Limitations</h3><ul>' + M.limits.map(l => '<li style="margin:6px 0">' + l + '</li>').join('') + '</ul></div>';
renderQueue();
const first = A.findIndex(a => a.id === 'T-3449'); pick(first >= 0 ? first : 0);
</script></body></html>"""

out = (HTML.replace("__ALERTS__", json.dumps(rows).replace("</", "<\\/"))
           .replace("__METRICS__", json.dumps(METRICS))
           .replace("__META__", json.dumps(META)))
open("dashboard.html", "w", encoding="utf-8").write(out)
print("wrote dashboard.html")