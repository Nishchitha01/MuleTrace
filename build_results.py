S = [
 ("Transaction level (test split: 1,015,669 transactions, 1,798 labelled laundering)",
  ["System", "Flagged", "Precision", "Recall", "F1"],
  [["Tuned threshold baseline", "399,520", "0.0037", "0.8270", "0.0074"],
   ["Plain $10k rule", "165,520", "0.0045", "0.4182", "0.0090"],
   ["Baseline, top N by amount (same volume)", "109,895", "0.0009", "0.0539", "0.0017"],
   ["MuleTrace, all alerts", "109,895", "0.0022", "0.1363", "0.0044"],
   ["MuleTrace, top 50 alerts", "432", "0.0972", "0.0234", "0.0377"]]),
 ("Account level",
  ["System", "Flagged accounts", "Precision", "Recall", "F1"],
  [["Tuned threshold baseline", "167,811", "0.0120", "0.9128", "0.0236"],
   ["MuleTrace", "43,215", "0.0117", "0.2299", "0.0223"]]),
 ("Detectors (base rate about 0.0018)",
  ["Detector", "Transactions covered", "Precision"],
  [["Fan-out", "95,661", "0.0013"], ["Fan-in", "11,914", "0.0031"],
   ["Chains", "3,939", "0.0099"], ["Cycles", "116", "0.3621"]]),
 ("Ring recall (one alert holds at least 70% of a ring's accounts)",
  ["Ring type", "Found", "Total"],
  [["Bipartite", "9", "17"], ["Cycle", "2", "10"], ["Fan-in", "2", "12"],
   ["Fan-out", "2", "10"], ["Gather-scatter", "2", "16"], ["Random", "3", "12"],
   ["Scatter-gather", "4", "11"], ["Stack", "4", "7"], ["All", "28", "95"]]),
]
LIM = [
 "Synthetic benchmark (IBM AML HI-Small). About 0.18% of test transactions are labelled laundering.",
 "Whole-list F1 is below the threshold baseline: it flags 399,520 transactions for 83% recall; MuleTrace flags 109,895 for 14%.",
 "Strength: precision of ranked alerts. The top 50 alerts hold 42 labelled-laundering transactions in 432 (precision 0.097), driven mostly by the cycle detector.",
 "Ring recall is low for rings with more than 2 accounts; most rings found are 2-account rings.",
 "The fan-out detector is at or below base-rate precision.",
 "The alert ranking order was chosen on the validation split.",
]
h = ["<html><head><meta charset='utf-8'><title>MuleTrace results</title><style>",
     "body{font-family:Segoe UI,Arial;margin:30px;max-width:900px}",
     "table{border-collapse:collapse;margin-bottom:24px;width:100%}",
     "th,td{border:1px solid #ccc;padding:6px 10px;text-align:left}",
     "th{background:#14213d;color:#fff}</style></head><body>",
     "<h1>MuleTrace: test-split results</h1>"]
for title, head, rows in S:
    h.append("<h3>" + title + "</h3><table><tr>" + "".join("<th>" + c + "</th>" for c in head) + "</tr>")
    for r in rows:
        h.append("<tr>" + "".join("<td>" + c + "</td>" for c in r) + "</tr>")
    h.append("</table>")
h.append("<h3>Limitations</h3><ul>" + "".join("<li>" + l + "</li>" for l in LIM) + "</ul></body></html>")
open("results.html", "w", encoding="utf-8").write("\n".join(h))
print("wrote results.html")