# Half_PSD: evidence-backed GraphRAG demo

Inspected `output/half_psd_graph_data.json`: **181 nodes, 228 directed relationships, 52 saved communities**. There are **24 CAUSES**, **22 TRIGGERS**, and **7 PRECEDES** relationships. Page references below are physical PDF pages in `data/152E-GEN-8002(doc)-C.pdf`. Findings were checked against the PDF text, not only the extracted labels.

The strongest claim to demonstrate is: **connect dispersed symptoms, operating modes, components, and verification requirements, and show the evidence for each connection**. Superiority over vector RAG is a hypothesis to measure on these questions, not an established result for this application. A sufficiently broad or iterative vector retriever can answer them too.

## 1. Best opening question: compare fault scope

**Ask:** “Compare the manual's possible explanations when all platform doors fail to open versus when only one or a few fail. Which explanations concern shared infrastructure, and which concern an individual doorway? Does the symptom alone justify concluding that the DCU is faulty?”

**Expected answer:** Platform-wide opening failures include signalling-system failure and incoming power failure for all DCUs (p. 107). Individual-door failures include locking-block/limit-switch/solenoid faults, no power to the DCU, broken tooth belt, DCU or motor faults, and Isolated/Bypass modes (p. 108). The symptom does not uniquely identify a faulty DCU. The manual lists possible causes, not probabilities or a measured diagnosis.

**Graph evidence:** Incoming Power Failure and Signalling System Failure each `CAUSES` All PSD Don't Open. Locking Block Fault, Limit Switch Fault, Solenoid Fault, No Power to DCU, Broken Tooth Belt, DCU Fault, and Motor Fault each `CAUSES` One or More PSDs Don't Open.

**Why a graph may help:** Retrieve the incoming cause branches for both symptoms and compare them systematically. The architecture on pp. 23 and 32 explains shared signalling/PSCC versus per-door DCU control.

**Current gap:** The source lists Isolated and Bypass modes for failure to open, but the corresponding cause edges are missing for that symptom. They are present for failure to close. An answer grounded only in the graph must disclose the gap or retrieve p. 108. This is a good test of completeness rather than fluency.

## 2. Best cross-section explanation: mechanical success versus automatic readiness

**Ask:** “Following DCU reset or replacement, why is a successful manual open/close test not enough to demonstrate readiness for automatic operation? Connect the functional checks, mode settings, and normal command path.”

**Expected answer:** Successful manual movement establishes only part of the required behavior. The post-maintenance checks also require the DCU mode to be AUTO, the Normal/Bypass switch to be NORMAL, covers secured, and stable response to signalling commands. The normal path involves SIG, PSCC, and per-door DCUs; a manual test does not by itself exercise that whole path. The reset and replacement checklists specify monitoring three trains after correction (pp. 188 and 192). Treat this as a source explanation, not a substitute maintenance procedure.

**Graph evidence:** `Perform Open-Close Cycles --TESTS--> Door Normal Speed/Normal Speed Operation`; mode-switch state relationships; `Set DCU Mode Switch to AUTO --REQUIRES--> Steps 11 to 15 Passed`; and the PRECEDES chain ending at Monitor Doorway Operation. Signal relationships connect SIG, PSCC, commands, and the DCU (pp. 23, 32).

**Why a graph may help:** Join evidence from system operation, the controller description, replacement, and reset, rather than return a single maintenance paragraph. It can explain the reason for the checks as well as list them.

**Limits:** The three-train criterion is in the source excerpts, but is not an explicit graph relationship. Sequence and mode prerequisites are not proof of physical causation. Do not infer that a manual test passing guarantees every mechanical component is healthy.

## 3. Best directed path: command and feedback trace

**Ask:** “Trace how the automatic opening command reaches the DCU and how closed-and-locked status returns to signalling. Identify the intermediaries and distinguish hardwired signals from the CAN and SMT interfaces.”

**Expected answer:** SIG issues the automatic opening command; PSCC distributes Enable/Open commands to the doors. The DCU interfaces with PSCC using hardwired Enable/Open and Closed-and-Locked status, alongside the CAN data bus. PSCC sends the system closed-and-locked status to SIG once all doors are closed and locked. The portable SMT connects serially for configuration and diagnostics (pp. 23 and 32).

**Exact graph pattern:**

```text
Signalling System --SENDS--> Open Command <--RECEIVES-- Door Control Unit
Platform Screen Control Cabinet --SENDS--> Open Command
Platform Screen Control Cabinet --SENDS--> Enable Command <--RECEIVES-- Door Control Unit
Door Control Unit --SENDS--> Doors Closed and Locked Status <--RECEIVES-- Platform Screen Control Cabinet
Platform Screen Control Cabinet --SENDS--> Doors Closed and Locked Status <--RECEIVES-- Signalling System
```

These are joins through signal nodes. `RECEIVES` points from receiver to signal; do not reverse its meaning just to draw a left-to-right path. The generic Open Command node merges different command sources and contexts, so p. 23 is needed to establish the actual intermediary sequence.

**Why a graph may help:** Explicit sender/receiver joins expose the complete route and interface distinctions. This is an information-flow explanation, not a chain of CAUSES edges.

## 4. Best conditional-behavior question: slow movement after restart

**Ask:** “A door moves slowly after a DCU power interruption. Does that establish a motor fault? Compare first-time/NEW SETUP, local-mode recovery, and Auto/ESDC recovery, including when obstacle detection is active.”

**Expected answer:** Slow opening and closing can be calibration behavior. First startup/NEW SETUP measures door width, with obstacle detection inactive. Local-mode recovery after power restoration also has inactive obstacle detection during calibration. The manual explicitly says obstacle detection is active during Auto/ESDC recovery. Completion of the relevant calibration leads to normal operation (pp. 103–104). Slow motion alone is insufficient to diagnose a motor fault.

**Graph evidence:** Power Interrupt `TRIGGERS` DCU Initialization; DCU control of slow movement; Closed and Locked Position Detected `TRIGGERS` Normal Operation State. Commands have both slow and normal speed targets, depending on context.

**Why a graph may help:** Combine state, event, and observation evidence to challenge a premature diagnosis.

**Limits:** The conditional distinction is better preserved in p. 103 than in the current graph. The graph has no complete, explicit directed causal path joining all of these facts. Because the source passage is compact, a strong vector-RAG baseline may do equally well or better. Use this as a condition-preservation test, not a guaranteed GraphRAG win.

## 5. Obstruction sequence: visually useful, weaker retrieval comparison

**Ask:** “Explain the documented obstruction-recycling behavior, the evidence used to detect obstruction, when PSCC receives obstructed status, and when the report clears. Preserve the operating conditions.”

**Expected answer:** The source describes detection through speed reduction and increased motor current. For a closing obstruction within 500 mm of the fully closed position, the recycling behavior involves a stop and delayed reduced-speed reclose. Continued obstruction after the second attempt leads to the stated constant-force behavior and obstructed status to PSCC. Clearing the obstruction permits closing to resume; reaching closed and locked clears the report (p. 22). Do not generalize this sequence to every obstruction in every mode.

**Graph evidence:** Motor Current / Door Speed `INDICATES` Obstruction Detected; Obstruction Detected `TRIGGERS` Door Stop; Door Stop `PRECEDES` Door Reclose; DCU `SENDS` Door Obstructed Status; PSCC `RECEIVES` that status; Obstruction Cleared `TRIGGERS` Doors Resume Closing.

**Why show it:** It makes monitoring, control response, reporting, and state transitions visible. Most evidence is on one page, so it is not persuasive evidence that GraphRAG retrieves better than vanilla RAG.

## Causal paths actually present

Searching the exported directed edges yields only **one path of two or more edges using CAUSES alone**:

```text
Set Doorway in Bypass Mode --CAUSES--> Bypass Mode
Bypass Mode --CAUSES--> One or More PSD Don't Close
```

Interpretation: selecting bypass is a documented possible explanation for lack of normal automatic closure; it does not imply physical damage or inability to close manually. The action-to-mode relationship is supported in the replacement procedure; the mode-to-symptom relationship is in p. 111. Do not describe this as a learned causal effect or a probability estimate.

Allowing both CAUSES and TRIGGERS gives **16 multi-edge directed paths**, including prefixes of longer paths. Examples:

```text
Doorway Fault --TRIGGERS--> Local Control Mode --CAUSES--> Door Isolation
Close Command --TRIGGERS--> Slow Close --TRIGGERS--> Door Closed and Locked
Check Door Position --TRIGGERS--> Manually Close Doorway --CAUSES--> Door Closed and Locked
```

These require conditions from the relationship descriptions and source: local intervention may be needed; slow close is initialization behavior; manual closure follows a failed position check. A graph path is not automatically a valid causal argument. PRECEDES expresses procedural order, INDICATES expresses diagnostic evidence, and SENDS/RECEIVES expresses information flow.

## Fix before relying on live answers

1. **The saved summaries contain reverse relationships absent from the directed JSON.** Community 44 says failure to close causes loss of DCU power. Community 6 says resuming closing triggers clearance of the obstruction. Community 46 says fault rectification precedes informing the department as well as the reverse. Twenty-five summaries contain “bidirectional,” “reciprocal,” or “reverse causal.” This is an audit flag count, not a claim that every symmetric relationship is wrong.
2. **There is a concrete code explanation:** `GraphRAGStore.to_networkx()` uses an undirected `nx.Graph`. `_collect_community_info()` walks each node's neighbors and emits a directional statement from whichever node it is visiting. That creates reverse statements. The simple graph also collapses different predicates between the same entity pair. Keep the undirected projection for clustering if needed, but build summary relationships from the original directed property relations. Preserve cross-community connections needed by queries. Regenerate summaries after fixing; changing source code alone does not repair saved summaries.
3. **The current query engine answers from summaries, not explicit path retrieval.** `GraphRAGQueryEngine.custom_query()` asks every saved community summary and aggregates the answers. With 52 summaries, that is normally 52 community calls plus an aggregation call. It does not currently fetch source excerpts or verify paths at query time. Consequently the live QA can repeat the summary defects, and its latency/cost should be measured.
4. **Some extracted edges need correction:** Door Isolation `PREVENTS` Platform Operation reverses the intended effect: isolation prevents disruption of platform operation. Door Closed and Locked `TRIGGERS` Obstruction Report means *clears* the report in its description, not creates it. Locking Block/Limit Switch/Solenoid Fault cause edges to failure-to-close appear to mix p. 108 opening faults into p. 111 closing faults. Review before presenting these as source-backed.
5. **Entity aliases fragment paths:** ESDC / Emergency Screen Door Control; AUTO Mode / Auto Mode; MCB / Miniature Circuit Breaker; several Closed and Locked variants. Canonicalization would make traversal more complete. Keep device-level and platform-level status distinctions where they matter.
6. **Provenance is chunk-level.** An edge's provenance list includes the pages of its source chunk, not a claim that each page supports that edge. Open the cited excerpt and identify the actual supporting sentence.

No graph, summary, model, or application code was changed for this review, and no live QA benchmark was run.

## Five-minute presentation and fair comparison

1. State the distinction: “The aim is to connect evidence across the manual and make the connections inspectable.”
2. Run Question 1 in both systems. Show the two symptom nodes and incoming cause branches. Ask whether either system overdiagnosed the DCU or missed an operating mode.
3. Run Question 2. Show evidence from pp. 23/32 and 188/192. Look for an explanation of automatic readiness rather than just a list of replacement steps.
4. Use Question 3 to visualize commands and returning status. Follow the relationship labels and then open source text.
5. Use Question 4 as a challenge: a credible answer preserves the local/Auto obstacle-detection distinction. If it does not, show the limitation.

Use the same source sections, chunking, generation model, answer prompt, and comparable evidence-token budgets. Report retrieval settings, end-to-end latency, and total token/model-call cost; your all-community method has substantially more intermediate calls than simple top-k retrieval. Include a reranked or expanded-context vector baseline if that is the intended alternative. Do not deliberately restrict vanilla RAG until it fails.

Score each answer on (a) required source-backed facts found, (b) correct relationship direction, (c) preserved operating conditions, (d) source support, and (e) unsupported claims. Use a human-reviewed answer checklist from the expected answers above. Include a control question such as “What is the maximum door speed?” where ordinary RAG should perform well. If both methods answer equally well, the supported demo claim is improved traceability or organization, not greater answer accuracy.

Research supports GraphRAG for global sensemaking, but does not establish a win for this small maintenance subset: [original GraphRAG paper](https://arxiv.org/abs/2404.16130), [global search documentation](https://microsoft.github.io/graphrag/query/global_search/).
