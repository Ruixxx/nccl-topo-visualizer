# NCCL Topology Visualizer

Parse NCCL-TEST logs and generate physical + logical topology diagrams with Graphviz.

## What It Does

Given an NCCL-TEST log (single-node or multi-node), this tool produces three kinds of diagrams:

| Diagram | Content |
|---|---|
| **Physical topology** | Per-node hardware layout: CPU sockets, PCIe switches, GPUs (with rank numbers), NVSwitch, NIC. Shows NVL/PCI/SYS link bandwidth. |
| **Ring topology** | Every ring channel reconstructed into a full cycle, with each rank labeled as `hostname:GPUx`. Edges colored green (P2P, intra-node) or red (RDMA, inter-node). |
| **Tree topology** | Each distinct double-tree (Tree0 = uplink, Tree1 = cross-node) drawn separately. Root nodes use `box3d` shape. Same P2P/RDMA coloring. |

## Prerequisites

- Python 3.8+
- Graphviz (`dot` binary) — install with `apt install graphviz` or `brew install graphviz`

## Generating NCCL-TEST Logs

This tool relies on the topology dump that NCCL prints at `INFO` level during communicator initialization. When running NCCL-TEST, set these environment variables to capture the required output:

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH ./build/all_reduce_perf -b 8 -e 8 -g 8
```

- `NCCL_DEBUG=INFO` — enables INFO-level logging, which includes the `=== System ===` topology tree, `Ring`, `Trees`, and `Channel ... via` lines.
- `NCCL_DEBUG_SUBSYS=INIT,GRAPH` — restricts output to the INIT and GRAPH subsystems, keeping the log concise while still containing everything this tool needs.

Redirect stdout/stderr to a file, then pass it to the visualizer:

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH mpirun -np 8 ./all_reduce_perf -b 8 -e 8 -g 8 2>&1 | tee nccl_log.txt
python3 nccl_topo_viz.py nccl_log.txt
```

## Usage

```bash
python3 nccl_topo_viz.py <log_file> [options]
```

### Options

```
log_file                         NCCL test log file
--output-dir, -o <dir>           Output directory (default: same as log file)
--format, -f {png,svg,pdf}       Output format (default: png)
--summary, -s                    Print text summary to stdout
```

### Examples

```bash
# Basic: generate PNGs next to the log file
python3 nccl_topo_viz.py example/H800_allreduce.txt

# SVG format into a specific directory
python3 nccl_topo_viz.py example/2xNode-H800_allreduce.log -f svg -o output/

# Print parsed summary (ranks, rings, trees, connections) without rendering
python3 nccl_topo_viz.py example/H800_allreduce.txt -s
```

## Output Files

For a log file named `foo.log`, the tool generates:

```
foo_physical_<hostname>.dot   + .png/.svg/.pdf   (one per node)
foo_rings.dot                  + .png/.svg/.pdf
foo_trees.dot                  + .png/.svg/.pdf
```

## Log Format Reference

The parser extracts data from these NCCL log sections:

### Rank Mapping

```
#  Rank  0 Group  0 Pid 199434 on  cu74 device  0 [0000:0f:00] NVIDIA H800
```

Parses rank → hostname, GPU device index, PCI BDF, GPU model.

### Physical Topology

```
=== System : maxBw 164.8 totalBw 164.8 ===
CPU/0-0 (1/1/3)
+ PCI[48.0] - PCI/0-c000 (1000c0301000100b)
              + PCI[48.0] - NIC/0-e000
              + PCI[48.0] - DEV/0-f000 (10de232410de17a6)
                            + LOC[5000.0] - GPU/0-f000
                            + NVL[164.8] - NVS/0-0
+ SYS[22.0] - CPU/0-1
```

The indentation-based tree is parsed to reconstruct:
- CPU sockets connected via `SYS` (UPI)
- PCIe switches under each CPU (`PCI[48.0]`)
- GPUs and NIC hanging off PCIe switches
- NVSwitch connected to all GPUs via `NVL[164.8]`

### Ring Topology

Each rank logs its 3-node segment:
```
Ring 00 : 1 -> 2 -> 3
```
The tool follows `next` pointers to reconstruct the full ring cycle.

### Tree Topology

Double-tree algorithm entries per channel:
```
Trees [0] 1/8/-1->0->-1 [1] 1/8/-1->0->-1 [2] 1/-1/-1->0->8 [3] 1/-1/-1->0->8
```
Format: `[channel] up0/up1/up2->self->down0/down1/down2`

- **Tree 0**: built from `up[0]` (parent) and `down[0]` (child) — the primary uplink tree
- **Tree 1**: built from `up[1]` (parent) — the secondary/cross-node tree

Channels with identical tree structures are grouped together (e.g., `Ch0-15 Tree0` means channels 0–15 share the same tree).

### Connection Type

```
Channel 00/0 : 0[0] -> 1[1] via P2P/CUMEM        # intra-node P2P
Channel 00/0 : 8[0] -> 0[0] [receive] via NET/IB/0/GDRDMA   # inter-node RDMA
```

`P2P/CUMEM` → green edge (intra-node NVLink/PCIe P2P)
`NET/IB/.../GDRMA` → red edge (inter-node GPUDirect RDMA)

## Diagram Legend

### Physical Topology

| Shape | Color | Element |
|---|---|---|
| box3d | Sky blue | CPU socket |
| box | Light gray | PCIe Switch |
| box | Light green | GPU (labeled with rank, model, BDF) |
| box | Gold | NIC (mlx5_0/RoCE) |
| diamond | Plum | NVSwitch |
| dashed gray | — | SYS/UPI inter-socket link |

### Ring & Tree Topology

| Color | Meaning |
|---|---|
| Green edge | P2P (intra-node, via NVLink/PCIe) |
| Red edge | RDMA (inter-node, via GPUDirect RDMA) |
| Sky blue node | Rank on first host |
| Pink node | Rank on second host |
| box3d node | Tree root |

## How It Works

```
NCCL Log
  │
  ├─ "Using devices" lines → rank mapping (rank ↔ host:GPU)
  ├─ "=== System ===" blocks → physical topology tree per host
  ├─ "Ring NN : a -> b -> c" lines → ring segments (reconstructed into full cycles)
  ├─ "Trees [N] ..." lines → double-tree parent/child edges per channel
  └─ "Channel NN/0 : X[x] -> Y[y] via ..." → P2P vs RDMA classification
       │
       ▼
  Graphviz DOT → PNG/SVG/PDF
```

## Example Output

Running on the bundled examples:

```bash
# Single node (8x H800, 1 machine)
python3 nccl_topo_viz.py example/H800_allreduce.txt -s

#  Ranks: 8     Hosts: cu74
#  Rings: 16    Tree groups: 1
#  Ring 00: 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 0  (all P2P)

# Two nodes (16x H800, 2 machines)
python3 nccl_topo_viz.py example/2xNode-H800_allreduce.log -s

#  Ranks: 16    Hosts: cu74, cu37
#  Rings: 4     Tree groups: 4
#  Ring 00: 0 → 7 → 6 → 5 → 4 → 3 → 2 → 1 → 8 → 15 → 14 → 13 → 12 → 11 → 10 → 9 → 0
#  RDMA bridge: rank 1↔8 (cu74↔cu37)
```
