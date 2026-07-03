#!/usr/bin/env python3
"""NCCL Topology Visualizer

Parses NCCL test logs and generates Graphviz diagrams:
1. Physical topology (GPU, PCIe Switch, CPU, NVSwitch, NIC)
2. Ring topology (all rings, horizontally arranged)
3. Tree topology (all distinct trees, horizontally arranged)

Usage:
    python3 nccl_topo_viz.py <log_file> [--output-dir <dir>]
"""

import re
import sys
import os
import argparse
import subprocess
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RankInfo:
    rank: int
    hostname: str
    device: int
    bdf: str        # e.g., "0000:0f:00"
    topo_id: str    # e.g., "f000"
    gpu_model: str
    pid: int


@dataclass
class TopoNode:
    """A node in the physical topology tree."""
    node_type: str          # CPU, PCI, DEV, GPU, NIC, NET, NVS, GIN, RMA
    node_id: str            # e.g., "CPU/0-0", "GPU/0-f000"
    link_type: str = ""     # PCI, SYS, NVL, LOC, NET (link from parent)
    bandwidth: str = ""     # link bandwidth value
    pci_bdf: str = ""       # for PCI/DEV nodes, the PCI address
    children: list = field(default_factory=list)
    indent: int = 0         # indentation level in the log


@dataclass
class RingSegment:
    """A 3-node segment of a ring: prev -> self -> next."""
    ring_id: int
    prev_rank: int
    self_rank: int
    next_rank: int


@dataclass
class TreeEntry:
    """A tree entry from Trees [N] line for one rank, one channel."""
    channel: int
    rank: int
    up: list       # [up0, up1, up2] parent ranks for trees 0, 1, 2
    self_rank: int
    down: list     # [down0, down1, down2] child ranks (may be truncated)


@dataclass
class Connection:
    """A connection between two ranks."""
    src_rank: int
    dst_rank: int
    via: str          # "P2P/CUMEM", "NET/IB/0/GDRDMA", etc.
    src_gpu: int
    dst_gpu: int


# ═══════════════════════════════════════════════════════════════════════════
# Parser
# ═══════════════════════════════════════════════════════════════════════════

class NCCLLogParser:
    def __init__(self, filepath):
        self.filepath = filepath
        self.ranks = {}                                   # rank -> RankInfo
        self.rank_lookup = {}                             # (hostname, local_rank) -> global_rank
        self.physical_topo = {}                           # hostname -> list of TopoNode (root-level)
        self.topo_gpu_map = {}                            # hostname -> {topo_id -> rank}
        self.rings = defaultdict(list)                    # ring_id -> list of RingSegment
        self.trees = defaultdict(list)                    # (channel, tree_idx) -> list of (rank, parent, child)
        self.trees_combined = defaultdict(list)           # (channel, ) -> list of TreeEntry
        self.connections = {}                             # (src_rank, dst_rank) -> Connection
        self.nic_info = {}                                # hostname -> NIC name
        self.hostnames = []                               # ordered list of hostnames
        self.num_ranks = 0

    def parse(self):
        with open(self.filepath) as f:
            lines = f.readlines()
        self._parse_ranks(lines)
        self._build_rank_lookup()
        self._parse_topology(lines)
        self._parse_rings(lines)
        self._parse_trees(lines)
        self._parse_connections(lines)
        self._parse_nics(lines)
        self._build_gpu_map()
        return self

    # ── Rank parsing ──────────────────────────────────────────────────────

    def _parse_ranks(self, lines):
        pattern = re.compile(
            r'#\s+Rank\s+(\d+)\s+Group\s+\d+\s+Pid\s+(\d+)\s+on\s+(\S+)\s+'
            r'device\s+(\d+)\s+\[([0-9a-f:]+)\]\s+(.+)'
        )
        for line in lines:
            m = pattern.match(line)
            if m:
                rank = int(m.group(1))
                pid = int(m.group(2))
                hostname = m.group(3)
                device = int(m.group(4))
                bdf = m.group(5)
                gpu_model = m.group(6).strip()
                topo_id = self._bdf_to_topo_id(bdf)
                self.ranks[rank] = RankInfo(
                    rank=rank, hostname=hostname, device=device,
                    bdf=bdf, topo_id=topo_id, gpu_model=gpu_model, pid=pid
                )
                if hostname not in self.hostnames:
                    self.hostnames.append(hostname)
        self.num_ranks = len(self.ranks)

    def _build_rank_lookup(self):
        for rank, info in self.ranks.items():
            self.rank_lookup[(info.hostname, info.device)] = rank

    @staticmethod
    def _bdf_to_topo_id(bdf):
        """Convert BDF '0000:0f:00' to topology ID 'f000'."""
        parts = bdf.split(":")
        if len(parts) >= 3:
            bus = parts[1]               # e.g., "0f"
            dev_func = parts[2]           # e.g., "00"
            if "." in dev_func:
                dev, func = dev_func.split(".")
            else:
                dev, func = dev_func, "0"
            raw = f"{bus}{dev}{int(func):x}"
            return format(int(raw, 16), "x")
        return bdf

    # ── Physical topology parsing ─────────────────────────────────────────

    def _parse_topology(self, lines):
        """Parse the '=== System ===' block for each unique hostname."""
        in_topo = False
        current_hostname = None
        topo_lines = []

        for line in lines:
            if "=== System :" in line:
                m = re.match(r'(\S+):\d+:\d+\s+\[(\d+)\]\s+NCCL INFO === System', line)
                if m:
                    hostname = m.group(1)
                    if hostname not in self.physical_topo:
                        in_topo = True
                        current_hostname = hostname
                        topo_lines = []
                    else:
                        in_topo = False
                        current_hostname = None
                    continue
            if in_topo:
                if "======" in line:
                    if current_hostname and topo_lines:
                        self.physical_topo[current_hostname] = self._build_topology_tree(topo_lines)
                    in_topo = False
                    current_hostname = None
                    topo_lines = []
                    continue
                # Extract the NCCL INFO content
                m = re.match(r'\S+:\d+:\d+\s+\[\d+\]\s+NCCL INFO\s(.*)', line)
                if m:
                    topo_lines.append(m.group(1))

    def _build_topology_tree(self, lines):
        """Build a topology tree from indented log lines."""
        nodes = []
        stack = []  # (indent, TopoNode)

        for line in lines:
            parsed = self._parse_topo_line(line)
            if parsed is None:
                continue
            indent, node = parsed

            # Pop stack to find parent
            while stack and stack[-1].indent >= indent:
                stack.pop()

            if stack:
                stack[-1].children.append(node)
            else:
                nodes.append(node)

            stack.append(node)

        return nodes

    @staticmethod
    def _parse_topo_line(line):
        """Parse a single topology line, returning (indent, TopoNode) or None."""
        line = line.rstrip()
        if not line.strip():
            return None

        stripped = line.lstrip()
        leading_spaces = len(line) - len(stripped)

        indent = leading_spaces
        if stripped.startswith('+'):
            indent += 14

        m = re.match(
            r'(?:\+\s+(\w+)\[([\d.]+)\]\s+-\s+)?'
            r'(\w+)/(\S+)'
            r'(?:\s+\(([^)]+)\))?',
            stripped
        )
        if not m:
            return None

        link_type = m.group(1) or ""
        bandwidth = m.group(2) or ""
        node_type = m.group(3)
        node_id = m.group(4)
        extra = m.group(5) or ""

        node = TopoNode(
            node_type=node_type,
            node_id=f"{node_type}/{node_id}",
            link_type=link_type,
            bandwidth=bandwidth,
            pci_bdf=extra,
            indent=indent,
        )
        return (indent, node)

    # ── Ring parsing ──────────────────────────────────────────────────────

    def _parse_rings(self, lines):
        pattern = re.compile(
            r'(\S+):\d+:\d+\s+\[(\d+)\]\s+NCCL INFO Ring\s+(\d+)\s*:\s*(\d+)\s*->\s*(\d+)\s*->\s*(\d+)'
        )
        for line in lines:
            m = pattern.search(line)
            if m:
                hostname = m.group(1)
                local_rank = int(m.group(2))
                ring_id = int(m.group(3))
                prev_rank = int(m.group(4))
                self_rank = int(m.group(5))
                next_rank = int(m.group(6))

                global_rank = self.rank_lookup.get((hostname, local_rank))
                if global_rank is None:
                    continue

                self.rings[ring_id].append(RingSegment(
                    ring_id=ring_id,
                    prev_rank=prev_rank,
                    self_rank=global_rank,
                    next_rank=next_rank,
                ))

    # ── Tree parsing ──────────────────────────────────────────────────────

    def _parse_trees(self, lines):
        # Parse "Trees [N] up0/up1/up2->self->down" lines
        # These appear as: Trees [0] val [1] val [2] val ...
        trees_pattern = re.compile(
            r'(\S+):\d+:\d+\s+\[(\d+)\]\s+NCCL INFO Trees\s+(.+)'
        )

        for line in lines:
            m = trees_pattern.search(line)
            if not m:
                continue
            hostname = m.group(1)
            local_rank = int(m.group(2))
            rest = m.group(3)

            global_rank = self.rank_lookup.get((hostname, local_rank))
            if global_rank is None:
                continue

            # Parse channel-value pairs: [0] val [1] val ...
            channel_vals = re.findall(r'\[(\d+)\]\s+(\S+)', rest)
            for ch_str, val in channel_vals:
                channel = int(ch_str)
                parsed = self._parse_tree_value(val)
                if parsed:
                    up, self_val, down = parsed
                    self.trees_combined[channel].append(TreeEntry(
                        channel=channel,
                        rank=global_rank,
                        up=up,
                        self_rank=self_val,
                        down=down,
                    ))

    @staticmethod
    def _parse_tree_value(val):
        """Parse 'up0/up1/up2->self->down0/down1/down2' (down may be truncated)."""
        parts = val.split('->')
        if len(parts) != 3:
            return None

        up_str = parts[0].strip()
        self_str = parts[1].strip()
        down_str = parts[2].strip()

        up = [int(x) for x in up_str.split('/')]
        while len(up) < 3:
            up.append(-1)

        down_parts = down_str.split('/')
        down = [int(x) for x in down_parts]
        while len(down) < 3:
            down.append(-1)

        return (up, int(self_str), down)

    # ── Connection parsing ────────────────────────────────────────────────

    def _parse_connections(self, lines):
        """Parse 'Channel NN/T : src[gpu] -> dst[gpu] [direction] via TYPE' lines."""
        pattern = re.compile(
            r'(\S+):\d+:\d+\s+\[(\d+)\]\s+NCCL INFO Channel\s+(\d+)/\d+\s*:\s*'
            r'(\d+)\[(\d+)\]\s*->\s*(\d+)\[(\d+)\]'
            r'(?:\s+\[(\w+)\])?\s*via\s+(\S+)'
        )
        for line in lines:
            m = pattern.search(line)
            if m:
                src_rank = int(m.group(4))
                src_gpu = int(m.group(5))
                dst_rank = int(m.group(6))
                dst_gpu = int(m.group(7))
                via = m.group(9)

                key = (src_rank, dst_rank)
                if key not in self.connections:
                    self.connections[key] = Connection(
                        src_rank=src_rank,
                        dst_rank=dst_rank,
                        via=via,
                        src_gpu=src_gpu,
                        dst_gpu=dst_gpu,
                    )

    # ── NIC parsing ───────────────────────────────────────────────────────

    def _parse_nics(self, lines):
        pattern = re.compile(
            r'(\S+):\d+:\d+\s+\[\d+\]\s+NCCL INFO NET/IB\s+:\s+Using\s+\[\d+\](\S+)'
        )
        for line in lines:
            m = pattern.search(line)
            if m:
                hostname = m.group(1)
                nic_name = m.group(2)
                if hostname not in self.nic_info:
                    self.nic_info[hostname] = nic_name

    # ── GPU map ───────────────────────────────────────────────────────────

    def _build_gpu_map(self):
        """Map hostname + full topo node ID -> rank for physical topology labeling."""
        for rank, info in self.ranks.items():
            if info.hostname not in self.topo_gpu_map:
                self.topo_gpu_map[info.hostname] = {}
            # Topology node IDs use format "0-f000" (domain-busId), topo_id is "f000"
            self.topo_gpu_map[info.hostname][f"0-{info.topo_id}"] = rank

    # ── Reconstruction ────────────────────────────────────────────────────

    def reconstruct_ring(self, ring_id):
        """Reconstruct the full ring from per-rank segments."""
        segments = self.rings.get(ring_id, [])
        if not segments:
            return []

        # Build a map: self_rank -> (prev, next)
        next_map = {}
        prev_map = {}
        for seg in segments:
            next_map[seg.self_rank] = seg.next_rank
            prev_map[seg.self_rank] = seg.prev_rank

        # Start from rank 0 if available, otherwise use the first segment's rank
        start = 0 if 0 in next_map else segments[0].self_rank
        # Follow next pointers to build the ring
        ring = [start]
        current = start
        visited = {start}
        while True:
            nxt = next_map.get(current)
            if nxt is None or nxt in visited:
                break
            ring.append(nxt)
            visited.add(nxt)
            current = nxt

        return ring

    def get_ring_channels(self):
        """Return sorted list of ring IDs."""
        return sorted(self.rings.keys())

    def get_tree_groups(self):
        """Group channels by their tree structure and return distinct tree configurations.

        Returns a list of (group_name, tree_index, edges) where edges is a list of (parent, child, rank).
        """
        if not self.trees_combined:
            return []

        # For each channel, build tree 0 and tree 1 structures.
        # NCCL Trees format: up0/up1/up2->self->down0/down1/down2
        # up = parent, down = child.
        # Tree 0 is the reduce tree: data flows child->parent (leaf->root).
        # Tree 1 is the broadcast tree: data flows parent->child (root->leaf).
        # Edges are drawn in data flow direction.

        channel_groups = {}  # key: (tree0_sig, tree1_sig) -> list of channels

        for channel in sorted(self.trees_combined.keys()):
            entries = self.trees_combined[channel]

            tree0_edges = set()
            tree1_edges = set()

            for entry in entries:
                rank = entry.rank
                # Tree 0 (reduce): data flows child -> parent, so arrows point self -> up
                if entry.up[0] != -1:
                    tree0_edges.add((rank, entry.up[0]))
                if entry.down[0] != -1:
                    tree0_edges.add((entry.down[0], rank))
                # Tree 1 (broadcast): data flows parent -> child, so arrows point up -> self
                if entry.up[1] != -1:
                    tree1_edges.add((entry.up[1], rank))

            # Create signature for grouping
            tree0_sig = frozenset(tree0_edges)
            tree1_sig = frozenset(tree1_edges)
            key = (tree0_sig, tree1_sig)

            if key not in channel_groups:
                channel_groups[key] = {
                    'channels': [],
                    'tree0_edges': tree0_edges,
                    'tree1_edges': tree1_edges,
                }
            channel_groups[key]['channels'].append(channel)

        result = []
        for key, data in channel_groups.items():
            channels = data['channels']
            ch_str = f"Ch{channels[0]}" if len(channels) == 1 else f"Ch{channels[0]}-{channels[-1]}"
            if data['tree0_edges']:
                result.append((f"{ch_str} Tree0", data['tree0_edges']))
            if data['tree1_edges']:
                result.append((f"{ch_str} Tree1", data['tree1_edges']))

        return result


# ═══════════════════════════════════════════════════════════════════════════
# DOT Generators
# ═══════════════════════════════════════════════════════════════════════════

def _rank_label(rank_info):
    """Generate a node label for a rank."""
    return f"Rank {rank_info.rank}\\n{rank_info.hostname}:GPU{rank_info.device}\\n{rank_info.gpu_model}"


def _rank_color(rank_info, hostnames):
    """Color based on hostname for multi-node visualization."""
    colors = ['#A8D8EA', '#FFB7B2', '#B5EAD7', '#FFDFBA', '#E2C2FF', '#C7F0DB']
    idx = hostnames.index(rank_info.hostname) % len(colors)
    return colors[idx]


def _parse_rdma_info(via):
    """Extract RDMA device ID and GDRDMA flag from a 'via' string.

    Examples:
        NET/IB/0/GDRDMA       -> ('0', True)
        NET/IB/0(0)/GDRDMA    -> ('0', True)
        P2P/CUMEM             -> (None, False)
    """
    import re
    m = re.match(r'NET/IB/(\d+)(?:\(\d+\))?/(GDRDMA)?', via)
    if m:
        return m.group(1), m.group(2) is not None
    return None, False


def _conn_label(conn, parser=None):
    """Generate edge label for a connection."""
    if 'P2P' in conn.via:
        return 'P2P'
    elif 'NET' in conn.via or 'IB' in conn.via or 'GDRDMA' in conn.via:
        dev_id, gdrdma = _parse_rdma_info(conn.via)
        parts = ['RDMA']
        if parser:
            src_host = parser.ranks[conn.src_rank].hostname
            nic = parser.nic_info.get(src_host, 'mlx5_0')
            if ':' in nic:
                nic = nic.split(':')[0]
            parts.append(nic)
        if dev_id is not None:
            parts.append(f'IB/{dev_id}')
        if gdrdma:
            parts.append('GDRDMA')
        return '\\n'.join(parts)
    return conn.via


def _conn_color(conn):
    """Edge color based on connection type."""
    if 'P2P' in conn.via:
        return '#2ca02c'  # green
    elif 'NET' in conn.via or 'IB' in conn.via or 'GDRDMA' in conn.via:
        return '#d62728'  # red
    return '#666666'


def generate_physical_topo_dot(parser, hostname, topo_nodes, output_path):
    """Generate Graphviz DOT for physical topology of a single node."""
    lines = []
    lines.append(f'digraph phys_topo_{hostname} {{')
    lines.append('  rankdir=TB;')
    lines.append('  graph [fontname="Helvetica", fontsize=12, labelloc="t"];')
    lines.append(f'  label="Physical Topology - {hostname}";')
    lines.append('  node [fontname="Helvetica", fontsize=10, style=filled];')
    lines.append('  edge [fontname="Helvetica", fontsize=8];')
    lines.append("")

    defined_nodes = set()

    def make_node_name(node):
        return node.node_id.replace('/', '_').replace('-', '_')

    def add_node(node, effective_parent=None, override_link=None):
        skip_types = ('DEV', 'NET', 'GIN', 'RMA')

        if node.node_type in skip_types:
            if node.node_type == 'DEV':
                gpu_child = None
                other_children = []
                for child in node.children:
                    if child.node_type == 'GPU':
                        gpu_child = child
                    else:
                        other_children.append(child)
                pci_parent = effective_parent
                if gpu_child:
                    add_node(gpu_child, pci_parent, override_link=('PCI', node.bandwidth or '48.0'))
                for child in other_children:
                    if child.node_type == 'NVS' and gpu_child:
                        add_node(child, gpu_child)
                    else:
                        add_node(child, pci_parent)
            else:
                for child in node.children:
                    add_node(child, effective_parent)
            return

        if node.link_type == 'SYS' and node.node_type == 'CPU':
            return

        name = make_node_name(node)

        if node.node_type == 'CPU':
            label = f"CPU\\n{node.node_id}"
            color = '#87CEEB'
            shape = 'box3d'
        elif node.node_type == 'PCI':
            label = f"PCIe Switch\\n{node.node_id}"
            if node.pci_bdf:
                label += f"\\n({node.pci_bdf[:8]})"
            color = '#D3D3D3'
            shape = 'box'
        elif node.node_type == 'GPU':
            topo_id = node.node_id.split('/')[-1]
            rank = parser.topo_gpu_map.get(hostname, {}).get(topo_id, -1)
            if rank >= 0:
                ri = parser.ranks[rank]
                label = f"GPU {ri.device} (Rank {rank})\\n{ri.gpu_model}\\n{ri.bdf}"
            else:
                label = f"GPU\\n{node.node_id}"
            color = '#90EE90'
            shape = 'box'
        elif node.node_type == 'NIC':
            nic_name = parser.nic_info.get(hostname, 'mlx5_0')
            if ':' in nic_name:
                nic_name = nic_name.split(':')[0]
            label = f"NIC\\n{nic_name}\\n{node.node_id}"
            color = '#FFD700'
            shape = 'box'
        elif node.node_type == 'NVS':
            label = f"NVSwitch\\n{node.node_id}\\nNVL 164.8 GB/s"
            color = '#DDA0DD'
            shape = 'diamond'
        else:
            label = f"{node.node_id}"
            color = '#FFFFFF'
            shape = 'box'

        if name not in defined_nodes:
            lines.append(f'  {name} [label="{label}", fillcolor="{color}", shape={shape}];')
            defined_nodes.add(name)

        if effective_parent:
            parent_name = make_node_name(effective_parent)
            link_type = override_link[0] if override_link else node.link_type
            link_bw = override_link[1] if override_link else node.bandwidth
            link_label = ''
            if link_type and link_bw:
                link_label = f'{link_type} {link_bw}'
            elif link_type:
                link_label = link_type
            edge_key = f'{parent_name}->{name}'
            if edge_key not in defined_nodes:
                lines.append(f'  {parent_name} -> {name} [label="{link_label}"];')
                defined_nodes.add(edge_key)

        for child in node.children:
            add_node(child, node)

    for node in topo_nodes:
        add_node(node)

    cpu_nodes = [n for n in topo_nodes if n.node_type == 'CPU']
    if len(cpu_nodes) >= 2:
        cpu0_name = make_node_name(cpu_nodes[0])
        cpu1_name = make_node_name(cpu_nodes[1])
        lines.append(f'  {cpu0_name} -> {cpu1_name} [label="SYS/UPI 22.0", style=dashed, color=gray, constraint=false];')

    lines.append("}")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    return output_path


def generate_rings_dot(parser, output_path):
    """Generate Graphviz DOT for all rings, arranged horizontally."""
    lines = []
    lines.append('digraph rings {')
    lines.append('  graph [fontname="Helvetica", fontsize=14, labelloc="t", rankdir=LR];')
    lines.append(f'  label="Ring Topology - {parser.num_ranks} ranks";')
    lines.append('  node [fontname="Helvetica", fontsize=9, style=filled];')
    lines.append('  edge [fontname="Helvetica", fontsize=8];')
    lines.append("")

    ring_ids = parser.get_ring_channels()
    hostnames = parser.hostnames

    for ring_id in ring_ids:
        ring = parser.reconstruct_ring(ring_id)
        if not ring:
            continue

        cluster_name = f'cluster_ring_{ring_id:02d}'
        lines.append(f'  subgraph {cluster_name} {{')
        lines.append(f'    label="Ring {ring_id:02d}";')
        lines.append('    style=dashed;')
        lines.append('    rankdir=LR;')
        lines.append("")

        # Create nodes for each rank in the ring
        for i, rank in enumerate(ring):
            ri = parser.ranks.get(rank)
            if not ri:
                continue
            color = _rank_color(ri, hostnames)
            label = _rank_label(ri)
            node_name = f'r{ring_id:02d}_{rank}'
            lines.append(f'    {node_name} [label="{label}", fillcolor="{color}"];')

        # Create edges for the ring (each rank -> next rank)
        for i in range(len(ring)):
            src_rank = ring[i]
            dst_rank = ring[(i + 1) % len(ring)]
            ri_src = parser.ranks.get(src_rank)
            ri_dst = parser.ranks.get(dst_rank)
            if not ri_src or not ri_dst:
                continue

            conn = parser.connections.get((src_rank, dst_rank))
            node_name_src = f'r{ring_id:02d}_{src_rank}'
            node_name_dst = f'r{ring_id:02d}_{dst_rank}'

            if conn:
                label = _conn_label(conn, parser)
                color = _conn_color(conn)
            else:
                # Same node → P2P, different node → RDMA
                if ri_src.hostname == ri_dst.hostname:
                    label = 'P2P'
                    color = '#2ca02c'
                else:
                    label = 'RDMA'
                    color = '#d62728'

            lines.append(f'    {node_name_src} -> {node_name_dst} [label="{label}", color="{color}"];')

        lines.append('  }')
        lines.append("")

    # Legend
    lines.append('  subgraph cluster_legend {')
    lines.append('    label="Legend";')
    lines.append('    style=dashed;')
    lines.append('    legend_p2p [label="P2P (intra-node)", fillcolor="#FFFFFF", shape=box];')
    lines.append('    legend_rdma [label="RDMA (inter-node)", fillcolor="#FFFFFF", shape=box];')
    lines.append('    legend_p2p -> legend_rdma [label="P2P", color="#2ca02c"];')
    lines.append('    legend_rdma -> legend_p2p [label="RDMA", color="#d62728"];')
    lines.append('  }')

    lines.append('}')

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    return output_path


def generate_trees_dot(parser, output_path):
    """Generate Graphviz DOT for all distinct trees, arranged horizontally."""
    lines = []
    lines.append('digraph trees {')
    lines.append('  graph [fontname="Helvetica", fontsize=14, labelloc="t", rankdir=TB];')
    lines.append(f'  label="Tree Topology - {parser.num_ranks} ranks";')
    lines.append('  node [fontname="Helvetica", fontsize=9, style=filled];')
    lines.append('  edge [fontname="Helvetica", fontsize=8];')
    lines.append("")

    tree_groups = parser.get_tree_groups()
    hostnames = parser.hostnames

    for idx, (group_name, edges) in enumerate(tree_groups):
        cluster_name = f'cluster_tree_{idx}'
        lines.append(f'  subgraph {cluster_name} {{')
        lines.append(f'    label="{group_name}";')
        lines.append('    style=dashed;')
        lines.append("")

        # Collect all ranks in this tree
        tree_ranks = set()
        for parent, child in edges:
            tree_ranks.add(parent)
            tree_ranks.add(child)

        # Create nodes
        for rank in sorted(tree_ranks):
            ri = parser.ranks.get(rank)
            if not ri:
                continue
            color = _rank_color(ri, hostnames)
            label = _rank_label(ri)
            node_name = f't{idx}_{rank}'
            # Mark root nodes (reduce target: no outgoing edge)
            roots = set()
            for r in tree_ranks:
                has_child = any(p == r for p, c in edges)
                if not has_child:
                    roots.add(r)
            shape = 'box3d' if rank in roots else 'box'
            lines.append(f'    {node_name} [label="{label}", fillcolor="{color}", shape={shape}];')

        # Create edges
        for parent, child in sorted(edges):
            ri_parent = parser.ranks.get(parent)
            ri_child = parser.ranks.get(child)
            if not ri_parent or not ri_child:
                continue

            node_name_parent = f't{idx}_{parent}'
            node_name_child = f't{idx}_{child}'
            conn = parser.connections.get((parent, child))

            if conn:
                label = _conn_label(conn, parser)
                color = _conn_color(conn)
            else:
                if ri_parent.hostname == ri_child.hostname:
                    label = 'P2P'
                    color = '#2ca02c'
                else:
                    label = 'RDMA'
                    color = '#d62728'

            lines.append(f'    {node_name_parent} -> {node_name_child} [label="{label}", color="{color}"];')

        lines.append('  }')
        lines.append("")

    # Legend
    lines.append('  subgraph cluster_legend {')
    lines.append('    label="Legend";')
    lines.append('    style=dashed;')
    lines.append('    legend_root [label="Root (box3d)", fillcolor="#FFFFFF", shape=box3d];')
    lines.append('    legend_node [label="Node", fillcolor="#FFFFFF", shape=box];')
    lines.append('    legend_root -> legend_node [label="P2P", color="#2ca02c"];')
    lines.append('    legend_node -> legend_root [label="RDMA", color="#d62728"];')
    lines.append('  }')

    lines.append('}')

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    return output_path


# ═══════════════════════════════════════════════════════════════════════════
# Rendering
# ═══════════════════════════════════════════════════════════════════════════

def render_dot(dot_path, output_format='png'):
    """Render a DOT file to the specified format using graphviz."""
    output_path = dot_path.rsplit('.', 1)[0] + '.' + output_format
    try:
        subprocess.run(
            ['dot', f'-T{output_format}', '-o', output_path, dot_path],
            check=True, capture_output=True, text=True
        )
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"  Warning: Failed to render {dot_path}: {e.stderr}", file=sys.stderr)
        return None
    except FileNotFoundError:
        print(f"  Warning: 'dot' command not found. DOT file saved but not rendered.", file=sys.stderr)
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Summary Printer
# ═══════════════════════════════════════════════════════════════════════════

def print_summary(parser):
    """Print a text summary of the parsed topology."""
    print(f"\n{'='*70}")
    print(f"NCCL Topology Summary: {parser.filepath}")
    print(f"{'='*70}")
    print(f"Ranks: {parser.num_ranks}")
    print(f"Hosts: {', '.join(parser.hostnames)}")
    print()

    # Rank mapping
    print("Rank Mapping:")
    for rank in sorted(parser.ranks.keys()):
        ri = parser.ranks[rank]
        print(f"  Rank {rank:2d} → {ri.hostname}:GPU{ri.device} ({ri.bdf} {ri.gpu_model})")
    print()

    # Physical topology summary
    for hostname in parser.hostnames:
        topo = parser.physical_topo.get(hostname, [])
        print(f"Physical Topology ({hostname}):")
        for node in topo:
            _print_topo_node(node, indent=2)
        print()

    # Ring summary
    ring_ids = parser.get_ring_channels()
    print(f"Rings: {len(ring_ids)} (IDs: {', '.join(f'{r:02d}' for r in ring_ids)})")
    for ring_id in ring_ids:
        ring = parser.reconstruct_ring(ring_id)
        if ring:
            print(f"  Ring {ring_id:02d}: {' → '.join(str(r) for r in ring)} → {ring[0]}")
    print()

    # Tree summary
    tree_groups = parser.get_tree_groups()
    print(f"Tree Groups: {len(tree_groups)}")
    for name, edges in tree_groups:
        print(f"  {name}: {len(edges)} edges")
        for parent, child in sorted(edges):
            ri_p = parser.ranks.get(parent)
            ri_c = parser.ranks.get(child)
            if ri_p and ri_c:
                p2p = 'P2P' if ri_p.hostname == ri_c.hostname else 'RDMA'
                print(f"    {parent}({ri_p.hostname}:GPU{ri_p.device}) → {child}({ri_c.hostname}:GPU{ri_c.device}) [{p2p}]")
    print()

    # Connection summary
    p2p_conns = [(k, v) for k, v in parser.connections.items() if 'P2P' in v.via]
    rdma_conns = [(k, v) for k, v in parser.connections.items() if 'P2P' not in v.via]
    print(f"Connections: {len(parser.connections)} total ({len(p2p_conns)} P2P, {len(rdma_conns)} RDMA)")
    if rdma_conns:
        print("  RDMA connections:")
        for (src, dst), conn in sorted(rdma_conns):
            ri_s = parser.ranks.get(src)
            ri_d = parser.ranks.get(dst)
            if ri_s and ri_d:
                print(f"    Rank {src}({ri_s.hostname}:GPU{ri_s.device}) → Rank {dst}({ri_d.hostname}:GPU{ri_d.device}) via {conn.via}")
    print()


def _print_topo_node(node, indent=0):
    """Recursively print a topology node."""
    prefix = ' ' * indent
    link = f'[{node.link_type} {node.bandwidth}]' if node.link_type else ''
    extra = f' ({node.pci_bdf})' if node.pci_bdf else ''
    print(f"{prefix}{node.node_id}{link}{extra}")
    for child in node.children:
        _print_topo_node(child, indent + 2)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser_arg = argparse.ArgumentParser(description='NCCL Topology Visualizer')
    parser_arg.add_argument('log_file', help='NCCL test log file')
    parser_arg.add_argument('--output-dir', '-o', default=None,
                            help='Output directory (default: same as log file)')
    parser_arg.add_argument('--format', '-f', default='png', choices=['png', 'svg', 'pdf'],
                            help='Output format (default: png)')
    parser_arg.add_argument('--summary', '-s', action='store_true',
                            help='Print text summary')
    args = parser_arg.parse_args()

    if not os.path.isfile(args.log_file):
        print(f"Error: File not found: {args.log_file}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.log_file))
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(args.log_file))[0]

    print(f"Parsing {args.log_file}...")
    parser = NCCLLogParser(args.log_file).parse()

    print(f"  Ranks: {parser.num_ranks}")
    print(f"  Hosts: {', '.join(parser.hostnames)}")
    print(f"  Rings: {len(parser.get_ring_channels())}")
    print(f"  Tree groups: {len(parser.get_tree_groups())}")
    print(f"  Connections: {len(parser.connections)}")

    if args.summary:
        print_summary(parser)

    # Generate physical topology
    for hostname in parser.hostnames:
        topo = parser.physical_topo.get(hostname)
        if topo:
            dot_path = os.path.join(output_dir, f'{base_name}_physical_{hostname}.dot')
            generate_physical_topo_dot(parser, hostname, topo, dot_path)
            print(f"  Physical topology ({hostname}): {dot_path}")
            out = render_dot(dot_path, args.format)
            if out:
                print(f"    → {out}")

    # Generate rings
    ring_dot = os.path.join(output_dir, f'{base_name}_rings.dot')
    generate_rings_dot(parser, ring_dot)
    print(f"  Rings: {ring_dot}")
    out = render_dot(ring_dot, args.format)
    if out:
        print(f"    → {out}")

    # Generate trees
    tree_dot = os.path.join(output_dir, f'{base_name}_trees.dot')
    generate_trees_dot(parser, tree_dot)
    print(f"  Trees: {tree_dot}")
    out = render_dot(tree_dot, args.format)
    if out:
        print(f"    → {out}")

    print(f"\nDone! Output files in {output_dir}/")


if __name__ == '__main__':
    main()
