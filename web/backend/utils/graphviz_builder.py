from __future__ import annotations

from dataclasses import dataclass, field
from typing_extensions import List, Optional, Union, Set, Literal, Dict, cast
import graphviz
from web.common.graphviz_typing import GraphAttr, NodeAttr, EdgeAttr


class Node:
    def __init__(self, name: str, attrs: Optional[NodeAttr] = None):
        self.name = name
        self.attrs = attrs

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, value: Node) -> bool:
        return self.name == value.name


class Edge:
    def __init__(self, src: Node, dst: Node, attrs: Optional[EdgeAttr] = None):
        self.src = src
        self.dst = dst
        self.attrs = attrs


@dataclass
class GraphvizBuilder:
    graph_type: Literal['digraph', 'graph'] = 'digraph'
    graph_attr: Optional[GraphAttr] = field(default=None)
    edge_attr: Optional[EdgeAttr] = field(default=None)
    node_attr: Optional[NodeAttr] = field(default=None)
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)

    def add_node(self, node: Node):
        self.nodes[node.name] = node

    def update_node(self, node: Node, attrs: NodeAttr):
        if node.attrs:
            node.attrs = cast(NodeAttr, {**node.attrs, **attrs})
        else:
            node.attrs = attrs

    def update_edge(self, edge: Edge, attrs: EdgeAttr):
        if edge.attrs:
            edge.attrs = cast(EdgeAttr, {**edge.attrs, **attrs})
        else:
            edge.attrs = attrs

    def add_edge(self, edge: Edge):
        self.edges.append(edge)

    def compile(self) -> Union[graphviz.Digraph, graphviz.Graph]:
        if self.graph_type == 'digraph':
            graph = graphviz.Digraph(graph_attr=self.graph_attr, edge_attr=self.edge_attr, node_attr=self.node_attr)
        else:
            graph = graphviz.Graph(graph_attr=self.graph_attr, edge_attr=self.edge_attr, node_attr=self.node_attr)

        for node in self.nodes.values():
            if node.attrs is None:
                graph.node(node.name)
            else:
                graph.node(node.name, **node.attrs)

        for edge in self.edges:
            if edge.attrs is None:
                graph.edge(edge.src.name, edge.dst.name)
            else:
                graph.edge(edge.src.name, edge.dst.name, **edge.attrs)

        return graph
