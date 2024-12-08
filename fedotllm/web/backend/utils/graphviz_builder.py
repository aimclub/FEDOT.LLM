from __future__ import annotations

from dataclasses import dataclass, field

import graphviz
from typing_extensions import List, Optional, Union, Literal, Dict, cast

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
    name: str
    graph_type: Literal['digraph', 'graph'] = 'digraph'
    graph_attr: Optional[GraphAttr] = field(default=None)
    edge_attr: Optional[EdgeAttr] = field(default=None)
    node_attr: Optional[NodeAttr] = field(default=None)
    subgraphs: List[GraphvizBuilder] = field(default_factory=list)
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)

    def add_node(self, node: Node):
        self.nodes[node.name] = node

    @staticmethod
    def update_node(node: Node, attrs: NodeAttr):
        if node.attrs:
            node.attrs = cast(NodeAttr, {**node.attrs, **attrs})
        else:
            node.attrs = attrs

    @staticmethod
    def update_edge(edge: Edge, attrs: EdgeAttr):
        if edge.attrs:
            edge.attrs = cast(EdgeAttr, {**edge.attrs, **attrs})
        else:
            edge.attrs = attrs

    def add_edge(self, edge: Edge):
        self.edges.append(edge)

    def compile(self) -> Union[graphviz.Digraph, graphviz.Graph]:
        if self.graph_type == 'digraph':
            graph = graphviz.Digraph(name=f"cluster_{self.name}", graph_attr=self.graph_attr, edge_attr=self.edge_attr,
                                     node_attr=self.node_attr)
        else:
            graph = graphviz.Graph(graph_attr=self.graph_attr, edge_attr=self.edge_attr, node_attr=self.node_attr)

        for subgraph_item in self.subgraphs:
            new_subgraph = subgraph_item.compile()
            graph.subgraph(graph=new_subgraph)

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
