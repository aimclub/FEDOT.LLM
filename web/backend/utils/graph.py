from pydantic import BaseModel, Field
from typing_extensions import List


class GraphvizBuilder(BaseModel):
    template: str = (
        'digraph {{\n'
        'bgcolor="transparent"\n'
        'rankdir=LR\n'
        'edge [color="white"]\n'
        'node [style="filled", shape="box"]\n'
        '{node_configuration}\n'
        '{dependencies}\n'
        '}}'
    )
    nodes_id: List[str] = Field(default_factory=list)
    node_configuration: List[str] = Field(default_factory=list, description='List of node configurations')
    dependencies: List[str] = Field(default_factory=list, description='List of node dependencies')

    def add_node(self, label: str, fillcolor: str = 'white'):
        node_id = ''.join(label.split())
        self.node_configuration.append(f'{node_id} [fillcolor="{fillcolor}", label="{label}"]')
        if len(self.nodes_id) > 0:
            self.dependencies.append(f'{self.nodes_id[-1]} -> {node_id}')
        self.nodes_id.append(node_id)

    def get_graph(self):
        return self.template.format(
            node_configuration='\n'.join(self.node_configuration),
            dependencies='\n'.join(self.dependencies)
        )
