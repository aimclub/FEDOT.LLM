# Used GPT to do this. Errors may occur.

from typing_extensions import TypedDict, Union

class GraphAttr(TypedDict, total=False):
    _background: str
    """A string in the xdot format specifying an arbitrary background."""
    
    bb: str
    """Bounding box of drawing in points. For write only."""
    
    beautify: bool
    """Whether to draw leaf nodes uniformly in a circle around the root node in sfdp. For sfdp only."""
    
    bgcolor: str
    """Canvas background color."""
    
    center: bool
    """Whether to center the drawing in the output canvas"""
    
    charset: str
    """Character encoding used when interpreting string input as a text label."""
    
    class_: str
    """Classnames to attach to the node, edge, graph, or cluster's SVG element. For svg only."""
    
    clusterrank: str
    """Mode used for handling clusters. For dot only."""
    
    colorscheme: str
    """A color scheme namespace: the context for interpreting color names."""
    
    comment: str
    """Comments are inserted into output."""
    
    compound: bool
    """If true, allow edges between clusters. For dot only."""
    
    concentrate: bool
    """If true, use edge concentrators."""
    
    Damping: float
    """Factor damping force motions. For neato only."""
    
    defaultdist: float
    """The distance between nodes in separate connected components. For neato only."""
    
    dim: int
    """Set the number of dimensions used for the layout. For neato, fdp, sfdp only."""
    
    dimen: int
    """Set the number of dimensions used for rendering. For neato, fdp, sfdp only."""
    
    diredgeconstraints: Union[str, bool]
    """Whether to constrain most edges to point downwards. For neato only."""
    
    dpi: float
    """Specifies the expected number of pixels per inch on a display device. For bitmap output, svg only."""
    
    epsilon: float
    """Terminating condition. For neato only."""
    
    esep: float
    """Margin used around polygons for purposes of spline edge routing. For neato, fdp, sfdp, osage, circo, twopi only."""
    
    fontcolor: str
    """Color used for text."""
    
    fontname: str
    """Font used for text."""
    
    fontnames: str
    """Allows user control of how basic fontnames are represented in SVG output. For svg only."""
    
    fontpath: str
    """Directory list used by libgd to search for bitmap fonts."""
    
    fontsize: float
    """Font size, in points, used for text."""
    
    forcelabels: bool
    """Whether to force placement of all xlabels, even if overlapping."""
    
    gradientangle: int
    """If a gradient fill is being used, this determines the angle of the fill."""
    
    href: str
    """Synonym for URL. For map, postscript, svg only."""
    
    id: str
    """Identifier for graph objects. For map, postscript, svg only."""
    
    imagepath: str
    """A list of directories in which to look for image files."""
    
    inputscale: float
    """Scales the input positions to convert between length units. For neato, fdp only."""
    
    K: float
    """Spring constant used in virtual physical model. For fdp, sfdp only."""
    
    label: str
    """Text label attached to objects."""
    
    label_scheme: int
    """Whether to treat a node whose name has the form |edgelabel|* as a special node representing an edge label. For sfdp only."""
    
    labeljust: str
    """Justification for graph & cluster labels."""
    
    labelloc: str
    """Vertical placement of labels for nodes, root graphs and clusters."""
    
    landscape: bool
    """If true, the graph is rendered in landscape mode."""
    
    layerlistsep: str
    """The separator characters used to split attributes of type layerRange into a list of ranges."""
    
    layers: str
    """A linearly ordered list of layer names attached to the graph."""
    
    layerselect: str
    """Selects a list of layers to be emitted."""
    
    layersep: str
    """The separator characters for splitting the layers attribute into a list of layer names."""
    
    layout: str
    """Which layout engine to use."""
    
    levels: int
    """Number of levels allowed in the multilevel scheme. For sfdp only."""
    
    levelsgap: float
    """strictness of neato level constraints. For neato only."""
    
    lheight: float
    """Height of graph or cluster label, in inches. For write only."""
    
    linelength: int
    """How long strings should get before overflowing to next line, for text output."""
    
    lp: str
    """Label center position. For write only."""
    
    lwidth: float
    """Width of graph or cluster label, in inches. For write only."""
    
    margin: str
    """For graphs, this sets x and y margins of canvas, in inches."""
    
    maxiter: int
    """Sets the number of iterations used. For neato, fdp only."""
    
    mclimit: float
    """Scale factor for mincross (mc) edge crossing minimiser parameters. For dot only."""
    
    mindist: float
    """Specifies the minimum separation between all nodes. For circo only."""
    
    mode: str
    """Technique for optimizing the layout. For neato only."""
    
    model: str
    """Specifies how the distance matrix is computed for the input graph. For neato only."""
    
    newrank: bool
    """Whether to use a single global ranking, ignoring clusters. For dot only."""
    
    nodesep: float
    """In dot, nodesep specifies the minimum space between two adjacent nodes in the same rank, in inches."""
    
    nojustify: bool
    """Whether to justify multiline text vs the previous text line (rather than the side of the container)."""
    
    normalize: bool
    """normalizes coordinates of final layout. For neato, fdp, sfdp, twopi, circo only."""
    
    notranslate: bool
    """Whether to avoid translating layout to the origin point. For neato only."""
    
    nslimit: int
    """Sets number of iterations in network simplex applications. For dot only."""
    
    nslimit1: int
    """Sets number of iterations in network simplex applications. For dot only."""
    
    oneblock: bool
    """Whether to draw circo graphs around one circle. For circo only."""
    
    ordering: str
    """Constrains the left-to-right ordering of node edges. For dot only."""
    
    orientation: str
    """node shape rotation angle, or graph orientation."""
    
    outputorder: str
    """Specify order in which nodes and edges are drawn."""
    
    overlap: str
    """Determines if and how node overlaps should be removed. For fdp, neato, sfdp, circo, twopi only."""
    
    overlap_scaling: float
    """Scale layout by factor, to reduce node overlap. For prism, neato, sfdp, fdp, circo, twopi only."""
    
    overlap_shrink: bool
    """Whether the overlap removal algorithm should perform a compression pass to reduce the size of the layout. For prism only."""
    
    pack: bool
    """Whether each connected component of the graph should be laid out separately, and then the graphs packed together."""
    
    packmode: str
    """How connected components should be packed."""
    
    pad: float
    """Inches to extend the drawing area around the minimal area needed to draw the graph."""
    
    page: str
    """Width and height of output pages, in inches."""
    
    pagedir: str
    """The order in which pages are emitted."""
    
    quadtree: str
    """Quadtree scheme to use. For sfdp only."""
    
    quantum: float
    """If quantum > 0.0, node label dimensions will be rounded to integral multiples of the quantum."""
    
    rankdir: str
    """Sets direction of graph layout. For dot only."""
    
    ranksep: float
    """Specifies separation between ranks. For dot, twopi only."""
    
    ratio: str
    """Sets the aspect ratio (drawing height/drawing width) for the drawing."""
    
    remincross: bool
    """If there are multiple clusters, whether to run edge crossing minimization a second time. For dot only."""
    
    repulsiveforce: float
    """The power of the repulsive force used in an extended Fruchterman-Reingold. For sfdp only."""
    
    resolution: float
    """Synonym for dpi. For bitmap output, svg only."""
    
    root: Union[str, bool]
    """Specifies nodes to be used as the center of the layout. For twopi, circo only."""
    rotate: int
    """If rotate=90, sets drawing orientation to landscape."""
    
    rotation: float
    """Rotates the final layout counter-clockwise by the specified number of degrees. For sfdp only."""
    
    scale: float
    """Scales layout by the given factor after the initial layout. For neato, twopi only."""
    
    searchsize: int
    """During network simplex, the maximum number of edges with negative cut values to search when looking for an edge with minimum cut value. For dot only."""
    
    sep: float
    """Margin to leave around nodes when removing node overlap. For fdp, neato, sfdp, osage, circo, twopi only."""
    
    showboxes: int
    """Print guide boxes for debugging. For dot only."""
    
    size: str
    """Maximum width and height of drawing, in inches."""
    
    smoothing: str
    """Specifies a post-processing step used to smooth out an uneven distribution of nodes. For sfdp only."""
    
    sortv: int
    """Sort order of graph components for ordering packmode packing."""
    
    splines: str
    """Controls how, and if, edges are represented."""
    
    start: str
    """Parameter used to determine the initial layout of nodes. For neato, fdp, sfdp only."""
    
    style: str
    """Set style information for components of the graph."""
    
    stylesheet: str
    """A URL or pathname specifying an XML style sheet, used in SVG output. For svg only."""
    
    target: str
    """If the object has a URL, this attribute determines which window of the browser is used for the URL. For map, svg only."""
    
    TBbalance: str
    """Which rank to move floating (loose) nodes to. For dot only."""
    
    tooltip: str
    """Tooltip (mouse hover text) attached to the node, edge, cluster, or graph. For cmap, svg only."""
    
    truecolor: bool
    """Whether internal bitmap rendering relies on a truecolor color model or uses. For bitmap output only."""
    
    URL: str
    """Hyperlinks incorporated into device-dependent output. For map, postscript, svg only."""
    
    viewport: str
    """Clipping window on final drawing."""
    
    voro_margin: float
    """Tuning margin of Voronoi technique. For neato, fdp, sfdp, twopi, circo only."""
    
    xdotversion: str
    """Determines the version of xdot used in output. For xdot only."""
    

class NodeAttr(TypedDict, total=False):
    area: float
    """Indicates the preferred area for a node or empty cluster. For patchwork only."""

    class_: str
    """Classnames to attach to the node, edge, graph, or cluster's SVG element. For svg only."""

    color: str
    """Basic drawing color for graphics, not text."""

    colorscheme: str
    """A color scheme namespace: the context for interpreting color names."""

    comment: str
    """Comments are inserted into output."""

    distortion: float
    """Distortion factor for shape=polygon."""

    fillcolor: str
    """Color used to fill the background of a node or cluster."""

    fixedsize: bool
    """Whether to use the specified width and height attributes to choose node size (rather than sizing to fit the node contents)."""

    fontcolor: str
    """Color used for text."""

    fontname: str
    """Font used for text."""

    fontsize: float
    """Font size, in points, used for text."""

    gradientangle: int
    """If a gradient fill is being used, this determines the angle of the fill."""

    group: str
    """Name for a group of nodes, for bundling edges avoiding crossings. For dot only."""

    height: float
    """Height of node, in inches."""

    href: str
    """Synonym for URL. For map, postscript, svg only."""

    id: str
    """Identifier for graph objects. For map, postscript, svg only."""

    image: str
    """Gives the name of a file containing an image to be displayed inside a node."""

    imagepos: str
    """Controls how an image is positioned within its containing node."""

    imagescale: Union[bool, str]
    """Controls how an image fills its containing node."""

    label: str
    """Text label attached to objects."""

    labelloc: str
    """Vertical placement of labels for nodes, root graphs and clusters."""

    layer: str
    """Specifies layers in which the node, edge or cluster is present."""

    margin: Union[float, str]
    """For graphs, this sets x and y margins of canvas, in inches."""

    nojustify: bool
    """Whether to justify multiline text vs the previous text line (rather than the side of the container)."""

    ordering: str
    """Constrains the left-to-right ordering of node edges. For dot only."""

    orientation: Union[float, str]
    """node shape rotation angle, or graph orientation."""

    penwidth: float
    """Specifies the width of the pen, in points, used to draw lines and curves."""

    peripheries: int
    """Set number of peripheries used in polygonal shapes and cluster boundaries."""

    pin: bool
    """Keeps the node at the node's given input position. For neato, fdp only."""

    pos: str
    """Position of node, or spline control points. For neato, fdp only."""

    rects: str
    """Rectangles for fields of records, in points. For write only."""

    regular: bool
    """If true, force polygon to be regular."""

    root: Union[bool, str]
    """Specifies nodes to be used as the center of the layout. For twopi, circo only."""

    samplepoints: int
    """Gives the number of points used for a circle/ellipse node."""

    shape: str
    """Sets the shape of a node."""

    shapefile: str
    """A file containing user-supplied node content."""

    showboxes: int
    """Print guide boxes for debugging. For dot only."""

    sides: int
    """Number of sides when shape=polygon."""

    skew: float
    """Skew factor for shape=polygon."""

    sortv: int
    """Sort order of graph components for ordering packmode packing."""

    style: str
    """Set style information for components of the graph."""

    target: str
    """If the object has a URL, this attribute determines which window of the browser is used for the URL. For map, svg only."""

    tooltip: str
    """Tooltip (mouse hover text) attached to the node, edge, cluster, or graph. For cmap, svg only."""

    URL: str
    """Hyperlinks incorporated into device-dependent output. For map, postscript, svg only."""

    vertices: str
    """Sets the coordinates of the vertices of the node's polygon, in inches. For write only."""

    width: float
    """Width of node, in inches."""

    xlabel: str
    """External label for a node or edge."""

    xlp: str
    """Position of an exterior label, in points. For write only."""

    z: float
    """Z-coordinate value for 3D layouts and displays."""

class EdgeAttr(TypedDict, total=False):
    arrowhead: str
    """Style of arrowhead on the head node of an edge."""

    arrowsize: float
    """Multiplicative scale factor for arrowheads."""

    arrowtail: str
    """Style of arrowhead on the tail node of an edge."""

    class_: str
    """Classnames to attach to the node, edge, graph, or cluster's SVG element. For svg only."""

    color: str
    """Basic drawing color for graphics, not text."""

    colorscheme: str
    """A color scheme namespace: the context for interpreting color names."""

    comment: str
    """Comments are inserted into output."""

    constraint: bool
    """If false, the edge is not used in ranking the nodes. For dot only."""

    decorate: bool
    """Whether to connect the edge label to the edge with a line."""

    dir: str
    """Edge type for drawing arrowheads."""

    edgehref: str
    """Synonym for edgeURL. For map, svg only."""

    edgetarget: str
    """Browser window to use for the edgeURL link. For map, svg only."""

    edgetooltip: str
    """Tooltip annotation attached to the non-label part of an edge. For cmap, svg only."""

    edgeURL: str
    """The link for the non-label parts of an edge. For map, svg only."""

    fillcolor: str
    """Color used to fill the background of a node or cluster."""

    fontcolor: str
    """Color used for text."""

    fontname: str
    """Font used for text."""

    fontsize: float
    """Font size, in points, used for text."""

    head_lp: str
    """Center position of an edge's head label. For write only."""

    headclip: bool
    """If true, the head of an edge is clipped to the boundary of the head node."""

    headhref: str
    """Synonym for headURL. For map, svg only."""

    headlabel: str
    """Text label to be placed near head of edge."""

    headport: str
    """Indicates where on the head node to attach the head of the edge."""

    headtarget: str
    """Browser window to use for the headURL link. For map, svg only."""

    headtooltip: str
    """Tooltip annotation attached to the head of an edge. For cmap, svg only."""

    headURL: str
    """If defined, headURL is output as part of the head label of the edge. For map, svg only."""

    href: str
    """Synonym for URL. For map, postscript, svg only."""

    id: str
    """Identifier for graph objects. For map, postscript, svg only."""

    label: str
    """Text label attached to objects."""

    labelangle: float
    """The angle (in degrees) in polar coordinates of the head & tail edge labels."""

    labeldistance: float
    """Scaling factor for the distance of headlabel / taillabel from the head / tail nodes."""

    labelfloat: bool
    """If true, allows edge labels to be less constrained in position."""

    labelfontcolor: str
    """Color used for headlabel and taillabel."""

    labelfontname: str
    """Font for headlabel and taillabel."""

    labelfontsize: float
    """Font size of headlabel and taillabel."""

    labelhref: str
    """Synonym for labelURL. For map, svg only."""

    labeltarget: str
    """Browser window to open labelURL links in. For map, svg only."""

    labeltooltip: str
    """Tooltip annotation attached to label of an edge. For cmap, svg only."""

    labelURL: str
    """If defined, labelURL is the link used for the label of an edge. For map, svg only."""

    layer: str
    """Specifies layers in which the node, edge or cluster is present."""

    len: float
    """Preferred edge length, in inches. For neato, fdp only."""

    lhead: str
    """Logical head of an edge. For dot only."""

    lp: str
    """Label center position. For write only."""

    ltail: str
    """Logical tail of an edge. For dot only."""

    minlen: int
    """Minimum edge length (rank difference between head and tail). For dot only."""

    nojustify: bool
    """Whether to justify multiline text vs the previous text line (rather than the side of the container)."""

    penwidth: float
    """Specifies the width of the pen, in points, used to draw lines and curves."""

    pos: str
    """Position of node, or spline control points. For neato, fdp only."""

    samehead: str
    """Edges with the same head and the same samehead value are aimed at the same point on the head. For dot only."""

    sametail: str
    """Edges with the same tail and the same sametail value are aimed at the same point on the tail. For dot only."""

    showboxes: int
    """Print guide boxes for debugging. For dot only."""

    style: str
    """Set style information for components of the graph."""

    tail_lp: str
    """Position of an edge's tail label, in points. For write only."""

    tailclip: bool
    """If true, the tail of an edge is clipped to the boundary of the tail node."""

    tailhref: str
    """Synonym for tailURL. For map, svg only."""

    taillabel: str
    """Text label to be placed near tail of edge."""

    tailport: str
    """Indicates where on the tail node to attach the tail of the edge."""

    tailtarget: str
    """Browser window to use for the tailURL link. For map, svg only."""

    tailtooltip: str
    """Tooltip annotation attached to the tail of an edge. For cmap, svg only."""

    tailURL: str
    """If defined, tailURL is output as part of the tail label of the edge. For map, svg only."""

    target: str
    """If the object has a URL, this attribute determines which window of the browser is used for the URL. For map, svg only."""

    tooltip: str
    """Tooltip (mouse hover text) attached to the node, edge, cluster, or graph. For cmap, svg only."""

    URL: str
    """Hyperlinks incorporated into device-dependent output. For map, postscript, svg only."""

    weight: Union[int, float]
    """Weight of edge."""

    xlabel: str
    """External label for a node or edge."""

    xlp: str
    """Position of an exterior label, in points. For write only."""
