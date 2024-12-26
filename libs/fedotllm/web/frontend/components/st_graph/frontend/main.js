import Streamlit from "./streamlit-component-lib.js";

var first_run = true;
var margin = 20;

var graphviz = d3.select("#graph")
    .graphviz()
    .attributer(attributer)

function attributer(datum, index, nodes) {
    var selection = d3.select(this);
    if (datum.tag == "svg") {
        var width = window.innerWidth;
        var height = window.innerHeight;
        console.log(height)
        selection
            .attr("width", width)
            .attr("height", height)
        datum.attributes.width = width - margin;
        datum.attributes.height = height - margin;
    }
}

function resetZoom() {
    console.log('Resetting zoom');
    console.log(d3.select("#graph").selectWithoutDataPropagation("svg"))
    graphviz
        .resetZoom(d3.transition().duration(1000));
}

function resizeSVG() {
    console.log('Resize');
    var width = window.innerWidth;
    var height = window.innerHeight;
    d3.select("#graph").selectWithoutDataPropagation("svg")
        .transition(function () {
            return d3.transition("main")
                .ease(d3.easeLinear)
                .duration(1500);
        })
        .duration(700)
        .attr("width", width - margin)
        .attr("height", height - margin);
}

d3.select(window).on("resize", resizeSVG);
d3.select(window).on("click", resetZoom);


function render_dot(dot) {
    graphviz
        .renderDot(dot)
        .transition(function () {
            return d3.transition("main")
                .ease(d3.easeLinear)
                .duration(1500);
        });
}

function render(new_dot, prev_dot) {
    console.log(window.innerWidth)
    if (new_dot) {
        if (prev_dot && prev_dot !== new_dot) {
            graphviz
                .width(window.innerWidth)
                .renderDot(prev_dot)
                .on("end", function () {
                        render_dot(new_dot)
                    }
                );
        } else {
            graphviz
                .renderDot(new_dot)
        }
    }
}


// data is any JSON-serializable value you sent from Python,
// and it's already deserialized for you.
function onDataFromPython(event) {
    if (event.data.type !== "streamlit:render") return;
    var dots = event.data.args.dot;  // Access values sent from Python here!
    var prev_dot = event.data.args.prev_dot;
    render(dots, prev_dot)
    console.log(d3.select("#graph").node().getBoundingClientRect().height)
}

function onRender(event) {
    if (first_run) {

        first_run = false;
    }
    console.log('Render')
    if (!window.rendered) {
        window.rendered = true
    }
}

document.addEventListener('DOMContentLoaded', function () {
    // Render the component whenever python send a "render event"
    Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender)
    // Tell Streamlit that the component is ready to receive events
    Streamlit.setComponentReady()
    // Render with the correct height, if this is a fixed-height component
    window.addEventListener("message", onDataFromPython);
    Streamlit.setFrameHeight(200)
})


