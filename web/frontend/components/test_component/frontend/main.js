import Streamlit from "./streamlit-component-lib.js";

function sendValue(value) {
    Streamlit.setComponentValue(value)
  }
  
export function handleClick(component){
    sendValue(component.srcElement.id);
}

document.addEventListener('DOMContentLoaded', function(){ 
    document.querySelector('#yin').addEventListener('click', handleClick);
    document.querySelector('#yang').addEventListener('click', handleClick);
})
  
var first_run = true;

function onRender(event) {
    if (first_run) {
        
        first_run = false;
    }
    if (!window.rendered) {
        window.rendered = true
    }
}

  // Render the component whenever python send a "render event"
  Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender)
  // Tell Streamlit that the component is ready to receive events
  Streamlit.setComponentReady()
  // Render with the correct height, if this is a fixed-height component
  Streamlit.setFrameHeight(50)