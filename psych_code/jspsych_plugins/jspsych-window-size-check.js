var jsPsychWindowSizeCheck = (function (jspsych) {
  'use strict';

  const info = {
    name: "window-size-check",
    parameters: {
      minimum_width: {
        type: jspsych.ParameterType.INT,
        default: 0,
      },
      minimum_height: {
        type: jspsych.ParameterType.INT,
        default: 0,
      },
      resize_message: {
        type: jspsych.ParameterType.HTML_STRING,
        default: `
        <h3>Window Size Alert</h3>
        <p>Please resize your browser window to meet the minimum requirements:</p>
        <ul>
          <li>Minimum width: <span id="window-check-min-width"></span> px</li>
          <li>Minimum height: <span id="window-check-min-height"></span> px</li>
        </ul>
        <p>Your current window size:</p>
        <ul>
          <li>Current width: <span id="window-check-actual-width"></span> px</li>
          <li>Current height: <span id="window-check-actual-height"></span> px</li>
        </ul>
        <p>The experiment will continue automatically once your window meets the size requirements.</p>
        `,
      },
    },
  };

  class WindowSizeCheckPlugin {
    constructor(jsPsych) {
      this.jsPsych = jsPsych;
    }

    trial(display_element, trial) {
      const checkWindowSize = () => {
        const width = window.innerWidth;
        const height = window.innerHeight;

        if (width >= trial.minimum_width && height >= trial.minimum_height) {
          this.jsPsych.finishTrial({});
        } else {
          display_element.innerHTML = trial.resize_message;
          document.getElementById('window-check-min-width').textContent = trial.minimum_width;
          document.getElementById('window-check-min-height').textContent = trial.minimum_height;
          document.getElementById('window-check-actual-width').textContent = width;
          document.getElementById('window-check-actual-height').textContent = height;

          const checkInterval = setInterval(() => {
            const newWidth = window.innerWidth;
            const newHeight = window.innerHeight;
            
            document.getElementById('window-check-actual-width').textContent = newWidth;
            document.getElementById('window-check-actual-height').textContent = newHeight;

            if (newWidth >= trial.minimum_width && newHeight >= trial.minimum_height) {
              clearInterval(checkInterval);
              this.jsPsych.finishTrial({ 
                resize_required: true,
                initial_width: width,
                initial_height: height,
                final_width: newWidth,
                final_height: newHeight
              });
            }
          }, 200);
        }
      };

      checkWindowSize();
    }
  }

  WindowSizeCheckPlugin.info = info;

  return WindowSizeCheckPlugin;
})(jsPsychModule);