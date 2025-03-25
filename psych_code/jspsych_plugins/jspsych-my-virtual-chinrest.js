
const jsPsychMyVirtualChinrest = (function (jspsych) {
  'use strict';

  const info = {
      name: "virtual-chinrest", parameters: {

          /** Any content here will be displayed above the card stimulus. */
          credit_card_prompt: {
              type: jspsych.ParameterType.HTML_STRING, pretty_name: "Adjustment prompt", default: `

               <h3>We will now measure how large your monitor is.</h3>
              To help us calibrate our experiment, we would like to measure the size of your monitor. Please follow the instructions below. 
`,
          },
          credit_card_instructions: {
              type: jspsych.ParameterType.HTML_STRING, pretty_name: "Credit card instructions", default: `
<h3>Instructions:</h3>
<ul style="margin-top:1px; text-indent:4px">
  <li><b>Click and drag</b> the upper right <span style="color:red">corner</span> of the image until it is the same size as a credit card held up to
      the screen.
  </li>
  <li>You can use any card that is the same size as a credit card, like a driver's license.</li>
  <li>If you do not have access to a card, you can use a ruler: resize the card until it is 3.37 inches wide (8.56 cm).
  </li>
</ul>
`,
          },
          /** Content of the button displayed below the card stimulus. */
          credit_card_button_text: {
              type: jspsych.ParameterType.HTML_STRING,
              pretty_name: "Credit card button prompt",
              default: "Click here when the image is the same size as a credit card.",
          },
          /** How many times to measure the credit card size?  */
          credit_card_reps: {
              type: jspsych.ParameterType.INT,
              pretty_name: "Number of repetitions for credit card size measurement",
              default: 1,
          },
          /** Path to an image to be shown in the resizable item div. */
          credit_card_image_path: {
              type: jspsych.ParameterType.IMAGE, pretty_name: "Item path", default: null, preload: false,
          },
          /** The height of the item to be measured, in mm. */
          item_height_mm: {
              type: jspsych.ParameterType.FLOAT, pretty_name: "Item height (mm)", default: 53.98,
          },
          /** The width of the item to be measured, in mm. */
          item_width_mm: {
              type: jspsych.ParameterType.FLOAT, pretty_name: "Item width (mm)", default: 85.6,
          },
          /** The initial size of the card, in pixels, along the largest dimension. */
          credit_card_init_size: {
              type: jspsych.ParameterType.INT, pretty_name: "Initial Size", default: 250,
          },
          /** How many times to measure the blindspot location? If 0, blindspot will not be detected, and viewing distance and degree data not computed. */
          blindspot_reps: {
              type: jspsych.ParameterType.INT, pretty_name: "Blindspot measurement repetitions", default: 5,
          },
          /** HTML-formatted prompt to be shown on the screen, before the blindspot instructions. */
          blindspot_prompt: {
              type: jspsych.ParameterType.HTML_STRING, pretty_name: "Blindspot instructions", default: `
                     <h3>We will now measure how far away you are sitting.</h3>
      Everyone's eye has a small "blind spot", located in their peripheral vision.
      Measuring your blind spot lets us estimate how far you are sitting from your monitor, using trigonometry. This process should take no longer than a few moments. Please follow the instructions below. 
              `,
          },
          /** HTML-formatted prompt to be shown on the screen during blindspot estimates. */
          blindspot_instructions: {
              type: jspsych.ParameterType.HTML_STRING, pretty_name: "Blindspot prompt", default: `
      <h3>Instructions:</h3> 
          <ol style="margin-top:1px; text-indent:4px">
            <li>Place your left hand on the <b>space bar</b>.</li>
            <li>Then, cover your right eye with your right hand.</li>
            <li>Using your left eye, focus on the black square. Keep your focus on the black square.</li>
            <li>The <span style="color: red; font-weight: bold;">red ball</span> will seem to disappear as it moves from right to left, as it enters your blind spot. </li>
            <li>Press the space bar as soon as you see <b> the <span style="color: red;">ball</span> disappear</b> in your peripheral vision.</li>
          </ol>
          <span>You can redo these measurements by pressing the restart button.</span> <b>Press the space bar once you are ready.</b>
        `,
          },
          /** Text accompanying the remaining measurements counter. */
          blindspot_button_text: {
              type: jspsych.ParameterType.HTML_STRING,
              pretty_name: "Blindspot measurements prompt",
              default: "Remaining measurements: ",
          },
          /** HTML-formatted string for reporting the distance estimate. It can contain a span with ID 'distance-estimate', which will be replaced with the distance estimate. If "none" is given, viewing distance will not be reported to the participant. */
          viewing_distance_report: {
              type: jspsych.ParameterType.HTML_STRING,
              pretty_name: "Viewing distance report",
              default: `
                      <p>
                      Based on your responses, you are sitting about <span id='distance-estimate' style='font-weight: bold;'></span> from the screen.
                      
                      Does that seem about right?
                      </p>
                      `,
          },
          /** Label for the button that can be clicked on the viewing distance report screen to re-do the blindspot estimate(s). */
          redo_measurement_button_label: {
              type: jspsych.ParameterType.HTML_STRING,
              pretty_name: "Re-do measurement button label",
              default: "Restart",
          },
          /** Label for the button that can be clicked on the viewing distance report screen to accept the viewing distance estimate. */
          blindspot_done_prompt: {
              type: jspsych.ParameterType.HTML_STRING,
              pretty_name: "Blindspot done prompt",
              default: "Continue",
          },
      },
  };

  /**
   * **virtual-chinrest**
   *
   * jsPsych plugin for estimating physical distance from monitor and optionally resizing experiment content, based on Qisheng Li 11/2019. /// https://github.com/QishengLi/virtual_chinrest
   *
   * @author Gustavo Juantorena
   * 08/2020 // https://github.com/GEJ1
   * Contributions from Peter J. Kohler: https://github.com/pjkohler
   * @see {@link https://www.jspsych.org/plugins/jspsych-virtual-chinrest/ virtual-chinrest plugin documentation on jspsych.org}
   */
  class VirtualChinrestPlugin {
      constructor(jsPsych) {
          this.jsPsych = jsPsych;
          this.ball_size = 30;
          this.ball = null;
          this.container = null;
          this.credit_card_reps_remaining = 0;
          this.blindspot_reps_remaining = 0;
          this.ball_animation_frame_id = null;
      }

      trial(display_element, trial) {

          display_element.innerHTML = `<div id="content" style="width: 900px; margin: 0 auto;"></div>`;

          /** some additional parameter configuration */
          this.credit_card_reps_remaining = trial.credit_card_reps;
          this.blindspot_reps_remaining = trial.blindspot_reps;


          /** Assemble the trial response object */
          let trial_data = {
              credit_card_response_data: [],
              blindspot_response_data: [],
              outputs: {},
          };

          /** Credit card */
          let credit_card_aspect_ratio = trial.item_width_mm / trial.item_height_mm;
          let timestamp_start_credit_card_phase;
          const initialize_credit_card_html = (init_size_px) => {
              /** Render then set content for the credit card phase */

              const start_div_height = credit_card_aspect_ratio < 1 ? init_size_px : Math.round(init_size_px / credit_card_aspect_ratio);
              const start_div_width = credit_card_aspect_ratio < 1 ? Math.round(init_size_px * credit_card_aspect_ratio) : init_size_px;
              const adjust_size = Math.round(start_div_width * 0.1);


              //let background_color = '#7F7F7F'

              const pagesize_content = `
                      <div id="page-size">
<div style="margin-bottom:20px">
<div style="
text-align: left;
background-color : #E4E4E4;
border-color: #7F7F7F;
border-radius: 8px;
max-width: 100%;
width:auto;
display:flex;
align-items:center;
flex-direction:column;
margin: 0 auto;
padding: 2%; 
">
${trial.credit_card_prompt}
  </div>
  <div style="margin-bottom:20px">
               <div style="
                    text-align: left;
                    background-color : #E4E4E4;
                    border-color: #7F7F7F;
                    border-radius: 8px;
                    max-width: 100%;
                    width:auto;
                    padding-left: 2%;
                    padding-right: 2%;
                    padding-bottom:1%;
                    display:flex;
                    align-items:center;
                    flex-direction:column;
                    margin-top:10px; 
                    ">
${trial.credit_card_instructions}
  </div>
</div>
                      <div id="item" style="
                          border: none;
                          height: ${start_div_height}px; 
                          width: ${start_div_width}px; 
                          margin: 30px auto; 
                           
                          position: relative;
                          ${trial.credit_card_image_path === null ? "" : `background-image: url(${trial.credit_card_image_path}); background-size: 100% auto; background-repeat: no-repeat;`}">
                          
                          <div 
                              id="jspsych-resize-handle" 
                              style="
                                  cursor: nesw-resize; 
                                  width: ${adjust_size}px; 
                                  height: ${adjust_size}px;
                                  border: 5px solid red; 
                                  border-radius:1px; 
                                  border-left: 0;
                                  border-bottom: 0;
                                  position: absolute;
                                  top: 0;
                                  right: 0;"
                                  >
                          </div>
                        </div>
                        
                      
                      <div style="
                      margin: 10px auto;
                       max-width: 50%; 
                       border-width: 2px; 
                       border-color: black; 
                       border-radius: 8px; ">
                      
   <div style="display: inline-block; margin: 0 auto; padding: 4px; border-radius: 8px;">
      <div style="padding:2px; margin: 1px auto; visibility: ${trial.credit_card_reps > 1 ? 'visible' : 'hidden'}">
          ${trial.blindspot_button_text}
          <span id="creditcardrepcount" style="color: black"> ${this.credit_card_reps_remaining} </span>
      </div>
      <button id="end_resize_phase" class="jspsych-btn" style="margin: 5px auto;"> ${trial.credit_card_button_text}</button>
  </div>
                       
                      </div>
                      </div>
`;

              display_element.querySelector("#content").innerHTML = pagesize_content;

              // Event listeners for mouse-based resize
              let dragging = false;
              let origin_x, origin_y;
              let item_width_cur_px, item_height_cur_px;
              const scale_div = display_element.querySelector("#item");

              function mouseupevent() {
                  dragging = false;
              }

              document.addEventListener("mouseup", mouseupevent);

              function mousedownevent(e) {
                  e.preventDefault();
                  dragging = true;
                  origin_x = e.pageX;
                  origin_y = e.pageY;
                  item_width_cur_px = parseInt(scale_div.style.width);
                  item_height_cur_px = parseInt(scale_div.style.height);
              }

              display_element
                  .querySelector("#jspsych-resize-handle")
                  .addEventListener("mousedown", mousedownevent);

              function resizeevent(e) {
                  if (dragging) {
                      let dx = e.pageX - origin_x;
                      //let dy = e.pageY - origin_y;

                      let minsize_px = 50;
                      let new_width = item_width_cur_px + dx * 2
                      new_width = Math.max(minsize_px, new_width) // Safety
                      let new_height = new_width / credit_card_aspect_ratio

                      scale_div.style.width = Math.round(new_width) + "px";
                      scale_div.style.height = Math.round(new_height) + "px";
                  }
              }

              display_element.addEventListener("mousemove", resizeevent);
              display_element
                  .querySelector("#end_resize_phase")
                  .addEventListener("click", submitCreditCardMeasurement);

              timestamp_start_credit_card_phase = performance.now()
          }

          function startCreditCardPhase() {

              // Increment the data counter
              let nprevious_reps = trial_data['credit_card_response_data'].length;
              let increment_new = true;
              if (nprevious_reps > 0) {
                  if (trial_data['credit_card_response_data'][nprevious_reps - 1].length === 0) {
                      increment_new = false;
                  }
              }
              if (increment_new) {
                  trial_data['credit_card_response_data'].push([]);
              }

              initialize_credit_card_html(trial.credit_card_init_size);
          }

          const submitCreditCardMeasurement = () => {
              // add item width info to data
              const item_width_px = document.querySelector("#item").getBoundingClientRect().width;
              const reaction_time_msec = performance.now() - timestamp_start_credit_card_phase
              const credit_card_trial_outcome = {
                  'reported_item_width_px': item_width_px,
                  'reaction_time_msec': reaction_time_msec,
                  'rel_timestamp_credit_card_phase_start': timestamp_start_credit_card_phase,
                  'initial_size_px': trial.credit_card_init_size,
                  'item_width_mm': trial.item_width_mm,
                  'item_height_mm': trial.item_height_mm, //card dimension: 85.60 × 53.98 mm (3.370 × 2.125 in)
              }
              let i = trial_data['credit_card_response_data'].length - 1;
              trial_data['credit_card_response_data'][i].push(credit_card_trial_outcome)

              this.credit_card_reps_remaining--
              if (this.credit_card_reps_remaining > 0) {
                  // Reinitialize the credit card size measurement
                  initialize_credit_card_html(trial.credit_card_init_size);
              } else {
                  finishCreditCardPhase();
              }
          }

          const finishCreditCardPhase = () => {
              // Finish credit card phase here
              let i_latest = trial_data['credit_card_response_data'].length - 1;
              let results = trial_data['credit_card_response_data'][i_latest];

              let sum = 0;
              let n = 0;
              for (let i_rep = 0; i_rep < results.length; i_rep++) {
                  const v = results[i_rep]['reported_item_width_px'];
                  sum += (typeof v !== 'undefined') ? v : 0
                  n += (typeof v !== 'undefined') ? 1 : 0
              }

              let average_item_width_px = sum / n;
              let pixels_per_mm = average_item_width_px / trial["item_width_mm"]; // Pixels per mm

              // Save calculated outputs
              trial_data['outputs']['pixels_per_mm'] = pixels_per_mm;
              trial_data['outputs']['average_reported_item_width_px'] = average_item_width_px;
              _store_jspsych_cookie(trial_data);

              // Check what to do next
              if (trial.blindspot_reps > 0) {
                  // Proceed to blind spot phase
                  generateBlindspotPhaseContent();
                  beginBlindSpotPhase()
              } else {
                  // End the entire trial here
                  endTrial();
              }
          }

          /** Create content for second screen, blind spot */
              // Add the blindspot content to the page
          const generateBlindspotPhaseContent = () => {
                  console.log('Initializing blindspot phase')

                  let blindspot_content = `
              <div id="blind-spot">
  <div style="margin-bottom:20px">
      <div style="
                    text-align: left;
                    background-color : #E4E4E4;
                    border-color: #7F7F7F;
                    border-radius: 8px;
                    max-width: 100%;
                    width:auto;
                    padding-left: 2%;
                      padding-right: 2%;
                      padding-bottom:2%;
                    display:flex;
                    align-items:center;
                    flex-direction:column;
                    margin: 0 auto;
                    ">
          ${trial.blindspot_prompt}
      </div>
        <div style="
text-align: left;
background-color : #E4E4E4;
border-color: #7F7F7F;
border-radius: 8px;
max-width: 100%;
width:auto;
display:flex;
align-items:center;
flex-direction:column;
margin: 0 auto;
padding: 2%; 
">

      ${trial.blindspot_instructions}
                  </div>
  </div>
  <div id="svgDiv" style="height:100px; position:relative;"></div>
  <div style="display: inline-block; margin: 0 auto; padding: 4px; border-radius: 8px;">
      <div style="padding:2px; margin: 1px auto">
          ${trial.blindspot_button_text}
          <span id="blindspot_rep_counter" style="color: black"> ${trial.blindspot_reps} </span>
      </div>
      <button class="jspsych-btn" id="restart_blindspot_button">${trial.redo_measurement_button_label}</button>
  </div>
</div>
`;

                  display_element
                      .querySelector("#content").innerHTML = blindspot_content;

                  display_element
                      .querySelector("#restart_blindspot_button")
                      .addEventListener("click", function () {
                          console.log('User hit restart button');
                          beginBlindSpotPhase();

                      });
                  this.container = display_element.querySelector("#svgDiv");

                  drawBallAndSquare();
              };

          const drawBallAndSquare = () => {
              this.container.innerHTML = `
                      <div id="virtual-chinrest-circle" style="position: absolute; background-color: #f00; width: ${this.ball_size}px; height: ${this.ball_size}px; border-radius:${this.ball_size}px;"></div>
                      <div id="virtual-chinrest-square" style="position: absolute; background-color: #000; width: ${this.ball_size}px; height: ${this.ball_size}px;"></div>
                      `;
              const ball = this.container.querySelector("#virtual-chinrest-circle");
              const square = this.container.querySelector("#virtual-chinrest-square");
              const rectX = this.container.getBoundingClientRect().width - this.ball_size;
              const ballX = rectX * 0.85; // define where the ball is

              // Vertically center the ball and square
              const bounding_rect = (this.container.getBoundingClientRect());
              const topY = bounding_rect.height / 2 - this.ball_size / 2;

              ball.style.left = `${ballX}px`;
              square.style.left = `${rectX}px`;
              ball.style.top = `${topY}px`;
              square.style.top = `${topY}px`;

              this.ball = ball;
              this.square = square;
          };

          const beginBlindSpotPhase = () => {

              // Reset animatinos and event listeners
              this.jsPsych.pluginAPI.cancelAllKeyboardResponses();
              cancelAnimationFrame(this.ball_animation_frame_id);

              // Increment the data counter
              let nprevious_reps = trial_data['blindspot_response_data'].length;
              let increment_new = true;
              if (nprevious_reps > 0) {
                  if (trial_data['blindspot_response_data'][nprevious_reps - 1].length === 0) {
                      increment_new = false;
                  }
              }
              if (increment_new) {
                  trial_data['blindspot_response_data'].push([]);
              }

              // Reset the number of repetitions
              this.blindspot_reps_remaining = trial.blindspot_reps;
              document.querySelector("#blindspot_rep_counter").textContent = Math.max(this.blindspot_reps_remaining, 0).toString();

              // Draw the ball and fixation square
              resetAndWaitForBallStart();
          }

          const resetAndWaitForBallStart = () => {
              const rectX = this.container.getBoundingClientRect().width - this.ball_size;
              const ball_initialX = rectX * 0.85; // Reset the initial position of the ball
              this.ball.style.left = `${ball_initialX}px`;

              // Wait for a spacebar keypress to begin the ball trial
              this.jsPsych.pluginAPI.cancelAllKeyboardResponses();
              this.jsPsych.pluginAPI.getKeyboardResponse({
                  callback_function: startBall,
                  valid_responses: [" "],
                  rt_method: "performance",
                  allow_held_key: false,
                  persist: false,
              });
          };

          let timestamp_ball_start;
          let last_timestamp_frame;
          let get_ball_pixels_per_second = function () {
              return Math.max(window.screen.width / 16, 50);
          }
          let ball_pixels_per_second;
          let monitor_width_px;
          let monitor_height_px;
          let rel_timestamp_ball_start;

          const startBall = () => {
              monitor_width_px = window.screen.width;
              monitor_height_px = window.screen.height;
              ball_pixels_per_second = get_ball_pixels_per_second();
              rel_timestamp_ball_start = performance.now();
              this.jsPsych.pluginAPI.getKeyboardResponse({
                  callback_function: submitBallPosition,
                  valid_responses: [" "],
                  rt_method: "performance",
                  allow_held_key: false,
                  persist: false,
              });
              timestamp_ball_start = performance.now();
              last_timestamp_frame = timestamp_ball_start;
              ball_pixels_per_second = get_ball_pixels_per_second();
              this.ball_animation_frame_id = requestAnimationFrame(animateBall);
          };

          const animateBall = (timestamp_frame) => {
              let dx_default = -2;
              let telapsed_msec = timestamp_frame - last_timestamp_frame
              let dx = -(telapsed_msec / 1000 * ball_pixels_per_second) || dx_default
              const x = parseInt(this.ball.style.left);
              this.ball.style.left = `${x + dx}px`;

              last_timestamp_frame = timestamp_frame
              this.ball_animation_frame_id = requestAnimationFrame(animateBall);
          };

          const submitBallPosition = () => {
              console.log("ball submitted")
              let rel_timestamp_submit = performance.now();

              cancelAnimationFrame(this.ball_animation_frame_id);

              let ball_position = accurateRound(getElementCenter(this.ball).x, 2)
              let square_position = accurateRound(getElementCenter(this.square).x, 2);

              let blindspot_trial_outcome = {
                  'ball_position_px': ball_position,
                  'square_position_px': square_position,
                  'ball_speed_px_per_sec': ball_pixels_per_second,
                  'monitor_width_px': monitor_width_px,
                  'monitor_height_px': monitor_height_px,
                  'rel_timestamp_ball_start': rel_timestamp_ball_start,
                  'reaction_time_msec': rel_timestamp_submit - timestamp_ball_start,
              }

              let i_cur_instance = trial_data['blindspot_response_data'].length - 1;
              trial_data["blindspot_response_data"][i_cur_instance].push(blindspot_trial_outcome);

              // Decrement counter
              this.blindspot_reps_remaining--;
              document.querySelector("#blindspot_rep_counter").textContent = Math.max(this.blindspot_reps_remaining, 0).toString();

              // Finish blind spot phase or perform another trial
              if (this.blindspot_reps_remaining <= 0) {
                  // Add continue button
                  finishBlindSpotPhase();
              } else {
                  resetAndWaitForBallStart();
              }
          };


          const finishBlindSpotPhase = () => {
              const angle = 13.5;
              const blindspot_response_data = trial_data['blindspot_response_data']
              let i_latest_instance = trial_data['blindspot_response_data'].length - 1;
              let ball_position_data = blindspot_response_data[i_latest_instance] // Array of each instance of the blind spot experiment. Will be length=1 if user does not elect to repeat the blind spot phase

              const nreps_collected = ball_position_data.length;
              let sum = 0;
              let n = 0;
              for (let i_rep = 0; i_rep < nreps_collected; i_rep++) {
                  let reported_ball_x = ball_position_data[i_rep]['ball_position_px']
                  let square_x = ball_position_data[i_rep]['square_position_px']
                  let dist = square_x - reported_ball_x
                  sum += dist
                  n += 1
              }

              let average_ball_to_square_distance_pixels = accurateRound(sum / n, 2);

              // Calculate viewing distance in mm
              const viewing_distance_pixels = average_ball_to_square_distance_pixels / Math.tan(deg_to_radians(angle));
              trial_data['outputs']["viewing_distance_px"] = accurateRound(viewing_distance_pixels, 2);

              if (trial.viewing_distance_report === "none") {
                  endTrial();
              } else {
                  showReport();
              }
          };

          function accurateRound(value, decimals) {
              return Number(Math.round(Number(value + "e" + decimals)) + "e-" + decimals);
          }

          function getElementCenter(el) {
              const box = el.getBoundingClientRect();
              return {
                  x: box.left + box.width / 2,
                  y: box.top + box.height / 2,
              };
          }

          const deg_to_radians = (degrees) => {
              return (degrees * Math.PI) / 180;
          };


          /** Create content for final report screen */
          const showReport = () => {
              let viewing_distance_px = trial_data['outputs']["viewing_distance_px"];
              let pixels_per_mm = trial_data['outputs']["pixels_per_mm"];

              let skip_report = false;
              if (isNaN(parseFloat(pixels_per_mm)) === true) {
                  skip_report = true;
              }
              if (isNaN(parseFloat(viewing_distance_px)) === true) {
                  skip_report = true;
              }

              if (skip_report === true) {
                  endTrial();
              } else {
                  let report_content = `
                          <div id="distance-report"  style="
                              background-color:#E4E4E4;
                              text-align: left; 
                              border-color: #7F7F7F; 
                              border-radius: 8px; 
                              max-width: 70%; 
                              width:auto; 
                              padding-left: 2%; 
                              padding-right: 2%; 
                              padding-top: 2%; 
                              padding-bottom: 2%;
                            display:flex;
                            align-items:center;
                            flex-direction:column;
                            margin: 0 auto;
                            ">
                            <div id="info-h">
                              ${trial.viewing_distance_report}
                            </div>
                            <button id="proceed" class="jspsych-btn">${trial.blindspot_done_prompt}</button>
                            <button id="redo_blindspot" class="jspsych-btn">${trial.redo_measurement_button_label}</button>
                          </div>`;

                  // Display data
                  display_element.innerHTML = `<div id="content" style="width: 900px; margin: 0 auto;"></div>`;
                  display_element.querySelector("#content").innerHTML = report_content;
                  let viewing_distance_mm = accurateRound(viewing_distance_px / pixels_per_mm, 2);
                  display_element.querySelector("#distance-estimate").innerHTML = `${Math.round(viewing_distance_mm / 10)} cm (${Math.round(viewing_distance_mm * 0.0393701)} inches)`;
                  display_element
                      .querySelector("#redo_blindspot")
                      .addEventListener("click", initiate_trial);
                  display_element.querySelector("#proceed").addEventListener("click", endTrial);
              }
          }

          const endTrial = () => {
              // finish trial
              trial_data.rt = Math.round(performance.now() - overall_start_time);
              // remove lingering event listeners, just in case
              this.jsPsych.pluginAPI.cancelAllKeyboardResponses();

              // Clear the display
              display_element.innerHTML = "";

              // Finish the trial
              this.jsPsych.finishTrial(trial_data);
          };

          function initiate_trial() {
              if (trial.credit_card_reps > 0) {
                  startCreditCardPhase()
              } else {
                  // Skip the credit card phase
                  generateBlindspotPhaseContent();
                  beginBlindSpotPhase();
              }
          }

          // Run trial
          const overall_start_time = performance.now();
          initiate_trial();
      }
  }

  VirtualChinrestPlugin.info = info;
  return VirtualChinrestPlugin;

})(jsPsychModule);