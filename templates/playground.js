// Protocol and Host information ----------------------------

const protocol = window.location.protocol;
const host = window.location.host;

let baseURL;
// let baseStreamURL;
const AWSServerIP = "15.229.156.138";
const AWSServerURL = `http://${AWSServerIP}`;
const cloudRunServerURL = 'https://video-analytics-oayt5ztuxq-ue.a.run.app'; // cloud run server

// Check if the host is a server 
// Check if the host is hostgator or the local file system
if (host.includes("octacity.org") | host == "") {
    baseURL = AWSServerURL;
    // baseStreamURL = AWSServerURL;
}
else {
  baseURL = ".";
  // baseStreamURL = ".";
}

// Log the final base URL and host
console.log('Host:', host);
console.log('Base URL:', baseURL);
// console.log('Base Streaming URL:', baseStreamURL);

// Retrieve the DOM elements
const sourceInput = document.getElementById('source-input');
const modelInput = document.getElementById('model-input');
// Add more variables for other input fields

const runButton = document.getElementById('run-button');
const videoPlayer = document.getElementById('video-player');
const resultsContainer = document.getElementById('results-container');

// Function to update the video source
async function updateVideoSource(sourceUrl) {
  if (!sourceUrl.includes("stream=true")) {
    const res = await fetch(sourceUrl).then(res => res.json())
    return res
  }
  else {
    videoPlayer.src = sourceUrl;
    return null
  }
  
  // videoPlayer.load();
}

// Function to display the inference results
function displayResults(results) {
  // Log results
  console.log("Results: ", results);
    
  // Clear previous results
  resultsContainer.innerHTML = '';

  // Iterate through the results and create HTML elements to display them
  for (const result of results) {
    const resultElement = document.createElement('div');
    resultElement.textContent = JSON.stringify(result, null, 4);
    resultsContainer.appendChild(resultElement);
  }
}

// Convert an object into a URL query string
function objectToQueryString(obj) {
  const keyValuePairs = [];
  for (const key in obj) {
    if (obj.hasOwnProperty(key)) {
      const value = obj[key];
      if (value !== '' && !Number.isNaN(value)) {
        const encodedKey = encodeURIComponent(key);
        const encodedValue = encodeURIComponent(value);
        keyValuePairs.push(`${encodedKey}=${encodedValue}`);
      }
    }
  }
  return keyValuePairs.join('&');
}

// Example usage:
// const params = {
//   source: 'http://example.com/video.mp4',
//   model: 'yolov8s.pt',
//   max_frames: 10,
//   seconds: 60,
//   process: 'bigquery',
//   annotator: 'fps',
//   conf: NaN,
//   iou: 0.5,
//   max_det: NaN,
//   post_url: null,
//   post_scheme: undefined
// };

// const queryString = objectToQueryString(params);
// console.log(queryString);

// Retrieve the DOM elements for additional parameters
const postUrlInput = document.getElementById('post-url-input');
const postSchemeInput = document.getElementById('post-scheme-input');
const taskSelect = document.getElementById('task-input');
const maxFramesInput = document.getElementById('max-frames-input');
const secondsInput = document.getElementById('seconds-input');
const executionSecondsInput = document.getElementById('execution-seconds-input');
const logSecondsInput = document.getElementById('log-seconds-input');
const fpsInput = document.getElementById('fps-input');
const processSelect = document.getElementById('process-input');
const annotatorSelect = document.getElementById('annotator-input');
const streamCheckbox = document.getElementById('stream-input');
const objectsInput = document.getElementById('objects-input');
const classesInput = document.getElementById('classes-input');
const confInput = document.getElementById('conf-input');
const iouInput = document.getElementById('iou-input');
const maxDetInput = document.getElementById('max-det-input');
const vidStrideInput = document.getElementById('vid-stride-input');
const imgszInput = document.getElementById('imgsz-input');
const deviceSelect = document.getElementById('device-input');
const trackerSelect = document.getElementById('tracker-input');
const persistCheckbox = document.getElementById('persist-input');
const augmentCheckbox = document.getElementById('augment-input');
const saveCheckbox = document.getElementById('save-input');
const showCheckbox = document.getElementById('show-input');
const verboseCheckbox = document.getElementById('verbose-input');


// Update the function to prepare the request parameters based on user input
async function makeAPIRequest() {
  // Prepare the request parameters based on user input
  const query = {
    source: sourceInput.value,
    post_url: postUrlInput.value,
    post_scheme: postSchemeInput.value,
    model: modelInput.value,
    task: taskSelect.value,
    max_frames: parseInt(maxFramesInput.value),
    seconds: parseInt(secondsInput.value),
    execution_seconds: parseInt(executionSecondsInput.value),
    log_seconds: parseInt(logSecondsInput.value),
    fps: parseInt(fpsInput.value),
    process: processSelect.value,
    annotator: annotatorSelect.value,
    stream: streamCheckbox.checked,
    objects: objectsInput.value,// .split(',').map(obj => obj.trim()),
    classes: classesInput.value, // .split(',').map(cls => parseInt(cls.trim())),
    conf: parseFloat(confInput.value),
    iou: parseFloat(iouInput.value),
    max_det: parseInt(maxDetInput.value),
    vid_stride: parseInt(vidStrideInput.value),
    imgsz: parseInt(imgszInput.value),
    device: deviceSelect.value,
    tracker: trackerSelect.value,
    persist: persistCheckbox.checked,
    augment: augmentCheckbox.checked,
    save: saveCheckbox.checked,
    show: showCheckbox.checked,
    verbose: verboseCheckbox.checked
  };
    
  const queryString = objectToQueryString(query);
  const queryURL = `${baseURL}/track?${queryString}`;
    
  console.log("QUERY PARAMETERS:", query);
  console.log("QUERY URL:", queryString);
  
  const results = await updateVideoSource(queryURL)
  if (results)
      displayResults(results)
    
  // Make the API request and handle the response
//   fetch('/track', {
//     method: 'POST',
//     body: JSON.stringify(query)
//   })
//     .then(response => response.json())
//     .then(results => displayResults(results))
//     .catch(error => console.error(error));
}

// Event listener for the run button click
runButton.addEventListener('click', () => {
  makeAPIRequest();
});

// Call the function to update the video source with the provided URL
// updateVideoSource('http://example.com/video.mp4');