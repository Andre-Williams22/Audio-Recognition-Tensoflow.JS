const INPUT_SHAPE =[NUM_FRAMES, 232, 1];
let model;

// one frame is 23 ms
const NUM_FRAMES = 3;
let examples = []

//collect() associates a label with the output of recognizer.listen(). Since includeSpectrogram is true, 
//recognizer.listen() gives the raw spectrogram (frequency data) for 1 sec of audio, divided into 43 frames, 
//so each frame is ~23ms of audio

function collect(label) {
    if (recognizer.islistening()){
        return recognizer.stopListening();
    }
    if (label == null) {
        return;
    }
    recognizer.listen(async ({spectrogram: {frameSize, data}}) => {
//Since we want to use short sounds instead of words to control the slider, we are taking into consideration only the last 3 frames (~70ms):
        let vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
        //Finally, each training example will have 2 fields:
        examples.push({vals, label});
        document.querySelector('#console').textContent = `${examples.length} examples collected`;
    }, {
        overlapFactor: 0.999,
        includeSpectrogram: true,
        invokeCallbackOnNoiseAndUnknown: true
    });
    
    function normalize (x) {
    // And to avoid numerical issues, we normalize the data to have an average of 0 and a standard deviation of 1. 
    // In this case, the spectrogram values are usually large negative numbers around -100 and deviation of 10:
        const mean = -100
        const std = 10;
        return x.map(x => (x - mean) / std);
    }
}

// train() trains the model using the collected data.

// The training goes 10 times (epochs) over the data using a batch size of 16 (processing 16 examples at a time) 
// and shows the current accuracy in the UI
async function train() {
    toggleButtons(false);
    const ys = tf.oneHot(examples.map(e => e.label), 3);
    const xsShape = [examples.length, ...INPUT_SHAPE];
    const xs = tf.tensor(flatten(examples.map(e => e.vals)), xsShape);
   
    await model.fit(xs, ys, {
      batchSize: 16,
      epochs: 10,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          document.querySelector('#console').textContent =
              `Accuracy: ${(logs.acc * 100).toFixed(1)}% Epoch: ${epoch + 1}`;
        }
      }
    });
    tf.dispose([xs, ys]);
    toggleButtons(true);
   }

// BuildModel() defines the model architecture   
   function buildModel() {
    // Model Architecture
    //The model has 4 layers: a convolutional layer that processes the audio data (represented as a spectrogram), 
    // a max pool layer, a flatten layer, and a dense layer that maps to the 3 actions:
    model = tf.sequential();
    model.add(tf.layers.depthwiseConv2d({
      depthMultiplier: 8,
      kernelSize: [NUM_FRAMES, 3],
      activation: 'relu', // uses relu activation function
      inputShape: INPUT_SHAPE
    }));
    model.add(tf.layers.maxPooling2d({poolSize: [1, 2], strides: [2, 2]}));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({units: 3, activation: 'softmax'}));
// The input shape of the model is [NUM_FRAMES, 232, 1] where each frame is 23ms of audio containing 232 numbers that 
//correspond to different frequencies (232 was chosen because it is the amount of frequency buckets needed to capture the human voice).
// we are using samples that are 3 frames long (~70ms samples) since we are making sounds instead of 
//speaking whole words to control the slider

//compile our model to get it ready for training
    const optimizer = tf.train.adam(0.01);
    model.compile({
      optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });
   }
   // use Adam Optimizer a common optimizer used in deep learning, and `categoricalCrossEntropy` for loss, the standard loss function used for 
   //classification. In short, it measures how far the predicted probabilities (one probability per class) are from having 100% probability 
   //in the true class, and 0% probability for all the other classes. We also provide accuracy as a metric to monitor, which will give us the 
   //percentage of examples the model gets correct after each epoch of training.
   
   function toggleButtons(enable) {
    document.querySelectorAll('button').forEach(b => b.disabled = !enable);
   }
   
   function flatten(tensors) {
    const size = tensors[0].length;
    const result = new Float32Array(tensors.length * size);
    tensors.forEach((arr, i) => result.set(arr, i * size));

    return result;
   }
/*
let recognizer; 

function predictWord() {
    // Array of words the recognizer is trained to recognize and identify
    const words = recognizer.wordLabels();
    recognizer.listen(({scores}) => {
        // turn score into a list of (score, word) pairs.
        scores = Array.from(scores).map((s, i) => ({score: s, word: words[[i]]}));
        // Find the most probable word 
        scores.sort((s1, s2) => s2.score - s1.score);
        document.querySelector("#console").textContent = scores[0].word;
    }, {probabilityThreshold: 0.75});} // 0.75 means the model will fire when it feels like it's more than 75% confident
*/

//moveSlider() decreases the value of the slider if the label is 0 ("Left") , increases it if the label is 1 ("Right") 
//and ignores if the label is 2 ("Noise").
async function moveSlider(labelTensor) {
    const label = (await labelTensor.data())[0];
    document.getElementById('console').textContent = label;
    if (label == 2) {
      return;
    }
    let delta = 0.1;
    const prevValue = +document.getElementById('output').value;
    document.getElementById('output').value =
        prevValue + (label === 0 ? -delta : delta);
   }
   // Real Time Predicitions 
//listen() listens to the microphone and makes real time predictions. The code is very similar to the collect() method, 
//which normalizes the raw spectrogram and drops all but the last NUM_FRAMES frames. The only difference is that we also 
//call the trained model to get a prediction
   function listen() {
    if (recognizer.isListening()) {
      recognizer.stopListening();
      toggleButtons(true);
      document.getElementById('listen').textContent = 'Listen';
      return;
    }
    toggleButtons(false);
    document.getElementById('listen').textContent = 'Stop';
    document.getElementById('listen').disabled = false;
   
    recognizer.listen(async ({spectrogram: {frameSize, data}}) => {
      const vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
      const input = tf.tensor(vals, [1, ...INPUT_SHAPE]);
      const probs = model.predict(input);
      const predLabel = probs.argMax(1);
      await moveSlider(predLabel);
      tf.dispose([input, probs, predLabel]);
    }, {
      overlapFactor: 0.999,
      includeSpectrogram: true,
      invokeCallbackOnNoiseAndUnknown: true
    });
   }

async function app() {
    recognizer = speechCommands.create('BROWSER.FFT');
    await recognizer.ensureModeLoaded();
    // builds the model
    buildModel();

}


app();


// source https://codelabs.developers.google.com/codelabs/tensorflowjs-audio-codelab/index.html#0