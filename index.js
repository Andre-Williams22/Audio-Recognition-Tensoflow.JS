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
       // Since we want to use short sounds instead of words to control the slider, we are taking into consideration only the last 3 frames (~70ms):
        let vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
        examples.push({vals, label});
        document.querySelector('#console').textContent = `${examples.length} examples collected`;
    }, {
        overlapFactor: 0.999,
        includeSpectrogram: true,
        invokeCallbackOnNoiseAndUnknown: true
    });
    
    function normalize (x) {
        const mean = -100
        const std = 10;
        return x.map(x => (x - mean) / std);
    }
}

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

async function app() {
    recognizer = speechCommands.create('BROWSER.FFT');
    await recognizer.ensureModeLoaded();

}

app();


