# osu-beatmap-generator
CNN-Transformer hybrid model for osu! beatmap generation 

note: system requires ffmpeg library: 'sudo apt install ffmpeg'


# Plan
osu! beatmap generation is planned to be split into two separate problems:

## 1 - Onset Detection
Onset detection refers to determining which timestamps within an input audio file should have a hitsound (i.e. a corresponding hitsound from some hitobject)

The solution designed for this task involves generating mel-filtered and log-scaled spectrograms of the audio, then passing this data through [CNNs](#onset-detection---cnn) and an [Encoder-only transformer](#onset-detection---encoder-transformer) to produce binary outputs representing the identification of a hitobject onset

### Onset Detection - CNN

### Onset Detection - Encoder Transformer

## 2 - Object Creation
Object creation involves taking the aforementioned sequence of onsets created by the [Onset Detection model](#1---onset-detection), then producing a sequence of osu! hitobjects

The solution proposed for this tasks requires tokenizing osu objects. Due to the infinite nature of possible slider objects, the tokenizer developed represents objects with each respective component, along with specified beginning and end tokens to guide the model with explicit structure. 

Since rhythmic context is required, similar to Onset Detection data is passed through CNNs, then into an encoder transformer. A decoder transformer then auto-regressively generates hitobjects, using the encoder's output as musical context to condition the output.
