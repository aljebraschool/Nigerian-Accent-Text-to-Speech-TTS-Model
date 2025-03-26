# Nigerian-Accent-Text-to-Speech-TTS-Model

## Project Overview
This project fine-tunes a pre-trained SpeechT5 model to generate Nigerian-accented English speech. The model is trained using a dataset of 500 Nigerian English speech samples. The primary goal is to adapt an existing TTS model to better capture the pronunciation and rhythm characteristic of Nigerian speakers.

## Prerequisites
- Python 3.10.16
- Google Colab (recommended)

## Required Libraries:
- soundfile
- librosa
- pandas
- tqdm
- tensorflow
- torch
- tf-keras
- datasets
- speechbrain
- numpy
- torchaudio
- accelerate
- Transformers (HuggingFace)
Find the full list in requirements.txt file

## Dataset Structure
The dataset consists of 500 audio-text pairs. Each sample includes:
- A transcript of Nigerian English text.
- A corresponding speech audio file, resampled to 16kHz and converted to mono.
- Speaker annotations for better generalization across different Nigerian accents.

  tts_data/
│
├── pleshy_1/
│   ├── recorder.tsv
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
│
├── pleshy_3/
│   ├── recorder.tsv
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...

### Preprocessing Steps
Preprocessing involves:
1. **Resampling audio to 16kHz**: Ensures uniform sample rate.
2. **Converting stereo to mono**: Standardizes audio format.
3. **Tokenizing text**: Prepares text for model input.
4. **Aligning speech and text data**: Ensures paired training.

## Preprocessing Steps

## Audio Resampling:
- Convert all audio files to 16kHz sample rate
- Convert stereo to mono if necessary
- Use high-quality Kaiser Best resampling method


## Metadata Preparation:

- Create a standardized metadata CSV file
- Columns: audio_path, text, speaker_id

## Preprocessing Code
- The preprocessing involves two main functions:
  - accurate_resample_audio(): Resample audio files
  - batch_resample_and_update_metadata(): Process multiple speaker directories

The preprocessing script is located in `preprocess_audio_and_metadata.ipynb`.

## Installation

1. Clone the repository:
   ``` bash
   git clone https://github.com/yourusername/nigerian-english-tts.git
   cd nigerian-english-tts
   ```
2. Install dependencies:
   ``` bash
    pip install -r requirements.txt
   ```

## Training
- Model Selection
  - Recommended: Use pre-trained models from HuggingFace Model Hub
  - Suggested Model: SpeechT5 model

## Training on Google Colab (or PC WITH GPU)
  - Upload your dataset
  - Use transfer learning/fine-tuning approach
  - Train on the 500 sample Nigerian English dataset


The model is fine-tuned using the Hugging Face `transformers` library with the following training arguments:

```python
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./speecht5_tts_nigerian_accent",
    
    # Batch Size Optimization
    per_device_train_batch_size=4,  # Slightly increased
    per_device_eval_batch_size=2,   # Matched or reduced
    gradient_accumulation_steps=2,  # Keep for virtual batch size increase
    
    # Learning Parameters
    learning_rate=5e-5,
    warmup_steps=500,
    max_steps=4000,
    
    # Performance and Memory Optimization
    gradient_checkpointing=True,
    fp16=True,
    
    # Evaluation Strategy
    evaluation_strategy="steps",
    save_steps=1000,
    eval_steps=1000,
    logging_steps=100,
    
    # Model Saving and Reporting
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    greater_is_better=False,
    label_names=["labels"],
    
    # Hugging Face Hub
    push_to_hub=True,
    
    # Additional Optimizations for Single GPU
    dataloader_num_workers=2,
    dataloader_pin_memory=True
)
```
## Training Progress
![image](https://github.com/user-attachments/assets/4e4eb1be-3a68-4afe-a8a9-1249728a8c53)
The final model checkpoint is uploaded to the Hugging Face Hub.

## Model Deployment
- The fine-tuned model can be loaded from Hugging Face as follows:

  ``` bash
  from transformers import SpeechT5ForTextToSpeech
  model = SpeechT5ForTextToSpeech.from_pretrained("aljebra/speecht5_tts_nigerian_accent")
  
  ```
## Limitations
- Limited Training Data: With only 500 samples, the model has limited exposure to the full variability of Nigerian-accented English. This leads to relatively high training loss and may impact the quality and generalization of the generated speech.
- High Model Loss: The observed training and validation losses remain higher than desired, likely due to the data scarcity. In real-world applications, increasing the dataset size could significantly improve performance.
- Accent Nuances: Some subtle accent nuances might not be fully captured given the limited diversity in the dataset.
- Compute Resources: Although optimizations have been applied, training on more extensive datasets would require more compute resources.

## Contributing
Feel free to open issues and submit pull requests if you find any improvements.

## License

This project is released under the MIT License.
