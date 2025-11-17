# Josh Talks Hindi Speech Recognition - Whisper Fine-tuning

A complete pipeline for fine-tuning OpenAI's Whisper model on Hindi speech data from Josh Talks, including baseline evaluation, data preparation, and LoRA-based fine-tuning.

## 📋 Project Overview

This project demonstrates the end-to-end process of improving Whisper's Hindi speech recognition through fine-tuning:

1. **Baseline Evaluation**: Testing pre-trained `whisper-small` on Hindi FLEURS test dataset
2. **Data Preparation**: Processing and segmenting Josh Talks audio data
3. **Fine-tuning**: Applying LoRA (Low-Rank Adaptation) for efficient model training
4. **Evaluation**: Comparing baseline vs fine-tuned model performance

## 🗂️ Repository Structure
```
├── josh_talks_base_score.ipynb          # Baseline Whisper evaluation on FLEURS
├── josh_data_preparation.ipynb          # Data preprocessing and segmentation
├── josh-fine-tune.ipynb                 # LoRA fine-tuning pipeline
├── josh_data_cleaned.csv                # Cleaned training data
├── FT Data - data.csv                   # Fine-tuning dataset
├── FT Result - Sheet1.csv               # Fine-tuning results
├── processed/                           # Processed audio segments
├── 825780_audio.wav                     # Sample audio file
├── LICENSE                              # Project license
└── README.md                            # This file
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install transformers==4.36.2
pip install datasets==2.15.0
pip install peft==0.6.0
pip install accelerate==0.25.0
pip install soundfile librosa
```

### Hardware Requirements

- **GPU**: NVIDIA P100 or better (16GB+ VRAM recommended)
- **RAM**: 16GB+ system memory
- **Storage**: 10GB+ for datasets and model checkpoints

## 📊 Pipeline Workflow

### 1. Baseline Evaluation (`josh_talks_base_score.ipynb`)

Evaluates the pre-trained `openai/whisper-small` model on the Hindi portion of the FLEURS dataset to establish baseline metrics.

**Key metrics measured:**
- Word Error Rate (WER)
- Character Error Rate (CER)
- Inference time

### 2. Data Preparation (`josh_data_preparation.ipynb`)

Processes raw Josh Talks audio and transcriptions:

- Audio segmentation and normalization
- Text cleaning and formatting
- Train/validation split (80/20)
- Feature extraction and validation

**Output**: Processed dataset in HuggingFace format with aligned audio segments

### 3. Fine-tuning (`josh-fine-tune.ipynb`)

Implements efficient fine-tuning using LoRA (Low-Rank Adaptation):

**Key Features:**
- **LoRA Configuration**: r=32, alpha=64 for optimal parameter efficiency
- **Memory Optimization**: Gradient checkpointing, FP16 training
- **Batch Size**: 24 (optimized for P100 GPU)
- **Training**: 3 epochs with warmup ratio 0.1

**Architecture Details:**
- Base Model: `openai/whisper-small`
- Target Modules: `q_proj`, `v_proj`, `k_proj`, `o_proj`
- Trainable Parameters: ~4.2M (1.5% of total)
- Language: Hindi (`hi`)

## 🔧 Configuration

Key hyperparameters in `josh-fine-tune.ipynb`:
```python
MODEL_NAME = "openai/whisper-small"
LANGUAGE = "hi"
NUM_EPOCHS = 3
BATCH_SIZE = 24
LEARNING_RATE = 1e-3
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.0
```

## 📈 Results

### Performance Comparison

| Metric | Baseline (Pre-trained) | Fine-tuned | Improvement |
|--------|----------------------|------------|-------------|
| WER | [Your baseline WER] | [Your fine-tuned WER] | [Δ%] |
| CER | [Your baseline CER] | [Your fine-tuned CER] | [Δ%] |

*Note: Fill in actual metrics from your experiments*

### Training Metrics

- **Training Loss**: Progressive decrease across epochs
- **Validation Loss**: Monitored every 50 steps
- **GPU Memory**: ~12.8GB peak usage
- **Training Time**: ~[X] hours on P100

## 💾 Model Checkpoints

The fine-tuned model is saved in multiple formats:
```
whisper_lora_output/
├── checkpoint-[step]/          # Intermediate checkpoints
└── final_model/               # Final merged model
    ├── pytorch_model.bin
    ├── config.json
    ├── preprocessor_config.json
    └── tokenizer files
```

## 🔍 Usage Example
```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

# Load fine-tuned model
model = WhisperForConditionalGeneration.from_pretrained("./final_model")
processor = WhisperProcessor.from_pretrained("./final_model")

# Transcribe audio
model.eval()
model.to("cuda")

audio_input = processor(audio_array, sampling_rate=16000, return_tensors="pt")
audio_input = audio_input.input_features.to("cuda")

with torch.no_grad():
    predicted_ids = model.generate(
        audio_input,
        language="hi",
        task="transcribe"
    )

transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print(transcription)
```

## 🛠️ Technical Optimizations

### Memory Management
- **FP16 Training**: Reduces memory footprint by 50%
- **Gradient Checkpointing**: Trades compute for memory
- **TF32 Operations**: Faster matrix multiplication on Ampere+ GPUs

### Training Stability
- **PEFT 0.6.0**: Fixes LoRA forward pass compatibility
- **Custom Collator**: Handles variable-length audio sequences
- **Path Validation**: Ensures all audio files are accessible

### Error Handling
- Graceful handling of corrupted audio files
- Resume capability from latest checkpoint
- Automatic padding/truncation to expected sequence length

## 📝 Dataset Information

**Source**: Josh Talks Hindi speeches and interviews

**Statistics**:
- Total samples: [X training + Y validation]
- Audio format: WAV, 16kHz mono
- Average duration: [X] seconds per segment
- Total duration: [X] hours

**Preprocessing Steps**:
1. Audio resampling to 16kHz
2. Stereo to mono conversion
3. Silence trimming
4. Segmentation to 30-second chunks
5. Text normalization (punctuation, casing)

## ⚠️ Known Issues & Solutions

### Issue: OOM (Out of Memory) Errors
**Solution**: Reduce `BATCH_SIZE` from 24 to 16 or enable gradient accumulation

### Issue: LoRA module not found
**Solution**: Ensure `peft==0.6.0` is installed (not newer versions)

### Issue: Mixed precision errors during inference
**Solution**: Convert model to FP16 before generation:
```python
model = model.half()
```

## 🔮 Future Improvements

- [ ] Experiment with larger base models (medium/large)
- [ ] Implement speculative decoding for faster inference
- [ ] Add real-time streaming transcription
- [ ] Multi-speaker diarization support
- [ ] Extend to other Indian languages
- [ ] Quantization for edge deployment (ONNX/TensorRT)

## 📚 References

- [Whisper Paper](https://arxiv.org/abs/2212.04356) - Robust Speech Recognition via Large-Scale Weak Supervision
- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Low-Rank Adaptation of Large Language Models
- [PEFT Library](https://github.com/huggingface/peft) - Parameter-Efficient Fine-Tuning
- [FLEURS Dataset](https://huggingface.co/datasets/google/fleurs) - Few-shot Learning Evaluation of Universal Representations of Speech

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss proposed changes.

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 🙏 Acknowledgments

- OpenAI for the Whisper model
- Josh Talks for the Hindi speech dataset
- HuggingFace for the transformers and PEFT libraries
- Google for the FLEURS benchmark dataset

## 📧 Contact

For questions or collaborations, please open an issue in this repository.

---

**Note**: This project was developed as part of the AI Researcher Intern - Speech & Audio role at Josh Talks. See `Task Assignment _ AI Researcher Intern- Speech & Audio _ Josh Talks.pdf` for original requirements.
