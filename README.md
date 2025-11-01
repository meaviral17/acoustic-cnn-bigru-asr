# 🧠 Acoustic CNN-BiGRU ASR with KenLM

End-to-end speech recognition system powered by:
- CNN + BiGRU acoustic model
- CTC loss training
- Beam search decoding
- 5-gram KenLM Language Model (trained on WikiText103)
- Optional EMA, SpecAugment, and curriculum learning

## 🏗 Setup

```bash
git clone https://github.com/<your-username>/acoustic-cnn-bigru-asr.git
cd acoustic-cnn-bigru-asr
pip install -r requirements.txt
