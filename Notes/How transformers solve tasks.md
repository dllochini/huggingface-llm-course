- **Wav2Vec2** for audio classification and automatic speech recognition (ASR)
- **Vision Transformer (ViT)** and **ConvNeXT** for image classification
- **DETR** for object detection
- **Mask2Former** for image segmentation
- **GLPN** for depth estimation
- **BERT** for NLP tasks like text classification, token classification and question answering that use an encoder
- **GPT2** for NLP tasks like text generation that use a decoder
- **BART** for NLP tasks like summarization and translation that use an encoder-decoder

1. two main approaches for training a transformer model
- **Masked language modeling (MLM)**:
    - used by encoder models(BERT)
    - randomly masks some tokens in the input
    - trains the model to predict the original tokens based on the *surrounding context* (looks at words both before and after the masked word)
- **Causal language modeling (CLM)**:
    - used by decoder models(GPT)
    - predicts the next token based on all *previous tokens* in the sequence.