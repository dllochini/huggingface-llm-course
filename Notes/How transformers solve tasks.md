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

2. Text Generation
- GPT-2 breaks text into tokens using Byte Pair Encoding (BPE)
- Each token is converted into a token embedding (numbers)
- Positional encodings are added so the model knows the order of tokens
- The embeddings are passed through multiple decoder blocks
- Each decoder block uses masked self-attention
- Masked self-attention allows GPT-2 to look only at previous tokens, not future ones
- This is different from BERT, which uses a [MASK] token
- After the decoder blocks, the model produces a final hidden state
- A language modeling head converts this into logits (scores for each token)
- The model predicts the next token in the sequence
- During training, outputs are shifted by one position
- Cross-entropy loss is used to measure prediction error

3. Text Classification
- Assigns predefined labels to text (e.g., sentiment analysis, topic classification, spam detection).
- **BERT Overview**:
  - Encoder-only transformer model.
  - Uses **deep bidirectional attention** to understand context from both left and right.
- **Tokenization & Special Tokens**:
  - Uses **WordPiece tokenization**.
  - `[CLS]` token added at the beginning (used for classification).
  - `[SEP]` token separates sentences.
  - **Segment embeddings** indicate sentence A or B in sentence pairs.
- **Pretraining Objectives**:
  - **Masked Language Modeling (MLM)**:
    - Random tokens are masked.
    - Model predicts masked words using context from both sides.
  - **Next Sentence Prediction (NSP)**:
    - Predicts whether sentence B follows sentence A.
    - Binary classification: `IsNext` or `NotNext`.
- **Model Architecture**:
  - Input embeddings pass through multiple **encoder layers**.
  - Outputs final hidden states.
- **Text Classification with BERT**:
  - A **sequence classification head** (linear layer) is added on top.
  - Uses `[CLS]` token output for prediction.
  - **Cross-entropy loss** used to train the model.
- **Fine-tuning**:
  - Pretrained BERT (or DistilBERT) can be fine-tuned for specific classification tasks.

4. Token Classification
- Assigns a label to each token in a sentence.
- Common use case: **Named Entity Recognition (NER)**.
- **Using BERT for Token Classification**:
  - Add a **token classification head** on top of the base BERT model.
  - The head is a **linear layer** applied to each token’s final hidden state.
- **Prediction Process**:
  - Linear layer converts hidden states into **logits** for each token.
  - Each token is classified independently.
- **Training**:
  - **Cross-entropy loss** is computed between predicted logits and true labels for each token.
- **Fine-tuning**:
  - Pretrained models like **DistilBERT** can be fine-tuned for token classification and inference.

5. Question answering
-  finds the answer to a question within a given context or passage.
- ***Using BERT for Question Answering**:
    - Add a **span classification head** on top of the base BERT model.
    - The head is a **linear layer** applied to final hidden states.
- **Prediction Process**:
    - Model predicts **start logits** and **end logits** for each token.
    - These logits represent the possible beginning and ending of the answer span.
- **Training**:
    - **Cross-entropy loss** is calculated between predicted logits and true start/end positions.
    - Model learns to select the most likely answer span in the text.

6. **Summarization**:
 - Condenses long text into a shorter version.
 - Preserves key information and overall meaning.
- **Models for Summarization**:
  - Uses **encoder–decoder (sequence-to-sequence)** models.
  - Common models: **BART** and **T5**.
- **BART Encoder**:
  - Architecture similar to **BERT**.
  - Takes **token embeddings** and **positional embeddings** as input.
  - Pretrained by **corrupting input text** and learning to reconstruct it.
- **Pretraining Strategy**:
  - Supports multiple corruption types.
  - **Text infilling** works best:
    - Replaces spans of text with a single `[MASK]` token.
    - Teaches the model to predict both **missing tokens and span length**.
- **Encoder Output**:
  - Produces final hidden states.
  - No final feedforward layer for word prediction (unlike BERT).
- **Decoder**:
  - Uses encoder outputs as context.
  - Predicts masked and unmasked tokens to reconstruct original text.
- **Language Modeling Head**:
  - Applies a linear layer to decoder outputs to produce **logits**.
- **Training**:
  - Uses **cross-entropy loss**.
  - Labels are the **tokens shifted one position to the right**.

7. Translation:
- Converts text from one language to another.
- Preserves the original meaning.
- A **sequence-to-sequence** task.
- **Models for Translation**:
 - Uses **encoder–decoder** models.
 - Common models: **BART** and **T5**.
- **BART for Translation**:
 - Adds a **separate, randomly initialized source encoder**.
 - This encoder maps the **source language** into representations suitable for decoding.
- **Encoding Process**:
 - Source encoder embeddings are passed to the **pretrained BART encoder**.
 - Original word embeddings are replaced by source encoder outputs.
- **Training Strategy**:
 - **Step 1**:
   - Freeze BART model parameters.
   - Train only the **source encoder**, positional embeddings, and input embeddings.
 - **Step 2**:
   - Train **all model parameters together**.
- **Multilingual Extension**:
 - **mBART** is a multilingual version of BART.
 - Pretrained on **multiple languages** for translation tasks.

8. Speech and Audio Tasks:
 - Deal with audio input instead of text or images.
 - Pose unique challenges for Transformer models.
- **Whisper Model**:
 - An **encoder–decoder (sequence-to-sequence) Transformer**.
 - Pretrained on **680,000 hours of labeled audio data**.
 - Supports **zero-shot performance** across many languages.
- **Why Whisper Works Well**:
 - Large-scale, diverse pretraining enables strong generalization.
 - Can perform tasks **without additional fine-tuning**.
- **Model Architecture**:
 - **Encoder**:
   - Converts raw audio into a **log-Mel spectrogram**.
   - Processes spectrogram using a Transformer encoder.
 - **Decoder**:
   - Autoregressively predicts **text tokens**.
   - Uses encoder outputs and previous tokens.
   - Special tokens guide tasks like transcription, translation, or language detection.
- **Pretraining Strategy**:
 - Uses **weakly supervised data** collected from the web.
 - Diversity of data improves robustness to accents, languages, and tasks.
- **Usage**:
 - Can be used **out of the box** for zero-shot inference.
 - Can also be **fine-tuned** for tasks like ASR or speech translation.
- **Key Innovation**:
 - Training on an **unprecedented scale of diverse audio data**.
 - Enables strong performance without task-specific fine-tuning.

9. Automatic Speech Recognition (ASR)
- Converts spoken audio into written text.
- **Using Whisper for ASR**:
    - Uses the full **encoder–decoder** architecture.
    - **Encoder** processes audio input.
    - **Decoder** generates the transcript **token by token** (autoregressively).
- **Training / Fine-tuning**:
  - Uses a **sequence-to-sequence loss** (typically cross-entropy).
  - Learns to predict correct text tokens from audio input.
- **Inference**:
  - The easiest way to run ASR is via a **pipeline**.
  - Handles preprocessing, model inference, and decoding automatically.
- **Output**:
  - Returns a dictionary containing the transcribed text.
- **Key Advantage**:
  - Whisper works well **out of the box**.
  - Can be further **fine-tuned** for better performance on specific datasets.

10. Computer vision
- there are 2 ways to approach computer vision tasks
    1. Split an image into a sequence of patches and process them in parallel with a Transformer.
    2. Use a modern CNN, like ConvNeXT, which relies on convolutional layers but adopts modern network designs.
- ViT and ConvNeXT - image classification
- DETR - object detection
- Mask2Former - segmentation
- GLPN -  depth estimation