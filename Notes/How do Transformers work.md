1. **language models** - transformer model that have been trained on large amounts of raw text in a self-supervised fashion
2. **Self-supervised learning** -  type of training in which the objective is automatically computed from the inputs of the model.(humans are not needed to label data)
3. **transfer learning / fine-tuning** -  
4. **causal language modeling** - model is predicting the next word in a sentence having read the n previous words
5. **masked language modeling** - model predicts a masked word in the sentence.
6.  [ML CO2 Impact](https://mlco2.github.io/impact/) or [Code Carbon](https://codecarbon.io) - tools to evaluate the carbon footprint of your model's training.
7. **pretraining** - act of training a model from scratch.
- weights are randomly initialized
- training starts without any prior knowledge
- very large amounts of data (training can take up to several weeks)
8. **Fine-tuning** - the training done after a model has been pretrained
9.  the knowledge the pretrained model has acquired is “transferred” -> **transfer learning**
- lower time,
- lower data,
- lower financial and
- lower environmental costs
10. model has 2 blocks
- *Encoder* - recieves an input, builds representation of it.
- *Decoder* - uses encoder’s representation (features) + other inputs -> generate a target sequence
11. these can be used independently.
- **Encoder-only model** - Good for tasks that require understanding of the input (sentence classification & NER)
- **Decoder-only models** - Good for generative tasks(text generation)
- **Encoder-decoder models**/**sequence-to-sequence models** - Good for generative tasks (translation/summarization)
12. **Attention layers**
- key feature of transformer models
- tells the model to pay specific attention to certain words in the sentence you passed it (and more or less ignore the others) when dealing with the representation of each word.
13. **Architecture** - This is the skeleton of the model
14. **Checkpoints** -  weights that will be loaded in a given architecture
15. **Model** -  This is an umbrella term that isn’t as precise as “architecture” or “checkpoint”.