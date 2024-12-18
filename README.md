# RedditGPT Tutorial: Train a GPT Model on Reddit Comments

Welcome to **RedditGPT**, a hands-on framework for learning how to train and fine-tune GPT models. In this tutorial, you’ll use **Reddit comments** as your dataset to train a language model from scratch or fine-tune an existing one.

This guide is tailored for running on **Amazon SageMaker Studio** using a **g5.xlarge instance** with an **NVIDIA A10 GPU**. By the end, you’ll understand the basic steps of data preparation, training, fine-tuning, and generating text samples.

---

## Learning Objectives

1. Understand how to preprocess text data for GPT models.
2. Learn to train and fine-tune a GPT model using Python.
3. Gain experience generating text samples from your trained model.
4. Explore how to tweak configurations for better results.

---

## Step 1: Setup and Installation

### Prerequisites
1. Make sure you're using **Amazon SageMaker Studio** with a **g5.xlarge instance**.
2. Ensure Python 3.11 (or higher) is installed in your environment.

### Install Required Libraries
Run the following command in your SageMaker notebook terminal:

```bash
conda create --name RedditGPT_python311 -y python=3.11 ipykernel
source activate RedditGPT_python311;
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

These libraries will provide the tools necessary for training and fine-tuning GPT models.

---

Below is a revised version of the README that includes both the Fortnite and Valorant Reddit comment datasets as options for training your GPT model.

---

## Step 2: Prepare Your Dataset

You can choose between the Fortnite and Valorant Reddit comment datasets. After selecting a dataset, run its preparation script to preprocess and tokenize the data.

For **Fortnite**:
```bash
python data/fortnite_comments/prepare.py
```

For **Valorant**:
```bash
python data/valorant_comments/prepare.py
```

This will generate two files:
- `train.bin` (training data)
- `val.bin` (validation data)

These binary files store tokenized Reddit comments, ready for model training.

---

## Step 3: Training Your GPT Model

### Choose Your Configuration

Depending on the dataset you selected, use the corresponding training configuration file. Both configurations are optimized for a single A10 GPU and have similar parameters, such as model size and training time.

- For Fortnite:
  ```bash
  python train.py config/train_fortnite_comments.py
  ```

- For Valorant:
  ```bash
  python train.py config/train_valorant_comments.py
  ```

**Key Model Details:**
- **Model Architecture**: 6 layers, 6 attention heads, 384 features
- **Context Length**: 256 tokens
- **Training Time**: ~8-10 minutes for initial results on a single A10 GPU.

During training, you’ll see logs showing progress, including validation loss.

---

## Step 4: Generating Text Samples

Once training is complete, you can generate text samples from the model using the `sample.py` script. Just point `--out_dir` to your chosen model’s output directory:

For Fortnite:
```bash
python sample.py --out_dir=out-fortnite-comments
```

For Valorant:
```bash
python sample.py --out_dir=out-valorant-comments
```

### Example Output

Sample text from a Valorant-trained model might look like this:

```text
Jet is the best duelist on the way to win rate as it happens. Not saying you should be able to always get any effect same character. Maybe the more you will lose the game and sometimes you still get angry?
```

To start generation with a specific prompt:
```bash
python sample.py \
    --start="What do you think about the new update?" \
    --num_samples=3 --max_new_tokens=100
```

You can also try different prompts and parameters to explore the model’s behavior.

## Step 5: Fine-tuning a Pretrained Model

Fine-tuning allows you to build on an existing GPT model, such as GPT-2, by adapting it to your specific dataset.

1. **Prepare Your Data**: Ensure `train.bin` and `val.bin` are ready as described in Step 2.
2. **Modify Configurations**: Open `config/finetune_reddit_comments.py` and update parameters such as learning rate or model size.
3. **Start Fine-tuning**:

   ```bash
   python train.py config/finetune_reddit_comments.py
   ```

Fine-tuning typically takes less time than training from scratch. Use the same `sample.py` script to generate outputs from your fine-tuned model.

---

## Step 6: Tweaking and Experimentation

Here are some ways to customize and improve your model:
- **Model Size**: Adjust layers, heads, or features in the configuration file.
- **Training Data**: Use a larger or more diverse dataset for better results.
- **Hyperparameters**: Experiment with learning rates, batch sizes, and context lengths.
- **Inference**: Use `max_new_tokens` or `temperature` in the sampling script to control the output length and creativity.

---

## Troubleshooting Tips

- **Memory Errors**: Reduce the `batch_size` or `context_length` in the configuration files.
- **Slow Training**: Double-check GPU utilization in SageMaker Studio and optimize code if necessary.
- **Unclear Outputs**: Fine-tune with more data or experiment with hyperparameters.

---

## Summary

By completing this tutorial, you’ve:
1. Preprocessed Reddit comment data for GPT training.
2. Trained and fine-tuned a GPT model on a single GPU.
3. Generated and analyzed text samples.
4. Explored ways to improve and customize your model.

---

## Class Activity

### Task 1: Train Your Own RedditGPT
1. Use the provided data and configurations to train a small GPT model.
2. Generate text samples and share them with the class.

### Task 2: Fine-tune RedditGPT
1. Modify `config/finetune_reddit_comments.py` to use GPT-2 as a base model.
2. Fine-tune the model on the same dataset and compare outputs with classmates.

---

## Additional Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [PyTorch Official Guide](https://pytorch.org/tutorials/)
- [OpenAI GPT Model Paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

---