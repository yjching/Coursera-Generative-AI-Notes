# Generative AI with Large Language Models

## Week 1

Introduction:

* input data for LLMs = prompts, input into a context window ~1k words
* output = completion, act of generating text = inference
* augmenting LLMs to external databases, or connecting with APIs <--
* started with RNNs, now transformers after Attention Is All You Need
* transformers can be scaled + pay attention
* What does that mean? compared to RNNs which learnt the relevance of words next to each other, transformers learn relevance of words to every other word in a sentence and apply weights
* basic architecture:
  * inputs tokenized
  * embedding layer that creates mutli-dimensional vector space where each token is placed in relation to one another, encoded in meaning and context
  * positional embedding is done to not lose the position of the word in the sentence
  * self-attention layer (encoder + decoder) that analyses the relationship between tokens
  * multi-headed self-attention means multiple sets of weights are learnt independently of one another
  * weights are random - like CNN you don't specify what the attention heads learn
  * outputs passed through fully connected feed forward network which produces a vector of logits proportional to probability of each token
  * then passed through softmax layer where they are normalized into probability score of each word
  * Input -> Embedding -> Encoder -> Decoder, then input -> embedding -> decoder -> output
* encoder will encode prompts with contextual understanding and passes into decoder, which uses this understanding to generate new tokens
* bert = encoder only model
* gpt = decoder only
* prompt engineering is possible because of in-context learning
* when several examples, few-shot learning fails then the model should be fine tuned
* configuration parameters != training parameters, these allow the model output during inference to be changed
* greedy vs random for the output from softmax - if we choose the highest probability (greedy), chance of repeated words so we randomly sample - but be careful since this can induce incorrect, nonsensical answers
* types of sampling:
  * top-k - select output from the top-k results after applying random-weighted strategy
  * top-p limit to predictions where top-ranked consecutive rsults do not exceed cumulative p
* temperature is a scaling factor that impacts the probability distribution (i.e. higher temp = more randomness)
* GAI project lifecycle:
  * define use case (business problem)
  * choose existing model or pretrain own (develop)
  * evaluate through prompt engineering, fine-tuning, and human feedback (assessment)
  

LLM pre-training and scaling laws:

* roughly 1/3% of data is usable after data quality
* auto-encoding/encoding only - Masked Language Modelling
  * mask tokens in sentence to try and predict masked tokens to re-construct original sentence
  * used for sentiment analysis, named entity recognition, word classification
* decoder only/autoregressive - predict next token based on previous sequence of tokens
  * mask the input sequence and can only see the input tokens leading up to the token in question then iterates for the tokens one by one
  * text generation and other emergent behaviour because of zero-shot learning capabilities
* sequence-to-sequence/encoder + decoder models
  * T5 uses span corruption to mask sequences of tokens, replacing with a sentinel token
  * decoder then reconstructs the masked sequences
  * translation, summarisation, question answering
* quantization - from 32bit to 16bit to save on memory resource
* bfloat16 as an alternative to fp16, truncated 32-bit float (but not suited for integer calcualtions)
* objective of pretraining is to maximise model performance through minimising loss
* chinchilla paper - optimal training size is 20x larger than model params
* domain adaptation required for areas which is not using common day-to-day language


## Week 2

Fine-tuning LLMs with instruction:
* fine-tuning is a supervised learning process - use labelled dataset of prompt-completion pairs to update the weights of the LLM
* based of what task to tune, generate the appropriate prompt/completion e.g. for translation, have "Translate this sentence to ..." prompt with appropriate completion
* split into training, test, validation
* calculate loss and update the model weights using backprop
* often 500-1k for fine-tuning on a simple task is needed
* but can lead to catastrophic fine-tuning, by degrading performance on other tasks
* 50-100k for multi-task fine tuning
* FLAN = Fine-tuned LAnguage Net and referse to models that have gone through specific fine tuning
* including different ways to say the same instruction helps the model generalise and perform bettter
* evaluation metrics - ROUGE & BLEU score
  * ROUGE is for text summarization by comparing a summary to one or more reference summaries
  * BLEU score is for text translation by comparing to human-generated translations
* ROUGE-1 Recall = unigram matches / unigrams in reference
* ROUGE-1 Precision = unigram matchse / unigrams in output
* ROGUE-1 F1 = 2* (precision * recall) / precision + recall
* but these dont consider the order of the words and is only 1 word
* bigrams = ROUGE2 and implicitly take into account order now
* could also use LCS (longest common subsequence) as well to calculate ROUGE-L (how do you determine the longest common subsequence?)
* can clip ROUGE matches to prevent repetition scoring well
* BLEU = Avg(precision across range of n-gram sizes)
* HELM benchmark includes metrics for fairness, bias and toxicity

Parameter efficient fine-tuning:
* PEFT freezes the original LLM weights to make it easier (resource-wise) to train and makes the model less prone to catastrophic forgetting
* train a small number of model weights, that are trained for each task (summary, translation etc) and can be swapped out for inference
* selective = select subset of initial LLM params
* reparametrization = reduce the no. of params by creating new low rank transformations of the original weights, e.g. LoRA
* additive = add trainable layers (adapters) or soft prompts (add parameters to embeddings)
* LoRA = done in the self-attention layers
  * freeze original LLM weights
  * inject 2 rank decompsition matrices
  * train weights of the smaller matrics
  * to update the model, multiply the matrices which now have the same dimension as the original weights
  * then add to original weights
* e.g. if transformer weight have dimensions 512x64, with LoRA rank 8 you train A matrix 8x64 and B matrix 512x8
* fine tune a different set for different tasks and then swap them in and out for specific tasks
* 4-32 rank r
* prompt tuning refers to adding soft prompts (trainable tokens) to inputs
  * soft prompt vectors get prepended to embedding vectors
  * can change soft prompts for different inferences

## Week 3

Reinforcement learning from human feedback:
* utilise reinforcement learning with human feedback data
* RL = agent learns to make decisions related to a specific goal by taking actions in an environment, with the object being to maximise reward
* how does the reward get defined in RLHF?
* an alternative to manual human input is creating a reward model - which classifies outputs of LLM and evaluate the degree of alignment with human preferences
* rollout (rather than playout)
* but human preferences still have to be created and defined - the reward model will just carry out the classification and evaluation
* human preference data must be converted into paiwise training data
* reward model (often an LLM too) will learn to maximise the highest rank prompt/completion while minimising loss
* binary classification output consisting of logits (apply softmax to become probs)
* reward model -> RL algorithm -> RL-updated LLM
* RL algorithm can be PPO (proximal policy optmization) among others
* reward hacking - where the model maximises the reward to the point where the language diverges or even nonsensical text
* to avoid, use the original LLM as the reference model (freeze weights)
* then calculate KL divergence shift penalty - a measure of how different two probability distributions are and calculate how the models have diverged
* add KL divergence as a penalty to the PPO, so it penalises the reward model if it diverges too much
* can combine with PEFT - update only the weights of the PEFT adapter and not the LLM so you can swap it out
* this way you reduce memory footprint by not having to have 2 models
* Constitutional AI is a way to train a model with a set of principles - the model self critiques itself
* red teaming = ask the model to generate harmful responses, then ask it to critique its own responses (through prompt) - then pair the red-team prompt with the constitutional response

LLM-powered applications:
* Distillation - use an LLM teacher model to train a smaller student model for inference
* Pruning - removes redundant model parameters
* for distillation, freeze teacher model and use it to generate completions, in parallel with the student
* distillation loss is calculated 0 between soft labels (teacher) and soft predictions (student). temperature applied to softmax
* in parallel, student also has hard predictions where temperature is not varied, calculating a student loss between these and ground truth
* distillation and student loss then use to update the LLM student weights via backprop
* can perform post-training quantization for optimizing in deployment (not only training)
* for pruning, eliminate weights with values close to or equal to 0
* PEFT/LoRA are pruning methods
* retrieval augmented generation (RAG) is useful to give model access to additional data at inference
* data must fit into context window
* data must be in the format of embedding vectors
* can use vector stores or vector data bases for faster searching
* LLMs are bad at reasoning - but they can be prompted through chain-of-thought prompting to help them
* ReAct - a prompting strategy that combines chain-of-thought with action planning
* Qusetion -> Thought -> Action -> Observation
* LangChain package
* 