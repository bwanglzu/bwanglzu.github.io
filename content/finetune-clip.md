Title: Fine-tuned CLIP vs Zero-Shot CLIP for search, here are the results
Date: 2022-10-03 19:09
Tags: representation learning, finetuner
Category: thoughts
Authors: Bo Wang
Summary: Fine-tuning CLIP for retrieval could be effective

## Background

The [OpenAI CLIP](https://openai.com/blog/clip/) is a neural network trained on a wide variety of images with a wide variety of natural language supervision that’s abundantly available on the internet. It is capable to map text and image embeddings into the same semantic space and make them *comparable*. i.e. we can measure the similarity between a piece of text and image once we use the CLIP model by encoding them into embeddings.

*It holds the potential for building a powerful cross-modaity search application*.

How? Let's start from the beginning, and ask ourselves: *how do we search images with text*?

## Text-Image Retrieval: the Traditional Approach

Search images with text is non-trivial. Text queries normally consist of a list of tokens, while images to search are represented by matrices of values, such as RGB images are represented by three matrices of Red, Green and Blue.

A typical image is represented by three matrices of values.

How can we measure the *relatedness* between text query and these matrices? There is no way except a lot of manual work.

For instance, social media websites such as [Flickr](https://www.flickr.com/) or [Instagram](https://www.instagram.com/) allow users to tag their images. Tags can be used for indexing and matching future user queries.
If the tags are missing, some might use pre-trained machine learning models to recognise things within the image, such as dog, cat. Last but not least, if both user tags and classifier are not available, traditionally we have to use the surrounding text of the image to match against text queries, built on the assumption that surrounding text of an image, to some extent, reflected the semantic of the image itself.

## The OpenAI CLIP

In Jan 2021, Open AI introduced the CLIP (*Contrastive Language–Image Pre-training*), which efficiently learns how to recognise images of things from natural language supervision. As was introduced in the paper ([Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/pdf/2103.00020.pdf)),

> CLIP pre-trains an image encoder and a text encoder to predict which images were paired with which texts in our dataset. We then use this behaviour to turn CLIP into a zero-shot classifier. We convert all of a dataset’s classes into captions such as “a photo of a *dog*” and predict the class of the caption CLIP estimates best pairs with a given image.
> 

The author created a dataset consist of 400 million image-text pairs for the language-vision pre-training. For the image encoder, a ResNet or ViT was employed as the backbone model, while for text encoder the author used a text transformer. The author claims that:

> Our studies of CLIP in a zero-shot setting show that the model displays significant promise for widely-applicable tasks like image retrieval or search.
> 

How CLIP can help Neural Search? The fundenmental difference between Neural Search and Symbolic Search, is, Neural Search find matches on *semantics*, not *occurence*. As introduced above, we can use the CLIP text encoder to encode a user query into an embedding, while use the CLIP image encoder to encode all images into embeddings. Then we can apply different similarity/distance metrics to evaluate how similar a query is to all encoded images, and produce a ranking list to return to the user.

## CLIP Fine-tuning with Finetuner

As introduced in the previous section, CLIP was pre-trained on a large collection of image-text pairs crawled from the internet, this ensures the model itself has a good “zero-shot” capability: it generalise well on a lot of domains.

If you want to apply CLIP on your domain-of-interest, such as Fashion search, Anime search..you might encounter performance issues due to the distribution shift of the training data.

CLIP fine-tuning itself is non-trivial, it involves jointly optimising two models in parallel: the CLIP text encoder and CLIP image encoder on the customized CLIP loss. Let alone a carefully selection of a set of effective hyper-parameters and setting up all the computing devices, such as GPUs.

That’s why we offer [Finetuner](https://github.com/jina-ai/finetuner) at Jina AI. Finetuner aims to optimise the quality of embeddings for search tasks. It:

- Take care all the machine learning algorithms and deep learning techniques, such as contrastive learning, negative sampling, distributed training on top Pytorch Distributed Data Parallel (DDP) etc..
- Owns all the complexity to set up computing resources, submit jobs, manage experiments and runs in the cloud.
- Deliver an extremely simplified user interface. How easy it is? Take a look at the code block below:

```python
import finetuner

finetuner.login()

# step 1: fine-tune
run = finetuner.fit(
    model='openai/clip-vit-base-patch32',
    train_data='fashion-eval-train-clip',
    epochs=5,
    learning_rate= 1e-7,
    loss='CLIPLoss',
    device='cuda',
)

for entry in run.stream_logs():
    print(entry)

# step 2: inference
query = DocumentArray([Document(text='white sneaker')])
articles = DocumentArray([
    Document(uri='sneaker1.png'),
    Document(uri='sneaker2.png'),
    Document(uri='sneaker3.png'),
])
# download fine-tuned model
artifact = run.save_artifact('clip-model')
clip_text_encoder = finetuner.get_model(artifact=artifact, select_model='clip-text')
clip_image_encoder = finetuner.get_model(artifact=artifact, select_model='clip-vision')
# encode query and articles
finetuner.encode(model=clip_text_encoder, data=query)
finetuner.encode(model=clip_image_encoder, data=articles)
# find best match
query.match(articles, limit=1, metric='cosine')
```

## Does it work? Compare with the Benchmark

Recently, credits to the [Laion AI](https://laion.ai/) team, their engineers published an open source Github repo [CLIP Benchmark](https://github.com/LAION-AI/CLIP_benchmark) which evaluated varies CLIP models on three datasets and two tasks: image classification and image retrieval.

The Jina AI team adopted the codebase from CLIP Benchmark and apply Finetuner fine-tuned three variations of CLIP models:

1. `ViT-B-32#openai`
2. `ViT-B-32-quickgelu#laion400m`
3. `ViT-B-16-plus#laion400m`

on three datasets (same as CLIP Benchmark)

1. Flickr8k
2. Flickr30k
3. MS-COCO captions

Nine experiments in total, and here are the results (compared with zero-shot CLIP):

| model | dataset | imageRecall@5(zero-shot) | textRecall@5(zero-shot) | imageRecall@5(fine-tuned) | textRecall@5(fine-tuned) |
| --- | --- | --- | --- | --- | --- |
| ViT-B-32#openai | flickr8k | 0.532 | 0.699 | 0.865 | 0.908 |
| ViT-B-16-plus-240 | flickr8k | 0.644 | 0.792 | 0.878 | 0.920 |
| ViT-B-32-quickgelu#laion400m_e32 | flickr8k | 0.579 | 0.739 | 0.849 | 0.902 |
| ViT-B-32#openai | flickr30k | 0.834 | 0.949 | 0.902 | 0.948 |
| ViT-B-16-plus-240 | flickr30k | 0.889 | 0.971 | 0.917 | 0.971 |
| ViT-B-32-quickgelu#laion400m_e32 | flickr30k | 0.855 | 0.941 | 0.872 | 0.929 |
| ViT-B-32#openai | coco captions | 0.559 | 0.748 | 0.655 | 0.745 |
| ViT-B-16-plus-240 | coco captions | 0.662 | 0.810 | 0.712 | 0.814 |
| ViT-B-32-quickgelu#laion400m_e32 | coco captions | 0.608 | 0.768 | 0.671 | 0.764 |

Default hyper-parameters are: `learning_rate: 1e-6`, `epochs: 5`, `optimizer: Adam`.
Flickr models are evaluated on the Karpathy test set.
MS-COCO caption models are fine-tuned on a random subset (100k pairs) extracted from 2014 train images and evaluated on 2014 validation images.

## General insights when Fine-tuning CLIP

- Use a small learning rate, such as 1e-6, 1e-7.
- You do not need huge or complex models, the ViT-B-32 is good enough with fine-tuning.
- If your search case is close domain/different domain, fine-tuning might be a good idea, otherwise not.
- Use a Prompt template to turn your keyword into a sentence. For example, use `This is a photo of cat` instead of `cat` as your text descriptor.

## Credits

- Thanks the great work produced by *[CLIP-Benchmark](https://github.com/LAION-AI/CLIP_benchmark).*
- Thanks [Open-CLIP](https://github.com/mlfoundations/open_clip) for providing ~~all~~ pre-trained CLIP models with different weights.

If you want to reproduce our results or try Finetuner on your own data, we have created a G[oogle Colab](https://colab.research.google.com/drive/1fHSUML3UmrRltzn7xjgl5Db6QOqXjNHa?usp=sharing) together with this blog post. It is worth mentioning that, apart from CLIP, Finetuner is capable of tuning other models like ResNet, EfficientNet and even language models such as BERT.

Please visit our Github page and documentation page for more information:

- [https://github.com/jina-ai/finetuner](https://github.com/jina-ai/finetuner)
- [https://finetuner.jina.ai/](https://finetuner.jina.ai/)
