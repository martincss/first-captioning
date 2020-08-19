import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
plt.ion()
from metrics import BLEUMetric, CrossEntropyMetric, METEORMetric

from caption_generation import predict_batch
# from training import loss_function

def all_scores_single(predicted_logits, predicted_caption, val_cap_vector,
               val_caption, lossf):

    scores= {}

    scores['cross-entropy'] = float(lossf(val_cap_vector, predicted_logits).numpy())

    for n in range(1,5):
        scores['bleu-' + str(n)] = BLEUMetric.bleu_n(reference=val_caption,
                                          prediction=predicted_caption,
                                          n=n)
    return scores


def all_scores_all(predicted_logits_all, predicted_captions, val_cap_vectors,
               val_captions, lossf):

    n_examples = len(val_captions)

    scores_accum = {'cross-entropy':[], 'bleu-1':[],'bleu-2':[],'bleu-3':[],
               'bleu-4':[]}

    for i in range(n_examples):

        scores = all_scores_single(predicted_logits_all[i],
                                   predicted_captions[i],
                                   val_cap_vectors[i],
                                   val_captions[i],
                                   lossf)

        for name, score in scores.items():
            scores_accum[name].append(score)

    return {name:np.mean(score) for name, score in scores_accum.items()}


def validation_scores(dataset, models, tokenizer):

    metrics = {'cross-entropy':CrossEntropyMetric(),
               'bleu-1':BLEUMetric(n_gram=1),
               'bleu-2':BLEUMetric(n_gram=2),
               'bleu-3':BLEUMetric(n_gram=3),
               'bleu-4':BLEUMetric(n_gram=4),
               'meteor':METEORMetric()}

    for (batch, (img_tensors, cap_vectors, captions)) in enumerate(dataset):

        batch_size = cap_vectors.shape[0]
        caption_length = cap_vectors.shape[1]

        batch_logits = predict_batch(img_tensors, models, tokenizer, caption_length)
        batch_captions = [[] for _ in range(batch_size)]

        # add predicted word to each caption in batch
        for step in range(caption_length):

            predicted_ids = tf.random.categorical(batch_logits[:,:,step], 1)
            metrics['cross-entropy'](cap_vectors[:,step], batch_logits[:,:,step])

            for i in range(batch_size):
                next_word = tokenizer.index_word.get(predicted_ids[i,0].numpy())
                batch_captions[i].append(next_word)

        # cut each caption up to the <end> token
        for i in range(batch_size):
            try:
                end_index = batch_captions[i].index('<end>')

            except ValueError:
                end_index = len(batch_captions[i])

            finally:
                batch_captions[i] = batch_captions[i][:end_index]

        true_captions = [cap.decode('utf-8').split(' ')[1:-1] for cap in \
                         captions.numpy().tolist()]

        for name, metric in metrics.items():
            if name != 'cross-entropy':
                metrics[name](true_captions, batch_captions)

    results = {}
    for name, metric in metrics.items():
        results[name] = float(metric.result().numpy())

    return results







def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result//2, len_result//2, l+1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()
