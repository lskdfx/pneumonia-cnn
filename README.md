# RSNA Pneumonia Dataset
CNN made from the dataset used in the 2018 RSNA Pneumonia Detection Challenge
# Batch Size Test

| Epoch | Batch Size 4 | Batch Size 8 | Batch Size 16 | Batch Size 32 | Batch Size 64 |
| --- | --- | --- | --- | --- | --- |
| **1** | 0.6077 | 0.5876 | 0.6188 | 0.6359 | 0.6045 |
| **2** | 0.5573 | 0.5394 | 0.5548 | 0.5826 | 0.5669 |
| **3** | 0.5285 | 0.5060 | 0.5042 | 0.5656 | 0.5576 |
| **4** | 0.5031 | 0.5216 | 0.5080 | 0.5335 | 0.5236 |
| **5** | 0.4912 | 0.4783 | 0.4942 | 0.5061 | 0.5051 |
| **6** | 0.4918 | 0.4880 | 0.4751 | 0.4978 | 0.4982 |
| **7** | 0.4790 | 0.4895 | 0.4604 | 0.4979 | 0.4964 |
| **8** | 0.4764 | 0.4732 | 0.4632 | 0.4953 | 0.4856 |
| **9** | 0.4726 | 0.4612 | 0.4612 | 0.4802 | 0.4773 |
| **10** | 0.4467 | 0.4453 | 0.4314 | 0.4828 | 0.4740 |

This was the output from running the CNN over the following batch sizes: 4, 8, 16, 32, 64. From this,
I found that a batch size of 16 is the best for performance of the model, though that may be untrue due to the
small number of epochs I was able to run, since I am running it on a laptop.

# Why?
I made this CNN to learn more about machine learning, and CNN's specifically. I used RSNA's dataset since it 
was made specifically for a challenge, so I could trust that it would be a good base to work off of.
