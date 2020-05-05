# comma-speed-challenge
This is my attempt for solving the [comma.ai speed challenge](https://github.com/commaai/speedchallenge).

## speedchallenge_cnn.ipynb

This is the CNN based solution. The approach is to take the training video and 
extract pairs of consecutive frames. Since, there is a total of `20400` frames,
this gives us `20399` input samples. This is fed through a few conv layers and
subsequently through a couple of FC layers to get the result.

The MSE ~ `1e-4` trained to 25 epochs. However, the model still shows a good
growth curve and has potential to do better. This was trained on Google Colab
using GPU and it took about 40 min.
