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

## speedchallenge_cnnrnn.ipynb
This uses a TimeDistributed CNN layer feeding into a single RNN layer followed by FCs.
This approach used an RNN to capture the temporal difference while producing a similar result.
I see a lot of attempts being done using LSTMs and ConvLSTM. However, I think it is important
to note that in the current problem, the speed of the vehicle will is not dependent
on the speed of the vehicle minutes ago. Therefore, capturing the context over long periods
of time seems irrelevant which LSTMs are really capable of doing. 
