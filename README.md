# comma-speed-challenge
This is my attempt for solving the [comma.ai speed challenge](https://github.com/commaai/speedchallenge).

## CNN Approach

This is the CNN based solution. The approach is to take the training video and 
extract pairs of consecutive frames. Since, there is a total of `20400` frames,
this gives us `20399` input samples. This is fed through a few conv layers and
subsequently through a couple of FC layers to get the result.

The MSE ~ `5e-4` trained to 25 epochs. However, the model still shows a good
growth curve and has potential to do better. This was trained on Google Colab
using GPU and it took about 40 min.
```
Jupyter: speedchallenge_cnn.ipynb
```

## CNN RNN Approach
This uses a TimeDistributed CNN layer feeding into a single RNN layer followed by FCs.
This approach used an RNN to capture the temporal difference while producing a similar result.
I see a lot of attempts being done using LSTMs and ConvLSTM. However, I think it is important
to note that in the current problem, the speed of the vehicle will is not dependent
on the speed of the vehicle minutes ago. Therefore, capturing the context over long periods
of time seems irrelevant which LSTMs are really capable of doing.

This achieved an MSE of `4.2e-4` after 30 epochs with a `90/10` train/dev ratio. Regularization
doesn't seem to be a problem.

```
Jupyter: speedchallenge_cnnrnn.ipynb
```
