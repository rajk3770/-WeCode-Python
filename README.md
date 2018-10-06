# segvoice

Segvoice was originally used to extract user voices from customer service calls. And because after that we need perform speaker verification on massive user voices, this tool tries to ensure the acquired speaker voices is pure. 

If you have similar needs, then it also works for you. But you should notice that it only works well when the speech data have high quality, that' mean there are few noise, overlapping and third speaker voices.

####  How to get pure speaker voices
* use large training data
* only accept low score segments
* remove silent segments 
* merge segments if they belong to one speaker, and only keep long merge segments 
* thanks to many awesome tools, python_speech_features for extract mfcc feature, 
scikit_learn for train GMM models, numpy for matrix computing

#### How to use it 
* install python dependencies => pip install -r requirements 
* python main.py task model | wav 

####  Experiment
I choose some speech from thchs30 dataset and random cat small segments to simulate calls.
The following is the timeline of the call, the green part means user and red means customer service.
We want extract the green part, and the above bar shows the voice we extract by timeline. As the image shows, the voice we extract is pretty pure. 
![](https://github.com/lianghyv/segvoice/raw/master/demo.jpg)
