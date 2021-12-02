#!/usr/bin/env python
# coding: utf-8

# In[2]:


from keras.applications.resnet import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import image
from keras.models import load_model, Model

# import matplotlib.pyplot as plt
import pickle
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# In[3]:


model = load_model("./model_weights/model_9.h5")


# In[5]:


model_temp = ResNet50(weights="imagenet", input_shape=(224,224,3))


# In[6]:


model_resnet = Model(model_temp.input, model_temp.layers[-2].output)


# In[7]:


# Load the word_to_idx and idx_to_word from disk
with open("./storage/word_to_idx.pkl", "rb") as w2i:
    word_to_idx = pickle.load(w2i)

with open("./storage/idx_to_word.pkl", "rb") as i2w:
    idx_to_word = pickle.load(i2w)
    


# In[8]:


max_len = 33


# In[9]:


def preprocess_image(img):
    img = image.load_img(img, target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


# In[10]:


def encode_image(img):
    img = preprocess_image(img)
    feature_vector = model_resnet.predict(img)
    feature_vector = feature_vector.reshape(1, feature_vector.shape[1])
    return feature_vector


# In[11]:


def predict_caption(photo):
    in_text = "startseq"

    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=max_len, padding='post')

        ypred =  model.predict([photo,sequence])
        ypred = ypred.argmax()
        word = idx_to_word[ypred]
        in_text+= ' ' +word

        if word =='endseq':
            break


    final_caption =  in_text.split()
    final_caption = final_caption[1:-1]
    final_caption = ' '.join(final_caption)

    return final_caption


# In[12]:


def caption_this_image(input_img): 

    photo = encode_image(input_img)
    

    caption = predict_caption(photo)
    # keras.backend.clear_session()
    return caption


# In[14]:


# enc = encode_image("testimg.jpg");


# # In[15]:


# enc


# # In[16]:


# predict_caption(enc)


# # In[17]:


# caption_this_image("testimg.jpg")


# In[ ]:




