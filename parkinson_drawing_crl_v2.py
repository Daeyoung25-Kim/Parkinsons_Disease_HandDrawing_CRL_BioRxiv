# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score,recall_score, f1_score, precision_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization,InputLayer,Conv2DTranspose,UpSampling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.initializers import HeNormal, GlorotNormal
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

tf.__version__ #implemented 2.18.0

import kagglehub

# Download latest version, Run this cell twice to implement data using path:"/kaggle/input/parkinsons-drawings/-/-/-"(experiments in manuscript were based on this path), do use not "root/.cache/-" path.
path = kagglehub.dataset_download("kmader/parkinsons-drawings")

print("Path to dataset files:", path)

train_dir_pk = "/kaggle/input/parkinsons-drawings/spiral/training/parkinson"
train_dir_nr = "/kaggle/input/parkinsons-drawings/spiral/training/healthy"
test_dir_pk = "/kaggle/input/parkinsons-drawings/spiral/testing/parkinson"
test_dir_nr = "/kaggle/input/parkinsons-drawings/spiral/testing/healthy"

train_dir_pk_w = "/kaggle/input/parkinsons-drawings/wave/training/parkinson"
train_dir_nr_w = "/kaggle/input/parkinsons-drawings/wave/training/healthy"
test_dir_pk_w = "/kaggle/input/parkinsons-drawings/wave/testing/parkinson"
test_dir_nr_w = "/kaggle/input/parkinsons-drawings/wave/testing/healthy"

os.path.join(train_dir_pk,os.listdir(train_dir_pk)[0])

def load_images_and_labels(dir,label,g):
  LB = []
  G = []
  IMG = []
  for i in range(len(os.listdir(dir))):
    img_path = os.path.join(dir, os.listdir(dir)[i])
    if os.path.exists(img_path):
      img = cv2.imread(img_path)
      img = cv2.resize(img, (224, 224)) # Resize images
      IMG.append(img)
      LB.append(label)
      G.append(g)
      if len(LB)%10 == 0: print(i+1)
  return np.array(IMG), np.array(LB), np.array(G)

X_tr_pk, y_tr_pk, g_tr_pk = load_images_and_labels(train_dir_pk, "parkinson","spiral")
X_tr_h, y_tr_h, g_tr_h = load_images_and_labels(train_dir_nr, "healthy","spiral")
X_te_pk, y_te_pk, g_te_pk = load_images_and_labels(test_dir_pk, "parkinson","spiral")
X_te_h, y_te_h, g_te_h = load_images_and_labels(test_dir_nr, "healthy","spiral")

X_tr_pk_w, y_tr_pk_w, g_tr_pk_w = load_images_and_labels(train_dir_pk_w, "parkinson","wave")
X_tr_h_w, y_tr_h_w, g_tr_h_w = load_images_and_labels(train_dir_nr_w, "healthy","wave")
X_te_pk_w, y_te_pk_w, g_te_pk_w = load_images_and_labels(test_dir_pk_w, "parkinson","wave")
X_te_h_w, y_te_h_w, g_te_h_w = load_images_and_labels(test_dir_nr_w, "healthy","wave")

plt.imshow(X_te_pk[4,:,:,:])

#concatenating Data
X_tr1 = np.concatenate([X_tr_pk ,X_tr_h,X_tr_pk_w,X_tr_h_w],axis=0)
y_tr1 = np.concatenate([y_tr_pk ,y_tr_h,y_tr_pk_w,y_tr_h_w],axis=0)
g_tr1 = np.concatenate([g_tr_pk ,g_tr_h,g_tr_pk_w,g_tr_h_w],axis=0)
X_te1 = np.concatenate([X_te_pk ,X_te_h,X_te_pk_w,X_te_h_w],axis=0)
y_te1 = np.concatenate([y_te_pk ,y_te_h,y_te_pk_w,y_te_h_w],axis=0)
g_te1 = np.concatenate([g_te_pk ,g_te_h,g_te_pk_w,g_te_h_w],axis=0)

X_tr1 = X_tr1/255.0 #rescaling
X_te1 = X_te1/255.0 #rescaling

#reshuffling data
(X_tr2,X_v2,y_tr2,y_v2,g_tr2,g_v2) = train_test_split(X_tr1,y_tr1,g_tr1,test_size=0.1,shuffle=True,random_state=321)
X_tr = np.concatenate([X_tr2,X_v2],axis=0) #image
y_tr = np.concatenate([y_tr2,y_v2],axis=0) #PD label
g_tr = np.concatenate([g_tr2,g_v2],axis=0) #Geometrical label(spiral, wave)

(X_te2,X_v,y_te2,y_v,g_te2,g_v) = train_test_split(X_te1,y_te1,g_te1,test_size=0.1,shuffle=True,random_state=321)
X_te = np.concatenate([X_te2,X_v],axis=0)
y_te = np.concatenate([y_te2,y_v],axis=0)
g_te = np.concatenate([g_te2,g_v],axis=0)
len(g_tr)

len(y_te)

#Example of hand drawing image with labels
plt.imshow(X_tr[-1,:,:,:])
print(g_tr[-1])
print(y_tr[-1])

#Masking example using 100 by 100 sized mask
mask_ex = np.ones(224*224*3).reshape(224,224,3)
mask_ex[50:150,50:150,:] = 0 #50:150
plt.imshow(X_tr[-1,:,:,:]*mask_ex)



#Binarizing categorical variables Y
y_tr = np.array([0 if v =="healthy" else 1 for v in y_tr])
y_te = np.array([0 if v =="healthy" else 1 for v in y_te])

#Binarizing categorical variables G
g_tr = np.array([0 if v =="spiral" else 1 for v in g_tr])
g_te = np.array([0 if v =="spiral" else 1 for v in g_te])

X_tr.shape

X_tr_m = X_tr*1
for i in range(X_tr.shape[0]):
  X_tr_m[i,:,:,:] = X_tr[i,:,:,:]*mask_ex

plt.imshow(X_tr_m[4,:,:,:])

#Additive Modeling based graph autoencoder(GAE) definition

class GAE(tf.keras.Model):
    def __init__(self,lat_dim,d2):
      super(GAE, self).__init__()
      self.lat_dim = lat_dim
      self.d2 = d2

      mat1 = np.zeros(lat_dim*lat_dim*d2).reshape(lat_dim,lat_dim*d2)
      for i in range(lat_dim):
        mat1[i,(d2*i):(d2*(i+1))] = 1

      mask1 = np.array(mat1).reshape(lat_dim,lat_dim*d2)
      class mask_1(tf.keras.constraints.Constraint):
        def __call__(self,w):
          return tf.convert_to_tensor(mask1)*w


      mat2 = np.zeros(lat_dim*d2*lat_dim*d2).reshape(lat_dim*d2,lat_dim*d2)
      for i in range(lat_dim):
        mat2[(i*d2):((i+1)*d2),(d2*i):(d2*(i+1))] = 1

      mask2 = np.array(mat2).reshape(lat_dim*d2,lat_dim*d2)
      class mask_2(tf.keras.constraints.Constraint):
        def __call__(self,w):
          return tf.convert_to_tensor(mask2)*w

      mat3 = np.zeros(lat_dim*d2*lat_dim).reshape(lat_dim*d2,lat_dim)
      for i in range(lat_dim):
        mat3[(i*d2):((i+1)*d2),i] = 1

      mask3 = np.array(mat3).reshape(lat_dim*d2,lat_dim)
      class mask_3(tf.keras.constraints.Constraint):
        def __call__(self,w):
          return tf.convert_to_tensor(mask3)*w

      mat4 = np.ones(lat_dim*lat_dim).reshape(lat_dim, lat_dim)
      for i in range(lat_dim):
        mat4[i,i] = 0 #blacklisting diagonal elements
      mat4[lat_dim-1,:] = 0 #blacklisting outgoing edges from target variable
      mat4[3:,[0,1,2,(lat_dim-1)]] = 0 #blacklisting outgoing edges from Z3 to causal variables: Z0 to Z2

      mask4 = np.array(mat4).reshape(lat_dim, lat_dim)
      class mask_4(tf.keras.constraints.Constraint):
        def __call__(self, w):
          return tf.convert_to_tensor(mask4)*w


      k2 = tf.keras.initializers.HeNormal(123)
      self.ENC = Sequential([
          InputLayer(shape=(lat_dim,)),
          Dense(units = lat_dim*d2, use_bias=False, kernel_initializer=k2,kernel_constraint=mask_1()),
          Dense(units = lat_dim*d2, use_bias=False,kernel_initializer=k2,kernel_constraint=mask_2()),
          Dense(units=lat_dim,use_bias=False,kernel_initializer=k2,activation="linear",kernel_constraint=mask_3())
       ])

      self.DEC = Sequential([
          InputLayer(shape=(lat_dim,)),
          Dense(units=lat_dim, use_bias=False,kernel_constraint=mask_4()), #Weighted Adjacency matrix part
          Dense(units = lat_dim*d2, use_bias=False,kernel_initializer=k2,kernel_constraint=mask_1()),
          Dense(units = lat_dim*d2, use_bias=False,kernel_initializer=k2,kernel_constraint=mask_2()),
          Dense(units=lat_dim,use_bias=False,kernel_initializer=k2,activation="linear",kernel_constraint=mask_3())
      ])


    def EnC(self,x):
      z = self.ENC(x, training=True)
      return z

    def DeC(self,z_):
      x_hat = self.DEC(z_, training=True)
      return x_hat

    mse_loss = tf.keras.losses.MeanSquaredError()
    def ae_loss(model, x,input_dim):
      z_1 = model.ENC(x)
      x_hat = model.DEC(z_1)
      return mse_loss(x[:,0:3],x_hat[:,0:3]) + mse_loss(x[:,-1],x_hat[:,-1]) #AE_loss for latent causal variables

#VAE+GAE+IT block Structures

tf.keras.utils.set_random_seed(321)

class Causal_VAE(tf.keras.Model):
  def __init__(self,input_dim, latent_dim,h1,h2,h3,d1,d2):
    super(Causal_VAE, self).__init__()
    self.input_dim = input_dim
    self.latent_dim = latent_dim
    self.h1 = h1
    self.h2 = h2
    self.h3 = h3

    k = tf.keras.initializers.HeNormal(123)
    self.Enc = Sequential([
        InputLayer(shape=(input_dim,input_dim,3)),
        Conv2D(filters=h1,kernel_size=4,strides=(2,2),activation="relu"),
        Dropout(0.1,seed=321), #0.2
        Conv2D(filters=h2,kernel_size=3,strides=(3,3),activation="relu"),
        Dropout(0.1,seed=321),
        Conv2D(filters=h3,kernel_size=3,strides=(2,2),activation="relu"),
        Flatten(),
        Dense(units = d1,activation="elu",kernel_initializer=k),
        Dense(units = d2,activation="elu",kernel_initializer=k),
        Dense(units=latent_dim+latent_dim,kernel_initializer=k,activation="linear")
        ])

    self.Dec = Sequential([
        InputLayer(shape=(latent_dim,)),
        Dense(units=d2, activation="elu",kernel_initializer=k),
        Dense(units=d1, activation="elu",kernel_initializer=k),
        Dense(units=18*18*h3, activation="elu",kernel_initializer=k),
        tf.keras.layers.Reshape(target_shape=(18,18,h3)),
        Dropout(0.1,seed=321),
        Conv2DTranspose(filters=h2, kernel_size=3, strides=2, activation='relu',kernel_initializer=k),
        Dropout(0.1,seed=321),
        Conv2DTranspose(filters=h1, kernel_size=3, strides=3,activation='relu',kernel_initializer=k),
        Conv2DTranspose(filters=3, kernel_size=4, strides=2, kernel_initializer=k),
        ])


    GAE1 = GAE(latent_dim+1,6) # +1 Due to concatenation of label y.
    self.GAE = GAE1




  #@tf.function

  def enc(self, x):
    ec = self.Enc(x)
    mean, lv = tf.split(ec,num_or_size_splits=2,axis=1)
    return mean, lv #lv: log-variance


  def reparam(self, mean, lv):
    eps = tf.random.normal(shape=mean.shape) #seed not fixed for randomness
    return eps*tf.math.exp(lv*0.5) + mean #lv: log variance

  def dec(self,z, sigmoid=False):
    result = self.Dec(z)
    if sigmoid == True:
        result = tf.math.sigmoid(result)
        return result
    return result



  mse_loss2 = tf.keras.losses.MeanSquaredError()

  def C_ELBO_loss(model, x,X): #ELBO loss
    mean, lv = model.enc(x)
    z = model.reparam(mean, lv)
    x_hat = model.dec(z, sigmoid=True)
    return mse_loss2(X,x_hat)

  def DAG_loss(model, x,y): #GAE reconstruction loss
    mean, lv = model.enc(x)
    z = model.reparam(mean, lv)
    z2 = np.concatenate([z,y],axis=1)
    res = model.GAE.ae_loss(z2,model.latent_dim+1)
    return res

  def Block_loss(model,info,x_2): #IT block loss
    mean, lv = model.enc(x_2)
    z = model.reparam(mean, lv)
    a = [] ; b=0.0
    for i in range(model.latent_dim-1):
      a.append(mutual_info_classif(np.array(z[:,i]).reshape(-1,1),info,n_neighbors=5,random_state=111))
    for j in range(model.latent_dim-3):
      b += mutual_info_classif(np.array(z[:,(3+j)]).reshape(-1,1),info,n_neighbors=5,random_state=111)
    #b = mutual_info_classif(np.array(z[:,3]).reshape(-1,1),info,n_neighbors=5,random_state=111)
    l = np.mean(np.array(a))+(-1)*b
    return l

  def Block_loss2(model, info_t,x_2):
    #IT_target
    mean, lv = model.enc(x_2)
    z = model.reparam(mean, lv)
    a = [] ; b=0.0
    for i in range(model.latent_dim-1):
      a.append(mutual_info_classif(np.array(z[:,i]).reshape(-1,1),info_t,n_neighbors=5,random_state=111))
    for j in range(model.latent_dim-3):
      b += mutual_info_classif(np.array(z[:,(3+j)]).reshape(-1,1),info_t,n_neighbors=5,random_state=111)
    #b = mutual_info_classif(np.array(z[:,3]).reshape(-1,1),info,n_neighbors=5,random_state=111)
    l_ = b-np.mean(np.array(a))
    return l_

lat_dim_1 = 4
CVAE2 = Causal_VAE(224,lat_dim_1,32,128,16,1024,128)
CVAE2.Enc.summary()

CVAE2.Dec.summary()

#Updating CVAE+GAE model

tf.keras.utils.set_random_seed(321)
os.environ['TF_DETERMINISTIC_OPS']='1'

tf.executing_eagerly()
import tensorflow.keras.backend as K

Epochs = 400 #400
lat_dim_ = 4
CVAE = Causal_VAE(224,lat_dim_,64,32,16,1024,128)
mse_loss = tf.keras.losses.MeanSquaredError()
mse_loss2 = tf.keras.losses.MeanSquaredError()
g_info = np.array(g_tr)*1

alpha=0.6
i = 0
rho = 0.1
gamma=0.9
beta = 1.01
lamb = 2.0 #L1-regularization,2.0
v=1.0 #information weight, 0.5

loss_of_cs = []
loss_of_cv = []
loss_of_v = []
basic_opt = tf.keras.optimizers.Adam(learning_rate=0.0015)
basic_opt2 = tf.keras.optimizers.Adam(learning_rate=0.0015)
basic_opt3 = tf.keras.optimizers.Adam(learning_rate=0.005)
y_2 = tf.reshape(tf.convert_to_tensor(y_tr,dtype=tf.float32),[144,1])

while i < Epochs:
    with tf.GradientTape() as dv_t, tf.GradientTape() as dv_t2, tf.GradientTape() as dg_t:
      loss_ = CVAE.C_ELBO_loss(X_tr_m,X_tr)
      loss_2 = CVAE.DAG_loss(X_tr_m,y_2)
      loss_3 = CVAE.Block_loss(g_info,X_tr_m)
      h_a = tf.linalg.trace(tf.math.exp(tf.math.multiply(CVAE.GAE.DEC.weights[0], CVAE.GAE.DEC.weights[0])))-lat_dim_-1 # -1 due to concatenation of Y
      cs_l = loss_ + loss_2 + alpha*h_a+rho*0.5*tf.math.abs(h_a)**2+lamb*tf.norm(CVAE.GAE.DEC.weights[0], ord=1, axis=[-2,-1])+ loss_3*v
      d_l = loss_2 + alpha*h_a+rho*0.5*tf.math.abs(h_a)**2+lamb*tf.norm(CVAE.GAE.DEC.weights[0], ord=1, axis=[-2,-1])
    loss_of_cs.append(cs_l)
    loss_of_cv.append(cs_l-d_l)
    loss_of_v.append(loss_)
    grad_g_a = dv_t.gradient(cs_l, CVAE.Enc.trainable_variables)
    grad_g_b = dv_t2.gradient(cs_l, CVAE.Dec.trainable_variables)
    grad_adj = dg_t.gradient(d_l, CVAE.GAE.trainable_variables)
    basic_opt.apply_gradients(zip(grad_g_a, CVAE.Enc.trainable_variables))
    basic_opt2.apply_gradients(zip(grad_g_b, CVAE.Dec.trainable_variables))
    basic_opt3.apply_gradients(zip(grad_adj, CVAE.GAE.trainable_variables))
    h_a_new = tf.linalg.trace(tf.math.exp(tf.math.multiply(CVAE.GAE.DEC.weights[0], CVAE.GAE.DEC.weights[0])))-lat_dim_-1
    alpha =  alpha + rho * h_a_new
    if (tf.math.abs(h_a_new) <= gamma*tf.math.abs(h_a)):
        rho = beta*rho
    else:
        rho = rho
    if (i+1) %10 == 0: print(i+1, cs_l,loss_,loss_3)
    i = i+1

print(cs_l)
print(loss_3)
print(loss_)
print(loss_2) #1.73659, -0.1216, 0.00579, 0.874059

import seaborn as sns
sns.set_style("darkgrid")
plt.plot(loss_of_cs, color="black",label="total loss")
#plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

import seaborn as sns
sns.set_style("darkgrid")
plt.plot(loss_of_cv, color="darkblue",label="VAE+IB loss")
plt.plot(loss_of_v, color="red",label="VAE loss")
#plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

tf.keras.utils.set_random_seed(321)
m1,l1 = CVAE.enc(X_tr_m)
z1 = CVAE.reparam(m1,l1)
one1 = CVAE.dec(z1,sigmoid=True)

fig = plt.figure(figsize=(10, 5))
rows = 1;columns=5
ax = []
img_set = [1,5,37,25,94]

for i in range(columns*rows):
  ax.append(fig.add_subplot(rows,columns,i+1))
  ax[0].set_title("Original:")
  plt.axis("off")
  plt.imshow(X_tr[img_set[i],:,:,:])

plt.tight_layout(pad=0.00)
plt.subplots_adjust(top = 1, bottom = 0, right = 2, left = 1.1, hspace = 0.01, wspace = 0)
plt.show()

fig = plt.figure(figsize=(10, 5))
rows = 1;columns=5
ax = []
img_set = [1,5,37,25,94]

for i in range(columns*rows):
  ax.append(fig.add_subplot(rows,columns,i+1 ))
  ax[0].set_title("Masked:")
  plt.axis("off")
  plt.imshow(X_tr_m[img_set[i],:,:,:])

plt.tight_layout(pad=0.00)
plt.subplots_adjust(top = 1, bottom = 0, right = 2, left = 1.1, hspace = 0.01, wspace = 0)
plt.show()

fig = plt.figure(figsize=(10, 5))
rows = 1;columns=5
ax = []
img_set = [1,5,37,25,94]

for i in range(columns*rows):
  ax.append(fig.add_subplot(rows,columns,i+1 ))
  ax[0].set_title("Generated:")
  plt.axis("off")
  plt.imshow(one1[img_set[i],:,:,:])

plt.tight_layout(pad=0.00)
plt.subplots_adjust(top = 1, bottom = 0, right = 2, left = 1.1, hspace = 0.01, wspace = 0)
plt.show()



############################################################################################
# Checking disentanglement score for Z3

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

#Standardizing to have 0 mean and unit variance
st = StandardScaler()
z1_st = st.fit_transform(np.array(z1))
np.var(z1_st)
st2 = StandardScaler()
y_tr_st2 = st2.fit_transform(np.array(y_tr).reshape(-1,1))
np.var(y_tr_st2)
st3 = StandardScaler()
g_tr_st3 = st3.fit_transform(np.array(g_tr).reshape(-1,1))
np.var(g_tr_st3)

ls_1 = Lasso(alpha=0.1,random_state=789789)
ls_1.fit(z1_st,y_tr_st2)

ls_2 = Lasso(alpha=0.1,random_state=789789)
ls_2.fit(z1_st,g_tr_st3)

ls_2.coef_ #-0.26589417 ,  0.11313799,  0.05944834, -0.34640324

Prob_mat = np.zeros(2*4).reshape(2,4)
Prob_mat[0,:] = np.abs(ls_1.coef_)
Prob_mat[1,:] = np.abs(ls_2.coef_)
Prob_mat_fac = Prob_mat*1
Prob_mat_fac[0,:] = Prob_mat_fac[0,:]/sum(Prob_mat_fac[0,:])
Prob_mat_fac[1,:] = Prob_mat_fac[1,:]/sum(Prob_mat_fac[1,:])
Prob_mat_fac

Prob_mat = np.zeros(2*4).reshape(2,4)
Prob_mat[0,:] = np.abs(ls_1.coef_)
Prob_mat[1,:] = np.abs(ls_2.coef_)
Prob_mat_code = Prob_mat*1
for i in range(4):
  Prob_mat_code[:,i] = Prob_mat_code[:,i]/sum(Prob_mat_code[:,i])
Prob_mat_code #1.        , 0.56937397, 0.37741155, 0.94901207

p3 = Prob_mat_code[:,3]
D3 = 1+np.sum(p3*np.emath.logn(2,p3));D3
print(D3) #disentanglement score for Z3, 0.7094

from sklearn.metrics import r2_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

##############################################################################
# latent adjacency matrix extraction

import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(np.array(CVAE.GAE.DEC.weights[0]), cmap="vlag",center=0)
plt.show()

print(np.quantile(np.array(CVAE.GAE.DEC.weights[0]),0.5))
print(np.mean(np.abs(np.array(CVAE.GAE.DEC.weights[0])))) #threshold

filtered = np.where(np.abs(np.array(CVAE.GAE.DEC.weights[0]))>=0.0968,1,0) #binarization
sns.heatmap(filtered,cmap='binary',linewidths=.1,linecolor="black",)
plt.show()

import networkx as nx
dic = {}
for i in enumerate(['Z0','Z1','Z2','target']):
    dic[i[0]] = i[1]

f1 = filtered[[0,1,2,4],:]
f2 = f1[:,[0,1,2,4]]

tf.keras.utils.set_random_seed(111)
bin_A = f2*1
G = nx.from_numpy_array(bin_A, create_using=nx.DiGraph())
nx.draw(G,with_labels=True,connectionstyle="arc3, rad=0.2",node_size=0.8e+3,font_size=10,node_color="lightgray",labels=dic)
plt.show()

print(np.quantile(np.array(z1[:,0]),[0,.25,.5,.75,1]))
print(np.quantile(np.array(z1[:,1]),[0,.25,.5,.75,1]))
print(np.quantile(np.array(z1[:,2]),[0,.25,.5,.75,1]))
print(np.quantile(np.array(z1[:,3]),[0,.25,.5,.75,1]))

##############################################################################
# qualitative disentanglement checks under visualization

plt.imshow(X_tr[37,:,:,:])

#Disentanglement simulation
set3 = [-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1]
set4 = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2] #For Z

fig3 = plt.figure(figsize=(10,2))
rows = 1; columns=10
ax3 = []

#Modifying only individual Z3 with other latent variables fixed.
for i in range(9):
  distangle = np.array(z1[37,:])#two spiral images:93,32 , two wave images:37,23
  distangle[3] = set4[i]
  distg = tf.reshape(tf.convert_to_tensor(distangle, dtype=tf.float32),[1,4])
  distg = CVAE.dec(distg, sigmoid=True)
  ax3.append(fig3.add_subplot(rows,columns,i+1))
  ax3[-1].set_title("Z3:"+str(set4[i]))
  plt.axis("off")
  plt.imshow(distg[0,:,:,:])

plt.tight_layout(pad=0.00)
plt.subplots_adjust(top = 1, bottom = 0, right = 2, left = 1.1, hspace = 0.1, wspace = 0)
plt.show()

#Disentanglement simulation
set3 = [-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1]
set4 = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2] #For Z

fig3 = plt.figure(figsize=(10,2))
rows = 1; columns=10
ax3 = []

#Modifying only individual Z2 with other latent variables fixed.
for i in range(9):
  distangle = np.array(z1[37,:])#two spiral images:93,32 , two wave images:37,23
  distangle[2] = set3[i]
  distg = tf.reshape(tf.convert_to_tensor(distangle, dtype=tf.float32),[1,4])
  distg = CVAE.dec(distg, sigmoid=True)
  ax3.append(fig3.add_subplot(rows,columns,i+1))
  ax3[-1].set_title("Z2:"+str(set3[i]))
  plt.axis("off")
  plt.imshow(distg[0,:,:,:])

plt.tight_layout(pad=0.00)
plt.subplots_adjust(top = 1, bottom = 0, right = 2, left = 1.1, hspace = 0.1, wspace = 0)
plt.show()

#Disentanglement simulation
set2 = [-4.5,-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5]
set3 = [-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1]
set4 = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2] #For Z

fig3 = plt.figure(figsize=(10,2))
rows = 1; columns=10
ax3 = []

#Modifying only individual Z1 with other latent variables fixed.
for i in range(9):
  distangle = np.array(z1[37,:])#two spiral images:93,32 , two wave images:37,23
  distangle[1] = set3[i]
  distg = tf.reshape(tf.convert_to_tensor(distangle, dtype=tf.float32),[1,4])
  distg = CVAE.dec(distg, sigmoid=True)
  ax3.append(fig3.add_subplot(rows,columns,i+1))
  ax3[-1].set_title("Z1:"+str(set3[i]))
  plt.axis("off")
  plt.imshow(distg[0,:,:,:])

plt.tight_layout(pad=0.00)
plt.subplots_adjust(top = 1, bottom = 0, right = 2, left = 1.1, hspace = 0.1, wspace = 0)
plt.show()

#Disentanglement simulation
set3 = [-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1]
set4 = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2] #For Z

fig3 = plt.figure(figsize=(10,2))
rows = 1; columns=10
ax3 = []

#Modifying only individual Z0 with other latent variables fixed.
for i in range(9):
  distangle = np.array(z1[37,:])#two spiral images:93,32 , two wave images:37,23
  distangle[0] = set4[i]
  distg = tf.reshape(tf.convert_to_tensor(distangle, dtype=tf.float32),[1,4])
  distg = CVAE.dec(distg, sigmoid=True)
  ax3.append(fig3.add_subplot(rows,columns,i+1))
  ax3[-1].set_title("Z0:"+str(set4[i]))
  plt.axis("off")
  plt.imshow(distg[0,:,:,:])

plt.tight_layout(pad=0.00)
plt.subplots_adjust(top = 1, bottom = 0, right = 2, left = 1.1, hspace = 0.1, wspace = 0)
plt.show()

##############################################################################
# Downstream task. PD detection

# preprocessing test data(masking)
X_te_m = X_te*1
for i in range(X_te.shape[0]):
  X_te_m[i,:,:,:] = X_te[i,:,:,:]*mask_ex

# Train data encoding
tf.keras.utils.set_random_seed(1234)
m1,l1 = CVAE.enc(X_tr_m) #X_tr_m
z1 = CVAE.reparam(m1,l1)
pk_info_tr_total = np.array(z1[:,0:3])
for i in range(9): #3
  tf.keras.utils.set_random_seed(123*i+1)
  m1,l1 = CVAE.enc(X_tr_m) #X_tr_m
  z1 = CVAE.reparam(m1,l1)
  pk_info_tr = np.array(z1[:,0:3])
  pk_info_tr_total = np.concatenate([pk_info_tr_total, pk_info_tr],axis=0)
pk_info_tr_total.shape

# Train data labels
y_tr_1 = np.concatenate([y_tr,y_tr,y_tr,y_tr,y_tr,y_tr,y_tr,y_tr,y_tr,y_tr],axis=0)
g_tr_1 = np.concatenate([g_tr,g_tr,g_tr,g_tr,g_tr,g_tr,g_tr,g_tr,g_tr,g_tr],axis=0)
y_tr_1.shape

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Test data image encoding (spiral)
pk_info_mean = 0
for j in range(10):
  tf.keras.utils.set_random_seed(j*123+1) #654
  m2,l2 = CVAE.enc(X_te_m[g_te==0]) #X_te_m
  z2 = CVAE.reparam(m2,l2)
  pk_info = np.array(z2[:,0:3])
  pk_info_mean += pk_info
pk_info_mean = pk_info_mean/10

# Test data image encoding (wave)
pk_info_mean_w = 0
for j in range(10): #3
  tf.keras.utils.set_random_seed(j*123+1)
  m3,l3 = CVAE.enc(X_te_m[g_te==1]) #X_te_m
  z3 = CVAE.reparam(m3,l3)
  pk_info_w = np.array(z3[:,0:3])
  pk_info_mean_w += pk_info_w
pk_info_mean_w = pk_info_mean_w/10



tf.keras.utils.set_random_seed(321321)
os.environ['TF_DETERMINISTIC_OPS']='1'
k3 = tf.keras.initializers.HeNormal(123)

md_c = Sequential([
    InputLayer(shape=(3,)),
    BatchNormalization(),
    Dense(units = 16, kernel_initializer=k3, activation="elu"),
    BatchNormalization(),
    Dense(units = 16, kernel_initializer=k3, activation="elu"),
    BatchNormalization(),
    Dense(units=8,kernel_initializer=k3,activation="elu"),
    BatchNormalization(),
    Dense(units=1,kernel_initializer=k3, activation="sigmoid")
    ])

from tensorflow.keras.optimizers import RMSprop, Adam
md_c.compile(optimizer=Adam(learning_rate=5e-3), loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy"])

hist = md_c.fit(x=pk_info_tr_total, y=y_tr_1,epochs=350,batch_size=200)

import seaborn as sns
sns.set_style("darkgrid")
plt.plot(hist.history['loss'], color="red",label="loss")

f1_score(y_tr_1[g_tr_1==0], np.round(md_c.predict(pk_info_tr_total)[g_tr_1==0]))#0.907..

accuracy_score(y_tr_1[g_tr_1==0], np.round(md_c.predict(pk_info_tr_total)[g_tr_1==0]))

print(accuracy_score(y_tr_1[g_tr_1==0],np.round(md_c.predict(pk_info_tr_total))[g_tr_1==0])) #0:3
print(precision_score(y_tr_1[g_tr_1==0],np.round(md_c.predict(pk_info_tr_total))[g_tr_1==0])) #0:3
print(recall_score(y_tr_1[g_tr_1==0],np.round(md_c.predict(pk_info_tr_total))[g_tr_1==0])) #0:3
print(f1_score(y_tr_1[g_tr_1==0],np.round(md_c.predict(pk_info_tr_total))[g_tr_1==0])) #0:3

#Prediction results
prd = md_c.predict(pk_info_mean)
prd1 = []
for i in range(len(y_te[g_te==0])):
  if prd[i] < 0.5:
    prd1.append(0)
  else:
    prd1.append(1)

from sklearn.metrics import ConfusionMatrixDisplay
dis = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_te[g_te==0],prd1), display_labels=[0,1])
dis.plot()
plt.grid()
plt.show()

from sklearn.metrics import classification_report

print(classification_report(y_te[g_te==0],prd1,digits=4))

from sklearn.metrics import roc_auc_score

roc_auc_score(y_te[g_te==0],md_c.predict(pk_info_mean))

print(accuracy_score(y_te[g_te==0],np.round(md_c.predict(pk_info_mean)))) #0:3
print(precision_score(y_te[g_te==0],np.round(md_c.predict(pk_info_mean)))) #0:3
print(recall_score(y_te[g_te==0],np.round(md_c.predict(pk_info_mean)))) #0:3
print(f1_score(y_te[g_te==0],np.round(md_c.predict(pk_info_mean)))) #0:3



print(accuracy_score(y_tr_1[g_tr_1==1],np.round(md_c.predict(pk_info_tr_total))[g_tr_1==1])) #0:3
print(precision_score(y_tr_1[g_tr_1==1],np.round(md_c.predict(pk_info_tr_total))[g_tr_1==1])) #0:3
print(recall_score(y_tr_1[g_tr_1==1],np.round(md_c.predict(pk_info_tr_total))[g_tr_1==1])) #0:3
print(f1_score(y_tr_1[g_tr_1==1],np.round(md_c.predict(pk_info_tr_total))[g_tr_1==1])) #0:3

#Prediction results
prd_ = md_c.predict(pk_info_mean_w)
prd2 = []
for i in range(len(y_te[g_te==1])):
  if prd_[i] < 0.5:
    prd2.append(0)
  else:
    prd2.append(1)

from sklearn.metrics import ConfusionMatrixDisplay
dis = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_te[g_te==1],prd2), display_labels=[0,1])
dis.plot()
plt.grid()
plt.show()

from sklearn.metrics import classification_report

print(classification_report(y_te[g_te==1],prd2,digits=4))

from sklearn.metrics import roc_auc_score

roc_auc_score(y_te[g_te==1],md_c.predict(pk_info_mean_w))

print(accuracy_score(y_te[g_te==1],np.round(md_c.predict(pk_info_mean_w)))) #0:3
print(precision_score(y_te[g_te==1],np.round(md_c.predict(pk_info_mean_w)))) #0:3
print(recall_score(y_te[g_te==1],np.round(md_c.predict(pk_info_mean_w)))) #0:3
print(f1_score(y_te[g_te==1],np.round(md_c.predict(pk_info_mean_w)))) #0:3

