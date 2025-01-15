def BrainNetCNNModelBuilding(LayerParametersDict):
  '''
  The Layer Parameters Dictionary should contains what's needed for building the model. Otherwise it will use the default setting
  typical Layer Parameter Dict should look like:
  Dict = {"InputShape": (nn,nn,1), 
  "E2E": {"numLayers": n, "numfilter": f1, "leaky": a1, "dropprob": p1}, 
  "E2N": {"numfilter": f2, "leaky": a2, "dropprob": p2},
  "N2G": {"numfilter": f3, "leaky": a3, "dropprob": p3},
  "Dense": {"numLayers": n, "numfilter": f4, "leaky": a4, "dropprob": p4},
  "Output": {"numfilter": f5, "modeltype": "regression"}
  "reg" : {"type": ridge, "decay": 0.0000005}}
  '''
  if LayerParametersDict.get("InputShape") is None:
    raise ValueError("InputShape not specified")
  if LayerParametersDict.get("Output") is None or LayerParametersDict["Output"].get("numfilter") is None or LayerParametersDict["Output"].get("modeltype") is None:
    raise ValueError("Output Parameters or Output Type not specified")

  import numpy as np
  import tensorflow as tf
  from tensorflow import keras
  from keras import layers
  from keras import regularizers

  DefaultDict = {"InputShape": LayerParametersDict["InputShape"], 
                 "E2E": {"numLayers": 1, "numfilter": 32, "leaky": 0.2, "dropprob": 0.2}, 
                 "E2N": {"numfilter": 64, "leaky": 0.2, "dropprob": 0.2},
                 "N2G": {"numfilter": 128, "leaky": 0.0001, "dropprob": 0.01},
                 "Dense": {"numLayers": 1, "numfilter": 32, "leaky": 0.0001, "dropprob": 0.01},
                 "Output": LayerParametersDict["Output"],
                 "reg" : {"type": "ridge", "decay": 0.0000005}}
  
  #Sanity Check for the Dictionary
  ModelDict = {}
  for outer_keys in DefaultDict.keys():
    if LayerParametersDict.get(outer_keys) is None:
      ModelDict[outer_keys] = DefaultDict[outer_keys]
      continue
    ModelDict[outer_keys] = LayerParametersDict[outer_keys]
    if not isinstance(DefaultDict[outer_keys],dict):
      continue
    for inner_keys in DefaultDict[outer_keys].keys():
      if LayerParametersDict[outer_keys].get(inner_keys) is None:
        ModelDict[outer_keys][inner_keys] = DefaultDict[outer_keys][inner_keys]
  if ModelDict["reg"]["type"] == "ridge":
    reg = regularizers.l2(ModelDict["reg"]["decay"])
  else:
    reg = regularizers.l1(ModelDict["reg"]["decay"])


  
  #Start Model Building

  Inputs = keras.Input(shape = ModelDict["InputShape"], name="graph_input") 
  #Stacking E2E
  e2e_output = Inputs
  for e2e_num in range(ModelDict["E2E"]["numLayers"]):
    e2e_left = layers.Conv2D(ModelDict["E2E"]["numfilter"], (1, ModelDict["InputShape"][0]), kernel_regularizer=reg)(e2e_output)
    e2e_left = layers.UpSampling2D(size=(1, ModelDict["InputShape"][0]))(e2e_left)

    e2e_right = layers.Conv2D(ModelDict["E2E"]["numfilter"], (ModelDict["InputShape"][0], 1), kernel_regularizer=reg)(e2e_output)
    e2e_right = layers.UpSampling2D(size=(ModelDict["InputShape"][0], 1))(e2e_right)

    e2e_output = layers.add([e2e_left, e2e_right])
    e2e_output = layers.LeakyReLU(alpha = ModelDict["E2E"]["leaky"])(e2e_output)
    e2e_output = layers.Dropout(ModelDict["E2E"]["dropprob"])(e2e_output)
      
  
  #Stacking E2N
  e2n_output = layers.Conv2D(ModelDict["E2N"]["numfilter"], (ModelDict["InputShape"][0], 1), kernel_regularizer=reg)(e2e_output)
  e2n_output = layers.LeakyReLU(alpha = ModelDict["E2N"]["leaky"])(e2n_output)
  e2n_output = layers.Dropout(ModelDict["E2N"]["dropprob"])(e2n_output)

  #Stacking N2G
  n2g_output = layers.Flatten()(e2n_output)
  n2g_output = layers.Dense(ModelDict["N2G"]["numfilter"], kernel_regularizer = reg)(n2g_output)
  n2g_output = layers.LeakyReLU(alpha = ModelDict["N2G"]["leaky"])(n2g_output)
  n2g_output = layers.Dropout(ModelDict["N2G"]["dropprob"])(n2g_output)

  #Stacking Dense Layers
  dense_output = n2g_output
  for dense_num in range(ModelDict["Dense"]["numLayers"]):
    dense_output = layers.Dense(ModelDict["Dense"]["numfilter"], kernel_regularizer = reg)(dense_output)
    dense_output = layers.LeakyReLU(alpha = ModelDict["Dense"]["leaky"])(dense_output)
    dense_output = layers.Dropout(ModelDict["Dense"]["dropprob"])(dense_output)

  #Setup the output format
  if ModelDict["Output"]["modeltype"] == "classification":
    Outputs = layers.Dense(ModelDict["Output"]["numfilter"],activation = 'softmax')(dense_output)
  else:
    Outputs = layers.Dense(ModelDict["Output"]["numfilter"],activation = 'linear')(dense_output)

  model = keras.Model(Inputs, Outputs, name="MyBrainNetModel")

  print("\nThis is a " + ModelDict["Output"]["modeltype"] + " model. Make sure you choose the correct loss function when calling model.compile().")
  print("\nTo view the model specification and visualising the model arcitecture, use model.summary() and keras.utils.plot_model().")

  return model