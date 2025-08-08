#!/usr/bin/env python
# coding: utf-8
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# This allows you to connect to the Workspace you've previously deployed in Azure.
# Be sure to fill in the settings below which can be retrieved by running 'az quantum workspace show' in the terminal.
from typing import List
from azure.quantum import Workspace
from numpy import square

import probtoangle
import helper

import sys, json, math, random

# Copy the settings for your workspace below
workspace = Workspace (
  subscription_id = "59bbc1ed-aca7-4d3a-b86f-21fb948008c2",
  resource_group = "quantumworkshop",
  name = "QuantumWorkshop",
  location = "westeurope"
)

#ré-edit
DictCoeff = {"masque": 8, "duree" : 6, "aeration": 4, "contact" : 8, "distance": 4}

data=json.loads(sys.argv[1])
parsed_data = data["results"]

R = 1.34

proba_list = []
# Proba des cas
if(parsed_data["user-masque"] and parsed_data["other-masque"]):
    proba_list.append(0.15)
elif(parsed_data["user-masque"] == False and parsed_data["other-masque"]):
    proba_list.append(0.4)
elif(parsed_data["user-masque"] == False and parsed_data["other-masque"] == False):
    proba_list.append(0.9)
elif(parsed_data["user-masque"] and parsed_data["other-masque"] == False):
    proba_list.append(0.75)
     
if(parsed_data["contact"]):
    proba_list.append(0.9)
else:
    if(parsed_data["distance"] <= 1 ):
        proba_list.append(0.8)
    elif(parsed_data["distance"] == 2):
        proba_list.append(0.6)
    elif(parsed_data["distance"] == 3):
        proba_list.append(0.4)
    elif(parsed_data["distance"] == 4):
        proba_list.append(0.2) 
    elif(parsed_data["distance"] >= 5):
        proba_list.append(0.1)


if(parsed_data["duree"] >= 15):
    proba_list.append(0.8)
elif(parsed_data["duree"] < 15):
    proba_list.append(0.2)

if(parsed_data["lieu"] == "interieur-na"):#inté fermé
    proba_list.append(0.8)
elif(parsed_data["lieu"] == "interieur-a"):#inté ouvert
    proba_list.append(0.2)
elif(parsed_data["lieu"] == "exterieur"):#exté
    proba_list.append(0.1)


qc = helper.QuantumCircuit(4)

for i in range(0, len(proba_list)):
    # Add event
    qc.ry(probtoangle.prob_to_angle(proba_list[i]), i)

res = helper.run_circuit(qc)
proba_calc = res[list(res)[-1]] * R

if(parsed_data["vaccin"]):
    qc2 = helper.QuantumCircuit(4)
    qc2.ry(probtoangle.prob_to_angle(0.1), 0)
    qc2.ry(probtoangle.prob_to_angle(proba_calc), 1)
    
    res2 = helper.run_circuit(qc2)

    proba_calc2 = res2[list(res2)[-1]]
    percentage = (proba_calc2 * 100)
    print(str("%.2f" % percentage) + "%")
else:
    percentage = (proba_calc * 100)
    print(str("%.2f" % percentage) + "%")
