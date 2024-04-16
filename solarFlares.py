# Import necessary libraries
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
#importacion de libreria para poder obtener el dataset
from ucimlrepo import fetch_ucirepo 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder


class solarFlares:
    # Constructor
    def __init__(self):
        #Declaracion de variables
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=50, max_iter=5000),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier()
        }

        #llamada a funciones
        self.features,self.target = self.getData()
        self.features,self.target = self.limpiezaData(self)
        self.features_train, self.features_test, self.target_train, self.target_test = self.dataforTrainAndTest()
        self.evaluateModel()

    def getData():
        #se consulta la informacion del dataset proporcionado
        solar_flare = fetch_ucirepo(id=89) 
        return solar_flare.data.features,solar_flare.data.targets   
    
    def limpiezaData(self):
        self.data = pd.concat([self.features, self.target], axis=1, join="inner")
        self.data['moderate flares'] = self.data['moderate flares'].apply(lambda x: x * 10)
        self.data['severe flares'] = self.data['severe flares'].apply(lambda x: x * 20)

        columns_to_keep = ['modified Zurich class', 'largest spot size', 'spot distribution',
       'activity', 'evolution', 'previous 24 hour flare activity',
       'historically-complex', 'became complex on this pass', 'area',
       'area of largest spot']
        
        data2  = pd.melt(self.data, id_vars=columns_to_keep, value_vars=['common flares', 'moderate flares', 'severe flares'], var_name='flares', value_name='flares_value')
        
        target_data = data2.iloc[:, -2:]
        features_data = data2.iloc[:, :-2]

        label_encoder = LabelEncoder()
        for col in features_data.select_dtypes(include=["object"]).columns:    
            features_data.loc[:,col] = label_encoder.fit_transform(features_data[col])
            print(col, label_encoder.classes_)

        return features_data,target_data
    
    def dataforTrainAndTest(self):
        features_train, features_test, target_train, target_test = train_test_split(self.features, self.target['flares_value'], test_size=0.3, random_state=50)
        return features_train, features_test, target_train, target_test
    
    def trainModel(self):
        for name, model in self.models.items():
            model.fit(self.features, self.target)
            print(f"MODELO ENTRENADO POR {name}")

    def evaluateModel(self):
        print("\nModel Performance:")
        for name, model in self.models.items():
            y_pred = model.predict(self.features_test)
            accuracy = accuracy_score(self.target_test, y_pred)
            print(f"{name}:")
            print(classification_report(self.features_test, y_pred,zero_division=1))

        # Visualize Model Comparison
        model_names = list(self.models.keys())
        accuracies = [accuracy_score(self.target, model.predict(self.features_test)) for model in self.models.values()]
        accuracy_df = pd.DataFrame({'Model': model_names, 'Accuracy': accuracies})
        fig = px.bar(accuracy_df, x='Model', y='Accuracy', title='Model Comparison')
        fig.show()