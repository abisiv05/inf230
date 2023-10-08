#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 20:56:55 2021

@student number: XXXXX
"""

# Import Python modules/packages
import pandas as pd
import matplotlib.pyplot as plt
import pymysql as pm
import hoggorm as ho
import hoggormplot as hop
from sklearn.preprocessing import StandardScaler

# Import CSV file

data = pd.read_csv("all_weather_data.csv", sep = ";", index_col=0)

# changing the datatype of the timestamp to datetime

data["Date_Time"] = pd.to_datetime(data["Date_Time"]) 


""" PART 2A: PLOTS """

""" 6.1 STATION BIRK: plot 1
dew_point_temperature_set_1d & air_temp_set_1 as a function of Date_Time """

# Selecting dew_point_temperature_set_1d & air_temp_set_1 for station ID "BIRK".

birk_dew = data.loc[data["Station_ID"] == "BIRK", "dew_point_temperature_set_1d"]

birk_air = data.loc[data["Station_ID"] == "BIRK", "air_temp_set_1"] 

birk_date_time = data.loc[data["Station_ID"] == "BIRK", "Date_Time"] 

plot_1_birk = plt.figure()
akser = plot_1_birk.add_subplot()


plt.plot(birk_date_time, birk_dew, label = "dew_point_temperature_set_1d")
plt.plot(birk_date_time, birk_air, label = "air_temp_set_1")
plt.xticks(rotation=45)

akser.set_xlabel("Date Time")
akser.set_ylabel("Temperature Degree")
akser.set_title(" Dew point temperature set 1d and Air temperature set 1 for Station BIRK ")
akser.legend()
plt.xticks(rotation=45)
plt.show()

""" 6.2 STATION BIRK: plot 2
relative_humidity_set_1 as a function of Date_Time """

birk_humidity = data.loc[data["Station_ID"] == "BIRK", "relative_humidity_set_1"]


plot_2_birk = plt.figure()
akser = plot_2_birk.add_subplot()


plt.plot(birk_date_time, birk_humidity)
plt.xticks(rotation=45)

akser.set_xlabel("Date Time")
akser.set_ylabel("relative_humidity_set_1")
akser.set_title(" Relative humidity set 1 for Station BIRK ")
plt.xticks(rotation=45)
plt.show()

""" 6.3 STATION BIRK: plot 3
air_temp_set_1 as a function of dew_point_temperature_set_1d """

plot_3_birk = plt.figure()
akser = plot_3_birk.add_subplot()

plt.scatter(birk_dew, birk_air, s=10)


akser.set_xlabel("dew_point_temperature_set_1d")
akser.set_ylabel("air_temp_set_1")
akser.set_title(" Dew point temperature set 1d and Air temperature set 1 for Station BIRK ")
plt.show()

""" 6.4 STATION EGNH: plot 4
dew_point_temperature_set_1d & air_temp_set_1 as a function of Date_Time """

egnh_dew = data.loc[data["Station_ID"] == "EGNH", "dew_point_temperature_set_1d"]

egnh_air = data.loc[data["Station_ID"] == "EGNH", "air_temp_set_1"] 

egnh_date_time = data.loc[data["Station_ID"] == "EGNH", "Date_Time"] 

plot_1_egnh = plt.figure()
akser = plot_1_egnh.add_subplot()


plt.plot(egnh_date_time, egnh_dew, label = "dew_point_temperature_set_1d")
plt.plot(egnh_date_time, egnh_air, label = "air_temp_set_1")
plt.xticks(rotation=45)

akser.set_xlabel("Date Time")
akser.set_ylabel("Temperature Degree")
akser.set_title(" Dew point temperature set 1d and Air temperature set 1 for Station EGNH")
akser.legend()
plt.xticks(rotation=45)
plt.show()

""" 6.5 STATION EGNH: plot 5
relative_humidity_set_1 as a function of Date_Time """

egnh_humidity = data.loc[data["Station_ID"] == "EGNH", "relative_humidity_set_1"]


plot_2_egnh = plt.figure()
akser = plot_2_egnh.add_subplot()


plt.plot(egnh_date_time, egnh_humidity)
plt.xticks(rotation=45)

akser.set_xlabel("Date Time")
akser.set_ylabel("relative_humidity_set_1")
akser.set_title(" Relative humidity set 1 for Station EGNH ")
plt.xticks(rotation=45)
plt.show()

""" 6.6 STATION EGNH: plot 6
air_temp_set_1 as a function of dew_point_temperature_set_1d """

plot_3_egnh = plt.figure()
akser = plot_3_egnh.add_subplot()

plt.scatter(egnh_dew, egnh_air, s=10)


akser.set_xlabel("dew_point_temperature_set_1d")
akser.set_ylabel("air_temp_set_1")
akser.set_title(" Dew point temperature set 1d and Air temperature set 1 for Station EGNH ")
plt.show()


# Counting the no. of zero values in each column

columnname = data.columns

for column in columnname:
    no_of_zero = (data[column] == 0).sum()
    
    if no_of_zero > len(data)*0.6:
        print(column, no_of_zero)

        
print(data.info())



""" PART 2B: PCA ANALYSIS  """

# Connect to Workbench
username = "root"
password = "password"
host = "127.0.0.1"
db = "weather_data"

con = pm.connect(host = host, user=username, password=password, database=db )
cursor = con.cursor()

# Make the view into a pandas dataframe
sql_view = "SELECT * FROM weather_view"

data_view = pd.read_sql(sql_view, con)

# Extract the necessary features from the df

var_names = list(data_view.columns)
var_names.pop(0)

labels = data_view["Station_ID"].tolist()
values = data_view.loc[:, var_names].values
object_names = labels

# Scale the df

scaling = StandardScaler()
scaling.fit(values)
scaled_values = scaling.transform(values)

# PCA NO 1
model_1 = ho.nipalsPCA(arrX=scaled_values, Xstand=False, cvType=["loo"], numComp=4)

hop.plot(model_1, comp=[1,2], plots=[1, 2, 3, 4, 6],
         XvarNames=var_names, 
         objNames=object_names)


# remove VTUV (possible outlier)

data_view_2 = data_view.drop([73], axis=0)
labels_2 = data_view_2["Station_ID"].tolist()


values_2 = data_view_2.loc[:, var_names].values
object_names_2 = labels_2

# Scale the df

scaling_2 = StandardScaler()
scaling_2.fit(values_2)
scaled_values_2 = scaling_2.transform(values_2)


#PCA NO 2
model_2 = ho.nipalsPCA(arrX=scaled_values_2, Xstand=False, cvType=["loo"], numComp=4)

hop.plot(model_2, comp=[1,2], plots=[1, 2, 3, 4, 6],
         XvarNames=var_names, 
         objNames=object_names_2)
