import pandas as pd
#import datetime
import numpy as np
from zipfile import ZipFile
import requests
#import warnings
#import matplotlib.pyplot as plt

# Import algorithm packages
from statsmodels.tsa.holtwinters import ExponentialSmoothing
#import joblib as jb




## Clean CPI Data

#Load Data
URLs = ["https://www150.statcan.gc.ca/n1/tbl/csv/18100007-eng.zip","https://www150.statcan.gc.ca/n1/tbl/csv/18100005-eng.zip","https://www150.statcan.gc.ca/n1/tbl/csv/13100835-eng.zip"]
filenames = ['18100007.csv','18100005.csv','13100835.csv']


for URL, filename in zip(URLs,filenames):
    response = requests.get(URL)
    open("Data/temp.zip","wb").write(response.content)
    with ZipFile("Data/temp.zip", 'r') as z:
        z.extract(filename,'Data/')


URL_world = "https://thedocs.worldbank.org/en/doc/1ad246272dbbc437c74323719506aa0c-0350012021/original/Inflation-data.xlsx"
response_world = requests.get(URL_world)
open("Data/Inflation-data.xlsx", "wb").write(response_world.content)



FH = pd.read_csv("Files/food_hierarchy.csv")
CPI = pd.read_csv("Data/18100005.csv")
BW = pd.read_csv("Data/18100007.csv")

#Clean & Process Data

#get useful columns
CPI = CPI[["REF_DATE","GEO", "Products and product groups","VALUE"]]
#filter for 1995 and up
CPI = CPI[CPI['REF_DATE']>1994]

BW = BW[["REF_DATE","GEO", "Products and product groups","Price period of weight","VALUE"]]

#format ref date as date
#CPI["REF_DATE"] = pd.to_datetime(CPI["REF_DATE"], format='%Y-%m')
CPI["REF_DATE"] = pd.to_datetime(CPI["REF_DATE"], format='%Y')
BW["REF_DATE"] = BW["REF_DATE"].replace({1992: 1995})
BW["REF_DATE"] = pd.to_datetime(BW["REF_DATE"], format='%Y')



#rename columns
CPI = CPI.rename(columns = {"VALUE":"CPI"})
BW = BW.rename(columns = {"VALUE":"Basket Weight (%)"})

#Complete data for all the years without data for BW

#get max year and min year for each product type
max_year = CPI.groupby(["Products and product groups"])['REF_DATE'].max()
min_year = CPI.groupby(["Products and product groups"])['REF_DATE'].min()
max_year = pd.DataFrame({"Products and product groups":max_year.index,'max_year':max_year.values} )
min_year = pd.DataFrame({"Products and product groups":min_year.index,'min_year':min_year.values} )
#add to CPI
CPI = CPI.merge(min_year, on="Products and product groups")
CPI = CPI.merge(max_year, on="Products and product groups")


#process basket weights

BW = BW[(BW["Price period of weight"]=="Weight at basket reference period prices")&(BW["Basket Weight (%)"].notna())]
BW = BW.drop(columns = "Price period of weight")


#merge with food hierarchy
CPI = CPI.merge(FH, on = "Products and product groups", how = 'left')

#merge cpi with BW
CPI = CPI.merge(BW, on = ["Products and product groups","REF_DATE",'GEO'], how = 'left')

CPI['Basket Weight (%)'] = CPI.apply(lambda row : 100 if row['Products and product groups']=='All-items' else row['Basket Weight (%)'], axis=1)

CPI = CPI.sort_values(by=["Products and product groups","GEO","REF_DATE"])
CPI['Basket Weight (%)'] = CPI['Basket Weight (%)'].fillna(method='bfill')

# get only level 4 data

CPI_lvl4 = CPI[CPI['Level']==4]
CPI_lvl4 = CPI_lvl4.drop(columns = ["Level5","Level6"])

#Build network graph

# #create edges
# nodes = FH["Products and product groups"]

# #Create edges
# lvl12 = CPI[["Level1","Level2"]].dropna().rename(columns={"Level1":"Start","Level2":"End"}).groupby(["Start","End"]).count().reset_index()
# lvl23 = CPI[["Level2","Level3"]].dropna().rename(columns={"Level2":"Start","Level3":"End"}).groupby(["Start","End"]).count().reset_index()
# lvl34 = CPI[["Level3","Level4"]].dropna().rename(columns={"Level3":"Start","Level4":"End"}).groupby(["Start","End"]).count().reset_index()
# lvl45 = CPI[["Level4","Level5"]].dropna().rename(columns={"Level4":"Start","Level5":"End"}).groupby(["Start","End"]).count().reset_index()
# lvl56 = CPI[["Level5","Level6"]].dropna().rename(columns={"Level5":"Start","Level6":"End"}).groupby(["Start","End"]).count().reset_index()
# edges = pd.concat([lvl12,lvl23,lvl34,lvl45,lvl56])

#Export Data

# nodes.to_csv("nodes.csv",index = False)
# edges.to_csv("edges.csv", index = False)
#CPI.to_csv("CPI.csv", index = False)
CPI_lvl4.to_csv("Data/CPI_lvl4.csv", index = False)

# test= CPI_lvl4[CPI_lvl4["Products and product groups"]=="Alcoholic beverages purchased from stores"]

# # sample data
# df = pd.DataFrame([['red', 0, 0], ['red', 1, 1], ['red', 2, 2], ['red', 3, 3], ['red', 4, 4], ['red', 5, 5], ['red', 6, 6], ['red', 7, 7], ['red', 8, 8], ['red', 9, 9], ['blue', 0, 0], ['blue', 1, 1], ['blue', 2, 4], ['blue', 3, 9], ['blue', 4, 16], ['blue', 5, 25], ['blue', 6, 36], ['blue', 7, 49], ['blue', 8, 64], ['blue', 9, 81]], columns=['color', 'x', 'y'])

# # pivot the data into the correct shape
# df = test.pivot(index='REF_DATE', columns='GEO', values='Basket Weight (%)')

# # plot the pivoted dataframe; if the column names aren't colors, remove color=df.columns
# df.plot(figsize=(24, 12)).legend(loc='best')

## Clean Food Insecurity

FI = pd.read_csv("Data/13100835.csv")
FI = FI[["REF_DATE","GEO", "Age group and sex",
         "Household food security status",
         "Statistics", "VALUE"
        ]]
FI = FI.rename(columns = {"VALUE":"Percentage of persons"})
FI = FI[FI['Statistics']=="Percentage of persons"]
FI = FI.drop(columns = "Statistics")
FI.to_csv("Data/food_insecurity.csv", index = False)

## Exponential Smoothing

df = pd.read_csv("Data/CPI_lvl4.csv")
df['REF_DATE'] = pd.to_datetime(df['REF_DATE'])

exp_smooth_preds = []
counter = 0
for g in df['GEO'].unique():
    for l4 in df['Level4'].unique():
        counter += 1
        try:
            samp = df[(df['GEO']==g) & (df['Level4']==l4)]
            expmodel = ExponentialSmoothing(np.asarray(samp['CPI']),seasonal=None, trend="multiplicative",damped_trend=True, initialization_method='legacy-heuristic')
            fit = expmodel.fit()
            pred = fit.forecast(2)[1]
            exp_smooth_preds.append({'geo':g,'level4':l4,'pred':pred})
        except:
            continue

pd.DataFrame(exp_smooth_preds).to_csv('Data/2023_predictions_exponentialsmoothing.csv', index = False)
#####################################################
######## RMSE of Exponential Smoothing Model ########
#####################################################
test_pts = []


for g in df['GEO'].unique():
    for l4 in df['Level4'].unique():
        counter += 1
        try:
            samp = df[(df['GEO']==g) & (df['Level4']==l4)].reset_index(drop=True)
            train = samp.iloc[:len(samp)-1,:]
            test = samp.iloc[len(samp)-1,:]['CPI']
            expmodel = ExponentialSmoothing(np.asarray(samp['CPI']),seasonal=None)
            fit = expmodel.fit()
            pred = fit.forecast(1)[0]
            test_pts.append({'geo':g,'level4':l4,'pred':pred,'test_pt':test})
        except:
            continue
#RMSE = np.mean(np.sum((test_df['test_pt']-test_df['pred'])**2)/len(test_df))
#print("The RMSE of exponential smoothing modeling is ", RMSE)


## CPI with Predictions

CPI = pd.read_csv("Data/CPI_lvl4.csv")
CPI = CPI.rename(columns = {"Basket Weight (%)":"lvl4 BW"})
pred = pd.read_csv("Data/2023_predictions_exponentialsmoothing.csv")
pred = pred.rename(columns = {"level4":"Products and product groups",'geo':"GEO"})
CPI = CPI.merge(pred, on = ["Products and product groups",'GEO'], how = 'left')
CPI = CPI.rename(columns = {"pred":"lvl4 pred"})
CPI["REF_DATE"] = pd.to_datetime(CPI["REF_DATE"], format='%Y-%m-%d')
CPI["min_year"] = pd.to_datetime(CPI['min_year'], format='%Y-%m-%d')
CPI["max_year"] = pd.to_datetime(CPI['max_year'], format='%Y-%m-%d')

lvl3_BW = CPI[["REF_DATE","GEO",'Level3','lvl4 BW']].groupby(["REF_DATE","GEO",'Level3']).sum().reset_index()
lvl3_BW = lvl3_BW.rename(columns = {"lvl4 BW":"lvl3 BW"})
CPI = CPI.merge(lvl3_BW, on = ["REF_DATE",'GEO','Level3'], how = 'left')

lvl2_BW = CPI[["REF_DATE","GEO",'Level2','lvl4 BW']].groupby(["REF_DATE","GEO",'Level2']).sum().reset_index()
lvl2_BW = lvl2_BW.rename(columns = {"lvl4 BW":"lvl2 BW"})
CPI = CPI.merge(lvl2_BW, on = ["REF_DATE",'GEO','Level2'], how = 'left')

lvl1_BW = CPI[["REF_DATE","GEO",'Level1','lvl4 BW']].groupby(["REF_DATE","GEO",'Level1']).sum().reset_index()
lvl1_BW = lvl1_BW.rename(columns = {"lvl4 BW":"lvl1 BW"})
CPI = CPI.merge(lvl1_BW, on = ["REF_DATE",'GEO','Level1'], how = 'left')

CPI['lvl3 pred'] = CPI['lvl4 pred']*CPI["lvl4 BW"]/CPI["lvl3 BW"]
lvl3_pred = CPI[["REF_DATE","GEO",'Level3','lvl3 pred']].drop_duplicates()
lvl3_pred = lvl3_pred.groupby(["REF_DATE","GEO",'Level3']).sum().reset_index()
CPI = CPI.drop(columns ='lvl3 pred')
CPI = CPI.merge(lvl3_pred, on = ["REF_DATE",'GEO','Level3'], how = 'left')

CPI['lvl2 pred'] = CPI['lvl3 pred']*CPI["lvl3 BW"]/CPI["lvl2 BW"]
lvl2_pred = CPI[["REF_DATE","GEO",'Level2','lvl2 pred']].drop_duplicates()
lvl2_pred = lvl2_pred.groupby(["REF_DATE","GEO",'Level2']).sum().reset_index()
CPI = CPI.drop(columns ='lvl2 pred')
CPI = CPI.merge(lvl2_pred, on = ["REF_DATE",'GEO','Level2'], how = 'left')

CPI['lvl1 pred'] = CPI['lvl2 pred']*CPI["lvl2 BW"]/CPI["lvl1 BW"]
lvl1_pred = CPI[["REF_DATE","GEO",'Level1','lvl1 pred']].drop_duplicates()
lvl1_pred = lvl1_pred.groupby(["REF_DATE","GEO",'Level1']).sum().reset_index()
CPI = CPI.drop(columns ='lvl1 pred')
CPI = CPI.merge(lvl1_pred, on = ["REF_DATE",'GEO','Level1'], how = 'left')

def get_current_value(x):
    max_date = x['max_year']
    geo = x['GEO']
    prod = x['Products and product groups']
    return CPI['CPI'].loc[(CPI['REF_DATE']==max_date)&(CPI['GEO']==geo)&(CPI['Products and product groups']==prod)].values

CPI['most_recent_val']=CPI.apply(lambda x: get_current_value(x), axis = 1)
CPI['most_recent_val'] = CPI['most_recent_val'].apply(lambda x: x[0])
CPI['increase_to_most_recent_val'] = ((CPI['most_recent_val']/CPI['CPI'])-1)*100

CPI['Products and product groups'] = (CPI['Products and product groups']
                                      .replace('Alcoholic beverages purchased from stores','Alcoholic beverages, stores')
                                      .replace('Bakery and cereal products (excluding baby food)','Bakery & cereal products')
                                      .replace('Child care and housekeeping services','Child care & housekeeping')
                                      .replace("Children's footwear (excluding athletic)","Children's footwear")
                                      .replace("Clothing accessories (2013=100)","Clothing accessories")
                                      .replace("Clothing material and notions","Clothing material")
                                      .replace("Dairy products and eggs","Dairy & eggs")
                                      .replace("Fish, seafood and other marine products","Seafood")
                                      .replace("Food purchased from fast food and take-out restaurants","Fast food")
                                      .replace("Food purchased from table-service restaurants","Restaurant food")
                                      .replace("Fruit, fruit preparations and nuts","Fruit & nuts")
                                      .replace("Fuel oil and other fuels","Fuel")
                                      .replace("Furniture and household textiles","Furniture & household textiles")
                                      .replace("Home entertainment equipment, parts and services","Home entertainment")
                                      .replace("Homeowners' home and mortgage insurance","Home & mortgage insurance")
                                      .replace("Homeowners' maintenance and repairs","Home maintenance & repairs")
                                      .replace("Local and commuter transportation","Local & commuter transportation")
                                      .replace("Men's footwear (excluding athletic)","Men's footwear")
                                      .replace("Non-electric kitchen utensils, tableware and cookware","Non-electric kitchen tools")
                                      .replace("Other cultural and recreational services","Other cultural/recreational services")
                                      .replace("Other food products and non-alcoholic beverages","Other foods & beverages")
                                      .replace("Other household goods and services","Other household goods & services")
                                      .replace("Other tobacco products and smokers' supplies","Other somker products")
                                      .replace("Paper, plastic and aluminum foil supplies","Paper, plastic & foil supplies")
                                      .replace("Personal care supplies and equipment","Personal care products")
                                      .replace("Paper, plastic and aluminum foil supplies","Paper, plastic & foil supplies")
                                      .replace("Property taxes and other special charges","Property taxes & other charges")
                                      .replace("Purchase and operation of recreational vehicles","Recreational vehicles")
                                      .replace("Purchase, leasing and rental of passenger vehicles","Acquiring passenger vehicles")
                                      .replace("Reading material (excluding textbooks)","Reading material")
                                      .replace("Recreational cannabis (201812=100)","Recreational cannabis")
                                      .replace("Recreational equipment and services (excluding recreational vehicles)","Recreational products")
                                      .replace("Services related to household furnishings and equipment","Household furnishings & equipment services")
                                      .replace("Tenants' maintenance, repairs and other expenses","Tenants' other expenses")
                                      .replace("Tools and other household equipment","Tools")
                                      .replace("Vegetables and vegetable preparations","Vegetables")
                                      .replace("Women's footwear (excluding athletic)","Women's footwear")
                                      .replace("Food purchased from table-service restaurants","Table-service food")
                                      .replace('Alcoholic beverages served in licensed establishments','Alcoholic beverages, restaurants'))


CPI['Level4'] = (CPI['Level4'].replace('Alcoholic beverages purchased from stores','Alcoholic beverages, stores')
                                      .replace('Bakery and cereal products (excluding baby food)','Bakery & cereal products')
                                      .replace('Child care and housekeeping services','Child care & housekeeping')
                                      .replace("Children's footwear (excluding athletic)","Children's footwear")
                                      .replace("Clothing accessories (2013=100)","Clothing accessories")
                                      .replace("Clothing material and notions","Clothing material")
                                      .replace("Dairy products and eggs","Dairy & eggs")
                                      .replace("Fish, seafood and other marine products","Seafood")
                                      .replace("Food purchased from fast food and take-out restaurants","Fast food")
                                      .replace("Food purchased from table-service restaurants","Restaurant food")
                                      .replace("Fruit, fruit preparations and nuts","Fruit & nuts")
                                      .replace("Fuel oil and other fuels","Fuel")
                                      .replace("Furniture and household textiles","Furniture & household textiles")
                                      .replace("Home entertainment equipment, parts and services","Home entertainment")
                                      .replace("Homeowners' home and mortgage insurance","Home & mortgage insurance")
                                      .replace("Homeowners' maintenance and repairs","Home maintenance & repairs")
                                      .replace("Local and commuter transportation","Local & commuter transportation")
                                      .replace("Men's footwear (excluding athletic)","Men's footwear")
                                      .replace("Non-electric kitchen utensils, tableware and cookware","Non-electric kitchen tools")
                                      .replace("Other cultural and recreational services","Other cultural/recreational services")
                                      .replace("Other food products and non-alcoholic beverages","Other foods & beverages")
                                      .replace("Other household goods and services","Other household goods & services")
                                      .replace("Other tobacco products and smokers' supplies","Other somker products")
                                      .replace("Paper, plastic and aluminum foil supplies","Paper, plastic & foil supplies")
                                      .replace("Personal care supplies and equipment","Personal care products")
                                      .replace("Paper, plastic and aluminum foil supplies","Paper, plastic & foil supplies")
                                      .replace("Property taxes and other special charges","Property taxes & other charges")
                                      .replace("Purchase and operation of recreational vehicles","Recreational vehicles")
                                      .replace("Purchase, leasing and rental of passenger vehicles","Acquiring passenger vehicles")
                                      .replace("Reading material (excluding textbooks)","Reading material")
                                      .replace("Recreational cannabis (201812=100)","Recreational cannabis")
                                      .replace("Recreational equipment and services (excluding recreational vehicles)","Recreational products")
                                      .replace("Services related to household furnishings and equipment","Household furnishings & equipment services")
                                      .replace("Tenants' maintenance, repairs and other expenses","Tenants' other expenses")
                                      .replace("Tools and other household equipment","Tools")
                                      .replace("Vegetables and vegetable preparations","Vegetables")
                                      .replace("Women's footwear (excluding athletic)","Women's footwear")
                                      .replace("Food purchased from table-service restaurants","Table-service food")
                                      .replace('Alcoholic beverages served in licensed establishments','Alcoholic beverages, restaurants'))

CPI['Level2'] = (CPI['Level2']
                  .replace("Alcoholic beverages, tobacco products and recreational cannabis","Alcoholic beverages, tobacco & cannabis")
                  .replace("Clothing and footwear","Clothing & footwear")
                  .replace("Household operations, furnishings and equipment","Household operations & products")
                  .replace("Clothing and footwear","Clothing & footwear")
                  .replace("Recreation, education and reading","Recreation, education & reading")
                  .replace("Health and personal care","Health & personal care")
                 )

CPI['Level3'] = (CPI['Level3']
                  .replace("Clothing accessories, watches and jewellery","Accessories")
                  .replace("Clothing material, notions and services","Clothing material & services")
                  .replace("Education and reading","Education & reading")
                  .replace("Food purchased from restaurants","Food, restaurants")
                  .replace("Food purchased from stores","Food, stores")
                  .replace("Household furnishings and equipment","Household furnishings & equipment")
                  .replace("Tobacco products and smokers' supplies","Tobacco products & smoker supplies")
                  .replace("Water, fuel and electricity","Water, fuel & electricity")
                 )

CPI.to_csv("Data/CPI_with_predictions.csv", index = False)






################################################################
################################################################
##################### DATA FOR GLOBAL INFLATION ################
################################################################


world_inf = pd.read_excel("Data/Inflation-data.xlsx",sheet_name="hcpi_a")
world_inf = world_inf.drop(['IMF Country Code','Country Code','Indicator Type','Series Name'],axis=1)
world_inf = world_inf.iloc[:len(world_inf)-2,:]
final = pd.melt(world_inf,id_vars=['Country']).rename({'variable':'year'},axis=1)
final['Country'] = final['Country'].str.replace('Congo, Dem. Rep.','Democratic Republic of the Congo',regex=True)
final['Country'] = final['Country'].str.replace('Egypt, Arab Rep.','Egypt',regex=True)
final['Country'] = final['Country'].str.replace("Côte d'Ivoire",'Ivory Coast',regex=True)
final['Country'] = final['Country'].str.replace('Congo, Rep.','Republic of the Congo',regex=True)
final['Country'] = final['Country'].str.replace('Iran, Islamic Rep.','Iran',regex=True)
final['Country'] = final['Country'].str.replace('Korea, Rep.','South Korea',regex=True)
final['Country'] = final['Country'].str.replace('Russian Federation','Russia',regex=True)
final['Country'] = final['Country'].str.replace('United States','USA',regex=True)
final['Country'] = final['Country'].str.replace('Venezuela, RB','Venezuela',regex=True)
final['Country'] = final['Country'].str.replace('Yemen, Rep.','Yemen',regex=True)
final = final.drop(final[final['Country']=='St. Lucia'].index)
final = final.drop(final[final['Country']=='Micronesia, Fed. Sts.'].index)
final = final.drop(final[final['Country']=='St. Kitts and Nevis'].index)
final = final.drop(final[final['Country']=='St. Vincent and the Grenadines'].index)
final.to_csv("Data/Annual_Global_Inflation_Melted.csv")
