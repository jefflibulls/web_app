# load packages
import joblib
import pandas as pd

# load model object
model = joblib.load("C:/Users/smith/OneDrive/Documents/Masters Program/CSE6242/Project/model5.pkl")


# read in scoring dataset
df = pd.read_csv('C:/Users/smith/OneDrive/Documents/Masters Program/CSE6242/Project/TestData.csv')

# read in encoding mapping
Reporting_Airline_map = pd.read_csv('C:/Users/smith/OneDrive/Documents/Masters Program/CSE6242/Project/Reporting_Airline_encoding.csv')
Origin_map = pd.read_csv('C:/Users/smith/OneDrive/Documents/Masters Program/CSE6242/Project/Origin_encoding.csv')
Dest_map = pd.read_csv('C:/Users/smith/OneDrive/Documents/Masters Program/CSE6242/Project/Dest_encoding.csv')
DepTimeBlk_map = pd.read_csv('C:/Users/smith/OneDrive/Documents/Masters Program/CSE6242/Project/DepTimeBlk_encoding.csv')

# merge encoding mapping
df = pd.merge(df, Reporting_Airline_map, how='left', on=['Reporting_Airline'])
df = pd.merge(df, Origin_map, how='left', on=['Origin'])
df = pd.merge(df, Dest_map, how='left', on=['Dest'])
df = pd.merge(df, DepTimeBlk_map, how='left', on=['DepTimeBlk'])

# drop non-encoded variables and rename encoded variables
df = df.drop(['Reporting_Airline', 'Origin', 'Dest', 'DepTimeBlk'], axis=1)
df.rename(columns={'Reporting_Airline_encoding': 'ReportingAirline', 'Origin_encoding': 'Origin', 'Dest_encoding': 'Dest', 'DepTimeBlk_encoding': 'DepTimeBlk'}, inplace=True)

# predict for scoring dataset
model.predict(df)