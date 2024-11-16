#################################################
# README:
#
# This code file creates a class that:
# given an input dataset,
# scores all the flight delay prediction models
# provided by the team
#
#################################################


# Import necessary packages
import pandas as pd
import joblib
import numpy as np
import sklearn as sk
from amadeus import Client, ResponseError
from meteostat import Daily, Point


class flight_data_prep:

    def __init__(self, flight_api=False):
        
        self.flight_api=flight_api
        
        if not flight_api:
            # Read in flight api test data
            self.flight_api_test_df = pd.read_csv('flight_api_test_data.csv')

        # Read in city name to airport mapping data
        self.city_to_airport_mapping_df = pd.read_csv('city_to_airport_mapping.csv')

        # Read in airport location and runway data
        self.airport_info_df = pd.read_csv('airport_runway_info.csv')


    # Looking up airports based on city names
    def get_airports(self, city_name):

        data = self.city_to_airport_mapping_df
        airports = data[(data['CityName'] == city_name)]['Airport']

        return airports


    # Define function to get future flight info from Amadeus API
    def get_flight_info(self, dep_airport, dest_airport, dep_date, airline=None):

        amadeus = Client(
        client_id='DVnS2wuDkH2PJeBqXuUxDUkqi34JGeir',
        client_secret='dhCKd4UgKyYcQv3g'
        )   
        
        try:
            response = amadeus.shopping.flight_offers_search.get(
                originLocationCode=dep_airport,
                destinationLocationCode=dest_airport,
                departureDate=dep_date,
                # includedAirlineCodes=airline,
                adults=1,
                nonStop='true'
            )
            
            # Convert JSON data to DataFrame
            data = response.data  # Extract the JSON data from the response
            # Parse each flight offer into a row
            parsed_data = []
            for offer in data:
                row = {
                    'flight_nbr': offer['itineraries'][0]['segments'][-1]['number'],
                    'origin': offer['itineraries'][0]['segments'][0]['departure']['iataCode'],
                    'orig_airline': offer['itineraries'][0]['segments'][0]['carrierCode'],
                    'destination': offer['itineraries'][0]['segments'][-1]['arrival']['iataCode'],
                    'dest_airline': offer['itineraries'][0]['segments'][-1]['carrierCode'],
                    'departure_time': offer['itineraries'][0]['segments'][0]['departure']['at'],
                    'arrival_time': offer['itineraries'][0]['segments'][-1]['arrival']['at'],
                    'legs': len(offer['itineraries'][0]['segments'])
                    # Add more fields as necessary
                }
                parsed_data.append(row)

            # Create DataFrame
            df = pd.DataFrame(parsed_data)

            # Filter by airline, if specified
            if airline is not None:
                df = df.loc[df['orig_airline']==airline, ].copy()

            return df

        except ResponseError as error:
            print("Error code:", error.response.status_code)
            print("Error message:", error.response.result)  # Detailed error message
    

    def flight_data_transformers(self, flight_info):

        # Use test api dataset to get flights info
        flight_info['Reporting_Airline'] = flight_info['orig_airline']
        flight_info['Origin'] = flight_info['origin']
        flight_info['Dest'] = flight_info['destination']
        flight_info['departure_time'] = pd.to_datetime(flight_info['departure_time'])
        flight_info['dep_date'] = pd.to_datetime(flight_info['departure_time'].dt.date)
        flight_info['Month'] = flight_info['departure_time'].dt.month
        flight_info['DayofMonth'] = flight_info['departure_time'].dt.day 
        flight_info['DayOfWeek'] = flight_info['departure_time'].dt.dayofweek + 1
        flight_info['dep_hr_min'] = flight_info['departure_time'].dt.strftime('%H%M').astype(int)

        # Define bins and labels
        bins = [0, 559, 659, 759, 859, 959, 1059, 1159, 1259, 1359, 1459, 1559, 1659, 1759, 1859, 1959, 2059, 2159, 2259, 2359]
        labels = [
            '0001-0559', '0600-0659', '0700-0759', '0800-0859', '0900-0959',
            '1000-1059', '1100-1159', '1200-1259', '1300-1359', '1400-1459',
            '1500-1559', '1600-1659', '1700-1759', '1800-1859', '1900-1959',
            '2000-2059', '2100-2159', '2200-2259', '2300-2359'
        ]

        # Categorize hour_minute into the defined bins
        flight_info['DepTimeBlk'] = pd.cut(flight_info['dep_hr_min'], bins=bins, labels=labels, right=True)

        # Map airport info to departing airport
        flight_info = flight_info.merge(self.airport_info_df[['ARPT_ID','ARPT_NAME','LAT_AVG','LONG_AVG','RWY_CNT']],
                                            how='left', left_on='origin', right_on='ARPT_ID')
        flight_info.rename(columns={'ARPT_ID':'DEP_ARPT_ID',
                                        'ARPT_NAME':'DEP_ARPT_NAME',
                                        'LAT_AVG':'DEP_LAT',
                                        'LONG_AVG':'DEP_LON',
                                        'RWY_CNT':'ORIGIN_RWY_CNT'},
                                        inplace=True)

        # Map airport info to destination airport
        flight_info = flight_info.merge(self.airport_info_df[['ARPT_ID','ARPT_NAME','LAT_AVG','LONG_AVG','RWY_CNT']],
                                            how='left', left_on='destination', right_on='ARPT_ID')
        flight_info.rename(columns={'ARPT_ID':'DEST_ARPT_ID',
                                        'ARPT_NAME':'DEST_ARPT_NAME',
                                        'LAT_AVG':'DEST_LAT',
                                        'LONG_AVG':'DEST_LON',
                                        'RWY_CNT':'DEST_RWY_CNT'},
                                        inplace=True)

        return flight_info
    

    def get_weather_info(self, flight_info):

        # Initialize dataframes for columns to be returned from Meteostat API call
        orig_weather_dt = pd.DataFrame(columns=['tavg','tmin','tmax','prcp','snow','wdir','wspd','wpgt','pres','tsun'])
        dest_weather_dt = pd.DataFrame(columns=['tavg','tmin','tmax','prcp','snow','wdir','wspd','wpgt','pres','tsun'])

        # Repeat API calls for each flight in the input data 
        for i in range(flight_info.shape[0]):
            
            # Meteostat API calls for departing airports
            loc = Point(flight_info['DEP_LAT'][i], flight_info['DEP_LON'][i]) # Get nearest weather station based on lat/lon of airport
            data = Daily(loc, flight_info['dep_date'][i], flight_info['dep_date'][i], model=True) # Get weather forecast given lat/lon and flight date
            data = data.fetch()
            
            # Sometimes there are no weather stations near an airport.  If so, create empty row to append to output data for that location
            if data.empty:
                empty_row = pd.DataFrame([[None] * len(orig_weather_dt.columns)], columns=orig_weather_dt.columns)
                orig_weather_dt = pd.concat([orig_weather_dt, empty_row], axis=0, ignore_index=True)
            else:
                orig_weather_dt = pd.concat([orig_weather_dt, data], axis=0, ignore_index=True)
            
            # Meteostat API calls for departing airports
            loc = Point(flight_info['DEST_LAT'][i], flight_info['DEST_LON'][i])
            data = Daily(loc, flight_info['dep_date'][i], flight_info['dep_date'][i], model=True)
            data = data.fetch()
            
            if data.empty:
                empty_row = pd.DataFrame([[None] * len(dest_weather_dt.columns)], columns=dest_weather_dt.columns)
                dest_weather_dt = pd.concat([dest_weather_dt, empty_row], axis=0, ignore_index=True)
            else:
                dest_weather_dt = pd.concat([dest_weather_dt, data], axis=0, ignore_index=True)

        # Rename columns
        orig_weather_dt = orig_weather_dt.add_suffix('_origin')
        dest_weather_dt = dest_weather_dt.add_suffix('_dest')

        # Append weather data to input data
        flight_info = pd.concat([flight_info, orig_weather_dt], axis=1)
        flight_info = pd.concat([flight_info, dest_weather_dt], axis=1)

        return flight_info
    

    def get_flight_data(self, dep_airport, dest_airport, dep_date, airline=None):
        
        # If flight api is invoked, get data from Amadeous api call.  If not, use test data
        if self.flight_api:
            data = self.get_flight_info(dep_airport, dest_airport, dep_date, airline=None)
            data = self.flight_data_transformers(data)
            data = self.get_weather_info(data)
        else:
            # Filter by airline, if specified
            data = self.flight_api_test_df
            data = data.loc[
                (data['origin'] == dep_airport) &
                (data['destination'] == dest_airport)
            ].copy()
            if airline is not None:
                data = data.loc[data['orig_airline']==airline, ].copy()
            data = self.flight_data_transformers(data)
            data = self.get_weather_info(data)

        
        return data



class delay_predictions:

    def __init__(self):

        # Airline Delay Classification Model Inputs
        self.xgboost_classification = joblib.load('models/airlines_caused_flight_delay_classification/xgb_airline_class_smote.pkl')
        self.airline_mapping_df = pd.read_csv('models/airlines_caused_flight_delay_classification/airline_mapping.csv')
        self.origin_mapping_df = pd.read_csv('models/airlines_caused_flight_delay_classification/origin_mapping.csv')
        self.dest_mapping_df = pd.read_csv('models/airlines_caused_flight_delay_classification/dest_mapping.csv')
        self.time_mapping_df = pd.read_csv('models/airlines_caused_flight_delay_classification/time_mapping.csv')

        # Airline Delay Length Model
        self.carrier_delay_length_model = joblib.load('models/airlines_caused_flight_delay_length/carrier_delay_length_model.pkl')

        # Overall Delay Classification Model
        self.loaded_model = joblib.load('models/overall_flight_delay_classification/log_reg_model.pkl')
        self.loaded_encoder = joblib.load('models/overall_flight_delay_classification/ordinal_encoder.pkl')
        self.categorical_data = joblib.load('models/overall_flight_delay_classification/categorical_data.pkl')

        # Overall Delay Length Model
        self.model = joblib.load("models/overall_flight_delay_length/model5.pkl")
        self.Reporting_Airline_map = pd.read_csv('models/overall_flight_delay_length/Reporting_Airline_encoding.csv')
        self.Origin_map = pd.read_csv('models/overall_flight_delay_length/Origin_encoding.csv')
        self.Dest_map = pd.read_csv('models/overall_flight_delay_length/Dest_encoding.csv')
        self.DepTimeBlk_map = pd.read_csv('models/overall_flight_delay_length/DepTimeBlk_encoding.csv')


    def score_airline_delay_classification(self, data):
        
        # Prep input data for scoring
        airline_mapping = dict(zip(self.airline_mapping_df['Key'], self.airline_mapping_df['Value']))
        origin_mapping = dict(zip(self.origin_mapping_df['Key'], self.origin_mapping_df['Value']))
        dest_mapping = dict(zip(self.dest_mapping_df['Key'], self.dest_mapping_df['Value']))
        time_mapping = dict(zip(self.time_mapping_df['Key'], self.time_mapping_df['Value']))

        scoring_dt = data[['Month','DayofMonth','DayOfWeek','Reporting_Airline','Origin','Dest',
                           'DepTimeBlk','ORIGIN_RWY_CNT','DEST_RWY_CNT','prcp_origin','snow_origin',
                           'wspd_origin','prcp_dest','snow_dest','wspd_dest']].copy()

        scoring_dt['Reporting_Airline'] = scoring_dt['Reporting_Airline'].map(airline_mapping)
        scoring_dt['Origin'] = scoring_dt['Origin'].map(origin_mapping)
        scoring_dt['Dest'] = scoring_dt['Dest'].map(dest_mapping)
        scoring_dt['DepTimeBlk'] = scoring_dt['DepTimeBlk'].map(time_mapping)

        # Predict probability
        prob_pred = self.xgboost_classification.predict_proba(scoring_dt)

        # Return probability of class = 1
        return np.round(prob_pred[:, 1], 2)
    

    def score_airline_delay_length(self, data):

        # Prep input data for scoring
        scoring_dt = data[['Reporting_Airline','Origin','Dest','DepTimeBlk','Month','DayofMonth',
                           'ORIGIN_RWY_CNT','prcp_origin','wspd_origin','wdir_origin','DEST_RWY_CNT',
                           'prcp_dest','wspd_dest','wdir_dest']].copy()

        # Predict and return delay length
        return np.round(self.carrier_delay_length_model.predict(scoring_dt))
        

    def score_overall_classification(self, data):
        
        # Prep input data for scoring
        scoring_dt = data[['Reporting_Airline','Origin','Dest','Month','DayofMonth','DayOfWeek']].copy()
        scoring_dt[self.categorical_data] = self.loaded_encoder.transform(scoring_dt[self.categorical_data])

        # Predict probability
        prob_pred = self.loaded_model.predict_proba(scoring_dt)

        # Return probability of class = 1 
        return np.round(prob_pred[:,1],2)

    def score_overall_delay_length(self, data):
        
        # Prep input data for scoring
        scoring_dt = data[['Month','DayofMonth','Reporting_Airline','Origin','Dest',
                           'DepTimeBlk','ORIGIN_RWY_CNT','DEST_RWY_CNT','prcp_origin',
                           'wspd_origin','wpgt_origin','prcp_dest','wspd_dest','wpgt_dest']].copy()

        scoring_dt = pd.merge(scoring_dt, self.Reporting_Airline_map, how='left', on=['Reporting_Airline'])
        scoring_dt = pd.merge(scoring_dt, self.Origin_map, how='left', on=['Origin'])
        scoring_dt = pd.merge(scoring_dt, self.Dest_map, how='left', on=['Dest'])
        scoring_dt = pd.merge(scoring_dt, self.DepTimeBlk_map, how='left', on=['DepTimeBlk'])

        # drop non-encoded variables and rename encoded variables
        scoring_dt = scoring_dt.drop(['Reporting_Airline', 'Origin', 'Dest', 'DepTimeBlk'], axis=1)
        scoring_dt.rename(columns={'Reporting_Airline_encoding': 'ReportingAirline', 'Origin_encoding': 'Origin', 'Dest_encoding': 'Dest', 'DepTimeBlk_encoding': 'DepTimeBlk'}, inplace=True)

        # Predict and return delay length
        return np.round(self.model.predict(scoring_dt))
    

class sentiment_scoring:

    def __init__(self):

        self.data = pd.read_csv('models/airlines_sentiment_analysis/airline_sentiment_scores.csv')
    

    def get_sentiment_scores(self, airlines):

        # Create mapping dictionary
        sentiment_mapping = dict(zip(self.data['Airline'], self.data['Sentiment_Score']))

        return airlines.map(sentiment_mapping)
        
