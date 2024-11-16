##################################
# Load necessary libraries
##################################

import modeling_scoring as scoring
import streamlit as st
from streamlit_option_menu import option_menu  # Need to have PyArrow installed as well
import pandas as pd
# import importlib
# importlib.reload(scoring)

#..........................................
# Initialize sidebar webparts for web app
#..........................................

# Initialize sidebar
st.sidebar.title('SOFIA')

# Create selection buttons for the 2 main functions:
# 1) airline recommendation for trip planning
# 2) flight delay risk for given trip
with st.sidebar:
    selected = option_menu(
        menu_title = "Main Menu",
        options=["Home","Airline Recommender", "Flight Delay Risk Pred."],
        icons = ["house","activity", "gear"],
        # menu_icon = "cast",
        default_index = 0
    )

    # Toggle to specify whether active API will be used to pull flight data
    flight_api = st.toggle('Activate Flight API')
    # st.write(flight_api)

####################################################
# Initialize custom class objects for model scoring
####################################################

 # Initialize data prep class object for modeling scoring. 
 # This class will extract the data according to user inputs. (API or embedded test data)
 # This class will also transform the data to a form scorable by the models
prep_data_obj = scoring.flight_data_prep(flight_api)

# Initialize class containing all model objects for model scoring.
scoring_obj = scoring.delay_predictions()

# Initialize class for getting sentiment scores for airlines
sentiment_obj = scoring.sentiment_scoring()


#.....................................................
# Create content for Airline Recommendation Dashboard
#.....................................................

if selected == "Airline Recommender":

    # Dashboard title
    st.title("Airline Recommendation Dashboard")

    # Add space between title and content using markdown and CSS
    st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)  # Adds extra space

    # Add sider bar section for user inputs
    st.sidebar.header('Flight Criterion')
    dep_city = st.sidebar.text_input('Departure City', value='Chicago, IL')
    dest_city = st.sidebar.text_input('Destination City', value='New York, NY')
    dep_dt = st.sidebar.date_input('Departure Date')


    # Generate dashboard info based on user inputs
    if st.sidebar.button('Recommend Airline'): 
        if dep_city and dest_city and dep_dt:
            
            # Get all airports based on cities inputted
            dep_airports = prep_data_obj.get_airports(dep_city)
            dest_airports = prep_data_obj.get_airports(dest_city)

            # Get flight data for all flight routes from departing and destination airports above
            flight_offers = pd.DataFrame()
            for i in dep_airports:
                for j in dest_airports:
                    df = prep_data_obj.get_flight_data(i, j, dep_dt)
                    flight_offers = pd.concat([flight_offers, df], ignore_index=True)

            # Score the resulting flight data using constructed models
            flight_offers['airline_delay_prob'] = scoring_obj.score_airline_delay_classification(flight_offers)
            flight_offers['airline_delay_mins'] = scoring_obj.score_airline_delay_length(flight_offers)

            # Transform model scores to 1-10 scale for intuitiveness
            flight_offers['airline_exp_delay'] = flight_offers['airline_delay_prob'] * flight_offers['airline_delay_mins'] # Calculated expected delay for airlines
            
            # Scale
            a = 1
            b = 10
            
            # Domain
            # This is basically the max and min delay expectations to score the best(10) and worst(1) scores
            # Can be changed
            x_min = 2
            x_max = 50

            # Derive delay score based on specified scale and domains
            flight_offers['airline_delay_score'] = a + (b - a) * (x_max - flight_offers['airline_exp_delay']) / (x_max - x_min)

            
            # Get sentiment scores for airlines
            flight_offers['Sentiment_Score'] = sentiment_obj.get_sentiment_scores(flight_offers['Reporting_Airline'])


            # st.write(flight_offers)


            if not flight_offers.empty:
                # Find airline with highest composite score
                best_airline = flight_offers.loc[flight_offers['airline_delay_score'].idxmax()]

                # Custom font size for column headers and values
                header_style = "font-size:20px; font-weight:normal; color: black; text-align: center; text-decoration: underline;"
                score_style = "font-size:28px; font-weight:bold; color: green; text-align: center;"
                delay_style = "font-size:28px; font-weight:bold; color: red; text-align: center;"

                # Show airline recommendation
                st.markdown(f"<p style='{header_style}'>Recommended Airline:</p>", unsafe_allow_html=True)
                # st.markdown(f"<p style='{header_style}'>{best_airline['Reporting_Airline']}</p>", unsafe_allow_html=True)
                st.image(f"./carrier_logos/{best_airline['Reporting_Airline']}.png", use_column_width=True)

                # Create 2 columns
                col1, col2 = st.columns(2)

                # Display results in separate columns with custom font sizes
                with col1:
                    st.markdown(f"<p style='{header_style}'>Flight Delay Risk Score</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='{score_style}'>{round(best_airline['airline_delay_score'],1)}</p>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<p style='{header_style}'>Sentiment Score</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='{score_style}'>{round(best_airline['Sentiment_Score'],1)}</p>", unsafe_allow_html=True)

                with col1:
                    st.markdown(f"<p style='{header_style}'>Flight Delay Probability</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='{delay_style}'>{round(best_airline['airline_delay_prob']*100,0)}%</p>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<p style='{header_style}'>Expected Delay Minutes</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='{delay_style}'>{round(best_airline['airline_delay_mins'],0)}</p>", unsafe_allow_html=True)
                
                # Display filtered data as a table
                st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)  # Adds extra space
                st.markdown("<h5>All Airline Options:</h5>", unsafe_allow_html=True)

                fltrd_data = flight_offers.sort_values(by='airline_delay_score', ascending=False)
                fltrd_data['Airline'] = fltrd_data['Reporting_Airline']
                fltrd_data['Delay Prob.'] = fltrd_data['airline_delay_prob'].apply(lambda x: f"{round(x*100, 0)}%")
                fltrd_data['Exp. Delay Min.'] = round(fltrd_data['airline_delay_mins'],0)
                fltrd_data['Delay Risk Score'] = round(fltrd_data['airline_delay_score'],1)
                fltrd_data['Sentiment Score'] = round(fltrd_data['Sentiment_Score'],1)
                display_df = fltrd_data[['Airline','Delay Prob.','Exp. Delay Min.','Delay Risk Score','Sentiment Score']]

                # st.table(display_df.reset_index(drop=True))  # Alternative: use st.table(fltrd_data) for a static table
                st.markdown(display_df.to_html(index=False), unsafe_allow_html=True)


            else:
                st.write(f"No Airline recommendations found for given criterion")

    else:
        # Placeholder for airline recommendations
        st.write("Please enter all search criterion to get recommendation.")


#.............................................................
# Create content for flight delay risk notification Dashboard
#.............................................................

if selected == "Flight Delay Risk Pred.":

    # Dashboard title
    st.title("Flight Delay Risk Dashboard")

    # Add space between title and content using markdown and CSS
    st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)  # Adds extra space

    # Add sider bar section for user inputs
    st.sidebar.header('Flight Information')
    airline = st.sidebar.text_input('Airline', value='UA')
    dep_city = st.sidebar.text_input('Departure City', value='Chicago, IL')
    dest_city = st.sidebar.text_input('Destination City', value='New York, NY')
    dep_dt = st.sidebar.date_input('Departure Date')

    # Store flight offers in session state to avoid re-fetching
    if 'flight_offers_df' not in st.session_state:
        st.session_state.flight_offers_df = pd.DataFrame()

    # Only call API if all inputs are provided and a new search is triggered
    if st.sidebar.button('Search Flights'): 
        if dep_city and dest_city and dep_dt:

            # Get all airports based on cities inputted
            dep_airports = prep_data_obj.get_airports(dep_city)
            dest_airports = prep_data_obj.get_airports(dest_city)

            # Get flight data for all flight routes from departing and destination airports above
            flight_offers = pd.DataFrame()
            for i in dep_airports:
                for j in dest_airports:
                    df = prep_data_obj.get_flight_data(i, j, dep_dt)
                    flight_offers = pd.concat([flight_offers, df], ignore_index=True)

            # Score the resulting flight data using constructed models
            flight_offers['overall_delay_prob'] = scoring_obj.score_overall_classification(flight_offers)
            flight_offers['overall_delay_mins'] = scoring_obj.score_overall_delay_length(flight_offers)

            st.session_state.flight_offers_df = flight_offers
            

    # Display the flight offers only if there is data in session state
    if not st.session_state.flight_offers_df.empty:
        flight_offers_df = st.session_state.flight_offers_df

        # st.write(orig_airports)
        # st.write(dest_airports)
        # st.write(dep_dt)
        # st.write(st.session_state.flight_offers_df)
        
        # Convert each flight offer to a string for display
        flight_offers_df['display'] = (
            flight_offers_df['orig_airline'] + " " +
            flight_offers_df['flight_nbr'].astype(str) + " - " +
            flight_offers_df['origin'] + " to " +
            flight_offers_df['destination'] + " | Dep: " +
            flight_offers_df['departure_time'].astype(str)
        )

        # Allow user to select a specific flight
        options = ['Select a Flight'] + flight_offers_df['display'].tolist()
        selected_flight = st.selectbox("Select a Flight", options)

        # Show detailed info only if a specific flight is selected
        if selected_flight != "Select a Flight":
            st.markdown("### Flight Details")
            flight_info = flight_offers_df[flight_offers_df['display'] == selected_flight]

            if not flight_info.empty:
                st.write({
                "Flight Number": flight_info['flight_nbr'].values[0],
                "Origin": flight_info['origin'].values[0],
                "Origin Airline": flight_info['orig_airline'].values[0],
                "Destination": flight_info['destination'].values[0],
                "Destination Airline": flight_info['dest_airline'].values[0],
                "Departure Time": flight_info['departure_time'].values[0],
                "Arrival Time": flight_info['arrival_time'].values[0],
                "Number of Legs": flight_info['legs'].values[0]
                })

    else:
        # Placeholder for airline recommendations
        st.write("Please enter all flight information to get flight delay risk assessment.")


# Run Streamlit App locally
# streamlit run flight_planning_webapp.py  --> Run this in the TERMINAL