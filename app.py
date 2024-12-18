import streamlit as st
import pandas as pd
from preprocessing.cleaning_data import DataCleaner
import model.model as m
import preprocessing.cleaning_data as cl
import predict.prediction as pred

model = m.XGBoostModel()
model.load_model('model/immoprice_model.json')

data = cl.DataCleaner('geo_prosp_data.csv')
# Streamlit App
st.set_page_config(layout="centered")

st.title("Immo Price Prediction")
st.write("""Welcome to Immo Price Prediction app!
         This app allows you to predict the price of a 
         property in Belgium. Please fill in the 
         necessary data to proceed:
     """)


# Check for session state variables
if "property_type" not in st.session_state:
    st.session_state.property_type = None
    st.session_state.property_subtype = None

def main():

    property_type = None
    # First form: Property Type Selection
    if st.session_state.property_type is None:
        # Create the property type selection form
        with st.form(key='property_selection_form'):
            property_type = st.radio("Select Type of Property", ["House", "Apartment"])
            submit_property_button = st.form_submit_button(label="Submit Property Type")

            if submit_property_button:
                st.session_state.property_type = property_type  # Store selected property type
                st.success(f"You selected: "
                           f"{property_type}. Please fill "
                           f"out the remaining details.")

    # Second form: Additional Property Details
    if st.session_state.property_type is not None:
        with (st.form(key='property_details_form')):
            property_subtype = None
            # Display the property subtype selection in the second form only
            if st.session_state.property_type == "House":
                property_subtype = st.selectbox("Select House Subtype", ['house', 'villa', 'town-house', 'apartment-block', 'mansion', 'manor-house', 'mixed-use-building', 'bungalow', 'country-cottage', 'exceptional-property', 'other-property', 'farmhouse', 'castle', 'chalet', 'pavilion'])
            elif st.session_state.property_type == "Apartment":
                property_subtype = st.selectbox("Select Apartment Subtype", ['apartment', 'penthouse', 'triplex', 'loft', 'duplex', 'ground-floor', 'service-flat', 'flat-studio', 'kot'])

            # Number input fields corresponding to the model's expected inputs
            zip_code = st.text_input(
                "Enter Zip Code (4 digits):", max_chars=4)
            livable_surface = st.number_input(
                "Enter Livable Space (m²):", min_value=1,
                value=1, step=1)
            num_rooms = st.number_input(
                "Enter Number of Rooms:", min_value=1,
                value=1)

            fully_equipped_kitchen = st.selectbox(
                "Fully Equipped Kitchen?", ["No", "Yes"])
            furnished = st.selectbox("Furnished?",
                                     ["No", "Yes"])
            has_fireplace = st.selectbox("Fireplace?",
                                         ["No", "Yes"])
            terrace_area = st.number_input(
                "Terrace Area (m²):", min_value=0, value=0,
                step=1)
            garden_area = st.number_input(
                "Garden Area (m²):", min_value=0, value=0,
                step=1)
            swimming_pool = st.selectbox("Swimming Pool?",
                                         ["No", "Yes"])
            number_of_facades = st.number_input(
                "Number of Facades:", min_value=1,
                value=1, step=1)

            primary_energy_consumption = st.number_input(
                "Primary Energy Consumption (kWh/m²):",
                min_value=0, value=0, step=1)
            construction_year = st.number_input(
                "Construction year (format: YYYY, "
                "min.:1700, max:2030):",
                min_value=1700,
                max_value=2030, value=2000)

            # Building condition PEB options
            building_condition = st.selectbox(
                "Select Building Condition",
                ["Good", "To be renovated",
                 "Just renovated", "To be done up",
                 "To restore"])

            # Dummy inputs for PEB (Primary Energy Certification) ratings
            peb_rating = st.selectbox("PEB Rating",
                                      ["Unknown", "A", "B",
                                       "C",
                                       "D", "E",
                                       "F", "G"])

            # Submit button for the details form
            details_submit_button = st.form_submit_button(
                label="Submit & Predict")

            if details_submit_button:
                if int(zip_code) not in data.valid_zip_codes:
                    st.error(
                        "Invalid zip code provided. Please enter a valid zip code.")
                else:
                # Validate inputs
                    form_data = [
                        zip_code, num_rooms,
                        livable_surface,
                        fully_equipped_kitchen, furnished,
                        has_fireplace, terrace_area,
                        garden_area,
                        swimming_pool, number_of_facades,
                        primary_energy_consumption,
                        construction_year,
                        peb_rating, building_condition,
                        property_type, property_subtype
                    ]
                    # Call the prepare_data method
                    prepared_data = data.preprocess(
                        form_data, data)
                    print(f"Prepared data: {prepared_data}")

                    # Create the DataFrame
                    df = pd.DataFrame(prepared_data, index=[0])

                    # Make prediction
                    predicted_price = model.predict(df)[0]
                    predicted_price = round(predicted_price, 2)
                    # Display predicted price
                    st.success(
                        f"The predicted price for this "
                        f"property is: "
                        f"{predicted_price:,.2f} "
                        f"EUR")


if __name__ == "__main__":
    main()

