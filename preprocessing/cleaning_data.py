import pandas as pd
import streamlit as st

class DataCleaner:
    def __init__(self, geo_data):
        self.geo_data = pd.read_csv(geo_data)
        self.valid_zip_codes = self.valid_zips(
            self.geo_data)
    def valid_zips(self, geo_data):
        # Convert relevant columns to integers
        try:
            geo_data['Post code'] = geo_data[
                'Post code'].astype(int)
            geo_data['Municipality code'] = geo_data[
                'Municipality code'].astype(int)
            geo_data['Prosperity index'] = geo_data[
                'Prosperity index'].astype(int)
        except ValueError as e:
            st.error(
                f"Error converting columns to integers: {e}")

        # Create a set of valid zip codes
        valid_zip_codes = set(
            geo_data['Post code'].unique())

        return valid_zip_codes

    def preprocess(self, form_data, geo_data):

        (zip_code, num_rooms, livable_surface,
         fully_equipped_kitchen,
         furnished, has_fireplace, terrace_area,
         garden_area,
         swimming_pool, number_of_facades,
         primary_energy_consumption,
         construction_year, peb_rating, building_condition,
         property_type,
         property_subtype) = form_data

        # Validate zip code
        if int(zip_code) not in self.valid_zip_codes:
            st.error("Invalid zip code provided.")
            return None

        # Get municipality code and prosperity index
        municipality_code = self.geo_data.loc[
                self.geo_data['Post code'] == int(zip_code),
                'Municipality code'
            ].values

        municipality_code = municipality_code[
                0] if municipality_code.size > 0 else None

        prosperity_index = self.geo_data.loc[
                self.geo_data[
                    'Municipality code'] == municipality_code,
                'Prosperity index'
            ].values

        prosperity_index = prosperity_index[
                0] if prosperity_index.size > 0 else None

        data = {
            'Zip Code': int(
                zip_code) if zip_code.isdigit() else None,
            'Number of Rooms': int(num_rooms),
            'Livable Space (m2)': float(
                livable_surface),
            'Fully Equipped Kitchen': 1 if fully_equipped_kitchen == "Yes" else 0,
            'Furnished': 1 if furnished == "Yes" else 0,
            'Fireplace': 1 if has_fireplace == "Yes" else 0,
            'Terrace': 1 if terrace_area > 0 else 0,
            'Terrace Area (m2)': float(
                terrace_area),
            'Garden': 1 if garden_area > 0 else 0,
            'Garden Area (m2)': float(
                garden_area),
            'Swimming Pool': 1 if swimming_pool == "Yes" else 0,
            'Number of Facades': int(
                number_of_facades),
            'Primary Energy Consumption ('
            'kWh/m2)': float(primary_energy_consumption),
            'Building Age': 2024 - int(
                construction_year),
            # PEB
            'PEB_B': 1 if peb_rating == "A" else 0,
            'PEB_B': 1 if peb_rating == "B" else 0,
            'PEB_C': 1 if peb_rating == "C" else 0,
            'PEB_D': 1 if peb_rating == "D" else 0,
            'PEB_E': 1 if peb_rating == "E" else 0,
            'PEB_F': 1 if peb_rating == "F" else 0,
            'PEB_G': 1 if peb_rating == "G" else 0,
            # State of the Building
            'State of the Building_Good': 1 if building_condition == "Good" else 0,
            'State of the Building_Just renovated': 1 if building_condition == "Just renovated" else 0,
            'State of the Building_To be done up': 1 if building_condition == "To be done up" else 0,
            'State of the Building_To renovate': 1 if building_condition == "To renovate" else 0,
            'State of the Building_To restore': 1 if building_condition == "To restore" else 0,
            # Type
            'Type of Property_House': 1 if property_type
                                        == "House" else 0,
            # Subtypes
            'Subtype of Property_apartment-block': 0,
            'Subtype of Property_bungalow': 0,
            'Subtype of Property_castle': 0,
            'Subtype of Property_chalet': 0,
            'Subtype of Property_country-cottage': 0,
            'Subtype of Property_duplex': 0,
            'Subtype of Property_exceptional-property': 0,
            'Subtype of Property_farmhouse': 0,
            'Subtype of Property_flat-studio': 0,
            'Subtype of Property_ground-floor': 0,
            'Subtype of Property_house': 0,
            'Subtype of Property_kot': 0,
            'Subtype of Property_loft': 0,
            'Subtype of Property_manor-house': 0,
            'Subtype of Property_mansion': 0,
            'Subtype of Property_mixed-use-building': 0,
            'Subtype of Property_other-property': 0,
            'Subtype of Property_pavilion': 0,
            'Subtype of Property_penthouse': 0,
            'Subtype of Property_service-flat': 0,
            'Subtype of Property_town-house': 0,
            'Subtype of Property_triplex': 0,
            'Subtype of Property_villa': 0,
            'Municipality code': municipality_code,
            'Prosperity index': prosperity_index,
        }
        selected_subtype = property_subtype

        # Switch the selected subtype to 1
        if selected_subtype in data:
            data[
                f'Subtype of Property_{selected_subtype}'] = 1

        return data
