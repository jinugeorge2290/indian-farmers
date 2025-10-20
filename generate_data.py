import pandas as pd
import numpy as np

# Parameters
num_rows = 300000  # ~1 million rows for ~100 MB CSV
np.random.seed(42)

# Lists of Indian states and sample districts
states = ['Maharashtra','Punjab','Karnataka','Tamil Nadu','Uttar Pradesh','Madhya Pradesh','Rajasthan','Bihar','Andhra Pradesh','Gujarat']
districts = {
    'Maharashtra': ['Pune','Ahmednagar','Nagpur','Nashik','Satara'],
    'Punjab': ['Ludhiana','Amritsar','Patiala','Jalandhar','Bathinda'],
    'Karnataka': ['Bangalore','Mysore','Mangalore','Belgaum','Hubli'],
    'Tamil Nadu': ['Chennai','Coimbatore','Madurai','Salem','Tirunelveli'],
    'Uttar Pradesh': ['Lucknow','Kanpur','Varanasi','Agra','Meerut'],
    'Madhya Pradesh': ['Bhopal','Indore','Gwalior','Jabalpur','Ujjain'],
    'Rajasthan': ['Jaipur','Jodhpur','Udaipur','Bikaner','Ajmer'],
    'Bihar': ['Patna','Gaya','Bhagalpur','Muzaffarpur','Darbhanga'],
    'Andhra Pradesh': ['Vijayawada','Visakhapatnam','Guntur','Tirupati','Kurnool'],
    'Gujarat': ['Ahmedabad','Surat','Vadodara','Rajkot','Bhavnagar']
}

crops = ['Wheat','Rice','Cotton','Sugarcane','Maize','Groundnut','Soybean']

# Generate random base data
state_col = np.random.choice(states, num_rows)
year_col = np.random.randint(2010, 2025, num_rows)
crop_col = np.random.choice(crops, num_rows)
loan_amount = np.random.randint(20000, 150000, num_rows)
rainfall = np.random.randint(400, 1500, num_rows)
temp = np.random.randint(20, 35, num_rows)
irrigated_land = np.random.randint(50, 500, num_rows)
farmer_income = np.random.randint(50000, 300000, num_rows)
poverty_rate = np.random.uniform(5, 40, num_rows)

# Introduce correlation: Total_Suicides
# Base suicides
suicides = np.random.poisson(lam=50, size=num_rows)

# Increase suicides for:
# - High debt relative to income
# - Low rainfall
# - Low irrigated land
debt_ratio = loan_amount / farmer_income
suicides = suicides + ((debt_ratio > 0.5)*np.random.randint(20,50,num_rows))
suicides = suicides + ((rainfall < 600)*np.random.randint(10,40,num_rows))
suicides = suicides + ((irrigated_land < 150)*np.random.randint(5,30,num_rows))

# Add some random noise
suicides = suicides + np.random.randint(-5, 5, num_rows)
suicides = np.clip(suicides, 0, None)  # Ensure no negative suicides

# Assign districts based on state
district_col = [np.random.choice(districts[state]) for state in state_col]

# Create DataFrame
df = pd.DataFrame({
    'State': state_col,
    'District': district_col,
    'Year': year_col,
    'Total_Suicides': suicides,
    'Crop_Type': crop_col,
    'Loan_Amount': loan_amount,
    'Rainfall': rainfall,
    'Temp': temp,
    'Irrigated_Land': irrigated_land,
    'Farmer_Income': farmer_income,
    'Poverty_Rate': poverty_rate
})

# Save to CSV
df.to_csv('farmer_suicide_large_realistic_train.csv', index=False)
print("CSV file 'farmer_suicide_large_realistic_train.csv' generated successfully!")
