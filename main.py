import pandas as pd
import numpy as np

# Updated list of column names to include all relevant columns
column_names_facebook = ['domain', 'address', 'categories', 'city', 'country_code', 'country_name',
                'description', 'email', 'link', 'name', 'page_type', 'phone',
                'phone_country_code', 'region_code', 'region_name', 'zip_code']


column_names_google = ['address', 'category', 'city', 'country_code', 'country_name', 'name',
       'phone', 'phone_country_code', 'raw_address', 'raw_phone',
       'region_code', 'region_name', 'text', 'zip_code', 'domain']


column_names_website = ['root_domain', 'domain_suffix', 'language', 'legal_name', 'main_city',
                        'main_country', 'main_region', 'phone', 'site_name', 'tld', 's_category']


# Function to handle bad lines and fill them with NaN
def fill_bad_lines_with_nan_facebook(bad_line):
    return [np.nan] * len(column_names_facebook)  # Match NaN length to the number of columns

# Function to handle bad lines and fill them with NaN
def fill_bad_lines_with_nan_google(bad_line):
    return [np.nan] * len(column_names_google)  # Match NaN length to the number of columns

# Read the first CSV
df_facebook = pd.read_csv('datasets/facebook_dataset.csv',
                  names=column_names_facebook,
                  on_bad_lines=fill_bad_lines_with_nan_facebook,
                  engine='python',
                  quotechar='"')

# Read the second CSV
df_google = pd.read_csv('datasets/google_dataset.csv',
                  names=column_names_google,
                  on_bad_lines=fill_bad_lines_with_nan_google,
                  engine='python',
                  quotechar='"')

# Read the third CSV
df_website = pd.read_csv('datasets/website_dataset.csv',
                  names=column_names_website,
                  delimiter=';')

# Set Pandas to display all columns
pd.set_option('display.max_columns', None)

# # Display
# print(df_facebook.head(20))
# print(df_facebook.columns)
#
# # Display
# print(df_google.head(20))
# print(df_google.columns)
#
# # Display
# print(df_website.head(20))
# print(df_website.columns)

# Number of rows in each dataset
total_rows_facebook = len(df_facebook)
total_rows_google = len(df_google)
total_rows_website = len(df_website)

# Count the number of NaN/None values in each column of the dataset
nan_counts_facebook = df_facebook.isnull().sum()
nan_counts_google = df_google.isnull().sum()
nan_counts_website = df_website.isnull().sum()

# Calculate the percentage of NaN values per column
nan_percentage_facebook = (nan_counts_facebook / total_rows_facebook) * 100
nan_percentage_google = (nan_counts_google / total_rows_google) * 100
nan_percentage_website = (nan_counts_website / total_rows_website) * 100

# Total number of NaN values in each dataset
total_nan_facebook = nan_counts_facebook.sum()
total_nan_google = nan_counts_google.sum()
total_nan_website = nan_counts_website.sum()

# Display the counts and percentages for each dataset
print("NaN counts and percentages per column in Facebook dataset:")
print(pd.DataFrame({'NaN Count': nan_counts_facebook, 'NaN Percentage': nan_percentage_facebook}))
print(f"Total NaN values in Facebook dataset: {total_nan_facebook} ({(total_nan_facebook / (total_rows_facebook * len(df_facebook.columns))) * 100:.2f}% of all values)\n")

print("NaN counts and percentages per column in Google dataset:")
print(pd.DataFrame({'NaN Count': nan_counts_google, 'NaN Percentage': nan_percentage_google}))
print(f"Total NaN values in Google dataset: {total_nan_google} ({(total_nan_google / (total_rows_google * len(df_google.columns))) * 100:.2f}% of all values)\n")

print("NaN counts and percentages per column in Website dataset:")
print(pd.DataFrame({'NaN Count': nan_counts_website, 'NaN Percentage': nan_percentage_website}))
print(f"Total NaN values in Website dataset: {total_nan_website} ({(total_nan_website / (total_rows_website * len(df_website.columns))) * 100:.2f}% of all values)\n")

"""
Facebook has the highest percentage of missing data per column, followed by Google, and then the Website CSV.

Since we consider all data to be equally trustworthy, we will start with Facebook, merge it with Google,
clean the data and then merge it with the Website dataset, then clean the result. 

Like this, we will obtain data of high granularity, which is practical to use.
"""


"""
To join the datasets, we need a common key.
Since all datasets contain a domain or root domain field, this is the best option.
"""

# Step 1: Perform an outer join between Facebook and Google datasets on 'domain'
df_facebook_google = pd.merge(df_facebook, df_google, on='domain', suffixes=('_fb', '_g'), how='outer')

# Step 2: Combine common columns from Facebook and Google
df_facebook_google['final_address'] = df_facebook_google['address_fb'].combine_first(df_facebook_google['address_g'])
df_facebook_google['final_city'] = df_facebook_google['city_fb'].combine_first(df_facebook_google['city_g'])
df_facebook_google['final_country'] = df_facebook_google['country_name_fb'].combine_first(df_facebook_google['country_name_g'])
df_facebook_google['final_region'] = df_facebook_google['region_name_fb'].combine_first(df_facebook_google['region_name_g'])
df_facebook_google['final_zip_code'] = df_facebook_google['zip_code_fb'].combine_first(df_facebook_google['zip_code_g'])
df_facebook_google['final_phone'] = df_facebook_google['phone_fb'].combine_first(df_facebook_google['phone_g'])
df_facebook_google['final_category'] = df_facebook_google['categories'].combine_first(df_facebook_google['category'])
df_facebook_google['final_name'] = df_facebook_google['name_fb'].combine_first(df_facebook_google['name_g'])

# Step 3: Create a new DataFrame with only the necessary columns
df_facebook_google_cleaned = df_facebook_google[['domain', 'final_address', 'final_city', 'final_country', 'final_region',
                                                 'final_zip_code', 'final_phone', 'final_category', 'final_name']]

# # Display
# print(df_facebook_google_cleaned.head(20))
# print(df_facebook_google_cleaned.columns)

# Step 4: Perform an outer join between df_facebook_google_cleaned and df_website
df_final = pd.merge(df_facebook_google_cleaned, df_website, left_on='domain', right_on='root_domain', how='outer')

# Step 5: Combine common columns from df_facebook_google_cleaned and df_website
df_final['final_address'] = df_final['final_address'].combine_first(df_final['main_city'])
df_final['final_city'] = df_final['final_city'].combine_first(df_final['main_city'])
df_final['final_country'] = df_final['final_country'].combine_first(df_final['main_country'])
df_final['final_region'] = df_final['final_region'].combine_first(df_final['main_region'])
df_final['final_phone'] = df_final['final_phone'].combine_first(df_final['phone'])
df_final['final_category'] = df_final['final_category'].combine_first(df_final['s_category'])
df_final['final_name'] = df_final['final_name'].combine_first(df_final['legal_name'])

# Step 6: Combine domain/root_domain for final 'domain' column
df_final['final_domain'] = df_final['domain'].combine_first(df_final['root_domain'])

# Step 7: Create a new DataFrame with only the necessary columns
df_final_cleaned = df_final[['final_domain', 'final_address', 'final_city', 'final_country', 'final_region',
                             'final_zip_code', 'final_phone', 'final_category', 'final_name']]

# Updated function to check if a string looks like a valid domain
def is_valid_domain(domain):
    if pd.isnull(domain):
        return False
    # Simple check: valid domains contain at least one period, no spaces, and are not numeric-only
    return '.' in domain and not domain.isnumeric() and ' ' not in domain

# Step 8: Clean rows where the domain is incorrect
df_final_cleaned = df_final_cleaned[df_final_cleaned['final_domain'].apply(is_valid_domain)]

# Display the final cleaned DataFrame
print(df_final_cleaned.head(20))

print(df_final_cleaned.columns)

# print(df_final_cleaned.tail(20))