from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
import pandas as pd


# Load from HuggingFace (same way shown on their page)
dataset = load_dataset("Tobi-Bueck/customer-support-tickets")

# Convert to pandas DataFrame
df = pd.DataFrame(dataset['train'])

# Filter only English language tickets
df_en = df[df['language'] == 'en'].reset_index(drop=True)

# Save for reuse
#df_en.to_csv('/Users/aanchala/Downloads/Task4/data/english_tickets.csv', index=False)

print(f"Total tickets: {len(df)}")
print(f"English tickets: {len(df_en)}")


print(df_en.shape)
print(df_en.columns.tolist())
print(df_en.dtypes)
print(df_en.isnull().sum())
print(df_en.head())


#Step 4: Explore the structure

 # Data types and missing values
print(df.dtypes)
print("\n--- Missing Values ---")
print(df.isnull().sum())
print("\n--- Sample row ---")
df.iloc[0]



