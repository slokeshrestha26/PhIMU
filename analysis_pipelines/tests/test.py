import pandas as pd
import numpy as np

# Create a list of participant names
participants = ['Alice', 'Bob', 'Charlie', 'Dave', 'Eve']

# Create a dummy dataframe with participant names and some numerical data
df = pd.DataFrame({
    'participant': np.random.choice(participants, size=10),
    'score': np.random.randint(0, 100, size=10),
    'age': np.random.randint(18, 50, size=10),
})

print(df)
