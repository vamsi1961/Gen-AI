import pandas as pd
import numpy as np
# Create a random dataframe of size (3,4)
df = pd.DataFrame(np.random.rand(3, 4))
# Half the size of the dataframe
df_half = df.sample(frac=0.5)
# Get the shape of the halved dataframe
shape = df_half.shape
print(shape)
