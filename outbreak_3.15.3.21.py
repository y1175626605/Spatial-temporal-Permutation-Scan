import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Actual event generation icon generation
data1 = r".\temp\39\outbreak_3.15-3.21.xlsx"
data2 = r".\temp\39\outbreak_1.15-1.21.xlsx"

# Read file
df1 = pd.DataFrame(pd.read_excel(data1))
df2 = pd.DataFrame(pd.read_excel(data2))

# Combine the values corresponding to z_id
df1 = df1.groupby(['z_id'], as_index=False).sum()
df2 = df2.groupby(['z_id'], as_index=False).sum()

#print('df')
#print(df)

def df_process(df):
    # Store index value, initialize
    index = []
    # Store value value, initialize
    v = []

    for i in range(1,43):
        index.append(i)
        a = df[(df["z_id"] == i)]["num"].tolist()  # Select a value that satisfies the condition

        if a == []:  # If there is no data, fill it with 0
            v.append(0)
        else:
            v.append(int(a[0]))  # Have data filled with num
    return index,v

index1,v1 = df_process(df1)
index2,v2 = df_process(df2)

# Create a window with 8 x 6 points and set the resolution to 80 pixels per inch
plt.figure(figsize=(8, 6), dpi=80)

plt.subplot(211)
# The width of the column
width = 0.5
# Draw a histogram, each column is violet in color
p2 = plt.bar(index2, v2, width, label="case", color="#6495ED")
# Set the horizontal axis label
plt.xlabel('locations')
# Set the vertical axis label
plt.ylabel('observed case')
# Add title
plt.title('1.15-1.21 outbreak case')
# # Add a scale for the vertical and horizontal axes
plt.xticks(np.arange(0, 42, 2))
# plt.yticks(np.arange(0, 81, 10))
# Add a legend
plt.legend(loc="upper right")

plt.subplots_adjust(hspace=0.4)

plt.subplot(212)
#pl.legend(bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4)
# The width of the column
width = 0.5
# Draw a histogram, each column is violet in color
p2 = plt.bar(index1, v1, width, label="case", color="#9F79EE")

# Set the horizontal axis label
plt.xlabel('locations')
# Set the vertical axis label
plt.ylabel('observed case')

# Add title
plt.title('3.8-3.15 outbreak case')

# Add a scale for the vertical and horizontal axes
plt.xticks(np.arange(0, 42, 2))
# plt.yticks(np.arange(0, 81, 10))

# Add a legend
plt.legend(loc="upper right")

plt.show()
