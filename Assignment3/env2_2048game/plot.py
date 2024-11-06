import matplotlib.pyplot as plt

# Tile values and their counts
tiles = [8, 64, 128, 256, 512, 1024]
counts = [2, 3, 7, 26, 57, 5]

# Create a bar chart
plt.bar(range(len(tiles)), counts, width=0.5, color='teal')

# Set the x-axis labels to the tile values
plt.xticks(range(len(tiles)), tiles)

# Add labels and title
plt.xlabel('Tile Index')
plt.ylabel('Count')
plt.title('Tile Distribution in 2048')

# Display the plot
plt.show()
