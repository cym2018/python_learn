import numpy as np
import matplotlib.pyplot as plt

x = ["a","b","c","d","e"]
y = [1,2,3,4,5]
fig, ax = plt.subplots()
ax.set_xlabel("scenario name")
ax.set_ylabel("diff count")
ax.bar(x, y)