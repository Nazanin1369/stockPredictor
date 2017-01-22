import dataReader as dr
import matplotlib
import matplotlib.pyplot as plt
from pandas.tools.plotting import parallel_coordinates

matplotlib.style.use('ggplot')

df = dr.getDataFrame();

pdf = df.cumsum()

df.plot(legend=False)
