import model as model
# from preprocessing import preproc
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

my_model = model.ModelWrapper('../weights/v3.cbm')
#test=pd.read_json('train_data.json').loc[[0,0]]
test=pd.read_json('train_data.json')
test=pd.DataFrame([test])
print(test)
# print(test.columns)
# print(test)


