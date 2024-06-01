import joblib

model = joblib.load("/home/kira/Documents/Placement_Prediction/PlacementModel.joblib")

attributes = dir(model)
print(attributes)